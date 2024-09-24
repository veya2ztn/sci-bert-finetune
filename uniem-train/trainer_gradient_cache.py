from transformers.trainer import *
from transformers.trainer_pt_utils import distributed_concat
from tqdm.auto import tqdm
try:
    from optimizer.sophia import SophiaG
except:
    pass
from numpy.linalg import norm
def dummy_tqdm(x,*args,**kargs):return x

def get_local_rank():
    local_rank = int(os.environ["LOCAL_RANK"]) if "LOCAL_RANK" in os.environ else 0
    rank = int(os.environ["RANK"]) if "RANK" in os.environ else 0
    return local_rank + rank



class RealtimeEmbeddingTrainer(Trainer):
    def __init__(self, *args, knowledge_buffer=None,**kargs):
        super().__init__(*args, **kargs)
        self.knowledge_buffer_index = self.knowledge_buffer_embedding = self.extra_answer_needed_keys_runtime = None
        if knowledge_buffer is not None:
            self.knowledge_buffer_key2index, self.knowledge_buffer_embedding = knowledge_buffer
            self.knowledge_buffer_index2key = {v:k for k,v in self.knowledge_buffer_key2index.items()}
            self.knowledge_buffer_embedding= torch.from_numpy(self.knowledge_buffer_embedding).to(self.args.device)
        self.args.real_batch_size = self.args.per_device_train_batch_size  * self.accelerator.num_processes
        if self.args.eval_mode == 'precompute_embedding':
            self.extra_answer_needed_keys_runtime = []
    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        
        self.accelerator.free_memory()
        self._train_batch_size = batch_size
        logger.debug(f"Currently training with a batch size of: {self._train_batch_size}")
        
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = self._train_batch_size * args.gradient_accumulation_steps * args.world_size

        len_dataloader = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader 
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps and args.logging_steps < 1:
            args.logging_steps = math.ceil(max_steps * args.logging_steps)
        if args.eval_steps and args.eval_steps < 1:
            args.eval_steps = math.ceil(max_steps * args.eval_steps)
        if args.save_steps and args.save_steps < 1:
            args.save_steps = math.ceil(max_steps * args.save_steps)

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torch.distributed.launch)."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = (
            self.sharded_ddp is not None
            and self.sharded_ddp != ShardedDDPOption.SIMPLE
            or is_sagemaker_mp_enabled()
            or self.fsdp is not None
        )

        # We need to reset the scheduler, as its parameters may be different on subsequent calls
        if self._created_lr_scheduler:
            self.lr_scheduler = None
            self._created_lr_scheduler = False

        if self.is_deepspeed_enabled:
            self.optimizer, self.lr_scheduler = deepspeed_init(self, num_training_steps=max_steps)

        if not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None

        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps is not None:
            if args.logging_steps < 1:
                self.state.logging_steps = math.ceil(max_steps * args.logging_steps)
            else:
                self.state.logging_steps = args.logging_steps
        if args.eval_steps is not None:
            if args.eval_steps < 1:
                self.state.eval_steps = math.ceil(max_steps * args.eval_steps)
            else:
                self.state.eval_steps = args.eval_steps
        if args.save_steps is not None:
            if args.save_steps < 1:
                self.state.save_steps = math.ceil(max_steps * args.save_steps)
            else:
                self.state.save_steps = args.save_steps

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        model = self._wrap_model(self.model_wrapped)

        if is_sagemaker_mp_enabled() and resume_from_checkpoint is not None:
            self._load_from_checkpoint(resume_from_checkpoint, model)

        # as the model is wrapped, don't use `accelerator.prepare`
        # this is for unhandled cases such as
        # Fairscale Sharded DDP, FSDP-XLA, SageMaker MP/DP, DataParallel, IPEX
        use_accelerator_prepare = True if model is self.model else False

        if delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # prepare using `accelerator` prepare
        if use_accelerator_prepare:
            self.model.train()
            if hasattr(self.lr_scheduler, "step"):
                if self.use_apex:
                    model = self.accelerator.prepare(self.model)
                else:
                    model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
            else:
                # to handle cases wherein we pass "DummyScheduler" such as when it is specified in DeepSpeed config.
                model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                    self.model, self.optimizer, self.lr_scheduler
                )

        if self.is_fsdp_enabled:
            self.model = model

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        # backward compatibility
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model_wrapped

        # deepspeed ckpt loading
        if resume_from_checkpoint is not None and self.is_deepspeed_enabled:
            deepspeed_load_checkpoint(self.model_wrapped, resume_from_checkpoint)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model), etc.

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples:,}")
        logger.info(f"  Num Epochs = {num_train_epochs:,}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
        if self.args.per_device_train_batch_size != self._train_batch_size:
            logger.info(f"  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps:,}")
        logger.info(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch  = self.state.global_step % (num_update_steps_per_epoch)
                #steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first"
                    f" {steps_trained_in_current_epoch} batches in the first epoch."
                )

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        if self.hp_name is not None and self._trial is not None:
            # use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
            # parameter to Train when using DDP.
            self.state.trial_name = self.hp_name(self._trial)
        if trial is not None:
            assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not args.ignore_data_skip:
            for epoch in range(epochs_trained):
                for _ in train_dataloader:
                    break
        
        if self.args.create_embedding:
            assert self.args.embedding_offline_path
            self.create_offline_embedding()
            self.accelerator.print(f'save embedding to {self.args.embedding_offline_path}')
            self.accelerator.wait_for_everyone()
            exit()

        # do once metric computing
        if self.eval_dataset is not None: 
            metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)

        #################################################################################
        ########################### Main Loop ###########################################
        #################################################################################
        ProgressBar = tqdm if self.accelerator.is_main_process else dummy_tqdm
        total_batched_samples = 0
        real_gradient_accumulation_steps = args.gradient_accumulation_steps//2
        for epoch in range(epochs_trained, num_train_epochs):
            epoch_iterator = train_dataloader

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:self._past = None
            
            
            
            steps_in_epoch = args.gradient_accumulation_steps * len(epoch_iterator)
            
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            rng_to_sync = False
            steps_skipped = 0
            if steps_trained_in_current_epoch > 0:
                epoch_iterator = skip_first_batches(epoch_iterator, steps_trained_in_current_epoch)
                steps_skipped = steps_trained_in_current_epoch
                steps_trained_in_current_epoch = 0
                rng_to_sync = True

            step = -1
            
            for update_step, inputs in ProgressBar(enumerate(epoch_iterator), desc="run in dataset", position=1, leave=False):
                self.control = self.callback_handler.on_step_begin(args, self.state, self.control)
                ## one step will provide a batch of data
                assert len(inputs['question_ids'])%real_gradient_accumulation_steps == 0
                chunk_size = len(inputs['question_ids'])//real_gradient_accumulation_steps
                model.eval()
                with torch.no_grad():
                    question_index = inputs.pop('question_index')
                    answer_index   = inputs.pop('answer_index')

                    #### below code compute the embedding in each GPU and then dispatch to all GPU.
                    #### It is inefficient, since the answer may repeat and the computing will process multiple times.
                    #### TODO: we leave this feature update in the future.
                    whole_question_embedding_in_this_update= self.obtain_embedding(model, inputs['question_ids'], 'question') #(b, D)
                    whole_answer_embedding_in_this_update  = self.obtain_embedding(model, inputs['answer_ids'], 'answer') #(b, D)
                    
                    
                        
                    self.accelerator.wait_for_everyone()
                    question_index = distributed_concat(question_index )
                    answer_index   = distributed_concat(answer_index   )
                    whole_question_embedding_in_this_update = distributed_concat(whole_question_embedding_in_this_update) #(B, D) 
                    whole_answer_embedding_in_this_update   = distributed_concat(whole_answer_embedding_in_this_update) #(B, D)
                    self.accelerator.wait_for_everyone()
                    assert len(whole_question_embedding_in_this_update) == len(whole_answer_embedding_in_this_update)


                    if self.knowledge_buffer_embedding is not None:
                        ## Lets update the knowledge buffer, since the buffer is assume small (340M for unarchieve qh 8k, we shard it in each GPU)
                        ## the answer is just the knowlegde

                        answer_keys     = [self.train_dataset.answer_index2key[i.item()] for i in answer_index[:,0]]
                        knowledge_index = [self.knowledge_buffer_key2index[k] for k in answer_keys]
                        self.knowledge_buffer_embedding[knowledge_index] = whole_answer_embedding_in_this_update

                    whole_reference_question_embedding = whole_reference_answer_embedding = whole_reference_question_embedding = whole_reference_answer_embedding = None
                    if 'reference_question_embedding' in inputs:
                        whole_reference_question_embedding = inputs['reference_question_embedding'] #(b, D)
                        whole_reference_answer_embedding   = inputs['reference_answer_embedding'] #(b, D)

                        whole_reference_question_embedding = distributed_concat(whole_reference_question_embedding) #(B, D)
                        whole_reference_answer_embedding   = distributed_concat(whole_reference_answer_embedding) #(B, D)

                    assert whole_reference_question_embedding is not None, "so far please must use a reference"
                
                

                model.train() #<<--- don't forget 
                total_batched_samples += 1
                
                
                for accumulation_step in ProgressBar(range(real_gradient_accumulation_steps), desc="accumulation step", position=1, leave=False):
                    inputs_for_this_chunk = {k: v[accumulation_step*chunk_size:(accumulation_step+1)*chunk_size] for k, v in inputs.items()} #(b//L, D)
                    for split_idx, embeder_type in enumerate(['question','answer']):
                        inputs_for_this_type = {}
                        if embeder_type == 'question':
                            inputs_for_this_type['text_ids']                 = inputs_for_this_chunk['question_ids'] #(b//L, S)
                            inputs_for_this_type['conjugate_text_embedding'] = whole_answer_embedding_in_this_update #(B, D)
                            if 'reference_question_embedding' in inputs_for_this_chunk:
                                inputs_for_this_type['reference_text_embedding'] = inputs_for_this_chunk['reference_question_embedding']#(b//L, D)
                                inputs_for_this_type['reference_conjugate_text_embedding']= whole_reference_answer_embedding #(B, D)
                        elif embeder_type == 'answer':
                            inputs_for_this_type['text_ids']                 = inputs_for_this_chunk['answer_ids'] #(b//L, S)
                            inputs_for_this_type['conjugate_text_embedding'] = whole_question_embedding_in_this_update #(B, D)
                            if 'reference_answer_embedding' in inputs_for_this_chunk:
                                inputs_for_this_type['reference_text_embedding'] = inputs_for_this_chunk['reference_answer_embedding'] #(b//L, D)
                                inputs_for_this_type['reference_conjugate_text_embedding']= whole_reference_question_embedding #(B, D)

                        step = update_step * 2 * real_gradient_accumulation_steps + 2*accumulation_step + split_idx
                        #step += 1
                        if rng_to_sync:
                            self._load_rng_state(resume_from_checkpoint)
                            rng_to_sync = False

                        # Skip past any already trained steps if resuming training
                        if steps_trained_in_current_epoch > 0:
                            steps_trained_in_current_epoch -= 1
                            if steps_trained_progress_bar is not None:
                                steps_trained_progress_bar.update(1)
                            if steps_trained_in_current_epoch == 0:
                                self._load_rng_state(resume_from_checkpoint)
                            continue #<<--- skip the first batch in the first epoch
                        elif steps_trained_progress_bar is not None:
                            steps_trained_progress_bar.close()
                            steps_trained_progress_bar = None

                            

                        # self.accelerator.gradient_state._set_sync_gradients(accumulation_step == real_gradient_accumulation_steps - 1 and split_idx == 1)
                        # if self.accelerator.sync_gradients:
                        #     context = contextlib.nullcontext
                        # else:
                        #     context = self.accelerator.no_sync
                        # self.accelerator.accumulate: count the forward times and reset when meet set number. 
                        # If meet the set number, use normal pytorch context to sync gradient otherwise use accelerator.no_sync to skip sync and only accumulate the gradient
                        with self.accelerator.accumulate(model):  ## this will use args.gradient_accumulation_steps to accumulate the gradient, which is around 
                        # with context(model):
                            tr_loss_step = self.training_step(model, inputs_for_this_type)

                        if (
                            args.logging_nan_inf_filter
                            and not is_torch_tpu_available()
                            and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                        ):
                            # if loss is nan or inf simply add the average of previous logged losses
                            tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                        else:
                            tr_loss += tr_loss_step

                        self.current_flos += float(self.floating_point_ops(inputs))
                        self.control = self.callback_handler.on_substep_end(args, self.state, self.control)
                
                
                # Gradient clipping
                if args.max_grad_norm is not None and args.max_grad_norm > 0:
                    # deepspeed does its own clipping

                    if self.do_grad_scaling:
                        # Reduce gradients first for XLA
                        if is_torch_tpu_available():
                            gradients = xm._fetch_gradients(self.optimizer)
                            xm.all_reduce("sum", gradients, scale=1.0 / xm.xrt_world_size())
                        # AMP: gradients need unscaling
                        self.scaler.unscale_(self.optimizer)

                    if is_sagemaker_mp_enabled() and args.fp16:
                        self.optimizer.clip_master_grads(args.max_grad_norm)
                    elif hasattr(self.optimizer, "clip_grad_norm"):
                        # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                        self.optimizer.clip_grad_norm(args.max_grad_norm)
                    elif hasattr(model, "clip_grad_norm_"):
                        # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
                        model.clip_grad_norm_(args.max_grad_norm)
                    elif self.use_apex:
                        # Revert to normal clipping otherwise, handling Apex or full precision
                        nn.utils.clip_grad_norm_(
                            amp.master_params(self.optimizer),
                            args.max_grad_norm,
                        )
                    else:
                        self.accelerator.clip_grad_norm_(
                            model.parameters(),
                            args.max_grad_norm,
                        )

                # Optimizer step
                optimizer_was_run = True
                if is_torch_tpu_available():
                    if self.do_grad_scaling:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        # tpu-comment: accelerate wrapped optimizers call xm.optimizer_step
                        self.optimizer.step()
                elif self.do_grad_scaling:
                    scale_before = self.scaler.get_scale()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    scale_after = self.scaler.get_scale()
                    optimizer_was_run = scale_before <= scale_after
                else:
                    self.optimizer.step()
                    optimizer_was_run = not self.accelerator.optimizer_step_was_skipped

                if optimizer_was_run:
                    # Delay optimizer scheduling until metrics are generated
                    if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.lr_scheduler.step()

                model.zero_grad()
                self.state.global_step += 1
                
                self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
                self.control = self.callback_handler.on_step_end(args, self.state, self.control)
                self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)
            

            if step < 0:
                logger.warning(
                    "There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True
            
            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_tpu_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

        #################################################################################
        #################################################################################
        #################################################################################
        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sur the model has been saved by process 0.
            if is_torch_tpu_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.parallel_mode == ParallelMode.DISTRIBUTED:
                dist.barrier()
            elif is_sagemaker_mp_enabled():
                smp.barrier()

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step

        metrics = speed_metrics("train", start_time, num_samples=num_train_samples, num_steps=self.state.max_steps)
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
        if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if checkpoint != self.state.best_model_checkpoint:
                    logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        return TrainOutput(self.state.global_step, train_loss, metrics)

    def create_offline_embedding(self, should_update_keys=None):
        model = self._wrap_model(self.model_wrapped,training=False)
        # as the model is wrapped, don't use `accelerator.prepare`
        # this is for unhandled cases such as
        # Fairscale Sharded DDP, FSDP-XLA, SageMaker MP/DP, DataParallel, IPEX
        use_accelerator_prepare = True if model is self.model else False
        now_process   = self.accelerator.process_index
        total_process = self.accelerator.num_processes

        # prepare using `accelerator` prepare
        if use_accelerator_prepare:
            self.model.eval()
            model= self.accelerator.prepare(self.model)
        
        
        model.eval() #<-------do not forget this !!!!!!!!!!!!
        if should_update_keys is None:
            extra_answer_needed_keys = self.train_dataset.answer_tokens_unique_id.keys()
        else:
            extra_answer_needed_keys = should_update_keys
        
        self.train_dataset.cache_whole_the_tokens()
        necessary_update_answer_index_in_knowledge_buffer = np.array([self.train_dataset.answer_tokens_unique_id[k] for k in extra_answer_needed_keys])
        
        total_length  = len(extra_answer_needed_keys)
        now_process   = self.accelerator.process_index
        total_process = self.accelerator.num_processes

        split_indices = np.array_split(np.arange(total_length),total_process)
        max_length    = max([len(t) for t in split_indices])
        split_indices = split_indices[now_process]
        if len(split_indices) < max_length:
            split_indices = np.array([0]+list(split_indices))
        
        necessary_update_answer_index_for_this_process = necessary_update_answer_index_in_knowledge_buffer[split_indices]
        necessary_update_answer_token_for_this_process = self.train_dataset.answer_tokens[necessary_update_answer_index_for_this_process]

        
        necessary_update_answer_token_for_this_process = torch.from_numpy(necessary_update_answer_token_for_this_process.astype('int')).to(self.args.device)
        necessary_update_answer_embedding_for_this_process = self.obtain_embedding(model, necessary_update_answer_token_for_this_process, 'answer') #(b, D)
        self.accelerator.wait_for_everyone()
        if self.args.embedding_offline_path:
            assert self.args.create_embedding
            if not os.path.exists(self.args.embedding_offline_path):os.makedirs(self.args.embedding_offline_path)
            necessary_update_answer_index_for_this_process_keys = np.array([self.train_dataset.answer_index2key[i] for i in necessary_update_answer_index_for_this_process])
            np.save(os.path.join(self.args.embedding_offline_path, f'answer_embedding_{now_process}.idx.npy'), necessary_update_answer_index_for_this_process_keys)
            np.save(os.path.join(self.args.embedding_offline_path, f'answer_embedding_{now_process}.npy'), necessary_update_answer_embedding_for_this_process.cpu().numpy())
        necessary_update_answer_index_for_this_process = torch.from_numpy(necessary_update_answer_index_for_this_process).to(self.args.device)
        return (necessary_update_answer_index_for_this_process, 
                necessary_update_answer_embedding_for_this_process)

    def obtain_embedding(self,model, ids, embedder_type):
        origin_train = model.training
        model = self.accelerator.unwrap_model(model)
        model.eval()
        chunk_size = min(len(ids),self.args.generate_chunk_size)
        ids = torch.split(ids,chunk_size)
        ProgressBar = tqdm if self.accelerator.is_main_process else dummy_tqdm
        with torch.no_grad():
            with self.accelerator.autocast():
                embedding = []
                for _ids in ProgressBar(ids, desc=f"generate online {embedder_type} embedding", position=2, leave=False):
                    embedding.append(model.get_embedding(_ids, embedder_type=embedder_type))
                embedding = torch.cat(embedding)
        if origin_train:
            model.train()
        return embedding

    def _remove_unused_columns(self, dataset, description: Optional[str] = None):
        return dataset
    
    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]
            if self.args.optim == 'sophia':
                optimizer_cls = SophiaG
                optimizer_kwargs = {
                    'lr': self.args.learning_rate,
                    'rho': 0.05,
                    'weight_decay': self.args.weight_decay,
                    }
            else:
                optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            if self.sharded_ddp == ShardedDDPOption.SIMPLE:
                self.optimizer = OSS(
                    params=optimizer_grouped_parameters,
                    optim=optimizer_cls,
                    **optimizer_kwargs,
                )
            else:
                self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
                if optimizer_cls.__name__ == "Adam8bit":
                    import bitsandbytes

                    manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                    skipped = 0
                    for module in opt_model.modules():
                        if isinstance(module, nn.Embedding):
                            skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                            logger.info(f"skipped {module}: {skipped/2**20}M params")
                            manager.register_module_override(module, "weight", {"optim_bits": 32})
                            logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                    logger.info(f"skipped: {skipped/2**20}M params")

        if is_sagemaker_mp_enabled():
            self.optimizer = smp.DistributedOptimizer(self.optimizer)

        return self.optimizer
    
    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        question_index = inputs.pop('question_index')
        #question_keys  = [self.eval_dataset.question_index2key[i.item()] for i in question_index[:,0]]
        answer_index  = inputs.pop('answer_index') # answer for one question may have many. we will firstly pad them into (B,L_max)
        postive_length= (answer_index>=0).sum(dim=-1) # [n1, n2, n3, ...]
        answer_mask   = answer_index>=0
        #answer_keys  = [self.eval_dataset.answer_index2key[i.item()] for i in answer_index[:,0]]
        
        question_embedding = self.obtain_embedding(model, inputs['question_ids'], 'question') #'question_ids' : (B, L)
        
        if self.eval_answer_embedding is None:
            answer_embedding = self.obtain_embedding(model, inputs['answer_ids'][answer_mask], 'answer') #'answer_ids' : (B, S, L_max) -> (B*S,L_max)
            assert len(answer_embedding) == sum(postive_length)
            result_positive = torch.cosine_similarity(
                question_embedding.unsqueeze(1),
                answer_embedding.unsqueeze(0),
                dim=-1,
            ) # (B, 768) * (768, B*S) = (B, B*S)
                
            result_negative = torch.cosine_similarity(
                question_embedding.unsqueeze(1),
                self.knowledge_buffer_embedding.unsqueeze(0),
                dim=-1,
            ) # (N, M) # (B, 768) * (768, 38000) = (B, 38000)
            result = torch.cat([result_positive, result_negative], -1) #(B, 38000+B)
            B,total_knowledge_num =  result.shape
            # find the order of the silimarities between question and label_answer
            result = torch.argsort(result, dim=-1,descending=True ) ## give the array like [5 ,4, 3,9,10 ] it return [4 3 0 1 2], but we need the order of each slot, which is [2 3 4 1 0]
            top100_index = None
            result = torch.argsort(result, dim=-1,descending=False) # [4 3 0 1 2] -> [2 3 4 1 0]
            
            out = torch.zeros(B,4).to(result.device)
            start = 0
            for i, l in enumerate(postive_length):
                end = start + l
                out[i][0] = torch.max(result[i, start: end])
                out[i][1] = torch.median(result[i, start: end])
                out[i][2] = torch.mean(result[i, start: end].float())
                out[i][3] = torch.min(result[i, start: end])
                start = end
        else:
            result_positive = torch.cosine_similarity(
                question_embedding.unsqueeze(1),
                self.eval_answer_embedding.unsqueeze(0),
                dim=-1,
            ) # (B, 768) * (768, B*S) = (B, B*S)
            
            result_negative = torch.cosine_similarity(
                question_embedding.unsqueeze(1),
                self.knowledge_buffer_embedding.unsqueeze(0),
                dim=-1,
            ) # (N, M) # (B, 768) * (768, 38000) = (B, 38000)
            
            result = torch.cat([result_positive, result_negative], -1) #(B, 38000+B)
            #self.accelerator.print(result.shape)
            B,total_knowledge_num =  result.shape
            # find the order of the silimarities between question and label_answer
            result = torch.argsort(result, dim=-1,descending=True ) ## give the array like [5 ,4, 3,9,10 ] it return [4 3 0 1 2], but we need the order of each slot, which is [2 3 4 1 0]
            top100_index = result[:,:100] - len(self.eval_answer_embedding) ## the true index in the knowledge buffer, the -1, -2 ,.. -n means the index in the eval_answer_embedding
            result = torch.argsort(result, dim=-1,descending=False) # [4 3 0 1 2] -> [2 3 4 1 0]
            out    = torch.zeros(B,4).to(result.device)
            for i, index in enumerate(answer_index):
                index = index[index>=0]
                out[i][0] = torch.max(result[i, index])
                out[i][1] = torch.median(result[i, index])
                out[i][2] = torch.mean(result[i, index].float())
                out[i][3] = torch.min(result[i, index])
        
        
        
        return (None, out, top100_index)

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        batch_size = self.args.eval_batch_size

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = getattr(dataloader, "dataset", None)

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        preds_host = None
        labels_host = None
        inputs_host = None

        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None
        all_inputs = None
        
        self.eval_answer_embedding = None
        if args.eval_mode == 'precompute_embedding':
        ######## lets firstly compute all the necessary embedding #########
            extra_answer_needed_keys = self.extra_answer_needed_keys_runtime if self.extra_answer_needed_keys_runtime is not None else self.eval_dataset.extra_answer_needed_keys
            if len(extra_answer_needed_keys) >0:
                necessary_update_answer_index_for_this_process, necessary_update_answer_embedding_for_this_process = self.create_offline_embedding(extra_answer_needed_keys)
                necessary_update_answer_index_in_knowledge_buffer     = distributed_concat(necessary_update_answer_index_for_this_process )
                necessary_update_answer_embedding_in_knowledge_buffer = distributed_concat(necessary_update_answer_embedding_for_this_process) #(B, D) 
                ## Lets update the knowledge buffer, since the buffer is assume small (340M for unarchieve qh 8k, we shard it in each GPU)
                ## the answer is just the knowlegde

                answer_keys     = [self.train_dataset.answer_index2key[i.item()] for i in necessary_update_answer_index_in_knowledge_buffer]
                knowledge_index = [self.knowledge_buffer_key2index[k] for k in answer_keys]
                #print(torch.dist(self.knowledge_buffer_embedding[knowledge_index],necessary_update_answer_embedding_in_knowledge_buffer))
                self.knowledge_buffer_embedding[knowledge_index] = necessary_update_answer_embedding_in_knowledge_buffer

            total_length  = len(self.eval_dataset.whole_answer_index)
            now_process   = self.accelerator.process_index
            total_process = self.accelerator.num_processes

            split_indices = np.array_split(np.arange(total_length),total_process)
            max_length    = max([len(t) for t in split_indices])
            split_indices = split_indices[now_process]
            if len(split_indices) < max_length:
                split_indices = np.array([0]+list(split_indices))

            necessary_compute_answer_index_for_this_process = torch.from_numpy(self.eval_dataset.whole_answer_index[split_indices]).to(args.device)
            necessary_compute_answer_token_for_this_process = torch.from_numpy(self.eval_dataset.whole_answer_token[split_indices]).to(args.device)
            necessary_compute_answer_embedding_for_this_process = self.obtain_embedding(model, necessary_compute_answer_token_for_this_process, 'answer') #(b, D)
            
            necessary_compute_answer_index_in_eval     = distributed_concat(necessary_compute_answer_index_for_this_process)
            necessary_compute_answer_embedding_in_eval = distributed_concat(necessary_compute_answer_embedding_for_this_process)
            self.accelerator.wait_for_everyone()
            index_map = {k.item():i for i,k in enumerate(necessary_compute_answer_index_in_eval)}
            #make sure the order in self.eval_answer_embedding follow 0,1,2,3,4,....
            select_only_need_once_index =[ index_map[i] for i in range(len(self.eval_dataset.whole_answer_index))]
            self.eval_answer_embedding = necessary_compute_answer_embedding_in_eval[select_only_need_once_index]

        
        ####################################################################


        observed_num_examples = 0
        # Main evaluation loop
        ProgressBar = tqdm if self.accelerator.is_main_process else dummy_tqdm
        for step, inputs in ProgressBar(enumerate(dataloader), total=len(dataloader), desc="running evaluate", position=1, leave=False):

            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            main_input_name = getattr(self.model, "main_input_name", "input_ids")
            inputs_decode = self._prepare_input(inputs[main_input_name]) if args.include_inputs_for_metrics else None

            if is_torch_tpu_available():
                xm.mark_step()

            # Update containers on host
            if loss is not None:
                losses = self.accelerator.gather_for_metrics((loss.repeat(batch_size)))
                losses_host = losses if losses_host is None else nested_concat(losses_host, losses, padding_index=-100)
            if labels is not None:
                labels = self.accelerator.pad_across_processes(labels, dim=1, pad_index=-100)
            if inputs_decode is not None:
                inputs_decode = self.accelerator.pad_across_processes(inputs_decode, dim=1, pad_index=-100)
                inputs_decode = self.accelerator.gather_for_metrics((inputs_decode))
                inputs_host = (
                    inputs_decode
                    if inputs_host is None
                    else nested_concat(inputs_host, inputs_decode, padding_index=-100)
                )
            if logits is not None:
                logits = self.accelerator.pad_across_processes(logits, dim=1, pad_index=-100)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                logits = self.accelerator.gather_for_metrics((logits))
                preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)

            if labels is not None:
                labels = self.accelerator.gather_for_metrics((labels))
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)

            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if (
                args.eval_accumulation_steps is not None
                and (step + 1) % args.eval_accumulation_steps == 0
                and (self.accelerator.sync_gradients or version.parse(accelerate_version) > version.parse("0.20.3"))
            ):
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
                if preds_host is not None:
                    logits = nested_numpify(preds_host)
                    all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
                if inputs_host is not None:
                    inputs_decode = nested_numpify(inputs_host)
                    all_inputs = (
                        inputs_decode
                        if all_inputs is None
                        else nested_concat(all_inputs, inputs_decode, padding_index=-100)
                    )
                if labels_host is not None:
                    labels = nested_numpify(labels_host)
                    all_labels = (
                        labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
                    )

                # Set back to None to begin a new accumulation
                losses_host, preds_host, inputs_host, labels_host = None, None, None, None

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
        if inputs_host is not None:
            inputs_decode = nested_numpify(inputs_host)
            all_inputs = (
                inputs_decode if all_inputs is None else nested_concat(all_inputs, inputs_decode, padding_index=-100)
            )
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)

        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and getattr(eval_dataset, "num_examples", 0) > 0:
            num_samples = eval_dataset.num_examples
        else:
            if has_length(dataloader):
                num_samples = self.num_examples(dataloader)
            else:  # both len(dataloader.dataset) and len(dataloader) fail
                num_samples = observed_num_examples
        if num_samples == 0 and observed_num_examples > 0:
            num_samples = observed_num_examples
        
        if self.extra_answer_needed_keys_runtime is not None:
            top100_index = np.unique(all_labels[all_labels>0])
            self.extra_answer_needed_keys_runtime = [self.knowledge_buffer_index2key[i.item()] for i in top100_index]
            # print(f"""
            #     GPU:{self.accelerator.process_index} has {len(top100_index)} unique answer in the knowledge buffer.
            #     The first 10 are {top100_index[:10]}
            #     """)
            

        all_labels = all_preds
        all_preds  = 1.0*all_preds/(len(self.eval_answer_embedding) + len(self.knowledge_buffer_embedding))
        # Metrics!
        if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
            if args.include_inputs_for_metrics:
                metrics = self.compute_metrics(
                    EvalPrediction(predictions=all_preds, label_ids=all_labels, inputs=all_inputs)
                )
            else:
                metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels))
        else:
            metrics = {}
        
        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)
        # for k,v in metrics.items():
        #     self.accelerator.print(f"{k}: {v}")

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()
        if hasattr(self, "jit_compilation_time"):
            metrics[f"{metric_key_prefix}_jit_compilation_time"] = self.jit_compilation_time

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)
        self.accelerator.wait_for_everyone()
        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)