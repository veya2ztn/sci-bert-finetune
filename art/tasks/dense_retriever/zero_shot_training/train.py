from collections import OrderedDict
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from megatron.global_vars import get_args
from megatron.global_vars import get_timers
from megatron import mpu
from megatron import print_rank_0
from megatron.checkpointing import load_checkpoint
from megatron.checkpointing import save_checkpoint
from megatron.training import setup_model_and_optimizer
from megatron.training import train_step
from megatron.training import training_log
from megatron.utils import reduce_losses
from megatron.global_vars import get_t0_tokenizer, get_evidence_in_string
from megatron.indexer import IndexBuilder
from tasks.dense_retriever.supervised_training.evaluation.evaluate import OpenRetrievalEvaluator
from tasks.dense_retriever.supervised_training.evaluation.data import get_qa_dataset


class CustomDataLoader(DataLoader):
    def __init__(self, dataset, eval=False, **kwargs):
        if kwargs.get('collate_fn', None) is None:
            kwargs['collate_fn'] = self._collate_fn
        self.eval = eval
        self.t0_tokenizer = get_t0_tokenizer()
        self.vicuna_style = False
        if hasattr(self.t0_tokenizer,'pad_token'):
            self.t0_tokenizer.pad_token = self.t0_tokenizer.unk_token
            self.vicuna_style = True
        super().__init__(dataset, **kwargs)

    def _collate_fn(self, batch_data):
        tensorized = OrderedDict()
        for d in batch_data:
            for k, v in d.items():
                tensorized.setdefault(k, []).append(v)
        assert len(tensorized) == 6

        tensorized['query_uid']       = torch.LongTensor(tensorized['query_uid'])
        tensorized['query_ids_bert']  = torch.LongTensor(np.array(tensorized['query_ids_bert']))
        tensorized['query_types']     = torch.LongTensor(np.array(tensorized['query_types']))
        tensorized['query_mask_bert'] = torch.LongTensor(np.array(tensorized['query_mask_bert']))
        
        prefixed_query_ids_mask_t0 = self.t0_tokenizer(tensorized['prefixed_query_text'],
                                padding='longest',max_length=128,
                                pad_to_multiple_of=8,truncation=True,return_tensors='pt')
        if not self.vicuna_style:
            tensorized['prefixed_query_ids_t0'] = prefixed_query_ids_mask_t0.input_ids
            tensorized['prefixed_query_mask_t0'] = prefixed_query_ids_mask_t0.attention_mask
        else:
            tensorized['prefixed_query_ids_t0'] = prefixed_query_ids_mask_t0['input_ids']
            tensorized['prefixed_query_mask_t0'] = prefixed_query_ids_mask_t0['attention_mask']
        tensorized['prefixed_query_ids_t0_len'] = torch.sum(prefixed_query_ids_mask_t0.attention_mask, dim=1)


        # The final key is the reference, which is already appended.
        return tensorized


def process_batch(batch):
    query_uid = batch['query_uid'].cuda()
    query_ids_bert = batch['query_ids_bert'].cuda()
    query_types = batch['query_types'].cuda()
    query_mask_bert = (batch['query_mask_bert'] < 0.5).cuda()
    prefixed_query_ids_t0 = batch['prefixed_query_ids_t0'].cuda()
    prefixed_query_mask_t0 = batch['prefixed_query_mask_t0'].cuda()
    prefixed_query_ids_t0_len = batch['prefixed_query_ids_t0_len'].cuda()
    reference = batch['reference']

    return query_uid, query_ids_bert, query_types, query_mask_bert, \
           prefixed_query_ids_t0, prefixed_query_mask_t0, prefixed_query_ids_t0_len, reference


def _cross_entropy_forward_step(batch, model):
    """Simple forward step with cross-entropy loss."""
    args  = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch generator').start()
    try:
        batch_ = next(batch)
    except BaseException:
        batch_ = batch

    query_uid, query_ids_bert, query_types, query_mask_bert, \
    prefixed_query_ids_t0, prefixed_query_mask_t0, \
    prefixed_query_ids_t0_len, reference = process_batch(batch_)
    assert torch.all(query_uid < 0), "query uid can't be positive"

    timers('batch generator').stop()

    # Forward model.
    topk_log_probs, gold_log_probs = model(query_uid,
                                           query_ids_bert,
                                           query_types,
                                           query_mask_bert,
                                           prefixed_query_ids_t0,
                                           prefixed_query_ids_t0_len,timers=timers)
    #print(f"topk_log_probs:{topk_log_probs.tolist()}")
    #print(f"gold_log_probs:{gold_log_probs.tolist()}")
    # Retriever loss
    retriever_loss = torch.FloatTensor([0]).cuda()
    if args.update_retriever:
        topk_log_probs = topk_log_probs.float()
        gold_log_probs = gold_log_probs.float()
        gold_log_probs_log_softmax = F.log_softmax(gold_log_probs, dim=1)
        loss_func = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
        retriever_loss = loss_func(topk_log_probs, gold_log_probs_log_softmax)

    net_loss = retriever_loss
    reduced_loss = reduce_losses([retriever_loss])

    return net_loss, {'retriever_loss': reduced_loss[0]}

def _binary_entropy_forward_step(batch, model):
    """Simple forward step with cross-entropy loss."""
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch generator').start()
    try:
        batch_ = next(batch)
    except BaseException:
        batch_ = batch

    query_uid, query_ids_bert, query_types, query_mask_bert, \
    prefixed_query_ids_t0, prefixed_query_mask_t0, \
    prefixed_query_ids_t0_len, reference = process_batch(batch_)
    
    timers('batch generator').stop()

    # Forward model.
    topk_probs_of_ART, topk_probs_of_LLM = model(query_uid,
                                           query_ids_bert,
                                           query_types,
                                           query_mask_bert,
                                           prefixed_query_ids_t0,
                                           prefixed_query_ids_t0_len,timers=timers)
    B, L = topk_probs_of_ART.shape
    recall_student_thinking = (topk_probs_of_ART.detach()[:,:L//2]>0.5).float().mean()
    recall_teacher_thinking = (topk_probs_of_LLM.detach()[:,:L//2]>0.5).float().mean()
    recall_reference        = (topk_probs_of_ART.detach()[:,L//2:]>0.5).float().mean()
    recall_student_thinking = reduce_losses([recall_student_thinking])
    recall_teacher_thinking = reduce_losses([recall_teacher_thinking])
    recall_reference        = reduce_losses([recall_reference])

    #print(f"topk_probs_of_ART:{topk_probs_of_ART.shape}")
    #print(f"topk_probs_of_LLM:{topk_probs_of_LLM.shape}")
    # topk_probs_of_ART (B,K) , each one is the probability of positive o
    # topk_probs_of_LLM (B,K) , each one is the probability of positive o
    # Retriever loss
    retriever_loss = torch.FloatTensor([0]).cuda()
    if args.update_retriever:
        topk_probs_of_ART = topk_probs_of_ART.float()
        topk_probs_of_LLM = topk_probs_of_LLM.float()
        loss_func = torch.nn.BCELoss(reduction="mean")
        retriever_loss = loss_func(topk_probs_of_ART, topk_probs_of_LLM)

    net_loss = retriever_loss
    reduced_loss = reduce_losses([retriever_loss])

    
    

    return net_loss, {'retriever_loss': reduced_loss[0], 
                      'recall_student_thinking': recall_student_thinking,
                      'recall_teacher_thinking': recall_teacher_thinking,
                      'recall_reference': recall_reference}

def accuracy_func_provider(single_dataset_provider, datapath):
    args = get_args()
    dataset = single_dataset_provider(datapath)
    drop_last = False

    dataloader = build_data_loader(dataset,
                                   args.eval_batch_size,
                                   num_workers=args.num_workers,
                                   drop_last=drop_last,
                                   shuffle=False)
    dataloaders = (dataset.dataset_name, dataloader)

    def metrics_func(model, epoch):
        print_rank_0('calculating metrics ...')

    return metrics_func


def build_data_loader(dataset, batch_size, num_workers, drop_last, shuffle=True):
    """Data loader. Note that batch-size is the local (per GPU) batch-size."""

    world_size = mpu.get_data_parallel_world_size()
    rank = mpu.get_data_parallel_rank()
    sampler = torch.utils.data.distributed.DistributedSampler(dataset,
                                                              num_replicas=world_size,
                                                              rank=rank,
                                                              shuffle=shuffle)
    # Data loader. Note that batch size is the per GPU batch size.
    data_loader = CustomDataLoader(dataset,
                                   batch_size=batch_size,
                                   sampler=sampler,
                                   shuffle=False,
                                   num_workers=num_workers,
                                   drop_last=drop_last,
                                   pin_memory=True)
    return data_loader


def _build_train_dataloader(train_dataset):
    """Train dataloader."""
    args = get_args()

    print_rank_0('building training dataloader ...')
    # Training dataset.
    train_dataloader = build_data_loader(train_dataset,
                                         args.batch_size,
                                         args.num_workers,
                                         not args.keep_last)
    # Set the training iterations.
    args.train_iters_per_epoch = len(train_dataloader)
    args.train_iters = args.epochs * args.train_iters_per_epoch
    return train_dataloader


def get_retrieval_score(mips_index=None, iteration_num=-1, model=None, datasystem=None):
    if not (args.qa_file_dev or args.qa_file_test):return
    args = get_args()
    evaluator = OpenRetrievalEvaluator(custom_load_path=args.load,
                                       key_list=['retriever/biencoder_model'],
                                       load_evidence_dataset=False, build_mips=False,
                                       use_faiss=False, model=model)
    evidence_id2text = get_evidence_in_string()

    if args.qa_file_dev is not None:
        evaluator.evaluate(args.qa_file_dev,"DEV",
                           mips_index=mips_index,
                           evidence_id2text=evidence_id2text,
                           iteration_num=iteration_num, datasystem=datasystem)
        torch.distributed.barrier()

    if args.qa_file_test is not None:
        evaluator.evaluate(args.qa_file_test,"TEST",
                           mips_index=mips_index,
                           evidence_id2text=evidence_id2text,
                           iteration_num=iteration_num, datasystem=datasystem)
        torch.distributed.barrier()

    if model is None:
        del evaluator.model
    del evaluator
    torch.cuda.empty_cache()



def call_evidence_index_builder(model=None):
    args = get_args()
    index_builder = IndexBuilder(custom_load_path=args.load,key_list=['retriever/biencoder_model'],model=model)
    index_builder.build_and_save_index()
    del index_builder
    torch.cuda.empty_cache()



def _train(model, optimizer, lr_scheduler, forward_step, train_dataloader):
    """Train the model."""
    args = get_args()
    timers = get_timers()

    # Turn on training mode which enables dropout.
    model.train()

    # Tracking loss.
    losses_dict_sum = {}

    # Starting epoch and iteration
    start_epoch = args.iteration // args.train_iters_per_epoch
    start_iteration = args.iteration % args.train_iters_per_epoch
    iteration = args.iteration

    # For async updates
    last_reload_iteration = None
    last_eval_iteration = iteration

    if args.compute_fresh_evidence_embeddings:
        last_reload_iteration = iteration

    # Memory reporting flag.
    report_memory_flag = True
    print_rank_0(f"build DEV and TEST dataset")
    qa_system = {}
    if args.qa_file_dev:
        for qa_file, split in [[args.qa_file_dev, 'DEV'], [args.qa_file_test, 'TEST']]:
            eval_dataset = get_qa_dataset(qa_file, split)
            qa_system[qa_file] = {}
            qa_system[qa_file][split] = eval_dataset
    # For each remaining epoch
    skip_times = 0
    once = False
    timers('interval time').start()
    for epoch in range(start_epoch, args.epochs):
        print_rank_0('working on epoch {} ...'.format(epoch + 1))

        # Set the data loader epoch to shuffle the index iterator.
        train_dataloader.sampler.set_epoch(args.seed + epoch)

        # For all the batches in the dataset.
        for iteration_, batch in enumerate(train_dataloader):

            # Ignore the iterations before starting value
            if iteration_ < start_iteration:
                continue
            # Set to zero so the next epoch does not skip any batches.
            start_iteration = 0

            # Evaluation
            if once or (args.compute_fresh_evidence_embeddings and iteration >= last_reload_iteration + args.index_reload_interval):
                once = False
                with torch.no_grad():
                    # get model without FP16 and/or TorchDDP wrappers
                    unwrapped_model = model
                    while hasattr(unwrapped_model, 'module'):unwrapped_model = unwrapped_model.module
                    print_rank_0("clean GPU cache for rebuiding evidence")
                    if unwrapped_model.evidence_retriever.mips_index is not None:
                        unwrapped_model.evidence_retriever.mips_index.clean_mips_index()
                    # Recompute evidence embeddings
                    call_evidence_index_builder(unwrapped_model.retriever_model)
                    print_rank_0("Training Group: Updating MIPS Index")
                    unwrapped_model.evidence_retriever.update_evidence_embedding()
                    print_rank_0("Training Group: MIPS Index Updated")
                    last_reload_iteration = iteration

            if args.qa_file_dev and iteration >= last_eval_iteration + args.eval_interval:
                print_rank_0('evaluation_on epoch {} ...'.format(epoch + 1))
                unwrapped_model = model
                while hasattr(unwrapped_model, 'module'):unwrapped_model = unwrapped_model.module
                get_retrieval_score(unwrapped_model.evidence_retriever.mips_index,iteration, datasystem=qa_system)
                last_eval_iteration = iteration

            # Train for one step.
         
            losses_dict, skipped_iter = train_step(forward_step, batch, model,optimizer, lr_scheduler)
            

            iteration += 1    
            
            if losses_dict is None:
                print_rank_0(f"OOM! skip epoch:{epoch} iteration:{iteration_}")
                continue
            # Logging.
            report_memory_flag = training_log(losses_dict, losses_dict_sum,
                                              optimizer.param_groups[0]['lr'],
                                              iteration, optimizer.loss_scale,
                                              report_memory_flag, skipped_iter)

            # Checkpointing
            if args.save and args.save_interval and \
                    iteration % args.save_interval == 0:
                save_checkpoint(iteration, model, optimizer, lr_scheduler)


            if args.exit_interval and iteration % args.exit_interval == 0:
                torch.distributed.barrier(mpu.get_data_parallel_group())
                rank = torch.distributed.get_rank()
                print_rank_0('rank: {} | exiting the program at iteration {}'.format(rank, iteration))
                sys.exit(0)

def train(train_dataset_provider, model_provider):

    args = get_args()
    timers = get_timers()

    # Train data-loaders.
    timers('train/valid/test dataset/dataloder').start()
    if args.epochs > 0:
        train_dataset = train_dataset_provider()
        train_dataloader = _build_train_dataloader(train_dataset)
    timers('train/valid/test dataset/dataloder').stop()

    # Build model, optimizer and learning rate scheduler.
    timers('model and optimizer').start()
    model, optimizer, lr_scheduler = setup_model_and_optimizer(model_provider)
    timers('model and optimizer').stop()

    # If pretrained checkpoint is provided and we have not trained for
    # any iteration (i.e., iteration is zero), then load the pretrained
    # checkpoint.
    timers('pretrained checkpoint').start()
    if args.iteration == 0 and args.pretrained_checkpoint is not None:
        original_load = args.load
        args.load = args.pretrained_checkpoint
        _ = load_checkpoint(model, None, None)
        args.load = original_load
        # This is critical when only model is loaded. We should make sure
        # master parameters are also updated.
        if args.fp16:
            optimizer._model_params_to_master_params()
    timers('pretrained checkpoint').stop()

    # Print setup timing.
    print_rank_0('done with setups ...')
    timers.log(['train/valid/test dataset/dataloder',
                'model and optimizer',
                'pretrained checkpoint'])
    print_rank_0('training ...')

    # Finetune the model.
    if args.task == 'Train_A_Judger':
        forward_step = _binary_entropy_forward_step
    else:
        forward_step = _cross_entropy_forward_step
    if args.epochs > 0 and args.art_training:
        _train(model,
               optimizer,
               lr_scheduler,
               forward_step,
               train_dataloader)

    print_rank_0('done :-)')
