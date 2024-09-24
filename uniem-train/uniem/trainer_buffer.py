from __future__ import annotations

from typing import Any, Callable, Sequence, Sized

import torch
from accelerate import Accelerator
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import time
import numpy as np
try:
    from torch.optim.lr_scheduler import LRScheduler
except ImportError:
    from torch.optim.lr_scheduler import _LRScheduler as LRScheduler

from .trainer import LossTracker, DistributedTqdmProgressBar




from uniem.criteria import PairInBatchNegSoftmaxContrastLoss
class Trainer:
    def __init__(
        self,
        *,
        model: torch.nn.Module,
        train_dataloader: DataLoader,
        optimizer: Optimizer,
        accelerator: Accelerator,
        validation_dataloader: DataLoader | None = None,
        buffer = None,
        sampler_num = None,
        epochs: int = 3,
        lr_scheduler: LRScheduler | None = None,
        log_interval: int = 50,
        save_on_epoch_end: bool = True,
        epoch_end_callbacks: Sequence[Callable[['Trainer'], None]] | None = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator
        self.epochs = epochs
        self.log_interval = log_interval
        self.save_on_epoch_end = save_on_epoch_end

        self.train_loss_tracker = LossTracker()
        self.validation_loss_tracker = LossTracker()
        if isinstance(self.train_dataloader.dataset, Sized):
            num_steps_per_epoch = len(self.train_dataloader)
        else:
            num_steps_per_epoch = None
        self.progress_bar = DistributedTqdmProgressBar(self.epochs, num_steps_per_epoch=num_steps_per_epoch)
        self.epoch_end_callbacks = epoch_end_callbacks or []
        self.current_step = 0
        unwrapped_model = model
        while hasattr(unwrapped_model, 'module'):unwrapped_model = unwrapped_model.module
        self.criterion = unwrapped_model.criterion
        self.buffer    = buffer
        self.sampler_num = sampler_num
        assert sampler_num is not None
        #assert criterion is not None
        assert buffer is not None   


    def train(self):
        for current_epoch in range(1, self.epochs + 1):
            self.model.train()
            self.progress_bar.on_epoch_start()
            data_loading = []
            model_train = []

            now = time.time()
            for batch_index, batch in enumerate(self.train_dataloader):
                data_loading.append(time.time() - now);now =time.time()
                question_index  = batch.pop('question_index').detach().cpu()
                answer_index = batch.pop('answer_index').detach().cpu()
                # question_ids = batch.pop('question_ids')
                # answer_ids   = batch.pop('answer_ids')
                
                with self.accelerator.accumulate(self.model):
                    self.optimizer.zero_grad()
                    batch_output = self.model(**batch)
                    question_embeddings = batch_output['question_embeddings']
                    answer_embeddings   = batch_output['answer_embedding']
                    (extra_question_index, extra_question_embedding,
                     extra_answer_index, extra_answer_embedding)= self.buffer.get_cached_question_and_answer(self.sampler_num, 
                                                  exclude_question_index=question_index.detach().cpu().numpy(),
                                                  exclude_answer_index=answer_index.detach().cpu().numpy())
                    
                    if extra_question_embedding is not None:
                        whole_question_embedding = torch.cat([question_embeddings,extra_question_embedding.to(question_embeddings.device)],dim=0)
                        whole_question_index     = torch.cat([question_index,extra_question_index],dim=0)
                    else:
                        whole_question_embedding = question_embeddings
                        whole_question_index     = question_index
                    if extra_answer_embedding is not None:
                        whole_answer_embedding   = torch.cat([answer_embeddings,extra_answer_embedding.to(answer_embeddings.device)],dim=0) 
                        whole_answer_index       = torch.cat([answer_index,extra_answer_index],dim=0)
                    else:
                        whole_answer_embedding   = answer_embeddings
                        whole_answer_index       = answer_index
                    
                    labels = self.buffer.get_ground_truth(whole_question_index.detach().cpu().numpy(),
                                                            whole_answer_index.detach().cpu().numpy())
                    labels = torch.from_numpy(labels).to(whole_question_embedding.device).long()
                    
                    #whole_index        = torch.cat([sample_index,extra_index],dim=0)
                    #print(whole_question_embedding.shape,whole_answer_embedding.shape,labels.shape)
                    loss = self.criterion(whole_question_embedding,whole_answer_embedding,labels)
                    self.accelerator.backward(loss)
                    self.optimizer.step()
                    if self.lr_scheduler is not None:self.lr_scheduler.step()
                    self.train_loss_tracker.update(loss)
                    self.buffer.update_cached_question(question_embeddings,question_index)
                    self.buffer.update_cached_answer(answer_embeddings,answer_index)

                model_train.append(time.time() - now);now =time.time()
                self.progress_bar.update()
                self.current_step += 1
                if batch_index % self.log_interval == 0:
                    log_dict = {'loss': self.train_loss_tracker.loss,
                                'data': np.mean(data_loading),
                                'model': np.mean(model_train),
                                #'time':self.timers.get_string()
                                }
                    # if self.accelerator.is_main_process:
                    #     print(log_dict)
                    self.log_metrics(log_dict,
                        step=self.current_step,
                    )
                    data_loading = []
                    model_train = []
                    train_metrics = self.add_prefix(log_dict, 'itering')
                    self.accelerator.log(train_metrics, step=self.current_step)

                # if self.current_step%1000==1 and self.accelerator.is_main_process:
                #     self.accelerator.save_state()
            train_metrics = self.add_prefix({'loss': self.train_loss_tracker.loss}, 'train')
            self.accelerator.log(train_metrics, step=current_epoch)
            self.train_loss_tracker.on_epoch_end()
            self.progress_bar.on_epoch_end()

            if self.validation_dataloader:
                validation_loss = evaluate(
                    self.model,
                    self.validation_dataloader,
                    self.validation_loss_tracker,
                )
                validation_metrics = self.add_prefix({'loss': validation_loss}, 'validation')
                self.accelerator.print(f'Epoch {current_epoch} Validation loss: {validation_loss:.4f}')
                self.accelerator.log(validation_metrics, step=current_epoch)

            # if self.save_on_epoch_end and self.accelerator.is_main_process:
            #     self.accelerator.save_state()

            if self.epoch_end_callbacks:
                for callback in self.epoch_end_callbacks:
                    callback(self)

        self.accelerator.end_training()

    def log_metrics(self, metrics: dict[str, float], step: int):
        self.accelerator.log(metrics, step=step)
        self.progress_bar.show_metrics(metrics)
        
    @staticmethod
    def add_prefix(values: dict[str, Any], prefix: str):
        return {f'{prefix}/{k}': v for k, v in values.items()}


def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    loss_tracker: LossTracker | None = None,
):
    model = model.eval()
    loss_tracker = loss_tracker or LossTracker()
    for batch in dataloader:
        with torch.inference_mode():
            batch_output = model(**batch)
            loss_tracker.update(batch_output['loss'])
    loss = loss_tracker.loss
    loss_tracker.on_epoch_end()
    return loss

