

import os
import transformers
from transformers import Trainer
import torch

from uniem.criteria import PairInBatchNegSoftmaxContrastLossLabel

from typing import List, Dict, Any, Union, Optional
from fullpaper_dataset import *

class PairQAmodel(torch.nn.Module):
    def __init__(self,embedder,return_loss=False):
        super().__init__()
        self.embedder=embedder
        self.criterion = None
        if return_loss:
            self.criterion = PairInBatchNegSoftmaxContrastLossLabel()
    
    @property
    def config(self): #<--- TODO: this is not good, please build a unique config for embedder model
        return self.embedder.encoder.config
    
    def forward(self, question_ids: torch.Tensor, answer_ids: torch.Tensor, **kargs) -> dict[str, torch.Tensor]:
        ### below code is for the checkpointing case, where the input must be one~~!!
        # batch_size = question_ids.shape[0]
        # question_ids_len = question_ids.shape[1]
        # answer_ids_len   = answer_ids.shape[1]
        # question_ids = torch.nn.functional.pad(question_ids,(0,answer_ids_len-question_ids_len,0,0))
        # whole_ids    = torch.cat([question_ids,
        #                           answer_ids],dim=0)
        # #print(f"input:=>{whole_ids.shape}")
        # whole_embedding = self.embedder(whole_ids)
        # #print(whole_embedding.shape)
        # text_embeddings, text_pos_embeddings = torch.split(whole_embedding,batch_size) 

        text_embeddings     = self.embedder(question_ids)
        text_pos_embeddings = self.embedder(answer_ids)
        if self.criterion is not None:
            loss = self.criterion(text_embeddings, text_pos_embeddings,labels=None)
            return {"loss":loss}
        else:
            return {'question_embeddings':text_embeddings,'answer_embedding':text_pos_embeddings}
    
    def gradient_checkpointing_enable(self):
        return self.embedder.encoder.gradient_checkpointing_enable()
    
class PairQAGradientCacheModel(torch.nn.Module):
    def __init__(self,embedder,return_loss=False, label_temperature=0.01):
        super().__init__()
        self.embedder=embedder
        self.criterion = None
        self.label_temperature = label_temperature
        if return_loss:
            self.criterion = PairInBatchNegSoftmaxContrastLossLabel()
    @property
    def config(self): #<--- TODO: this is not good, please build a unique config for embedder model
        return self.embedder.encoder.config
      
    def get_embedding(self, text_ids: torch.Tensor, embedder_type:str ) -> torch.Tensor:
        if embedder_type == 'question':
            return self.embedder(text_ids)
        elif embedder_type == 'answer':
            return self.embedder(text_ids)
        else:
            raise ValueError(f"embedder_type should be question or answer, but got {embedder_type}")

    def forward(self, text_ids: torch.Tensor, 
                      conjugate_text_embedding: torch.Tensor, 
                      reference_text_embedding: torch.Tensor = None,
                      reference_conjugate_text_embedding: torch.Tensor= None,
                      embedder_type:str = 'question',
                      **kargs) -> dict[str, torch.Tensor]:
        assert not conjugate_text_embedding.requires_grad
        assert embedder_type in ['question','answer']
        text_embeddings     = self.get_embedding(text_ids, embedder_type = embedder_type) # <--- with grad
        
        if self.criterion is not None:
            if reference_text_embedding is not None:
                labels = self.criterion(reference_text_embedding, reference_conjugate_text_embedding, labels=None, output_prob = True, temperature=self.label_temperature)
            else:
                labels = None
                assert len(text_embeddings) == len(conjugate_text_embedding)
            loss   = self.criterion(text_embeddings, conjugate_text_embedding, labels=labels)
            
            return {"loss":loss}
        else:
            return {f'{embedder_type}_embeddings':text_embeddings}
    
    def gradient_checkpointing_enable(self):
        return False
        #return self.embedder.encoder.gradient_checkpointing_enable()

class BufferWithTrainer(Trainer):
    def __init__(self, *args, 
                 negative_sampler_num=20, 
                 buffer=None,
                 **kargs):
        super().__init__(*args, **kargs)
        self.buffer  =  buffer#
        self.criterion = PairInBatchNegSoftmaxContrastLossLabel()
        self.negative_sampler_num = negative_sampler_num

    def compute_loss(self, model, inputs, return_outputs=False):
        assert not return_outputs, "return_outputs=True is not supported."
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        ### pass in the index of question and answer. 
        # It should be convert to its own true key like question_key and answer_key
        # pass the index value because dataloader/distributed-nccl can only pass the tensor ???
        question_index  = inputs.pop('question_index').detach().cpu()
        answer_index    = inputs.pop('answer_index').detach().cpu()
        
        extra_question_index  = inputs.pop('extra_question_index').detach().cpu() if 'extra_question_index' in inputs else None 
        extra_question_embedding  = inputs.pop('extra_question_embedding')  if 'extra_question_embedding' in inputs else None 

        extra_answer_index  = inputs.pop('extra_answer_index').detach().cpu() if 'extra_answer_index' in inputs else None 
        extra_answer_embedding  = inputs.pop('extra_answer_embedding') if 'extra_answer_embedding' in inputs else None 

        batch_output = model(**inputs)
        question_embeddings = batch_output['question_embeddings']
        answer_embeddings   = batch_output['answer_embedding']
        
        # (extra_question_index, extra_question_embedding,
        #  extra_answer_index  ,   extra_answer_embedding)= self.buffer.get_cached_question_and_answer(
        #                                 self.negative_sampler_num, 
        #                                 exclude_question_indexes=question_index.numpy(),
        #                                 exclude_answer_indexes  =answer_index.numpy())
        
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
        
        labels = self.buffer.get_ground_truth(whole_question_index.numpy(),
                                                whole_answer_index.numpy())
        labels = torch.from_numpy(labels).to(whole_question_embedding.device).long()
        
        #whole_index        = torch.cat([sample_index,extra_index],dim=0)
        #print(whole_question_embedding.shape,whole_answer_embedding.shape,labels.shape)
        loss = self.criterion(whole_question_embedding,whole_answer_embedding,labels)
        self.buffer.update_cached_question(question_embeddings,question_index)
        self.buffer.update_cached_answer(answer_embeddings,answer_index)
        return loss

    def _remove_unused_columns(self, dataset, description: Optional[str] = None):
        return dataset

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        if self.control.should_log:

            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["lr"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()
            if self.state.global_step%10==0:
                self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            if isinstance(self.eval_dataset, dict):
                metrics = {}
                for eval_dataset_name, eval_dataset in self.eval_dataset.items():
                    dataset_metrics = self.evaluate(
                        eval_dataset=eval_dataset,
                        ignore_keys=ignore_keys_for_eval,
                        metric_key_prefix=f"eval_{eval_dataset_name}",
                    )
                    metrics.update(dataset_metrics)
            else:
                metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self._report_to_hp_search(trial, self.state.global_step, metrics)

            # Run delayed LR scheduler now that metrics are populated
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                metric_to_check = self.args.metric_for_best_model
                if not metric_to_check.startswith("eval_"):
                    metric_to_check = f"eval_{metric_to_check}"
                self.lr_scheduler.step(metrics[metric_to_check])

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)


from transformers.models.llama import LlamaTokenizer, LlamaTokenizerFast

def get_local_rank():
    rank = int(os.environ.get("RANK",0))
    local_rank = int(os.environ.get("LOCAL_RANK",0))
    
    return rank + local_rank
from fullpaper_dataset import FullPaperDatasetWithIndex,QADataCollatorWithPadding
def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args,model_max_length=4000,return_mapping=False,dispatch_batches=False,
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
     
    max_token_set = model_max_length
    
    ROOTDIR = 'data/unarXive_quantum_physics/'
    with open(os.path.join(ROOTDIR,'query_full_paper.question_answer_map.json'),'r') as f:
        question_to_answer = json.load(f)
    question_to_answer = dict([(int(k),int(v)) for k,v in question_to_answer.items()])
    answer_token_path  = FullPaperDatasetWithIndex.get_filted_answer_path(max_token_set, ROOTDIR = ROOTDIR)
    #answer_token_path = ["data/unarXive_quantum_physics/llama_answer_token/llama_answer_token_28000_32000.npy"]
    question_token_path = os.path.join(ROOTDIR, 'llama_question_token/llama_question_token.npy')
    
    filter_question_to_answer,filter_answer_to_question = FullPaperDatasetWithIndex.get_QA_index_mapping(
        question_to_answer, answer_token_path
    )
    question_answer_pair = list(filter_question_to_answer.items())
    mapping_system = FullPaperDatasetWithIndex.create_entire_question_answer_mapping_from_pair(question_answer_pair)
    train_dataset  = FullPaperDatasetWithIndex(question_answer_pair, question_token_path, answer_token_path,
                                                dummy= get_local_rank()>0 and dispatch_batches,mapping_system=mapping_system)

    eval_dataset = None
    out_pool = dict(train_dataset=train_dataset, eval_dataset=eval_dataset)
    if return_mapping:
        out_pool['mapping_system']=mapping_system
    return  out_pool 

def make_supervised_data_module2(
    tokenizer: transformers.PreTrainedTokenizer, dummy_data=False,model_max_length=4000,
    datapair_path = 'data/unarXive_quantum_physics/pair.answer_version_b.question_version_a.json', 
    evalpair_tuple = None,
    use_reference = True  , dispatch_batches=False,  add_eval_dataset = False,
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
     
    max_token_set = model_max_length

    if datapair_path in ['data/unarXive_quantum_physics/pair.answer_version_b.question_version_a.json']:
        with open(os.path.join(datapair_path),'r') as f:
            question_answer_pair = json.load(f)
        ROOTDIR= os.path.dirname(datapair_path)
        if isinstance(tokenizer, (LlamaTokenizer, LlamaTokenizerFast)):
            question_token_path = os.path.join(ROOTDIR,"question_version_a/llama_question_token/llama_question_token.npy")
            answer_token_path   = os.path.join(ROOTDIR,"answer_version_b/llama_answer_token/llama_answer_token.npy")
        else:
            question_token_path = os.path.join(ROOTDIR,"question_version_a/jina_question_token/jina_question_token.npy")
            answer_token_path   = os.path.join(ROOTDIR,"answer_version_b/jina_answer_token/jina_answer_token.npy")
        if use_reference:
            reference_answer_embedding_path     = os.path.join(ROOTDIR,"answer_version_b/openai.ada_embedding/embedding.npy")
            reference_question_embedding_path   = os.path.join(ROOTDIR,"question_version_a/openai.ada2_question_embedding")
            reference_question_embedding_path = [os.path.join(reference_question_embedding_path,t) for t in os.listdir(reference_question_embedding_path) if 'idx' not in t and 'verify' not in t]
        else:
            reference_answer_embedding_path = reference_question_embedding_path = None
    elif datapair_path in ['data/unarXive_quantum_physics/pair.answer_version_a.question_version_a.json']:
        with open(os.path.join(datapair_path),'r') as f:
            question_answer_pair = json.load(f)
        ROOTDIR= os.path.dirname(datapair_path)
        if isinstance(tokenizer, (LlamaTokenizer, LlamaTokenizerFast)):
            question_token_path = os.path.join(ROOTDIR,"question_version_a/llama_question_token/llama_question_token.npy")
            answer_token_path   = os.path.join(ROOTDIR,"answer_version_a/llama_answer_token/llama_answer_token.npy")
        else:
            question_token_path = os.path.join(ROOTDIR,"question_version_a/jina_question_token/jina_question_token.npy")
            answer_token_path   = os.path.join(ROOTDIR,"answer_version_a/jina_answer_token/jina_answer_token.npy")
        assert not use_reference, "no reference so far"
        reference_answer_embedding_path = reference_question_embedding_path = None
    else:
        raise NotImplementedError
    
    

    train_dataset  = PaperDatasetWithReference(question_answer_pair, question_token_path=question_token_path, 
                                               answer_token_path=answer_token_path,
                                               reference_question_embedding_path=reference_question_embedding_path,
                                               reference_answer_embedding_path=reference_answer_embedding_path, tokenizer=tokenizer,
                                               dummy= dummy_data)
    eval_dataset = None
    if add_eval_dataset:
        datapair_path,eval_datapair_question_token_path,eval_datapair_answer_token_path = evalpair_tuple
        with open(os.path.join(datapair_path),'r') as f:
            question_answer_pair = json.load(f)
            if isinstance(question_answer_pair,dict):
                question_answer_pair = list(question_answer_pair.items())
        question_token_path = eval_datapair_question_token_path
        answer_token_path   = eval_datapair_answer_token_path
        reference_question_embedding_path=None
        reference_answer_embedding_path  =None
        eval_dataset = PaperDatasetWithReference(question_answer_pair, question_token_path=question_token_path, 
                                               answer_token_path=answer_token_path,
                                               reference_question_embedding_path=reference_question_embedding_path,
                                               reference_answer_embedding_path=reference_answer_embedding_path, tokenizer=tokenizer,
                                               dummy= dummy_data)
        

    out_pool = dict(train_dataset=train_dataset, eval_dataset=eval_dataset)
    return  out_pool 
