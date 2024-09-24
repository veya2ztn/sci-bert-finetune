import warnings
import math
import torch
import torch.nn.functional as F
from megatron.global_vars import get_args
from megatron  import print_rank_0
from megatron.checkpointing import load_dualencoder_checkpoint
from megatron.module import MegatronModule
from megatron.model.dualencoder_model import dualencoder_model_provider
from megatron.mpu import get_mips_group, get_node_first_rank
from megatron.global_vars import get_tokenizer
from megatron.tokenizer.tokenizer import vocab_size_with_padding
from megatron.data.art_index import OpenRetreivalDataStore, DistributedBruteForceIndex

from megatron.mpu.initialize import get_data_parallel_group
from megatron.global_vars import get_t0_model, get_t0_tokenizer, get_knowledge_pool
from transformers import BertTokenizer as HFBertTokenizer


class T0_Score_System:
    def __init__(self, args):
        self.args = args
        self.tokenizer = get_t0_tokenizer()
        self.model = get_t0_model()
        if 'vicuna' not in self.args.hf_model_name:
            self.model = self.model.cuda()


class EncoderDecoder_Matchness(T0_Score_System):
    def get_gold_log_probs(self, question_token,  answer_token):
        input_encoding = self.tokenizer.pad({'input_ids': question_token},
                                            padding='longest',
                                            max_length=512,
                                            pad_to_multiple_of=8,
                                            return_attention_mask=True,
                                            return_tensors='pt')
        assert input_encoding.input_ids.size(1) <= 512
        context_tensor, attention_mask = input_encoding.input_ids.cuda(), input_encoding.attention_mask.cuda()
        lm_output = self.model(input_ids=context_tensor, attention_mask=attention_mask, labels=answer_token,### ,--- label token is necessary
                                output_attentions=False, output_hidden_states=False)
        lm_logits = lm_output.logits.float()

        log_softmax = F.log_softmax(lm_logits, dim=-1)
        gold_log_probs = log_softmax.gather(2, answer_token.unsqueeze(2)).squeeze(2)

        return gold_log_probs

class DecoderOnly_Judger(T0_Score_System):
    def get_gold_log_probs(self, question_token,  answer_token):
        self.tokenizer.padding_side='left'
        inputs_ids = self.tokenizer.pad({'input_ids': question_token},padding='longest',
                                        max_length=336,pad_to_multiple_of=8,return_attention_mask=False,
                                        return_tensors='pt').input_ids.cuda()
        output_ids = self.model.generate(inputs_ids,return_dict_in_generate=True,output_scores=True, max_length=len(inputs_ids[0])+1)
        lm_logits  = output_ids.scores[0][:,[1939,3869]] # the first output and the logists for No/Yes # <- only work for vicuna
        # print(self.tokenizer.decode(question_token[0]))
        # print(lm_logits)
        # print(self.tokenizer.decode(output_ids.sequences[0].detach().cpu().numpy()))
        # raise
        softmax    = F.softmax(lm_logits, dim=-1)
        gold_log_probs = softmax[...,1]
        #print(f"The Yes Problity: {gold_log_probs}")
        return gold_log_probs

class DecoderOnly_Matchness(T0_Score_System):
    def get_gold_log_probs(self, question_token,  answer_token):
        real_answer_token = [[i for i in t if i !=0]+[self.tokenizer.eos_token_id] for t in answer_token[:,3:].tolist()]
        self.tokenizer.padding_side='left'
        inputs_ids = self.tokenizer.pad({'input_ids': question_token},padding='longest',
                                        max_length=336,pad_to_multiple_of=8,return_attention_mask=False,
                                        return_tensors='pt').input_ids.cuda()
        self.tokenizer.padding_side='right'
        labels_ids = self.tokenizer.pad({'input_ids': real_answer_token},padding='longest',
                                            max_length=64,pad_to_multiple_of=None,return_attention_mask=False,
                                            return_tensors='pt').input_ids.cuda()
        #print(f"now we generate the prob, the input is {inputs_ids.shape}, the label_id size is {labels_ids.shape}")
        output_ids = self.model.generate(inputs_ids,labels=labels_ids,
                                                    return_dict_in_generate=True,output_scores=True, 
                                                    max_length=600)
        lm_logits      = torch.stack(output_ids.scores,1)
        #print(f"=======>{lm_logits.shape} ==> {lm_logits.dtype}")
        log_softmax    = F.log_softmax(lm_logits, dim=-1)
        assert log_softmax.shape[1]==labels_ids.shape[1], f"get logits shape {log_softmax.shape} and labels_id shape {labels_ids.shape}, >>  you may not modify the greedy_search! <<"
        gold_log_masks = labels_ids==0
        gold_log_probs = log_softmax.gather(2, labels_ids.unsqueeze(2)).squeeze(2)
        gold_log_probs.masked_fill_(gold_log_masks,0)
        return gold_log_probs


class ARTModel(MegatronModule):
    def __init__(self, evidence_retriever=None):
        super(ARTModel, self).__init__()
        args = get_args()
        self.args = args
        self.topk = args.topk_retrievals

        if args.topk_retrievals > 0:
            bert_vocab_size = vocab_size_with_padding(
                get_tokenizer().vocab_size, args)
            print_rank_0('building Retriever for ART (Autoencoding-based Retriever Training) ...')
            self.retriever_model = dualencoder_model_provider(only_context_model=False, only_query_model=False, vocab_size=bert_vocab_size)
            print_rank_0('building Retriever for ART, done ...')
            self._retriever_model_key = 'retriever/biencoder_model'
            self.evidence_retriever = evidence_retriever

        # We have two tokenizers:
        # (1) for BERT as the retriever models are trained using BERT.
        # (2) for T0   as the pre-trained language model scorer uses T0 tokenization.
        self.hf_bert_tokenizer = None
        if 'vicuna' in self.args.hf_model_name:
            if self.args.art_mode in ['question_matchness', 'question_matchness_only_passage']:
                self.llm_system        = DecoderOnly_Matchness(args)
            elif self.args.art_mode == 'question_relative_check':
                self.llm_system        = DecoderOnly_Judger(args)
            else:
                raise NotImplementedError(f'The mode [{self.args.art_mode} is not defined]')
        elif 'bigscience' in self.args.hf_model_name:
            self.llm_system        = EncoderDecoder_Matchness(args)
        else:
            raise NotImplementedError(f'The mode [{self.args.hf_model_name} is not defined]')
    
    def retriever_embedder(self, tokens, mask, types, embedder_type, disable_dropout=False):
        unwrapped_model = self.retriever_model
        while not hasattr(unwrapped_model, 'embed_text'):
            unwrapped_model = unwrapped_model.module

        if embedder_type == "query":
            if disable_dropout:
                unwrapped_model.query_model.eval()
            logits = unwrapped_model.embed_text(unwrapped_model.query_model,
                                                tokens,
                                                mask,
                                                types)
            return logits
        elif embedder_type == "context":
            if disable_dropout:
                unwrapped_model.context_model.eval()
            logits = unwrapped_model.embed_text(unwrapped_model.context_model,
                                                tokens,
                                                mask,
                                                types)
            return logits
        else:
            raise ValueError("Invalid embedder type.")

    def get_query_embedding(self, query_tokens_for_retriever, query_mask_bert, query_types):
        # Compute "fresh" query logits
        args = self.args
        
        query_logits = self.retriever_embedder(query_tokens_for_retriever,
                                               query_mask_bert,
                                               query_types,
                                               embedder_type="query",
                                               disable_dropout=args.disable_retriever_dropout)
        if args.no_query_embedder_training:
            query_logits = query_logits.detach()
        return query_logits

    def get_topk_evidence(self, query_uid, query_embedding):
        # Get top-K evidence data for the BERT tokenized query
        with torch.no_grad():
            # GPU searching.
            topk_evidence_data, stale_topk_sim = self.evidence_retriever.get_topk(query_embedding.clone().detach())
        return topk_evidence_data
            

    def get_LLM_score(self, all_title_context_ids_for_LLM, prefixed_query_tuple):
        args = self.args
        topk = self.topk
        prefixed_query_token_for_LLM, prefixed_query_token_len_for_LLM = prefixed_query_tuple
        bsize = len(prefixed_query_token_for_LLM)
        #bsize, max_seq_len = query_tokens_for_retriever.shape

        decoder_prefix_tensor = torch.repeat_interleave(prefixed_query_token_for_LLM, topk, dim=0)
        log_prob_list = []
        #print(f"the whole input data size is {len(all_title_context_ids_for_LLM)} max is {max([len(t) for t in all_title_context_ids_for_LLM])} about {[len(t) for t in all_title_context_ids_for_LLM]}")
        for k in range(0, bsize * topk, topk):
            log_prob_list_one_question = []
            all_title_context_ids_one_question = all_title_context_ids_for_LLM[k: k + topk]
            decoder_prefix_tensor_one_question = decoder_prefix_tensor[k: k + topk]
            prefixed_query_ids_t0_len_one_question = prefixed_query_token_len_for_LLM[k // topk]
            for i in range(0, topk, args.shard_size):
                all_title_context_ids_view = all_title_context_ids_one_question[
                    i: i + args.shard_size]
                decoder_prefix_tensor_view = decoder_prefix_tensor_one_question[
                    i: i + args.shard_size]
                with torch.no_grad():
                    gold_log_probs = self.llm_system.get_gold_log_probs(all_title_context_ids_view,  decoder_prefix_tensor_view)
                    
                    if 'vicuna' in self.args.hf_model_name :
                        if self.args.art_mode in ['question_matchness', 'question_matchness_only_passage']:
                            # in decoder-only mode, we produce the masked gold_log_probs
                            teacher_log_probs = torch.sum(gold_log_probs, dim=1)/torch.sum(gold_log_probs!=0, dim=1) 
                        elif self.args.art_mode == 'question_relative_check':
                            teacher_log_probs = gold_log_probs
                        else:
                            raise NotImplementedError
                    else:
                        # this will work because the batch size is 1 and this implies all decoder labels have the same length
                        #### this is also confused, since the gold log probs provide the log prob of a sequence, should use sum rather than mean
                        #### or use a global divider..
                        teacher_log_probs = torch.mean(gold_log_probs[:, :prefixed_query_ids_t0_len_one_question], dim=1)
                    log_prob_list_one_question.append(teacher_log_probs)
            log_prob_list_one_question = torch.cat(log_prob_list_one_question).unsqueeze(0)
            log_prob_list.append(log_prob_list_one_question)
        gold_log_probs = torch.cat(log_prob_list, dim=0)
        return gold_log_probs

    def get_retriever_embedding(self, all_title_context_tokens_for_retriever,embedder_type = "context"):
        args = self.args
        topk = self.topk
        bsize_mul_topk = len(all_title_context_tokens_for_retriever)
        bsize = bsize_mul_topk//topk
        if self.hf_bert_tokenizer is None:
            self.hf_bert_tokenizer = HFBertTokenizer.from_pretrained("bert-large-uncased")
        input_encoding = self.hf_bert_tokenizer.pad({'input_ids': all_title_context_tokens_for_retriever},
                                                    padding='longest',
                                                    max_length=512,
                                                    pad_to_multiple_of=8,
                                                    return_attention_mask=True,
                                                    return_tensors='pt')
        assert input_encoding.input_ids.size(1) <= 512

        all_title_context_ids = input_encoding.input_ids.cuda()
        all_context_types     = torch.cuda.LongTensor(input_encoding.input_ids.size()).fill_(0)
        all_context_mask      = (all_title_context_ids[:, None, :] >= 1) * (all_title_context_ids[:, :, None] >= 1)
        # Inverting the mask
        all_context_mask = ~all_context_mask

        # Compute "fresh" context logits
        all_context_logits = self.retriever_embedder(all_title_context_ids,
                                                     all_context_mask,
                                                     all_context_types,
                                                     embedder_type=embedder_type,
                                                     disable_dropout=args.disable_retriever_dropout)
        all_context_logits = all_context_logits.reshape(bsize, topk, -1)

        if args.no_context_embedder_training:
            all_context_logits = all_context_logits.detach()

        return all_context_logits

    def compute_log_probs(self, query_embedding, all_context_embedding):
        args = self.args
        if 'vicuna' in self.args.hf_model_name and self.args.art_mode == 'question_relative_check':
            '''
            computer angle = a.b/(|a||b|
            '''
            ######### cossin simuliarity not good
            ## query_embedding = query_embedding/query_embedding.norm(1,keepdim=True) 
            ## all_context_embedding = all_context_embedding/all_context_embedding.norm(2,keepdim=True)
            ## topk_sim_scores = torch.einsum('bd,bkd->bk',query_embedding,all_context_embedding)# [B,K]
            ######### use orintation directly
            topk_sim_scores = torch.einsum('bd,bkd->bk',query_embedding,all_context_embedding)
            topk_sim_scores = topk_sim_scores/(args.inverse_temperature_multiplier * math.sqrt(args.hidden_size))
            topk_sim_scores = topk_sim_scores.sigmoid()
            return topk_sim_scores
        else:
            
            # query_embedding       = query_embedding.unsqueeze(1).float()# [B, 1, dim]
            # all_context_embedding = all_context_embedding.float()# [B, K, dim]
            # topk_sim_scores = torch.bmm(query_embedding, all_context_embedding.transpose(1, 2))# [B, 1, K]
            topk_sim_scores = torch.einsum('bd,bkd->bk',query_embedding,all_context_embedding).unsqueeze(1)# [B, 1, K]
            if args.retriever_score_scaling:topk_sim_scores = topk_sim_scores / (args.inverse_temperature_multiplier * math.sqrt(args.hidden_size))
            # [B, 1, K]
            topk_log_probs = F.log_softmax(topk_sim_scores, dim=2)
            # B x 1 x K -> B x K
            topk_log_probs = topk_log_probs.squeeze(1)
            return topk_log_probs

    def forward(self, query_uid, query_tokens_for_retriever, query_types, query_mask_bert,
                prefixed_query_token_for_LLM, prefixed_query_token_len_for_LLM, timers=None):

        #args = get_args()
        args = self.args
        # assert bsize == 1, "for auto-encoder pre-training, we assume a local batch size of 1"
        assert args.initialize_t0_model_tokenizer_evidence, "for auto-encoder pre-training, we need to pass the argument --initialize-t0-model-and-tokenizer"
        if timers:timers('forward-get-query-embedding').start()
        query_embedding = self.get_query_embedding(query_tokens_for_retriever, query_mask_bert, query_types)
        if timers:timers('forward-get-query-embedding').stop()

        # Coarse Search
        if timers:timers('forward-get-topk-embedding').start()
        topk_evidence_data = self.get_topk_evidence(query_uid, query_embedding)
        coarse_search_context_tokens_for_retriever, coarse_search_context_tokens_for_llm = self.evidence_retriever.postprocess(
            query_uid, topk_evidence_data,prefixed_query_token_for_LLM)
        if timers:timers('forward-get-topk-embedding').stop()

        ##  ==========  [Rank] the Coarsed Searched Result ===========
        #### build the embedding for those Coarsed Searched Result
        if timers:timers('forward-get-answer-embedding').start()
        all_context_embedding = self.get_retriever_embedding(coarse_search_context_tokens_for_retriever)
        if timers:timers('forward-get-answer-embedding').stop()
        #### compute the score of retriever ranking
        if timers:timers('forward-get-answer-score').start()
        topk_log_probs = self.compute_log_probs(query_embedding, all_context_embedding)
        if timers:timers('forward-get-answer-score').stop()

        

        ## ========== Use LLM [Rank] Coarsed Searched Result =========
        if timers:timers('forward-get-LLM-score').start()
        prefixed_query_tuple = (prefixed_query_token_for_LLM,prefixed_query_token_len_for_LLM)
        gold_log_probs = self.get_LLM_score(coarse_search_context_tokens_for_llm, prefixed_query_tuple)
        if timers:timers('forward-get-LLM-score').stop()



        return topk_log_probs, gold_log_probs

    def state_dict_for_save_checkpoint(self, destination=None, prefix='', keep_vars=False):
        """For easy load when model is combined with other heads, add an extra key."""
        state_dict_ = dict()
        state_dict_[self._retriever_model_key] = self.retriever_model.state_dict_for_save_checkpoint(destination,
                                                                                                     prefix,
                                                                                                     keep_vars)
        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Load the state dicts of each of the models"""
        self.retriever_model.load_state_dict(
            state_dict[self._retriever_model_key], strict)

    def init_state_dict_from_bert(self):
        """Initialize the state from pre-trained BERT model"""
        if self.retriever_model is not None:
            print_rank_0("Initializing retriever model from pretrained BERT")
            self.retriever_model.init_state_dict_from_bert()

    def init_state_dict_from_dualencoder(self):
        """Initialize the state from pre-trained DPR model and pre-trained T5 mode on iteration zero of pretraining"""
        args = get_args()

        if args.pretrained_dualencoder_load is None:
            assert args.bert_load is not None
            warnings.warn(
                "Pretrained dual-encoder checkpoints are not found. Initializing from random weights")
            return

        print_rank_0(
            "Initializing retriever model from pretrained Dual-Encoder checkpoint")
        load_dualencoder_checkpoint(self.retriever_model,
                                    custom_load_path=args.pretrained_dualencoder_load)


class JRTModel(ARTModel):
    """
    原本的 ART 框架写的过于 恶心 了, 需要以下几点:
    1. 统一两类 tokenize 的使用, 并且唯一指定起来
    2. 数据的注入建议直接使用 string 而不是 token, 因为瓶颈不是在 tokenize 上, 而是在 LLM 不过这样的话 torch 的多线程的 DistributedSample 会有问题。
       因为 Dataloader 只能用 Tensor 或者 numpy ?
    
    """
    def compute_log_probs(self, query_embedding, all_context_embedding):
        assert 'vicuna' in self.args.hf_model_name
        args = self.args
        assert self.args.art_mode == 'question_relative_check'
        '''
        computer angle = a.b/(|a||b|
        '''
        ######### cossin simuliarity not good
        ## query_embedding = query_embedding/query_embedding.norm(1,keepdim=True) 
        ## all_context_embedding = all_context_embedding/all_context_embedding.norm(2,keepdim=True)
        ## topk_sim_scores = torch.einsum('bd,bkd->bk',query_embedding,all_context_embedding)# [B,K]
        ######### use orintation directly

        query_embedding       = query_embedding/torch.norm(query_embedding,dim=-1,keepdim=True)
        all_context_embedding = all_context_embedding/torch.norm(all_context_embedding,dim=-1,keepdim=True)
        if len(query_embedding.shape)==2:
            topk_sim_scores       = torch.einsum('bd,bkd->bk',query_embedding,all_context_embedding)
        elif len(query_embedding.shape)==3:
            topk_sim_scores       = torch.einsum('bkd,bkd->bk',query_embedding,all_context_embedding)
        else:
            raise NotImplementedError
        ### in the origin implement, the query_embedding and all_context_embedding is free vector without normlization.
        ### Many pretrained model will provide large and positive embedding, thus should check the Positive and Maginitude
        #topk_sim_scores = topk_sim_scores/(args.inverse_temperature_multiplier * math.sqrt(args.hidden_size))#<--- if we normilized data, no need for this
        ### the topk_sim_scores stay in -1,1 and we need convert it to [0,1]
        #topk_sim_scores = torch.sigmoid(5*topk_sim_scores)
        topk_sim_scores = (topk_sim_scores + 1)/2
        return topk_sim_scores

    def get_LLM_score(self, all_title_context_ids_for_LLM, prefixed_query_tuple):
        assert self.args.art_mode == 'question_relative_check'
        args = self.args
        topk = self.topk
        prefixed_query_token_for_LLM, prefixed_query_token_len_for_LLM = prefixed_query_tuple
        bsize = len(prefixed_query_token_for_LLM)
        #bsize, max_seq_len = query_tokens_for_retriever.shape

        decoder_prefix_tensor = torch.repeat_interleave(prefixed_query_token_for_LLM, topk, dim=0)
        log_prob_list = []
        #print(f"the whole input data size is {len(all_title_context_ids_for_LLM)} max is {max([len(t) for t in all_title_context_ids_for_LLM])} about {[len(t) for t in all_title_context_ids_for_LLM]}")
        for k in range(0, bsize * topk, topk):
            log_prob_list_one_question = []
            all_title_context_ids_one_question = all_title_context_ids_for_LLM[k: k + topk]
            decoder_prefix_tensor_one_question = decoder_prefix_tensor[k: k + topk]
            prefixed_query_ids_t0_len_one_question = prefixed_query_token_len_for_LLM[k // topk]
            for i in range(0, topk, args.shard_size):
                all_title_context_ids_view = all_title_context_ids_one_question[i: i + args.shard_size]
                decoder_prefix_tensor_view = decoder_prefix_tensor_one_question[i: i + args.shard_size]
                with torch.no_grad():
                    gold_log_probs = self.llm_system.get_gold_log_probs(all_title_context_ids_view,  decoder_prefix_tensor_view)
                teacher_log_probs = gold_log_probs
                log_prob_list_one_question.append(teacher_log_probs)
            log_prob_list_one_question = torch.cat(log_prob_list_one_question).unsqueeze(0)
            log_prob_list.append(log_prob_list_one_question)
        gold_log_probs = torch.cat(log_prob_list, dim=0)
        return gold_log_probs


    def forward(self, query_uid, query_tokens_for_retriever, query_types, query_mask_bert,
                prefixed_query_token_for_LLM, prefixed_query_token_len_for_LLM, timers=None):

        #args = get_args()
        args = self.args
        # assert bsize == 1, "for auto-encoder pre-training, we assume a local batch size of 1"
        assert args.initialize_t0_model_tokenizer_evidence, "for auto-encoder pre-training, we need to pass the argument --initialize-t0-model-and-tokenizer"
        if timers:timers('forward-get-query-embedding').start()
        query_embedding = self.get_query_embedding(query_tokens_for_retriever, query_mask_bert, query_types)
        if timers:timers('forward-get-query-embedding').stop()

        # Coarse Search
        if timers:timers('forward-get-topk-embedding').start()
        topk_evidence_data = self.get_topk_evidence(query_uid, query_embedding)
        coarse_search_context_tokens_for_retriever, coarse_search_context_tokens_for_llm = self.evidence_retriever.postprocess(
            query_uid, topk_evidence_data,prefixed_query_token_for_LLM)
        if timers:timers('forward-get-topk-embedding').stop()

        ##  ==========  [Rank] the Coarsed Searched Result ===========
        #### build the embedding for those Coarsed Searched Result
        if timers:timers('forward-get-answer-embedding').start()
        all_context_embedding = self.get_retriever_embedding(coarse_search_context_tokens_for_retriever)
        if timers:timers('forward-get-answer-embedding').stop()

        
        


        #### compute the score of retriever ranking
        if timers:timers('forward-get-answer-score').start()
        topk_log_probs = self.compute_log_probs(query_embedding, all_context_embedding)
        if timers:timers('forward-get-answer-score').stop()

        

        ## ========== Use LLM [Rank] Coarsed Searched Result =========
        if timers:timers('forward-get-LLM-score').start()
        prefixed_query_tuple = (prefixed_query_token_for_LLM,prefixed_query_token_len_for_LLM)
        gold_log_probs = self.get_LLM_score(coarse_search_context_tokens_for_llm, prefixed_query_tuple)
        if timers:timers('forward-get-LLM-score').stop()

        flattened_reference_ids = self.evidence_retriever.get_the_reference_id(topk_evidence_data)
        all_reference_embedding = self.get_retriever_embedding(flattened_reference_ids,embedder_type = "query")
        reference_sim = self.compute_log_probs(all_reference_embedding,all_context_embedding)

        
        reference_seted= torch.ones_like(gold_log_probs)

        retriever_score = torch.concatenate([topk_log_probs,reference_sim ],-1) #(B, 2*k)
        LLM_score       = torch.concatenate([gold_log_probs,reference_seted ],-1) #(B, 2*k)
        return retriever_score, LLM_score



class PreComputedEvidenceDocsRetrieverbase(object):
    def __init__(self):
        self.args = args = get_args()
        self.topk = args.topk_retrievals
        self.embedding_size = args.hidden_size
        self.evidence_embedder_obj = None
        self.mips_index = None

        self.precomputed_index_wrapper()

        self.allow_trivial_doc = args.allow_trivial_doc
        if not args.allow_trivial_doc:
            self.topk = self.topk + 1

    def get_evidence_embedding(self, path):
        # must have valid initial evidence embedding. ~~~
        self.evidence_embedder_obj = OpenRetreivalDataStore(path, load_from_path=True)

    def precomputed_index_wrapper(self):
        args = get_args()
        if get_node_first_rank() == torch.distributed.get_rank():
            self.get_evidence_embedding(args.embedding_path)
            
            assert self.evidence_embedder_obj is not None
            self.mips_index = DistributedBruteForceIndex(embed_size=self.embedding_size, embed_data=self.evidence_embedder_obj)
        # Wait for the index to be initialized in all the GPUs
        torch.distributed.barrier(get_data_parallel_group())

    def update_evidence_embedding(self):
        """Reload index with new data loaded from disk. Should be performed after each indexer job completes."""
        if get_node_first_rank() == torch.distributed.get_rank():
            self.mips_index.update_index()

        # Wait for the MIPS index to be initialized in all the nodes
        torch.distributed.barrier(get_data_parallel_group())

    def get_topk(self, query_tensor):
        local_bsize = query_tensor.shape[0]
        input_ = torch.empty_like(query_tensor).copy_(query_tensor).detach_()
        tensor_list = [torch.empty_like(input_)
                       for _ in range(self.device_count)]
        torch.distributed.all_gather(
            tensor_list, query_tensor, group=get_mips_group())

        if get_node_first_rank() == torch.distributed.get_rank():
            assert self.mips_index is not None, "MIPS Index is not initialized"
            all_query_tensor = torch.cat(tensor_list, dim=0).contiguous()
            distance, topkindex = self.mips_index.search_mips_index(all_query_tensor,
                                                                    top_k=self.topk,
                                                                    reconstruct=False)
        else:
            distance = torch.empty(
                self.device_count * local_bsize, self.topk, dtype=torch.float16).cuda()
            topkindex = torch.empty(
                self.device_count * local_bsize, self.topk, dtype=torch.int32).cuda()

        torch.distributed.broadcast(
            distance, src=get_node_first_rank(), group=get_mips_group())
        torch.distributed.broadcast(
            topkindex, src=get_node_first_rank(), group=get_mips_group())

        distance = torch.split(distance, local_bsize, dim=0)[self.local_rank]
        topkindex = torch.split(topkindex, local_bsize, dim=0)[self.local_rank]
        topk_data = []
        for topkarray in topkindex:
            # The idx contains passage text and title
            topkarray = topkarray.tolist()
            topk_data_one_line = [topkarray]
            for token_type in self.evidence_used:
                text_list = []
                for idx in topkarray:
                    doctext_ids = self.evidence_pool[token_type]['passage'][idx - 1].tolist()
                    reference_ids   = self.evidence_pool[token_type]['tilte'][idx - 1].tolist() if self.evidence_pool[token_type]['tilte'] else None
                    text_list.append((doctext_ids, reference_ids))
                topk_data_one_line.append(text_list)
            topk_data.append(tuple(topk_data_one_line))

        return topk_data, distance

    def postprocess(self, query_uid, topk_evidence_data, answer_ids):
        args = get_args()
        query_uid = query_uid.tolist()
        query_for_each_token_type = dict([(token_type, []) for token_type in self.evidence_used])
        evidence_used = self.evidence_used
        for qid, topk_tuples,answer_id in zip(query_uid, topk_evidence_data,answer_ids):
            k = 0
            eids = topk_tuples[0]
            for i, eid in enumerate(eids):
                if not (qid != eid and k < args.topk_retrievals):
                    continue
                k += 1
                for token_type, context_ids_and_title_ids in zip(evidence_used, topk_tuples[1:]):
                    context_id, title_id = context_ids_and_title_ids[i]
                    if token_type in ['bert']:
                        query_for_each_token_type[token_type].append(self.query_template[token_type](title_id, context_id))
                    else:
                        query_for_each_token_type[token_type].append(self.query_template[token_type](title_id, context_id,answer_id))
        return [query_for_each_token_type[token_type] for token_type in self.evidence_used]

    @staticmethod
    def get_the_reference_id(topk_evidence_data):
        reference_answers = []
        for batch_row in topk_evidence_data:
            topk_index, bert_tokenized_data, llm_tokenized_data = batch_row
            #reference_answers.append([reference_answer for content, reference_answer in bert_tokenized_data])
            reference_answers.extend([reference_answer for content, reference_answer in bert_tokenized_data])
        return reference_answers

class PreComputedEvidenceDocsRetriever(PreComputedEvidenceDocsRetrieverbase):
    def __init__(self):
        super().__init__()
        args = self.args
        self.local_rank = args.local_rank
        world_size = torch.distributed.get_world_size()
        # Note: This will work only for 1 node. May need to fix this?
        self.device_count = 1 if world_size == 1 else min(
            torch.cuda.device_count(), args.max_training_rank)
        knowledge_pool = get_knowledge_pool()
        self.evidence_pool  = {
            'bert':{'passage': knowledge_pool['bert'].passages_map_bert,
                      'tilte': knowledge_pool['bert'].title_map_bert},
            't0':  {'passage': knowledge_pool['t0'].passages_map_bert,
                      'tilte': knowledge_pool['t0'].title_map_bert},
        }
        self.evidence_used  = ['bert', 't0']
        self.bert_tokenizer = get_tokenizer()
        self.t0_tokenizer   = get_t0_tokenizer()
        self.MAX_SEQUENCE_LEN = 256 #512 <========= influence the max memory cost
        self.verbalizer_head_ids = self.t0_tokenizer.encode(
            args.verbalizer_head, add_special_tokens=False)
        self.verbalizer_head_sep = self.t0_tokenizer.encode(
            ":\n", add_special_tokens=False)
        self.verbalizer_sep = self.t0_tokenizer.encode(
            "\"\"\"\n", add_special_tokens=False)
        self.verbalizer_ids = self.t0_tokenizer.encode(
            args.verbalizer, add_special_tokens=False)
        self.query_template = {      
            'bert': knowledge_pool['bert'].collect_knowledge,# self.combine2bert_tokens,
            't0': self.combine2t0_tokens,
        }

    def token_template(self, title_ids, context_ids,answer_id=None):
        MAX_SEQUENCE_LEN = self.MAX_SEQUENCE_LEN
        if 'vicuna' in self.args.hf_model_name:
            if self.args.art_mode == 'question_matchness':
                return self.vicuna_quesion_template(title_ids, context_ids,answer_id,MAX_SEQUENCE_LEN)
            elif self.args.art_mode == 'question_relative_check':
                return self.vicuna_check_is_relative_template(title_ids, context_ids,answer_id,MAX_SEQUENCE_LEN)
            elif self.args.art_mode == 'question_matchness_only_passage':
                return self.vicuna_only_passage_template(title_ids, context_ids, answer_id,MAX_SEQUENCE_LEN)
            else:
                raise NotImplementedError
        else:
            return self.t0_quesion_template(title_ids, context_ids,answer_id,MAX_SEQUENCE_LEN)
        
    def combine2t0_tokens(self, title_ids, context_ids,answer_id=None):
        return self.token_template(title_ids, context_ids,answer_id=answer_id)

    @staticmethod
    def vicuna_quesion_template(title_ids, context_ids,answer_id=None,MAX_SEQUENCE_LEN=512):
        title_ids = []
        #title_ids = title_ids[1:]    # remove the <s>
        context_ids = context_ids[1:]  # remove the <s>
        ## The template is
        '''
        <s>A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: Here is a passage about [<title_ids>]:\n"""\n<context_ids>\n"""\n Based on the given passage, formulate a question that aligns most accurately with the primary fact disclosed within the text. ASSISTANT:
        '''
        base = [1, 319, 13563, 1546, 263, 12758, 1404, 322, 385, 23116, 21082, 20255, 29889, 450, 20255, 4076, 8444, 29892, 13173,
                        29892, 322, 1248, 568, 6089, 304, 278, 1404, 29915, 29879, 5155, 29889, 3148, 1001, 29901]
        a = base + [2266, 338, 263, 13382, 1048, 518]
        b = [5387, 15945, 29908]
        c = [15945, 29908, 16564, 373, 278, 2183, 13382, 29892, 883, 5987, 263, 1139, 393, 7595, 29879,
                1556, 7913, 2486, 411, 278, 7601, 2114, 766, 15603, 2629, 278, 1426, 29889, 319, 1799, 9047, 13566, 29901]
        to_be_added_len = len(c)
        now_sequence = a + title_ids + b + context_ids
        now_length = len(now_sequence)
        if now_length + to_be_added_len >= MAX_SEQUENCE_LEN:
            truncate_len = now_length + to_be_added_len - MAX_SEQUENCE_LEN
            now_sequence = now_sequence[: -truncate_len]
        now_sequence += c
        #print(f"length of title_ids:{len(title_ids)} length of context_ids:{len(context_ids)} length of final:{len(now_sequence)}")
        return now_sequence

    @staticmethod
    def vicuna_check_is_relative_template(title_ids, context_ids,answer_id=None,MAX_SEQUENCE_LEN=512):
        #raise NotImplementedError
        #title_ids = title_ids[1:]    # remove the <s>
        context_ids = context_ids[1:]  # remove the <s>
        answer_id = answer_id[1:] # remove the <s>
        answer_id = [i for i in answer_id.tolist() if i>0] 
        #print(answer_id)
        '''template is 
        <s> A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user\'s questions. USER: Given the passage:""" {} """ Please check whether the passage is relative to the question: what is<s>what is mother . Return Yes or No. ASSISTANT:
        '''
        base = [1, 319, 13563, 1546, 263, 12758, 1404, 322, 385, 23116, 21082, 20255, 29889, 450, 20255, 4076, 8444, 29892, 13173,
                        29892, 322, 1248, 568, 6089, 304, 278, 1404, 29915, 29879, 5155, 29889, 3148, 1001, 29901]
        a = base + [11221, 278, 13382, 6160, 15945] 
        b = [15945, 29908, 3529, 1423, 3692, 278, 13382, 338, 6198, 304, 278, 1139, 29901]
        c = [869, 7106, 3869, 470, 1939,29889,  319, 1799, 9047, 13566, 29901]
        
        complete_sequence  = a + context_ids + b  + answer_id + c
        exceed_length = len(complete_sequence) - MAX_SEQUENCE_LEN
        if exceed_length >0: 
            context_ids = context_ids[:-exceed_length]
        return a + context_ids + b + answer_id + c
    
    @staticmethod
    def vicuna_only_passage_template(title_ids, context_ids, answer_id=None,MAX_SEQUENCE_LEN=1024):
        context_ids = context_ids[1:]  # remove the <s>
        '''template is 
        <s>A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: Here is a passage about [<title_ids>]:\n"""\n<context_ids>\n"""\n Based on the given passage, formulate a question that aligns most accurately with the primary fact disclosed within the text. ASSISTANT:
        '''
        base = [1, 319, 13563, 1546, 263, 12758, 1404, 322, 385, 23116, 21082, 20255, 29889, 450, 20255, 4076, 8444, 29892, 13173,
                        29892, 322, 1248, 568, 6089, 304, 278, 1404, 29915, 29879, 5155, 29889, 3148, 1001, 29901]
        a = base + [2266,338,263,13382,515,263,16021,5650,29901]
        b = [15945, 29908]
        c = [15945, 29908, 16564, 373, 278, 2183, 13382, 29892, 883, 5987, 263, 1139, 393, 7595, 29879,
                1556, 7913, 2486, 411, 278, 7601, 2114, 766, 15603, 2629, 278, 1426, 29889, 319, 1799, 9047, 13566, 29901]
        to_be_added_len = len(c)
        now_sequence = a  + b + context_ids
        now_length = len(now_sequence)
        if now_length + to_be_added_len >= MAX_SEQUENCE_LEN:
            truncate_len = now_length + to_be_added_len - MAX_SEQUENCE_LEN
            now_sequence = now_sequence[: -truncate_len]
        now_sequence += c
        return now_sequence

    def t0_quesion_template(self, title_ids, context_ids,answer_id=None,MAX_SEQUENCE_LEN=512):
        t0_tokenizer        = self.t0_tokenizer
        verbalizer_head_ids = self.verbalizer_head_ids
        verbalizer_ids      = self.verbalizer_ids
        t0_tokenizer        = self.t0_tokenizer

        a = verbalizer_head_ids + title_ids
        b = []
        c = verbalizer_ids + [t0_tokenizer.eos_token_id]
        to_be_added_len = len(c)
        now_sequence = a + title_ids + b + context_ids
        now_length = len(now_sequence)
        if now_length + to_be_added_len >= MAX_SEQUENCE_LEN:
            truncate_len = now_length + to_be_added_len - MAX_SEQUENCE_LEN
            now_sequence = now_sequence[: -truncate_len]
        now_sequence += c
        #print(f"length of title_ids:{len(title_ids)} length of context_ids:{len(context_ids)} length of final:{len(now_sequence)}")
        return now_sequence
    
        t0_tokenizer        = self.t0_tokenizer
        verbalizer_head_ids = self.verbalizer_head_ids
        verbalizer_ids      = self.verbalizer_ids
        t0_tokenizer        = self.t0_tokenizer

        a = verbalizer_head_ids + title_ids
        b = []
        c = verbalizer_ids + [t0_tokenizer.eos_token_id]
        to_be_added_len = len(c)
        now_sequence = a + title_ids + b + context_ids
        now_length = len(now_sequence)
        if now_length + to_be_added_len >= MAX_SEQUENCE_LEN:
            truncate_len = now_length + to_be_added_len - MAX_SEQUENCE_LEN
            now_sequence = now_sequence[: -truncate_len]
        now_sequence += c
        #print(f"length of title_ids:{len(title_ids)} length of context_ids:{len(context_ids)} length of final:{len(now_sequence)}")
        return now_sequence