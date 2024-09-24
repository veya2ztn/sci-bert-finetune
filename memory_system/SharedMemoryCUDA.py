import torch
from .art_index import DistributedBruteForceIndex
from ..art.megatron.mpu import get_data_parallel_group, get_mips_group, get_node_first_rank

class EmbeddingMemory:
    def build_memory_from_path(self, path):
        raise NotImplementedError
    
    def get_embedding_from_key(self, key):
        raise NotImplementedError
    
    def get_embedding_from_index(self, index):
        raise NotImplementedError
    
    def build_memory_from_tensor_and_index(self, tensor, index):
        raise NotImplementedError   


class CUDAEmbeddingMemoryBase(EmbeddingMemory):
    """
        Same as ART.
    """
    def __init__(self, config):
        self.config                = config 
        self.embedding_size        = config.hidden_size
        self.evidence_embedder_obj = None
        self.mips_index            = None

        self.precomputed_index_wrapper()

        

    def get_evidence_embedding(self, path):
        # must have valid initial evidence embedding. ~~~
        self.evidence_embedder_obj = OpenRetreivalDataStore(path, load_from_path=True)

    def precomputed_index_wrapper(self):
        config = get_config()
        if get_node_first_rank() == torch.distributed.get_rank():
            self.get_evidence_embedding(config.embedding_path)
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

    

    def postprocess(self, query_uid, topk_evidence_data, answer_ids):
        config = get_config()
        query_uid = query_uid.tolist()
        query_for_each_token_type = dict([(token_type, []) for token_type in self.evidence_used])
        evidence_used = self.evidence_used
        for qid, topk_tuples,answer_id in zip(query_uid, topk_evidence_data,answer_ids):
            k = 0
            eids = topk_tuples[0]
            for i, eid in enumerate(eids):
                if not (qid != eid and k < config.topk_retrievals):
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


class TopkReteriverCuda(CUDAEmbeddingMemoryBase):
    def __init__(self, config):
        super().__init__(config)
        self.topk = config.topk_retrievals
        self.allow_trivial_doc = config.allow_trivial_doc
        if not config.allow_trivial_doc:
            self.topk = self.topk + 1
            
    def get_topk(self, query_tensor):
        local_bsize = query_tensor.shape[0]
        input_ = torch.empty_like(query_tensor).copy_(query_tensor).detach_()
        tensor_list = [torch.empty_like(input_) for _ in range(self.device_count)]
        torch.distributed.all_gather( tensor_list, query_tensor, group=get_mips_group())

        if get_node_first_rank() == torch.distributed.get_rank():
            assert self.mips_index is not None, "MIPS Index is not initialized"
            all_query_tensor = torch.cat(tensor_list, dim=0).contiguous()
            distance, topkindex = self.mips_index.search_mips_index(all_query_tensor,
                                                                    top_k=self.topk,
                                                                    reconstruct=False)
        else:
            distance = torch.empty( self.device_count * local_bsize, self.topk, dtype=torch.float16).cuda()
            topkindex = torch.empty( self.device_count * local_bsize, self.topk, dtype=torch.int32).cuda()

        torch.distributed.broadcast( distance, src=get_node_first_rank(), group=get_mips_group())
        torch.distributed.broadcast( topkindex, src=get_node_first_rank(), group=get_mips_group())

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