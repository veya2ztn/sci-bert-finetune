import os
import shutil
import math
import json
import torch
from megatron.global_vars import get_args
from megatron  import print_rank_0, mpu
from megatron.global_vars import  get_tokenizer
from megatron.training import get_model
from megatron.mpu import get_node_first_rank
from megatron.mpu.initialize import get_data_parallel_group
from megatron.model.dualencoder_model import dualencoder_model_provider
from megatron.checkpointing import load_dualencoder_checkpoint
from megatron.data.orqa_wiki_dataset import get_open_retrieval_wiki_dataset
from tasks.dense_retriever.supervised_training.evaluation.data import get_qa_dataset, get_one_epoch_qa_dataloader, process_qa_batch
from tasks.dense_retriever.supervised_training.evaluation.qa_validation import calculate_matches
from megatron.data.art_index import OpenRetreivalDataStore, FaissMIPSIndex, DistributedBruteForceIndex
from tqdm import tqdm
from transformers import BertTokenizer as HFBertTokenizer

from sentence_transformers import SentenceTransformer
def get_model_from_name(args, custom_load_path, key_list):
    if args.retriever_model_name == 'dualencoder_model':
        only_query_model = True
        model = get_model(lambda: dualencoder_model_provider(only_query_model=True))
        if args.run_indexer and (args.bert_load is not None):
            unwrapped_model = model
            while hasattr(unwrapped_model, 'module'):unwrapped_model = unwrapped_model.module
            unwrapped_model.init_state_dict_from_bert()
        else:
            model = load_dualencoder_checkpoint(
                model, only_query_model=only_query_model, custom_load_path=custom_load_path, key_list=key_list)
        return model
    else:
        model = SentenceTransformer(args.retriever_model_name).cuda()
        return model
class OpenRetrievalEvaluator(object):
    def __init__(self, custom_load_path=None, key_list=None,
                 load_evidence_dataset=True,
                 use_faiss=True, build_mips=True, model = None):
        self.args = args = get_args()
        self.evidence_embedder_obj = None
        self.evidence_dataset = None
        self.mips_index = None
        self.hf_bert_tokenizer = None
        self.model = model
        # Load query encoder checkpoint
        if self.model is None:
            print("no preload model, we will build one from given path")
            self.model = get_model_from_name(args, custom_load_path, key_list)
            self.reuse_model = False
        else:
            print("unlike the origin code, we reuse the model in GPU directly rather than rebuild a new one")
            self.reuse_model = True
        self.model.eval()

        if load_evidence_dataset:
            self.get_evidence_dataset()

        if use_faiss:
            self.faiss_wrapper()
        elif build_mips:
            self.precomputed_index_wrapper()
        else:
            print(f"you need provide evidence when call")
        # Wait for the index to be initialized in all the nodes
        torch.distributed.barrier()

    def get_evidence_embedding(self):
        # This will load the embedding from the embedding path
        self.evidence_embedder_obj = OpenRetreivalDataStore(load_from_path=True)

    def get_evidence_dataset(self):
        self.evidence_dataset = get_open_retrieval_wiki_dataset()

    def faiss_wrapper(self):
        # Initialize FAISS wrapper on local rank = 0 as the evidence embeddings is distributed over all the GPUs in a node
        args = get_args()
        if args.local_rank == 0:
            self.get_evidence_embedding()
            assert self.evidence_embedder_obj is not None
            self.mips_index = FaissMIPSIndex(embed_size=args.hidden_size,
                                             embed_data=self.evidence_embedder_obj,
                                             use_gpu=args.faiss_use_gpu)

    def precomputed_index_wrapper(self):
        args = get_args()
        if get_node_first_rank() == torch.distributed.get_rank():
            self.get_evidence_embedding()
            assert self.evidence_embedder_obj is not None
            self.mips_index = DistributedBruteForceIndex(embed_size=args.hidden_size,
                                                         embed_data=self.evidence_embedder_obj)
        # Wait for the index to be initialized in all the GPUs
        torch.distributed.barrier(get_data_parallel_group())

    def generate_query_vectors(self, eval_dataset):
        dataloader = iter(get_one_epoch_qa_dataloader(eval_dataset))
        tokenizer = get_tokenizer()
        query_vectors = []
        query_list = []
        reference_list = []
        
       
        self.model.eval()
        while True:
            try:
                batch = next(dataloader)
            except (StopIteration, IndexError):
                break

            # batch also has query_tokens and query_pad_data
            query_tokens, query_mask, query_types, \
            query_len, reference = process_qa_batch(batch)

            unwrapped_model = self.model
            while hasattr(unwrapped_model, 'module'):
                unwrapped_model = unwrapped_model.module
            device = next(unwrapped_model.parameters()).device

            with torch.no_grad():
                if self.args.retriever_model_name == 'dualencoder_model':
                    query_logits = unwrapped_model.embed_text(unwrapped_model.query_model,
                                                            query_tokens,
                                                            query_mask,
                                                            query_types)
                else:
                    if self.hf_bert_tokenizer is None:self.hf_bert_tokenizer = HFBertTokenizer.from_pretrained("bert-large-uncased")
                    ostring  = [self.hf_bert_tokenizer.decode(t, skip_special_tokens=True) for t in query_tokens]
                    #print(ostring)
  
                    query_logits = torch.from_numpy(unwrapped_model.encode(ostring)).to(device).half()


            for i in range(len(query_tokens)):
                query_list.append(tokenizer.decode(query_tokens[i].tolist()[:query_len[i]]))

            reference_list.extend(reference)
            query_vectors.extend(query_logits.split(1, dim=0))
            #print(query_vectors[0].shape)
            if len(query_vectors) % 100 == 0:
                print_rank_0('Encoded queries {}'.format(len(query_vectors) * mpu.get_data_parallel_world_size()))
        self.model.train()
        query_tensor = torch.cat(query_vectors, dim=0)
        return query_list, query_tensor, reference_list

    def evaluate(self, qa_file, split, mips_index=None, evidence_id2text=None, iteration_num=-1,datasystem=None):
        args = get_args()
        if datasystem is None:
            eval_dataset = get_qa_dataset(qa_file, split)
        else:
            eval_dataset = datasystem[qa_file][split]
        
        
        query_list, query_tensor, reference_list = self.generate_query_vectors(eval_dataset)
        if mips_index is not None:
            mips_index_cls = mips_index
        else:
            mips_index_cls = self.mips_index

        local_rank = args.local_rank
        rank = torch.distributed.get_rank()
        device_count = torch.cuda.device_count()
        
        if torch.distributed.get_world_size() >1:
            num_nodes = torch.distributed.get_world_size() // device_count
            node_id = rank // device_count

            for node in range(num_nodes):
                start_rank = node * device_count
                end_rank   = (node + 1) * device_count
                ranks_list = list(range(start_rank, end_rank))
                node_group = torch.distributed.new_group(ranks=ranks_list)

                if node_id == node:
                    device_start_rank = start_rank
                    group = node_group

            input_ = torch.empty_like(query_tensor).copy_(query_tensor).detach_()
            all_query_tensor, allsizes = varsize_gather_nograd(input_, group)
        else:
            all_query_tensor = query_tensor
            #allsizes = torch.tensor([query_tensor.shape[0]], device=query_tensor.device, dtype=torch.int)
        print_rank_0(f"all_query_tensor datasize = {all_query_tensor.shape}, {all_query_tensor.dtype}")
        num_rows = len(all_query_tensor)
        print_rank_0(f"starting searching")
        if local_rank == 0 and mips_index_cls is not None:
            all_query_tensor = all_query_tensor.contiguous()
            all_distance, all_topkindex = [], []

            for i in tqdm(range(0, len(all_query_tensor), args.shard_size)):
                query_tensor_view = all_query_tensor[i: i + args.shard_size]

                distance, topkindex = mips_index_cls.search_mips_index(query_tensor_view,
                                                                       top_k=args.report_topk_accuracies[-1],
                                                                       reconstruct=False)
                if type(distance).__module__ == "numpy":
                    distance = torch.from_numpy(distance).half().cuda()
                    topkindex = torch.from_numpy(topkindex).int().cuda()

                all_distance.append(distance)
                all_topkindex.append(topkindex)

            distance = torch.cat(all_distance, dim=0)
            topkindex = torch.cat(all_topkindex, dim=0)

        if local_rank != 0:
            distance = torch.empty(len(all_query_tensor),
                                   args.report_topk_accuracies[-1],
                                   dtype=torch.float16).cuda()
            topkindex = torch.empty(len(all_query_tensor),
                                    args.report_topk_accuracies[-1],
                                    dtype=torch.int32).cuda()
        if torch.distributed.get_world_size() >1:
            torch.distributed.broadcast(distance, src=device_start_rank, group=group)
            torch.distributed.broadcast(topkindex, src=device_start_rank, group=group)

            distance = torch.split(distance, allsizes, dim=0)[local_rank]
            topkindex = torch.split(topkindex, allsizes, dim=0)[local_rank]

        del all_query_tensor

        topk_sim_scores = distance / math.sqrt(args.hidden_size)

        top_ids_and_scores = []
        for darray, topkarray in zip(topk_sim_scores, topkindex):
            top_ids_and_scores.append((topkarray.tolist(), darray.tolist()))

        if self.evidence_dataset is None:
            assert evidence_id2text is not None
            passages = evidence_id2text
        else:
            passages = self.evidence_dataset.id2text

        if args.trec_eval:
            self.trec_eval(top_ids_and_scores, reference_list)
            self.recall_cap(top_ids_and_scores, reference_list)

        else:
            
            match_stats = calculate_matches(passages,
                                            reference_list,
                                            top_ids_and_scores,
                                            workers_num=args.num_workers,
                                            match_type=args.match)

            doc_hits = match_stats.questions_doc_hits
            top_k_hits = torch.FloatTensor(match_stats.top_k_hits).cuda()

            # Accumulating and summing top-k hits scores from all the ranks
            if torch.distributed.get_world_size() >1:
                torch.distributed.all_reduce(top_k_hits, torch.distributed.ReduceOp.SUM)

            top_k_hits = [v / num_rows for v in top_k_hits]

            print_str = "{} SET RESULTS\tstep: {}\t".format(split, iteration_num)
            for i in args.report_topk_accuracies:
                print_str += "top-{}: {:.2f}\t".format(i, top_k_hits[i-1] * 100)

            print_rank_0(print_str)

            if args.save_topk_outputs_path is not None:
                all_data = []
                for i, (q, d, r) in enumerate(zip(query_list, doc_hits, reference_list)):
                    ctx_list = []
                    for j in range(args.topk_retrievals):

                        ctx = {"id": top_ids_and_scores[i][0][j],
                               "score": top_ids_and_scores[i][1][j],
                               "has_answer": d[j]}
                        ctx_list.append(ctx)
                    item = {"question": q,
                            "answers": r,
                            "ctxs": ctx_list}
                    all_data.append(item)

                temp_dir_name = os.path.join(args.save_topk_outputs_path,
                                             "_tmp_reranker_{}".format(os.getenv("SLURM_JOBID")))
                save_shard(all_data, temp_dir_name)
                del all_data
                
                torch.distributed.barrier()

                if mpu.get_data_parallel_rank() == 0:
                    file_name = os.path.splitext(os.path.basename(qa_file))[0]
                    all_data = merge_shards_and_save(args.save_topk_outputs_path, temp_dir_name, file_name)
                    # make sure that every single piece of data was embedded
                    assert len(all_data) == len(eval_dataset)
                    del all_data

        torch.distributed.barrier()
        return


    def trec_eval(self, top_ids_and_scores, reference_list):
        import pytrec_eval
        args = get_args()
        ndcg = {}
        recall = {}

        for k in args.report_topk_accuracies:
            ndcg[f"NDCG@{k}"] = 0.0
            recall[f"Recall@{k}"] = 0.0

        qrels = {}
        results = {}
        for i, ((id_list, score_list), reference) in enumerate(zip(top_ids_and_scores, reference_list)):
            qrels[str(i)] = {str(k): v for k, v in reference.items()}
            results[str(i)] = {str(id): score for id, score in zip(id_list, score_list)}

        ndcg_string = "ndcg_cut." + ",".join([str(k) for k in args.report_topk_accuracies])
        recall_string = "recall." + ",".join([str(k) for k in args.report_topk_accuracies])

        evaluator = pytrec_eval.RelevanceEvaluator(qrels, {ndcg_string, recall_string})
        scores = evaluator.evaluate(results)

        for query_id in scores.keys():
            for k in args.report_topk_accuracies:
                ndcg[f"NDCG@{k}"] += scores[query_id]["ndcg_cut_" + str(k)]
                recall[f"Recall@{k}"] += scores[query_id]["recall_" + str(k)]

        ndcg_tensor = torch.FloatTensor([ndcg[f"NDCG@{k}"] for k in args.report_topk_accuracies]).cuda()
        torch.distributed.all_reduce(ndcg_tensor, torch.distributed.ReduceOp.SUM)

        recall_tensor = torch.FloatTensor([recall[f"Recall@{k}"] for k in args.report_topk_accuracies]).cuda()
        torch.distributed.all_reduce(recall_tensor, torch.distributed.ReduceOp.SUM)

        n_queries = torch.FloatTensor([len(scores)]).cuda()
        torch.distributed.all_reduce(n_queries, torch.distributed.ReduceOp.SUM)

        if torch.distributed.get_rank() == 0:
            ndcg_tensor = ndcg_tensor / n_queries
            recall_tensor = recall_tensor / n_queries

            for i, k in enumerate(args.report_topk_accuracies):
                print_rank_0("NDCG@{}: {:.4f}".format(k, ndcg_tensor[i] * 100))
            print_rank_0("\n")

            for i, k in enumerate(args.report_topk_accuracies):
                print_rank_0("Recall@{}: {:.4f}".format(k, recall_tensor[i] * 100))


    def recall_cap(self, top_ids_and_scores, reference_list):
        args = get_args()
        capped_recall = {}
        for k in args.report_topk_accuracies:
            capped_recall[f"R_cap@{k}"] = 0.0

        k_max = max(args.report_topk_accuracies)

        qrels = {}
        results = {}
        for i, ((id_list, score_list), reference) in enumerate(zip(top_ids_and_scores, reference_list)):
            qrels[str(i)] = {str(k): v for k, v in reference.items()}
            results[str(i)] = {str(id): score for id, score in zip(id_list, score_list)}

        for query_id, doc_scores in results.items():
            top_hits = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)[0:k_max]
            query_relevant_docs = [doc_id for doc_id in qrels[query_id] if qrels[query_id][doc_id] > 0]
            for k in args.report_topk_accuracies:
                retrieved_docs = [row[0] for row in top_hits[0:k] if qrels[query_id].get(row[0], 0) > 0]
                denominator = min(len(query_relevant_docs), k)
                capped_recall[f"R_cap@{k}"] += (len(retrieved_docs) / denominator)

        capped_recall_tensor = torch.FloatTensor([capped_recall[f"R_cap@{k}"] for k in args.report_topk_accuracies]).cuda()
        torch.distributed.all_reduce(capped_recall_tensor, torch.distributed.ReduceOp.SUM)

        n_queries = torch.FloatTensor([len(results)]).cuda()
        torch.distributed.all_reduce(n_queries, torch.distributed.ReduceOp.SUM)

        if torch.distributed.get_rank() == 0:
            capped_recall_tensor = capped_recall_tensor / n_queries

            print_rank_0("\n")
            for i, k in enumerate(args.report_topk_accuracies):
                print_rank_0("Capped-Recall@{}: {:.4f}".format(k, capped_recall_tensor[i] * 100))
            print_rank_0("\n")

        return capped_recall



@torch.no_grad()
def varsize_gather_nograd(x, group):
    """gather tensors of different sizes along the first dimension"""
    
    #determine max size
    size = torch.tensor([x.shape[0]], device=x.device, dtype=torch.int)
    allsizes = [torch.zeros_like(size) for _ in range(8)]
    #print(f"[GPU:{torch.distributed.get_rank()}]: datashape {x.shape} world_size:{mpu.get_data_parallel_world_size()} group size:{torch.distributed.get_world_size(group)} size {size} all_sizes:{allsizes}")

    torch.distributed.all_gather(allsizes, size, group=group)
    max_size = max([size.cpu().max() for size in allsizes])

    padded = torch.empty(
                max_size,
                *x.shape[1:],
                dtype=x.dtype,
                device=x.device
            )
    padded[:x.shape[0]] = x

    output = [torch.zeros_like(padded) for _ in range(8)]
    torch.distributed.all_gather(output, padded, group=group)

    output = [tensor[:allsizes[k]] for k, tensor in enumerate(output)]
    output = torch.cat(output, dim=0)

    return output, allsizes


def save_shard(data, temp_dir_name):
    """
    Save the block data that was created this in this process
    """
    if not os.path.isdir(temp_dir_name):
        os.makedirs(temp_dir_name, exist_ok=True)

    outpath = os.path.join(temp_dir_name, "rank{}.json".format(mpu.get_data_parallel_rank()))
    with open(outpath, "w") as writer:
        writer.write(json.dumps(data, indent=4) + "\n")


def merge_shards_and_save(output_dir_path, temp_dir_name, file_name):
    """Combine all the shards made using self.save_shard()"""
    shard_names = os.listdir(temp_dir_name)
    all_data = []

    for fname in os.listdir(temp_dir_name):
        shard_size = 0
        old_size = len(all_data)
        fpath = '{}/{}'.format(temp_dir_name, fname)
        with open(fpath, 'r') as f:
            data = json.load(f)
            shard_size = len(data)
            all_data.extend(data)

        assert len(all_data) == old_size + shard_size
        os.remove(fpath)

    # save the consolidated shards
    outpath = os.path.join(output_dir_path, "{}.json".format(file_name))

    with open(outpath, 'w') as writer:
        writer.write(json.dumps(all_data, indent=4) + "\n")

    print("Finished merging {} shards for a total of {} embeds".format(
        len(shard_names), len(all_data)), flush=True)

    shutil.rmtree(temp_dir_name, ignore_errors=True)

    return all_data
