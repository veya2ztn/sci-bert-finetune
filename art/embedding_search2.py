import pandas as pd
import haystack
from tqdm.notebook import tqdm
import json
from haystack.document_stores import FAISSDocumentStore
from haystack.nodes import EmbeddingRetriever, PreProcessor
from haystack.pipelines import Pipeline
import tiktoken
import openai
import os
from pathlib import Path
import numpy as np

def test_acc_of_embedding(embedding_name, processed_docs, queries, labels, top_k=10, model_save_name=None, resume='auto', save_root=None):
    if save_root is None:save_root = 'data/abstract_scifact'
    model_save_name = embedding_name.split('/')[-1] if not model_save_name else model_save_name
    save_dir = os.path.join(save_root, model_save_name)
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    sql_url = os.path.join("sqlite:///"+save_dir, 'faiss_document_store.db')
    faiss_index = os.path.join(save_dir, 'index.faiss')
    faiss_config = os.path.join(save_dir, 'config.json')

    if resume == 'auto':
        resume = os.path.exists(faiss_index)

    if resume:
        try:
            document_store = FAISSDocumentStore.load(index_path=faiss_index, config_path=faiss_config)
            retriever = EmbeddingRetriever(
                embedding_model=embedding_name,
                model_format="sentence_transformers",
                document_store=document_store
            )
        except:
            raise ValueError("There is not such dataset whose index_path is {} and config_path is {}. Please set resume to False".format(faiss_index, faiss_config))
    else:

        retriever = EmbeddingRetriever(
            embedding_model=embedding_name,
            model_format="sentence_transformers"
        )
        embedding_dim = retriever.embed_queries(['a']).shape[-1]
        document_store = FAISSDocumentStore(sql_url=sql_url,
                                            embedding_dim=embedding_dim)
        document_store.delete_documents()
        document_store.write_documents(processed_docs)
        document_store.update_embeddings(retriever)
        retriever.document_store = document_store
        document_store.save(index_path=faiss_index, config_path=faiss_config)

    retrieve_documents = retriever.retrieve_batch(queries=queries, top_k=top_k)
    retrieve_result = []
    for search_doc, label in zip(retrieve_documents, labels):
        retrieve_result.append(
            {'search_result': [x.meta['doc_id'] for x in search_doc],
             'cited_doc_ids': label}
        )

    acc1 = 0
    acc5 = 0
    acc10 = 0
    for result in retrieve_result:
        id_rst = result['search_result']
        id_true = result['cited_doc_ids']
        if id_rst[0] in id_true:
            acc1 += 1
        if id_true[0] in id_rst[:5]:
            acc5 += 1
        if id_true[0] in id_rst[:10]:
            acc10 += 1

    acc1 /= len(retrieve_result)
    acc5 /= len(retrieve_result)
    acc10 /= len(retrieve_result)

    print(f"Successfully test the embedding model {embedding_name}, and the result is saved in {save_dir}")
    print(f"Total test number: {len(retrieve_result)}; acc@1: {acc1}, acc@5: {acc5}, acc@10: {acc10}")

    return retrieve_result


class Scifact_Retrieve_Test:
    def __init__(self, corpus_path, claims_path):
        doc_list = []
        data = pd.read_json(corpus_path, lines=True)
        for i in tqdm(range(len(data)), desc='Creating Documents'):
            abstract_i = data.loc[i, 'abstract']
            if isinstance(abstract_i, list):
                abstract_i = " ".join(abstract_i)

            doc_i = {
                'id': data.loc[i, 'doc_id'],
                'content': "title: " +data.loc[i, 'title'] + " abstract: " + abstract_i,
                'meta': data.loc[i].to_dict()
            }

            doc_i = haystack.Document(**doc_i)
            doc_list.append(doc_i)

        preprocessor = haystack.nodes.PreProcessor(
            clean_empty_lines=True,
            clean_whitespace=True,
            clean_header_footer=False,
            split_by="word",
            split_length=300,
            split_overlap=3,
            split_respect_sentence_boundary=False,
        )

        self.processed_docs = preprocessor.process(doc_list)

        with open(claims_path) as f:
            lines = f.readlines()
        claims_data = []
        for claimstr in lines:
            claims_data.append(json.loads(claimstr))

        self.claims = [x['claim'] for x in claims_data]
        self.labels = [x['cited_doc_ids'] for x in claims_data]


    def test_an_embedding(self, embedding_name, top_k=10, model_save_name=None, resume='auto', save_root=None):
        return test_acc_of_embedding(embedding_name=embedding_name,
                                      processed_docs=self.processed_docs,
                                      queries=self.claims,
                                      labels=self.labels,
                                      top_k=top_k,
                                      model_save_name=model_save_name,
                                      resume=resume,
                                      save_root=save_root)
        
        
corpus_path = 'data/eval/scifact/corpus.jsonl'
claims_path = 'data/eval/scifact/claims_train.jsonl'
retrieve_test = Scifact_Retrieve_Test(corpus_path=corpus_path, claims_path=claims_path)
embedding_name = "pretrain_weights/moka-ai_m3e/moka-ai_m3e-base"  # 很快，在cpu上运行即可，2~3分钟跑完
acc1, acc5, acc10 = retrieve_test.test_an_embedding(embedding_name)
print(f"acc1={acc1} acc5={acc5} acc10={acc10}")