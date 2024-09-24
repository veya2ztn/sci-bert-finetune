
from haystack import Document
from haystack.nodes import PreProcessor
import json
import pandas as pd
from tqdm import tqdm
import haystack
    
corpus_file = '/mnt/petrelfs/zhangtianning/projects/llm/art/data/eval/scifact/corpus.jsonl'
train_file = '/mnt/petrelfs/zhangtianning/projects/llm/art/data/eval/scifact/claims_train.jsonl'

with open(train_file) as f:
    # read the file as a list of lines
    lines = f.readlines()
    claims = []
    for claimstr in lines:
        claims.append(json.loads(claimstr))

data = pd.read_json(corpus_file, lines=True)
doc_list = []
for i in tqdm(range(len(data))):
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


preprocessor = PreProcessor(
 clean_empty_lines=True,
 clean_whitespace=True,
 clean_header_footer=False,
 split_by="word",
 split_length=1000,
 split_overlap=3,
 split_respect_sentence_boundary=False,
)

processed_docs = preprocessor.process(doc_list)

def get_search_id(retriever, query, top_k=10):
    documents = retriever.run_query(query=query, top_k=top_k)
    return [x.meta['doc_id'] for x in documents[0]['documents']]

def test_acc_of_embedding(embedding_name, processed_docs):
    retriever = haystack.nodes.EmbeddingRetriever(
        embedding_model=embedding_name,
        model_format="sentence_transformers"
    )
    embedding_dim = retriever.embed_queries(['a']).shape[-1]
    document_store = haystack.document_stores.FAISSDocumentStore(
        sql_url=f"sqlite:///data/faiss_document_store_{embedding_name.split('/')[-1]}.db",
        embedding_dim=embedding_dim)
    document_store.delete_documents()
    document_store.write_documents(processed_docs)
    document_store.update_embeddings(retriever)
    retriever.document_store = document_store
    document_store.save(index_path=f"data/index_{embedding_name.split('/')[-1]}.faiss", 
                       config_path=f"data/config_{embedding_name.split('/')[-1]}.faiss")
    results = []
    for claim in claims:
        search_result = get_search_id(retriever, claim['claim'])
        results.append({
            "cited_doc_ids": claim['cited_doc_ids'],
            "search_result": search_result
        })

    acc1 = 0
    acc5 = 0
    acc10 = 0
    for result in results:
        id_rst = result['search_result']
        id_true = result['cited_doc_ids']
        if id_rst[0] in id_true:
            acc1 += 1
        if id_true[0] in id_rst[:5]:
            acc5 += 1
        if id_true[0] in id_rst:
            acc10 += 1

    acc1 /= len(results)
    acc5 /= len(results)
    acc10 /= len(results)

    return acc1, acc5, acc10

embedding_name = "sentence-transformers/all-MiniLM-L6-v2" # 很快，在cpu上运行即可，2~3分钟跑完
acc1, acc5, acc10 = test_acc_of_embedding(embedding_name, processed_docs)
print(f"acc1={acc1} acc5={acc5} acc10={acc10}")