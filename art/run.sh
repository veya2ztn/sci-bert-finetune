export NCCL_INFO=ERROR;
torchrun --nproc_per_node 8 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 60000 tasks/run.py \
--train-data ./data/unArXiv.key_word_from_abstract.csv \
--qa-file-dev "" \
--qa-file-test "" \
--indexed-evidence-bert-tokenized-data-path ./data/unarXive/evidence-unarxiv-indexed-mmap/bert/unarxiv-keyword-evidence-bert_text_document \
--indexed-title-bert-tokenized-data-path ./data/unarXive/evidence-unarxiv-indexed-mmap/bert/unarxiv-keyword-evidence-bert_title_document \
--indexed-evidence-t0-tokenized-data-path ./data/unarXive/evidence-unarxiv-indexed-mmap/vicuna/unarxiv-keyword-evidence-vicuna_text_document \
--indexed-title-t0-tokenized-data-path ./data/unarXive/evidence-unarxiv-indexed-mmap/vicuna/unarxiv-keyword-evidence-vicuna_text_document \
--save-interval 50 \
--save /mnt/data/ai4earth/zhangtianning/LLM/checkpoints/arXiv-base-init-vicuna-judger \
--load /mnt/data/ai4earth/zhangtianning/LLM/checkpoints/arXiv-base-init-vicuna-judger \
--pretrained-dualencoder-load /mnt/data/ai4earth/zhangtianning/LLM/checkpoints/nq-mss-base-init-2 \
--embedding-path embedding-path/unarXiv.embedding/mss-retriever-base.pkl \
--evidence-data-path "" \
--vocab-file ./bert-vocab/bert-large-uncased-vocab.txt \
--log-interval 10 \
--eval-interval 100 \
--weight-decay 1.0e-1 \
--seq-length 512 \
--seq-length-retriever 256 \
--max-position-embeddings 512 \
--fp16 \
--num-workers 2 \
--distributed-backend nccl \
--checkpoint-activations \
--task Train_A_Judger \
--tokenizer-type BertWordPieceLowerCase \
--epochs 10 \
--sample-rate 1.0 \
--eval-batch-size 1 \
--lr 2e-5 \
--warmup 0.01 \
--DDP-impl local \
--lr-decay-style linear \
--max-training-rank 8 \
--topk-retrievals 16 \
--report-topk-accuracies 1 5 20 50 100 \
--art-training \
--retriever-score-scaling \
--update-retriever \
--allow-trivial-doc \
--batch-size 4 \
--shard-size 4 \
--hf-model-name pretrain_weights/vicuna-7b-v1.1 \
--hf-model-type compress \
--initialize-t0-model-tokenizer-evidence \
--t0-model-in-bf16 \
--index-reload-interval 200 \
--compute-fresh-evidence-embeddings \
--tensorboard-dir /mnt/data/ai4earth/zhangtianning/LLM/checkpoints/nq-mss-base-init-vicuna-judger/tb \
--num-layers 12 \
--hidden-size 768 \
--num-attention-heads 12 \
--kv-channels 64 \
--ffn-hidden-size 3072 \
--model-parallel-size 1 \
--art_mode question_relative_check