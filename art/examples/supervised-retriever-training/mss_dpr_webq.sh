#!/bin/bash

BASE_DIR="."

# BERT model configuration.
CONFIG="large"

# Path of the checkpoint and evidence embeddings
CHECKPOINT_PATH="${BASE_DIR}/checkpoints/dualencoder-mss-dpr-${CONFIG}-epochs20-webq"
EMBEDDING_PATH="${BASE_DIR}/embedding-path/psgs_w100-dualencoder-mss-dpr-${CONFIG}-epochs20-webq.pkl"
#rm -rf ${CHECKPOINT_PATH}

MSS_INIT="false"

# To train with hard negatives, mark this as true
TRAIN_WITH_NEG="true"

# Web Questions data path
DATA_DIR="${BASE_DIR}/data/webq"
TRAIN_DATA="${DATA_DIR}/biencoder-webquestions-train.json"
VALID_DATA="${DATA_DIR}/biencoder-webquestions-dev.json"

# BERT vocabulary path
#VOCAB_FILE="${BASE_DIR}/bert-vocab/bert-large-uncased-vocab.txt"
# Wikipedia evidence path (from DPR code) and WebQ evaluation questions
EVIDENCE_DATA_DIR="${BASE_DIR}/data/wikipedia-split/psgs_w100.tsv"
QA_FILE_DEV="${BASE_DIR}/data/qas/webq-dev.csv"
QA_FILE_TEST="${BASE_DIR}/data/qas/webq-test.csv"
WORLD_SIZE=1

#DISTRIBUTED_ARGS="-m torch.distributed.launch --nproc_per_node 16 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 6000"
DISTRIBUTED_ARGS='torchrun --nproc_per_node ${WORLD_SIZE} --nnodes 1 --node_rank 0 --master_addr localhost --master_port 60000'


function config_base() {
    export CONFIG_ARGS="--num-layers 12 \
--hidden-size 768 \
--num-attention-heads 12 \
--ffn-hidden-size 3072 \
--model-parallel-size 1"
}
function config_large() {
    export CONFIG_ARGS="--num-layers 24 \
--hidden-size 1024 \
--num-attention-heads 16 \
--kv-channels 64 \
--ffn-hidden-size 4096 \
--model-parallel-size 1"
}
VOCAB_FILE="${BASE_DIR}/bert-vocab/bert-large-uncased-vocab.txt"

if [ ${CONFIG} == "base" ]; then
    config_base;
    export BERT_LOAD_PATH="./checkpoints/megatron_bert_345m/release/mp_rank_00/model_optim_rng.pt";
elif [ ${CONFIG} == "large" ]; then
    config_large;
    export BERT_LOAD_PATH="./checkpoints/megatron_bert_345m/release/mp_rank_00/model_optim_rng.pt";
else
    echo "Invalid BERT model configuration"
    exit 1
fi


EXTRA_OPTIONS=""
if [ ${MSS_INIT} == "true" ]; then
    if [ ${CONFIG} == "base" ]; then
        PRETRAINED_CHECKPOINT="${BASE_DIR}/checkpoints/mss-emdr2-retriever-base-steps82k"
    elif [ ${CONFIG} == "large" ]; then
        PRETRAINED_CHECKPOINT="${BASE_DIR}/checkpoints/nq-ict-init-T0-3B-large"
    fi
    echo "This will work because the code will first load BERT checkpoints (default) and then will load PRETRAINED_CHECKPOINT"
    export EXTRA_OPTIONS+=" --finetune --pretrained-checkpoint ${PRETRAINED_CHECKPOINT}"
fi


if [ ${TRAIN_WITH_NEG} == "true" ]; then
    export EXTRA_OPTIONS+=" --train-with-neg --train-hard-neg 7"
fi

OPTIONS=" \
          --task RETRIEVER \
          --tokenizer-type BertWordPieceLowerCase \
          --train-data ${TRAIN_DATA} \
          --valid-data ${VALID_DATA} \
          --save ${CHECKPOINT_PATH} \
          --load ${CHECKPOINT_PATH} \
          --qa-file-dev ${QA_FILE_DEV} \
          --qa-file-test ${QA_FILE_TEST} \
          --evidence-data-path ${EVIDENCE_DATA_DIR} \
          --embedding-path ${EMBEDDING_PATH} \
          --vocab-file ${VOCAB_FILE} \
          --bert-load ${BERT_LOAD_PATH} \
          --save-interval 5000 \
          --log-interval 20 \
          --eval-iters 100 \
          --indexer-log-interval 1000 \
          --distributed-backend nccl \
          --faiss-use-gpu \
          --DDP-impl torch \
          --fp16 \
          --num-workers 2 \
          --sample-rate 1.00 \
          --report-topk-accuracies 1 5 10 20 50 100 \
          --seq-length 512 \
          --seq-length-ret 256 \
          --max-position-embeddings 512 \
          --attention-dropout 0.1 \
          --hidden-dropout 0.1 \
          --retriever-score-scaling \
          --epochs 20 \
          --batch-size 2 \
          --eval-batch-size 16 \
          --indexer-batch-size 128 \
          --lr 2e-5 \
          --warmup 0.01 \
          --lr-decay-style linear \
          --weight-decay 1e-1 \
          --clip-grad 1.0 \
          --max-training-rank ${WORLD_SIZE} \
          --val_av_rank_hard_neg 5 \
          --val_av_rank_other_neg 5 "


COMMAND="WORLD_SIZE=${WORLD_SIZE} ${DISTRIBUTED_ARGS} tasks/run.py ${OPTIONS} ${CONFIG_ARGS} ${EXTRA_OPTIONS}"
echo ${COMMAND}
eval ${COMMAND}
exit

set +x
