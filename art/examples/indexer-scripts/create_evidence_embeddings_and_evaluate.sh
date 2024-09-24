#!/bin/bash

INPUT_PATH=$1

WORLD_SIZE=8
RETRIEVER_CONFIG="base"
BASE_DIR="."
DATA_DIR="${BASE_DIR}/data"
EVIDENCE_DATA_PATH="${DATA_DIR}/wikipedia-split/psgs_w100.tsv"
VOCAB_FILE="${BASE_DIR}/bert-vocab/bert-large-uncased-vocab.txt"
QA_FILE_DEV="${DATA_DIR}/qas/nq-dev.csv"
QA_FILE_TEST="${DATA_DIR}/qas/nq-test.csv"

CHECKPOINT_PATH="$(dirname "${INPUT_PATH}")"
EMBEDDING_FILE_NAME="$(basename "${CHECKPOINT_PATH}").pkl"
EMBEDDING_PATH="${BASE_DIR}/embedding-path/art-finetuning-embedding-vicuna/mss-retriever-base.pkl"
CREATE_EVIDENCE_INDEXES="true"
EVALUATE_RETRIEVER_RECALL="true"

ITER_NUM="$(cut -d'_' -f2 <<<"$(basename "${INPUT_PATH}")")"
echo "${ITER_NUM}" > "${CHECKPOINT_PATH}/latest_checkpointed_iteration.txt"

# TMP_CHECKPOINT_PATH=${CHECKPOINT_PATH}"-tmp"
# python tools/save_art_retriever.py --load ${CHECKPOINT_PATH} --save ${TMP_CHECKPOINT_PATH} --submodel-name "retriever"
# CHECKPOINT_PATH=${TMP_CHECKPOINT_PATH}

#DISTRIBUTED_ARGS="-m torch.distributed.launch --nproc_per_node ${WORLD_SIZE} --nnodes 1 --node_rank 0 --master_addr localhost --master_port 60000"
DISTRIBUTED_ARGS='torchrun --nproc_per_node ${WORLD_SIZE} --nnodes 1 --node_rank 0 --master_addr localhost --master_port 60001'

function config_base() {
    export CONFIG_ARGS="--num-layers 12 \
--hidden-size 768 \
--num-attention-heads 12 \
--kv-channels 64 \
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


if [ ${RETRIEVER_CONFIG} == "base" ]; then
    config_base
elif [ ${RETRIEVER_CONFIG} == "large" ]; then
    config_large
else
    echo "Invalid model configuration"
    exit 1
fi


OPTIONS=" \
        --batch-size 64 \
        --checkpoint-activations \
        --seq-length 512 \
        --seq-length-retriever 256 \
        --max-position-embeddings 512 \
        --load ${CHECKPOINT_PATH} \
        --evidence-data-path ${EVIDENCE_DATA_PATH} \
        --indexed-evidence-bert-tokenized-data-path ${DATA_DIR}/evidence-wikipedia-indexed-mmap/bert/wikipedia-evidence-bert_text_document \
        --indexed-title-bert-tokenized-data-path ${DATA_DIR}/evidence-wikipedia-indexed-mmap/bert/wikipedia-evidence-bert_title_document \
        --embedding-path ${EMBEDDING_PATH} \
        --indexer-log-interval 1000 \
        --indexer-batch-size 64 \
        --vocab-file ${VOCAB_FILE} \
        --num-workers 64 \
        --fp16 \
        --max-training-rank ${WORLD_SIZE}"


if [ ${CREATE_EVIDENCE_INDEXES} == "true" ];
then
    #COMMAND="WORLD_SIZE=${WORLD_SIZE} python ${DISTRIBUTED_ARGS} create_doc_index.py ${OPTIONS} ${CONFIG_ARGS}"
    COMMAND="${DISTRIBUTED_ARGS} create_doc_index.py ${OPTIONS} ${CONFIG_ARGS}"
    echo "$COMMAND"
    exit
    eval "${COMMAND}"
fi
set +x

exit 1

OPTIONS=" \
    --checkpoint-activations \
    --seq-length 512 \
    --max-position-embeddings 512 \
    --load ${CHECKPOINT_PATH} \
    --evidence-data-path ${EVIDENCE_DATA_PATH} \
    --embedding-path ${EMBEDDING_PATH} \
    --batch-size 16 \
    --seq-length-retriever 256 \
    --vocab-file ${VOCAB_FILE} \
    --qa-file-dev ${QA_FILE_DEV} \
    --qa-file-test ${QA_FILE_TEST} \
    --num-workers 2 \
    --report-topk-accuracies 1 5 20 100 \
    --fp16 \
    --topk-retrievals 100 \
    --save-topk-outputs-path "${DATA_DIR}/retriever-topk-outputs" \
    --max-training-rank ${WORLD_SIZE} "


mkdir "${DATA_DIR}/retriever-topk-outputs"

if [ ${EVALUATE_RETRIEVER_RECALL} == "true" ];
then
#COMMAND="WORLD_SIZE=${WORLD_SIZE} python ${DISTRIBUTED_ARGS} evaluate_open_retrieval.py ${OPTIONS} ${CONFIG_ARGS}"
COMMAND="WORLD_SIZE=${WORLD_SIZE} ${DISTRIBUTED_ARGS} evaluate_open_retrieval.py ${OPTIONS} ${CONFIG_ARGS}"
echo "${COMMAND}"
eval "${COMMAND}"
fi

set +x
