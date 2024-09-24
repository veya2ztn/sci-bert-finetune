#!/bin/bash
#SBATCH -J Jina     # 作业在调度系统中的作业名为myFirstJob;
#SBATCH -o log/%j-Jina.out  # 脚本执行的输出将被保存在20210827-%j.out文件下，%j表示作业号;
#SBATCH -e log/%j-Jina.out  # 脚本执行的输出将被保存在20210827-%j.out文件下，%j表示作业号;
#SBATCH --partition=AI4Phys
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
for v in http_proxy https_proxy HTTP_PROXY HTTPS_PROXY; do export $v=http://zhangtianning.di:Sz3035286@10.1.8.50:33128; done
#export WANDB_WATCH=gradients
export WANDB_PROJECT=JinaEmbed
export WANDB_API_KEY=97f89f82c1c096c3ec08c67ee9abfc0b9c319960
export NCCL_INFO=ERROR

GPUS_PER_NODE=8
export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export NNODES=$SLURM_NNODES
export NODE_RANK=$SLURM_PROCID
export WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

function makehostfile() {
perl -e '$slots=split /,/, $ENV{"SLURM_STEP_GPUS"};
$slots=1 if $slots==0; # workaround 8 gpu machines
@nodes = split /\n/, qx[scontrol show hostnames $ENV{"SLURM_JOB_NODELIST"}];
print map { "$b$_ slots=$slots\n" } @nodes'
}
makehostfile > hostfile

H=`hostname`
THEID=`echo -e $HOSTNAMES  | python3 -c "import sys;[sys.stdout.write(str(i)) for i,line in enumerate(next(sys.stdin).split(' ')) if line.strip() == '$H'.strip()]"`

echo "WORLD_SIZE:" $WORLD_SIZE "NODE_RANK:" $NODE_RANK "DEVICES:" $CUDA_VISIBLE_DEVICES
echo "MASTER_ADDR:" $MASTER_ADDR "MASTER_PORT:" $MASTER_PORT "SLURM_PROCID:" $SLURM_PROCID
echo "NNODES": $NNODES

export CMD="train_qlora.py \
--model_name_or_path ./checkpoints/JinaEmbed_b2a_beta \
--output_dir ./checkpoints/JinaEmbed_b2a_gamma2 --model_max_length 8196 \
--num_train_epochs 2 --learning_rate 1e-3 --warmup_ratio 0.02 --lr_scheduler_type "cosine" --flash_attn --full_finetune --extrapolation_scaling 1 \
--per_device_train_batch_size 64 --gradient_accumulation_steps 16 \
--evaluation_strategy steps --eval_steps 100 \
--eval_datapair_path data/eval/RMP12/pair_question_to_answer_7k.json \
--eval_datapair_question_token_path data/eval/RMP12/question/jina_question_token.npy \
--eval_datapair_answer_token_path data/eval/RMP12/answer_7k/jina_answer_token.npy \
--knowledge_buffer data/unarXive_quantum_physics/answer_version_b/jina_embedding/beta/embedding.npy --add_eval_dataset \
--save_strategy steps --save_steps 100 --save_total_limit 3 \
--bf16 True --logging_steps 1 --tf32 False --q_lora False --ddp_find_unused_parameters False --dispatch_batches False --negative_sampler_num -1 \
--datapair_path data/unarXive_quantum_physics/pair.answer_version_b.question_version_a.json --use_reference_label False \
--weight_decay 0.1 --generate_chunk_size 32 --dataloader_num_workers 2 --optim sophia \
--report_to wandb tensorboard "

export ACCELERATE_USE_DEEPSPEED=true #<----use this enable accelerate deepspeed
export ACCELERATE_DEEPSPEED_CONFIG_FILE=config/deepspeed/deepspeed_config_s1.json #<----use this enable accelerate deepspeed

export LAUNCHER="torchrun \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
    "
echo $LAUNCHER
srun --jobid $SLURM_JOBID bash -c '$LAUNCHER --node_rank $SLURM_PROCID $CMD'
