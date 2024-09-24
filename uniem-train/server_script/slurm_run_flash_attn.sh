#!/bin/bash
#SBATCH -J Lora     # 作业在调度系统中的作业名为myFirstJob;
#SBATCH -o log/%j-Lora.out  # 脚本执行的输出将被保存在20210827-%j.out文件下，%j表示作业号;
#SBATCH -e log/%j-Lora.out  # 脚本执行的输出将被保存在20210827-%j.out文件下，%j表示作业号;
#SBATCH --job-name=Lora
#SBATCH --partition=AI4Phys
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16

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

export CMD="train_qlora.py --model_name_or_path pretrain_weights/models--jinaai--jina-embeddings-v2-base-en/snapshots/7302ac470bed880590f9344bfeee32ff8722d0e5 \
--bf16 True --output_dir ./checkpoints/JinaEmbed_b2a_gamma --num_train_epochs 2 --per_device_train_batch_size 60 \
--gradient_accumulation_steps 60 --evaluation_strategy "no" --save_strategy "steps" --save_steps 100 --save_total_limit 3 \
--learning_rate 1e-4 --warmup_ratio 0.02 --lr_scheduler_type "cosine" --logging_steps 1 --tf32 False --q_lora False \
--model_max_length 8196 --ddp_find_unused_parameters False --dispatch_batches False --negative_sampler_num -1 --full_finetune \
--datapair_path data/unarXive_quantum_physics/pair.answer_version_b.question_version_a.json \
--use_reference_label False --extrapolation_scaling 1 --flash_attn True --weight_decay 0.1 \
--generate_chunk_size 8 --dataloader_num_workers 2 \
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
