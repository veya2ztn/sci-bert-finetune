#!/bin/sh
#SBATCH -J EBD_LLma      # 作业在调度系统中的作业名为myFirstJob;
#SBATCH -o log/%j-EBD_LLma.out  # 脚本执行的输出将被保存在20210827-%j.out文件下，%j表示作业号;
#SBATCH -e log/%j-EBD_LLma.out  # 脚本执行的输出将被保存在20210827-%j.out文件下，%j表示作业号;
accelerate-launch --main_process_port 12334 --config_file config/accelerate/bf16_cards8.yaml train_qlora.py \
--model_name_or_path /mnt/petrelfs/zhangtianning.di/.cache/huggingface/hub/models--PY007--TinyLlama-1.1B-intermediate-step-240k-503b/snapshots/a016b960b941eb6eb7884e363b935370bafa5932/ \
--lora_r 8 --lora_alpha 16 --lora_dropout 0.05 \
--data_path ./data/dummy_conversation.json --bf16 True --output_dir ./checkpoints_longlora \
--num_train_epochs 3 --per_device_train_batch_size 2 --per_device_eval_batch_size 1 --gradient_accumulation_steps 1 \
--evaluation_strategy "no" --save_strategy "steps" --save_steps 10000 --save_total_limit 20 --learning_rate 2e-5 --weight_decay 0. \
--warmup_ratio 0.03 --lr_scheduler_type "cosine" --logging_steps 10 --tf32 False --q_lora True --model_max_length 12000 \
--ddp_find_unused_parameters False --dispatch_batches True --negative_sampler_num 200 \
--flash_attn --report_to wandb tensorboard --use_long_lora True