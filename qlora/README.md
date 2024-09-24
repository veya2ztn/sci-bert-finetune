# Qlora

Go to qlora official github page `https://github.com/artidoro/qlora` and fullfill the requirement.
### vicuna script

see Here: `https://github.com/lm-sys/FastChat/blob/main/docs/training.md`
use `accelerate` in python script automatively.

```
python qlora/train_lora.py --model_name_or_path pretrain_weights/vicuna/vicuna-7b-v1.5-16k/ --lora_r 8 \
--lora_alpha 16 --lora_dropout 0.05 --data_path ./data/dummy_conversation.json --bf16 True \
--output_dir ./checkpoints --num_train_epochs 3 --per_device_train_batch_size 1 \
--per_device_eval_batch_size 1 --gradient_accumulation_steps 1 --evaluation_strategy "no" \
--save_strategy "steps" --save_steps 1200 --save_total_limit 100 --learning_rate 2e-5 \
--weight_decay 0. --warmup_ratio 0.03 --lr_scheduler_type "cosine" --logging_steps 1 \
--tf32 False --q_lora True \
--model_max_length 2048 #<------- in 3090, it cannot be large
```
or use `deepspeed qlora/train_lora.py ............. --deepspeed playground/deepspeed_config_s2.json`

The sequence length is different from followed `official script`. The sequence length in this script is large~
### qlora official script
use 
- (`accelerate config` and  `accelerate-launch`) or
- (`accelerate config --config_file $accelerate_config_path` and `accelerate launch --config_file $accelerate_config_path` )
create N-GPU task.

use below commend start a naive pipline task that forward among gpus one by one
```
BATCHSIZE=32; accelerate-launch --config_file config/accelerate/bf16_single.yaml qlora/qlora.py \
--model_name_or_path pretrain_weights/llama2/llama2-7b-hf --use_auth --output_dir ./output/llama-2-guanaco-7b \
--logging_steps 10 --save_strategy steps --data_seed 42 --save_steps 500 --save_total_limit 40 \
--evaluation_strategy steps --eval_dataset_size 1024 --max_eval_samples 1000 --per_device_eval_batch_size $BATCHSIZE \
--max_new_tokens 32 --dataloader_num_workers 1 --group_by_length --logging_strategy steps \
--remove_unused_columns False --do_train --do_eval --lora_r 64 --lora_alpha 16 --lora_modules all \
--double_quant --quant_type nf4 --bf16 --bits 4 --warmup_ratio 0.03 --lr_scheduler_type constant \
--gradient_checkpointing --dataset oasst1 --source_max_len 16 --target_max_len 512 \
--per_device_train_batch_size $BATCHSIZE --gradient_accumulation_steps 1 --max_steps 1875 \
--eval_steps 187 --learning_rate 0.0002 --adam_beta2 0.999 --max_grad_norm 0.3 \
--lora_dropout 0.1 --weight_decay 0.0 --seed 0 --max_memory_MB 24000 \
--dataset_format oasst1
```
The sequence length in this script is small, thus it can afford large batch size.
