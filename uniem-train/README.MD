# usage 
不开checkpoing 的极限是 7B-qlora-4k-flash_attn. (70G at 4096) LongLora没有用 还是 (70G at 4096) 
启用全局Buffer 修改
```
data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args,model_max_length = training_args.model_max_length, return_mapping=True)
trainer = BufferWithTrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module) # modify the make_supervised_data_module
```
总的启动代码
```
accelerate-launch --main_process_port 12534 --config_file config/accelerate/bf16_cards8.yaml 
train_qlora.py --model_name_or_path ../pretrain_weights/vicuna/vicuna-7b-v1.5-16k/ --lora_r 8 --lora_alpha 16 --lora_dropout 0.05 --data_path ./data/dummy_conversation.json --bf16 True --output_dir ./checkpoints --num_train_epochs 3 --per_device_train_batch_size 64 --per_device_eval_batch_size 64 --gradient_accumulation_steps 1 --evaluation_strategy "no" --save_strategy "steps" --save_steps 1200 --save_total_limit 100 --learning_rate 2e-5 --weight_decay 0. --warmup_ratio 0.03 --lr_scheduler_type "cosine" --logging_steps 1 --tf32 False --q_lora True --model_max_length 4096 --ddp_find_unused_parameters False --dispatch_batches True --per_device_train_batch_size 1 --flash_attn
```
使用 DeepSpeed 不开 checkpointing 两个方案是一样的。 开了 DeepSpeed 就不能用 CPU-Numpy-Buffer 了，不兼容
```
srun -p AI4Phys  -N1 -c64 --gres=gpu:8 deepspeed  train_qlora.py --model_name_or_path ../pretrain_weights/vicuna/vicuna-7b-v1.5-16k/ --lora_r 8 --lora_alpha 16 --lora_dropout 0.05 --data_path ./data/dummy_conversation.json --bf16 True --output_dir ./checkpoints --num_train_epochs 3 --per_device_train_batch_size 64 --per_device_eval_batch_size 64 --gradient_accumulation_steps 1 --evaluation_strategy "no" --save_strategy "steps" --save_steps 1200 --save_total_limit 100 --learning_rate 2e-5 --weight_decay 0. --warmup_ratio 0.03 --lr_scheduler_type "cosine" --logging_steps 1 --tf32 False --q_lora True --model_max_length 4096 --ddp_find_unused_parameters False --dispatch_batches True --per_device_train_batch_size 1 --flash_attn --q_lora True --deepspeed config/deepspeed/deepspeed_config_s2.json
```
使用 checkpointing 需要将双边cat在一起走单边, 因为checkpoing 只有支持一条线路的计算。
```
        batch_size = question_ids.shape[0]
        question_ids_len = question_ids.shape[1]
        answer_ids_len   = answer_ids.shape[1]
     
        question_ids = torch.nn.functional.pad(question_ids,(0,answer_ids_len-question_ids_len,0,0))
        whole_ids    = torch.cat([question_ids,
                                  answer_ids],dim=0)
        whole_embedding = self.embedder(whole_ids)
        text_embeddings, text_pos_embeddings = torch.split(whole_embedding,batch_size)

```

首先，qlora 是一样的，原版 qlora 可能使用更加详细的配置，但是实测下来对内存没有帮助。
```
model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        load_in_4bit=args.bits == 4,
        load_in_8bit=args.bits == 8,
        device_map=device_map,
        max_memory=max_memory,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=args.bits == 4,
            load_in_8bit=args.bits == 8,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=args.double_quant,
            bnb_4bit_quant_type=args.quant_type,
        ),
        torch_dtype=(torch.float32 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)),
        trust_remote_code=args.trust_remote_code,
        use_auth_token=args.use_auth_token
    )
```