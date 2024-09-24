#!/bin/sh
#SBATCH -J PQues      # 作业在调度系统中的作业名为myFirstJob;
#SBATCH -o log/%j-PQues.out  # 脚本执行的输出将被保存在20210827-%j.out文件下，%j表示作业号;
#SBATCH -e log/%j-PQues.out  # 脚本执行的输出将被保存在20210827-%j.out文件下，%j表示作业号;
accelerate-launch --config_file config/accelerate/fp16_single.yaml precompute_embedding.py