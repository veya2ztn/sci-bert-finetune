for i in 0 1 2 3 4 5 6 7 8
do
    sbatch  -p AI4Phys  -N1 -c16 --gres=gpu:1 precompute_question.sh
    sleep 1
done
