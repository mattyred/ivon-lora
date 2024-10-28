#!/bin/bash

for taskname in winogrande_s winogrande_m ARC-Easy ARC-Challenge openbookqa boolq
do
    for seed in 21 42 87
    do
        python finetune.py --task_name=$taskname --seed=$seed --save_to=$1 --json_filename=$2
    done
done
