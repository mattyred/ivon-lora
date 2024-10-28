#!/bin/bash

for seed in 21 42 87
do
    python finetune.py --optimizer=ivon --task_name=winogrande_s --seed=$seed --ess=1e7 --hess_init=3e-4 --learning_rate=0.03 --save_to=$1 --json_filename=$2
    python finetune.py --optimizer=ivon --task_name=winogrande_m --seed=$seed --ess=1e8 --hess_init=3e-4 --learning_rate=0.03 --save_to=$1 --json_filename=$2
    python finetune.py --optimizer=ivon --task_name=ARC-Challenge --seed=$seed --ess=1e6 --hess_init=1e-3 --learning_rate=0.03 --save_to=$1 --json_filename=$2
    python finetune.py --optimizer=ivon --task_name=ARC-Easy --seed=$seed --ess=1e6 --hess_init=1e-3 --learning_rate=0.03 --save_to=$1 --json_filename=$2
    python finetune.py --optimizer=ivon --task_name=openbookqa --seed=$seed --ess=1e6 --hess_init=1e-3 --learning_rate=0.03 --save_to=$1 --json_filename=$2
    python finetune.py --optimizer=ivon --task_name=boolq --seed=$seed --ess=1e7 --hess_init=3e-4 --learning_rate=0.03 --save_to=$1 --json_filename=$2
done
