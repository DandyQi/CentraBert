#!/usr/bin/env bash

output_dir=model/glue/teacher
task=mrpc
learning_rate=2e-5,1e-4
fine_tuning_layers=6,8
exam_num=3

python result_summary.py \
    --job=plot \
    --key_param=learning_rate \
    --output_dir=${output_dir} \
    --task=${task} \
    --learning_rate=${learning_rate} \
    --fine_tuning_layers=${fine_tuning_layers} \
    --exam_num=${exam_num} \
    --dev=True \
    --version=teacher