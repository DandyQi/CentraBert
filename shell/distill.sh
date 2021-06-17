#!/usr/bin/env bash

# General param
bert_config_file=conf/uncased_bert_base/bert_config.json
vocab_file=conf/uncased_bert_base/vocab.txt
output_dir=model/glue/student
task_config=conf/glue_task_config.cfg
num_train_epoch=10
train_batch_size=64
gpu_id=6

# Teacher info
teacher_fine_tuning_layers=9
best_teacher_checkpoint=model/glue/teacher/rte/Lr-2e-05-Layers-9/ex-3/best_checkpoint/1623902794/model.ckpt-380

# Current task
task=rte

# Hyper param, separated by commas
learning_rate=2e-5
fine_tuning_layers=1,2

# Number of repetitions for each hyper parameter
exam_num=3

for lr in ${learning_rate//,/ }
do
    for layers in ${fine_tuning_layers//,/ }
    do
        for i in $(seq 1 ${exam_num})
        do
            python distill.py \
                --bert_config_file=${bert_config_file} \
                --vocab_file=${vocab_file} \
                --output_dir=${output_dir} \
                --best_teacher_checkpoint=${best_teacher_checkpoint} \
                --teacher_fine_tuning_layers=${teacher_fine_tuning_layers} \
                --task_config=${task_config} \
                --available_tasks=${task} \
                --current_task=${task} \
                --ex_idx=${i} \
                --num_train_epoch=${num_train_epoch} \
                --train_batch_size=${train_batch_size} \
                --learning_rate=${lr} \
                --student_fine_tuning_layers=${layers} \
                --gpu_id=${gpu_id}
        done
    done
done

# Result summary
python result_summary.py \
    --output_dir=${output_dir} \
    --task=${task} \
    --learning_rate=${learning_rate} \
    --fine_tuning_layers=${fine_tuning_layers} \
    --exam_num=${exam_num} \
    --dev=True \
    --keep_layers=$((12-teacher_fine_tuning_layers)) \
    --version=student