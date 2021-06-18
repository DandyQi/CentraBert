#!/usr/bin/env bash

# General param
bert_config_file=conf/uncased_bert_base/bert_config.json
vocab_file=conf/uncased_bert_base/vocab.txt
output_dir=model/glue/gather
init_checkpoint=model/glue/student/mrpc/Lr-2e-05-Layers-4-2/ex-2/best_checkpoint/1623900741/model.ckpt-572
task_config=conf/glue_task_config.cfg
branch_config=conf/branch.cfg
gather_from_student=True
gpu_id=3
input_file=data/glue/tmp_input_file.txt

# Current tasks
available_tasks=mrpc,rte

# new task info
update_checkpoint=model/glue/student/rte/Lr-2e-05-Layers-3-2/ex-3/best_checkpoint/1623910751/model.ckpt-382
update_scope=rte


python update.py \
    --bert_config_file=${bert_config_file} \
    --vocab_file=${vocab_file} \
    --output_dir=${output_dir} \
    --init_checkpoint=${init_checkpoint} \
    --update_checkpoint=${update_checkpoint} \
    --update_scope=${update_scope} \
    --task_config=${task_config} \
    --branch_config=${branch_config} \
    --available_tasks=${available_tasks} \
    --gather_from_student=${gather_from_student} \
    --gpu_id=${gpu_id} \
    --input_file=${input_file}