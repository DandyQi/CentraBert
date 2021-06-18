#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2020/6/23 下午3:26
# @Author  : qijianwei
# @File    : merge_branch.py
# @Usage: merge all branches to one model

import configparser
import os

import tensorflow as tf
from tensorflow.python.estimator.estimator import Estimator
from tensorflow.python.estimator.run_config import RunConfig

try:
    import centra_bert_core
    import tokenization
    from base_task import BaseExample
    from data_processor import BaseProcessor
    from modeling import BertConfig
except ImportError:
    from . import centra_bert_core
    from . import tokenization
    from .base_task import BaseExample
    from .data_processor import BaseProcessor
    from .modeling import BertConfig

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("bert_config_file", None, "The path of bert config file")

tf.flags.DEFINE_string("vocab_file", None, "The path of vocabulary file")

tf.flags.DEFINE_string("output_dir", None, "The path for model output")

tf.flags.DEFINE_string("init_checkpoint", None, "The path of initialize checkpoint")

tf.flags.DEFINE_string("task_config", None, "The path of task config file")

tf.flags.DEFINE_string("branch_config", None, "The path of branch config file")

tf.flags.DEFINE_string("available_tasks", None, "All available tasks")

tf.flags.DEFINE_string("log_level", "info", "The level of logs")

tf.flags.DEFINE_bool("gather_from_student", False, "Whether gather branch from student")

tf.flags.DEFINE_string("gpu_id", "", "GPU ID")

tf.flags.DEFINE_string("input_file", None, "The file of tmp input file")

tf.flags.DEFINE_integer("max_seq_length", 128, "Max sequence length")


def get_examples(input_file):
    examples = []
    with tf.gfile.GFile(input_file, "r") as fin:
        for line in fin.readlines():
            try:
                line = line.strip()
                examples.append(BaseExample(
                    query=tokenization.convert_to_unicode(line),
                    label=None
                ))
            except IOError as e:
                tf.logging.error("Load file error: " + e + ", line: " + line)
    fin.close()
    return examples


def main(_):
    log_level = FLAGS.log_level
    tf.logging.set_verbosity(log_level.upper())
    tf.logging.debug(tf.logging.get_verbosity())
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu_id

    # Load task config
    available_tasks = FLAGS.available_tasks.split(",")
    task_config = FLAGS.task_config
    all_tasks = centra_bert_core.load_available_task(
        available_tasks=available_tasks,
        task_config=task_config
    )

    # Load model config
    bert_config = BertConfig.from_json_file(FLAGS.bert_config_file)
    branch_config = configparser.ConfigParser()
    branch_config.read(FLAGS.branch_config)
    task_fine_tuning_layers = branch_config.items("layer_conf")
    for (task, layers) in task_fine_tuning_layers:
        if task in available_tasks:
            all_tasks[task].set_fine_tuning_layers([int(layer) for layer in layers.split(",")])
    available_tasks_bottom_layers = [int(layers.split(",")[0]) for (task, layers) in task_fine_tuning_layers
                                     if task in available_tasks]
    max_bottom_layer = max(available_tasks_bottom_layers)
    bert_config.num_hidden_layers = max_bottom_layer - 1

    tokenizer = tokenization.FullTokenizer(vocab_file=FLAGS.vocab_file, do_lower_case=True)
    all_label_map = {}
    for task in all_tasks:
        labels = all_tasks[task].get_labels()
        id2label_map = {}
        for idx, label in enumerate(labels):
            id2label_map[idx] = label
        all_label_map["%s_id2label" % task] = id2label_map

    task_init_checkpoints = branch_config.items("ckpt_conf")
    task_inits = {}
    for task_name, task_init_ckpt in task_init_checkpoints:
        task_inits[task_name] = task_init_ckpt

    output_dir = None
    for idx, task_name in enumerate(available_tasks):

        task_init_ckpt = task_inits[task_name.lower()]
        if idx == 0:
            output_dir = os.path.join(
                FLAGS.output_dir,
                task_name
            )
            tf.gfile.MakeDirs(output_dir)
            init_checkpoint = FLAGS.init_checkpoint
        else:
            init_checkpoint = os.path.join(
                output_dir,
                "model.ckpt-0"
            )
            output_dir = os.path.join(
                FLAGS.output_dir,
                task_name
            )

        tf.logging.debug("Init from %s, update %s from %s" % (init_checkpoint, task_name, task_init_ckpt))

        run_config = RunConfig(model_dir=output_dir)

        model_fn = centra_bert_core.model_fn_builder(
            bert_config=bert_config,
            init_checkpoint=init_checkpoint,
            max_seq_length=FLAGS.max_seq_length,
            tasks=all_tasks,
            update_checkpoint=task_init_ckpt,
            update_scope=task_name,
            update_from_distill=FLAGS.gather_from_student
        )

        dp = BaseProcessor(
            task_type=all_tasks[task_name].task_type,
            max_seq_length=all_tasks[task_name].max_seq_length,
            is_eng=all_tasks[task_name].is_eng
        )

        estimator = Estimator(
            model_fn=model_fn,
            params={
                "current_task": task_name,
                "train_batch_size": 32,
                "eval_batch_size": 8
            },
            config=run_config
        )

        examples = get_examples(FLAGS.input_file)
        features = dp.convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            label2id_map=None
        )
        input_fn = dp.input_fn_builder(features, is_training=False)
        save_hook = tf.train.CheckpointSaverHook(output_dir, save_secs=1)
        result = estimator.predict(input_fn=input_fn, hooks=[save_hook])

        output_file = os.path.join(output_dir, "predict_result.txt")
        with tf.gfile.GFile(output_file, "w") as writer:
            for res_idx, res in enumerate(result):
                query = examples[res_idx]
                content = {"query": query, "result": res}
                writer.write("%s\n" % content)

        writer.close()


if __name__ == "__main__":
    tf.app.run()
