#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2020/8/26 下午4:54
# @Author  : qijianwei
# @File    : centra_bert_distill.py
# @Usage: 模型蒸馏

import os
import json

import tensorflow as tf

try:
    import centra_bert_core
    import tokenization
except ImportError:
    from . import centra_bert_core
    from . import tokenization

FLAGS = tf.flags.FLAGS

# general parameter
tf.flags.DEFINE_string("bert_config_file", "/fds/conf/bert_config.json", "The path of bert config file")

tf.flags.DEFINE_string("vocab_file", "/fds/conf/vocab.txt", "The path of vocabulary file")

tf.flags.DEFINE_integer("eval_batch_size", 8, "The batch size of evaluation")

tf.flags.DEFINE_float("warmup_proportion", 0.1, "The proportion of warm up steps")

tf.flags.DEFINE_integer("save_checkpoints_steps", 1000, "The number of steps to save model")

tf.flags.DEFINE_integer("gpu_num", 1, "The number of GPU")

tf.flags.DEFINE_string("available_tasks", None, "All available tasks")

tf.flags.DEFINE_string("current_task", None, "Current task")

tf.flags.DEFINE_string("output_dir", None, "The path for model output")

tf.flags.DEFINE_string("task_config", "/fds/conf/qijianwei/task_config.cfg", "The path of task config file")

tf.flags.DEFINE_string("log_level", "info", "The level of logs")

tf.flags.DEFINE_string("teacher_fine_tuning_layers", None, "Which teacher layers to fine tuning")

tf.flags.DEFINE_string("student_fine_tuning_layers", None, "Which student layers to fine tuning")

tf.flags.DEFINE_bool("do_train", False, "Whether to train")

tf.flags.DEFINE_bool("do_eval", False, "Whether to evaluate")

tf.flags.DEFINE_bool("use_lexicon_bert", False, "Whether to use lexicon bert")

tf.flags.DEFINE_bool("is_eng", False, "Is english?")

# hyper parameter

tf.flags.DEFINE_integer("ex_idx", 0, "The experiment idx")

tf.flags.DEFINE_integer("num_train_epoch", 30, "The number of train epoch")

tf.flags.DEFINE_integer("train_batch_size", 64, "The batch size of train")

tf.flags.DEFINE_float("learning_rate", None, "Learning rate of train")

tf.flags.DEFINE_string("best_teacher_checkpoint_file", None, "Best teacher checkpoint file")

tf.flags.DEFINE_string("best_teacher_checkpoint", None, "Best teacher checkpoint")

tf.flags.DEFINE_bool("teacher_exist", True, "Whether teacher exist")

tf.flags.DEFINE_string("student_init_checkpoint", None, "student_init_checkpoint")


def main(_):
    log_level = FLAGS.log_level
    tf.logging.set_verbosity(log_level.upper())
    tf.logging.info(tf.logging.get_verbosity())

    available_tasks = FLAGS.available_tasks.split(",")
    task_config = FLAGS.task_config
    all_tasks = centra_bert_core.load_available_task(
        available_tasks=available_tasks,
        task_config=task_config
    )

    current_task = FLAGS.current_task

    teacher_fine_tuning_layers = []
    student_fine_tuning_layers = []
    keep_layers = 12 - int(FLAGS.teacher_fine_tuning_layers)

    for i in range(int(FLAGS.teacher_fine_tuning_layers)):
        teacher_fine_tuning_layers.append(keep_layers + i + 1)

    for i in range(int(FLAGS.student_fine_tuning_layers)):
        student_fine_tuning_layers.append(keep_layers + i + 1)

    teacher_fine_tuning_layers.sort()
    student_fine_tuning_layers.sort()

    if current_task in all_tasks:
        tf.logging.info("Available tasks: %s, current task: %s"
                        % (" ".join(all_tasks.keys()), current_task))
    else:
        raise ValueError("Current task: %s, is not in the available tasks: %s"
                         % (current_task, " ".join(all_tasks.keys())))

    all_tasks[current_task].set_fine_tuning_layers(teacher_fine_tuning_layers)
    all_tasks[current_task].set_student_fine_tuning_layers(student_fine_tuning_layers)

    output_dir = os.path.join(
        FLAGS.output_dir,
        current_task,
        "%s-%s" % (keep_layers, FLAGS.student_fine_tuning_layers),
        "%s" % FLAGS.learning_rate,
        "ex-%s" % FLAGS.ex_idx
    )
    tf.gfile.MakeDirs(output_dir)

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=True)

    if FLAGS.best_teacher_checkpoint:
        init_checkpoint = FLAGS.best_teacher_checkpoint
    elif FLAGS.best_teacher_checkpoint_file:
        with open(FLAGS.best_teacher_checkpoint_file, "r") as fin:
            init_checkpoint = json.loads(fin.readline())["best_checkpoint_path"]
            tf.logging.info("Initialize from %s" % init_checkpoint)
        fin.close()
    else:
        raise ValueError("No teacher checkpoint found")

    summary_file = os.path.join(output_dir, "result_summary.txt")
    with tf.gfile.GFile(summary_file, "w") as summary_writer:
        centra_bert_core.distill(
            bert_config_file=FLAGS.bert_config_file,
            init_checkpoint=init_checkpoint,
            output_dir=output_dir,
            all_tasks=all_tasks,
            current_task=current_task,
            tokenizer=tokenizer,
            train_batch_size=FLAGS.train_batch_size,
            eval_batch_size=FLAGS.eval_batch_size,
            num_train_epoch=FLAGS.num_train_epoch,
            warmup_proportion=FLAGS.warmup_proportion,
            gpu_num=FLAGS.gpu_num,
            save_checkpoints_steps=FLAGS.save_checkpoints_steps,
            learning_rate=FLAGS.learning_rate,
            summary_writer=summary_writer,
            do_train=FLAGS.do_train,
            do_eval=FLAGS.do_eval,
            use_lexicon_bert=FLAGS.use_lexicon_bert,
            is_eng=FLAGS.is_eng,
            teacher_exist=FLAGS.teacher_exist,
            student_init_checkpoint=FLAGS.student_init_checkpoint
        )

    summary_writer.close()


if __name__ == "__main__":
    tf.app.run()
