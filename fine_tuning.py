#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2020/8/25 上午9:38
# @Author  : qijianwei
# @File    : fine_tuning.py
# @Usage: 在下游任务上微调

import os

import tensorflow as tf

try:
    from modeling import BertConfig
    import centra_bert_core
    import tokenization
except ImportError:
    from .modeling import BertConfig
    from . import centra_bert_core
    from . import tokenization

FLAGS = tf.flags.FLAGS

# general parameter
tf.flags.DEFINE_string("bert_config_file", None, "The path of bert config file")

tf.flags.DEFINE_string("vocab_file", None, "The path of vocabulary file")

tf.flags.DEFINE_string("output_dir", None, "The path for model output")

tf.flags.DEFINE_string("init_checkpoint", None, "The path of initialize checkpoint")

tf.flags.DEFINE_string("task_config", None, "The path of task config file")

tf.flags.DEFINE_integer("eval_batch_size", 8, "The batch size of evaluation")

tf.flags.DEFINE_float("warmup_proportion", 0.1, "The proportion of warm up steps")

tf.flags.DEFINE_integer("save_checkpoints_steps", 1000, "The number of steps to save model")

tf.flags.DEFINE_string("available_tasks", None, "All available tasks")

tf.flags.DEFINE_string("current_task", None, "Current task")

tf.flags.DEFINE_string("log_level", "info", "The level of logs")

tf.flags.DEFINE_string("gpu_id", "", "GPU ID")

# hyper parameter

tf.flags.DEFINE_integer("ex_idx", 0, "The experiment idx")

tf.flags.DEFINE_integer("num_train_epoch", 30, "The number of train epoch")

tf.flags.DEFINE_integer("train_batch_size", 64, "The batch size of train")

tf.flags.DEFINE_float("learning_rate", None, "Learning rate of train")

tf.flags.DEFINE_integer("fine_tuning_layers", None, "Which layers to fine tuning")


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
    current_task = FLAGS.current_task
    if current_task in all_tasks:
        tf.logging.debug("Available tasks: %s, current task: %s"
                         % (" ".join(all_tasks.keys()), current_task))
    else:
        raise ValueError("Current task: %s, is not in the available tasks: %s"
                         % (current_task, " ".join(all_tasks.keys())))

    # Load model config
    bert_config = BertConfig.from_json_file(FLAGS.bert_config_file)
    fine_tuning_layers = []
    if FLAGS.fine_tuning_layers <= bert_config.num_hidden_layers:
        for i in range(int(FLAGS.fine_tuning_layers)):
            fine_tuning_layers.append(bert_config.num_hidden_layers - i)
    else:
        raise ValueError("Fine tuning layers exceed: %d vs %d"
                         % (FLAGS.fine_tuning_layers, bert_config.num_hidden_layers))
    fine_tuning_layers.sort()
    all_tasks[current_task].set_fine_tuning_layers(fine_tuning_layers)

    output_dir = os.path.join(
        FLAGS.output_dir,
        current_task,
        "Lr-%s-Layers-%s" % (FLAGS.learning_rate, FLAGS.fine_tuning_layers),
        "ex-%s" % FLAGS.ex_idx
    )
    tf.gfile.MakeDirs(output_dir)

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=True)

    if int(FLAGS.fine_tuning_layers) < 5:
        learning_rate = 1e-4
    else:
        learning_rate = FLAGS.learning_rate

    summary_file = os.path.join(output_dir, "result_summary.txt")
    with tf.gfile.GFile(summary_file, "w") as summary_writer:
        centra_bert_core.train(
            bert_config=bert_config,
            init_checkpoint=FLAGS.init_checkpoint,
            output_dir=output_dir,
            all_tasks=all_tasks,
            current_task=current_task,
            tokenizer=tokenizer,
            train_batch_size=FLAGS.train_batch_size,
            eval_batch_size=FLAGS.eval_batch_size,
            num_train_epoch=FLAGS.num_train_epoch,
            warmup_proportion=FLAGS.warmup_proportion,
            save_checkpoints_steps=FLAGS.save_checkpoints_steps,
            learning_rate=learning_rate,
            summary_writer=summary_writer,
            is_eng=all_tasks[current_task].is_eng
        )

    summary_writer.close()


if __name__ == "__main__":
    tf.app.run()
