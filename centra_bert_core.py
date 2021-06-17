#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/4/18 上午11:18
# @Author  : qijianwei
# @File    : centra_bert_core.py
# @Usage: CentraBERT网络结构主要代码


import configparser
import json
import os

import tensorflow as tf
from tensorflow.python.estimator.estimator import Estimator
from tensorflow.python.estimator.run_config import RunConfig

try:
    import best_exporter
    import slot_eval
    import tokenization
    from base_task import BaseTask
    from data_processor import BaseProcessor
    from model_builder import model_fn_builder
except ImportError:
    from . import best_exporter
    from . import slot_eval
    from . import tokenization
    from .base_task import BaseTask
    from .data_processor import BaseProcessor
    from .model_builder import model_fn_builder


def check_ignored_slots(tags, slots_to_ignore):
    if slots_to_ignore is None:
        return tags
    ret = list()
    for tag in tags:
        flag_ignore = False
        for slot in slots_to_ignore:
            if tag == "B-" + slot or tag == "I-" + slot:
                flag_ignore = True
                break
        if flag_ignore:
            ret.append("O")
        else:
            ret.append(tag)
    return ret


def evaluate_tagging(result, tokenizer, id2label, current_task, test_idx, output_file, is_student=False):
    truth_tags = []
    predict_tags = []

    with tf.gfile.GFile(output_file, "w") as detail_writer:
        for res_idx, res in enumerate(result):
            label_pred = res["%s" % "student/" + current_task if is_student else current_task]
            truth_label = res["%s_truth_label" % current_task]
            input_ids = res["%s_input_ids" % current_task]
            input_mask = res["%s_input_mask" % current_task]

            valid_input_ids = [input_id for input_id, mask in zip(input_ids, input_mask) if mask == 1]
            valid_truth_label = [l for l, m in zip(truth_label, input_mask) if m == 1]
            valid_label_pred = [l for l, m in zip(label_pred, input_mask) if m == 1]
            tokens = tokenizer.convert_ids_to_tokens(valid_input_ids)

            detail_writer.write("%s\t%s\t%s\n"
                                % (" ".join(tokens),
                                   " ".join([id2label[label] for label in valid_truth_label]),
                                   " ".join([id2label[label] for label in valid_label_pred]))
                                )
            preds = [id2label[label] for label in valid_label_pred]
            labels = [id2label[label] for label in valid_truth_label]

            truth_tags.extend(check_ignored_slots(labels, ("time",)))
            truth_tags.append("O")
            predict_tags.extend(check_ignored_slots(preds, ("time",)))
            predict_tags.append("O")

    prec, rec, f1 = slot_eval.evaluate(truth_tags, predict_tags, verbose=False)
    result = {"test_set": "%s_%s" % (current_task, test_idx), "f1": f1, "precision": prec, "recall": rec}

    return result


def evaluate_classification(result, tokenizer, id2label, current_task, test_idx, output_file, is_student=False):
    total_count = 0
    correct_count = 0.0
    with tf.gfile.GFile(output_file, "w") as detail_writer:
        for res_idx, res in enumerate(result):
            label_pred = res["%s_label" % ("student/" + current_task if is_student else current_task)]
            truth_label = res["%s_truth_label" % current_task]
            input_ids = res["%s_input_ids" % current_task]
            input_mask = res["%s_input_mask" % current_task]

            valid_input_ids = [input_id for input_id, mask in zip(input_ids, input_mask) if mask == 1]
            tokens = tokenizer.convert_ids_to_tokens(valid_input_ids)

            detail_writer.write("%s\t%s\t%s\n" % (" ".join(tokens), id2label[truth_label], id2label[label_pred]))

            if truth_label == label_pred:
                correct_count += 1
            total_count += 1

    accuracy = round(correct_count / total_count * 100, 3)
    result = {"test_set": "%s_%s" % (current_task, test_idx), "accuracy": accuracy}

    return result


def task_evaluate(task_name, task_type, idx, result, id2label, output_dir, tokenizer, is_student=False):
    """
    评测模型在下游任务上的表现，对于分类任务，输出 accuracy，对于槽位抽取任务，输出 P/R/F1
    Args:
        task_name: str, 当前任务名称
        task_type: str, 当前任务类型
        idx: int, 当前任务评测集序号
        result: Iterator(dict), 模型返回的结果
        id2label: Dict(int, str), label id 到 label 的映射
        output_dir: str, 评测时详细结果的输出路径
        tokenizer: FullTokenizer
        is_student: bool, 是否为student branch

    Returns:
        evaluate_result: json, 评测指标

    """
    output_predict_result = os.path.join(output_dir, "%s_results_%s.txt"
                                         % ("student_" + task_name if is_student else task_name, idx + 1))

    if task_type == "tagging":
        return evaluate_tagging(
            result=result,
            tokenizer=tokenizer,
            id2label=id2label,
            current_task=task_name,
            test_idx=idx,
            output_file=output_predict_result,
            is_student=is_student
        )
    elif task_type == "classification":
        return evaluate_classification(
            result=result,
            tokenizer=tokenizer,
            id2label=id2label,
            current_task=task_name,
            test_idx=idx,
            output_file=output_predict_result,
            is_student=is_student
        )
    else:
        raise ValueError("Task type %s is not supported, it have to be tagging or classification" % task_type)


def construct_estimator(bert_config, init_checkpoint, learning_rate, all_tasks, current_task,
                        all_train_steps, num_warmup_steps, output_dir, save_checkpoints_steps,
                        train_batch_size, eval_batch_size, use_distill=False, teacher_exist=True,
                        student_init_checkpoint=None):
    model_fn = model_fn_builder(
        bert_config=bert_config,
        init_checkpoint=init_checkpoint,
        learning_rate=learning_rate,
        max_seq_length=all_tasks[current_task].max_seq_length,
        num_train_steps=all_train_steps,
        num_warmup_steps=num_warmup_steps,
        tasks=all_tasks,
        use_distill=use_distill,
        teacher_exist=teacher_exist,
        student_init_checkpoint=student_init_checkpoint
    )

    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True
    session_config.allow_soft_placement = True

    run_config = RunConfig(
        model_dir=output_dir,
        save_checkpoints_steps=save_checkpoints_steps,
        session_config=session_config
    )

    tf.logging.debug("Use normal Estimator")
    estimator = Estimator(
        model_fn=model_fn,
        params={
            "current_task": current_task,
            "train_batch_size": train_batch_size,
            "eval_batch_size": eval_batch_size
        },
        config=run_config
    )

    return estimator


def train(bert_config, init_checkpoint, output_dir, all_tasks, current_task, tokenizer,
          train_batch_size, eval_batch_size, num_train_epoch, warmup_proportion, save_checkpoints_steps,
          learning_rate, summary_writer, is_eng):
    """
    执行下游任务训练
    Args:
        bert_config: BertConfig, Bert 相关配置
        init_checkpoint: str, 初始化模型文件路径
        output_dir: str, 训练输出路径
        all_tasks: dict(task_name, BaseTask), task name 到 BaseTask 的映射
        current_task: str, 当前任务名称
        tokenizer: FullTokenizer, Bert 源码中的切词器
        train_batch_size: int, 训练时的 batch size
        eval_batch_size: int, 评测时的 batch size
        num_train_epoch: int, 训练所需 epoch 数量
        warmup_proportion: float, 训练步数预热比例
        save_checkpoints_steps: int, 模型保存间隔
        learning_rate: float, 学习速率
        summary_writer: FileWriter, 评测结果写入文件
        is_eng: bool, 是否为英文，若为英文将按照英文进行分词

    Returns:

    """

    labels = all_tasks[current_task].get_labels()

    label2id_map = {}
    id2label_map = {}
    for idx, label in enumerate(labels):
        label2id_map[label] = idx
        id2label_map[idx] = label

    all_tasks[current_task].set_label2idx_map(label2idx_map=label2id_map)

    tf.logging.debug("Labels: %s" % " ".join(labels))

    tf.logging.debug("Use estimator config")

    train_example, test_examples = all_tasks[current_task].get_data()
    num_train_steps = int(len(train_example) / train_batch_size)
    all_train_steps = int(num_train_steps * num_train_epoch)
    num_warmup_steps = int(num_train_steps * warmup_proportion)

    # all_train_steps = max(10000, all_train_steps)

    dp = BaseProcessor(
        task_type=all_tasks[current_task].task_type,
        max_seq_length=all_tasks[current_task].max_seq_length,
        is_eng=is_eng
    )

    estimator = construct_estimator(
        bert_config=bert_config,
        init_checkpoint=init_checkpoint,
        learning_rate=learning_rate,
        all_tasks=all_tasks,
        current_task=current_task,
        all_train_steps=all_train_steps,
        num_warmup_steps=num_warmup_steps,
        save_checkpoints_steps=save_checkpoints_steps,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        output_dir=output_dir
    )

    tf.logging.debug("***** Running training *****")
    tf.logging.debug("  Num examples = %d", len(train_example))
    tf.logging.debug("  Batch size = %d", train_batch_size)
    tf.logging.debug("  Num steps = %d", all_train_steps)

    train_file = os.path.join(output_dir, "train.tf_record")
    dp.file_based_convert_examples_to_features(
        examples=train_example,
        label2id_map=label2id_map,
        tokenizer=tokenizer,
        output_file=train_file)

    train_input_fn = dp.file_based_input_fn_builder(
        input_file=train_file,
        is_training=True
    )

    test_input_fn_list = []

    for test_example in test_examples:
        test_input_fn_list.append(
            dp.input_fn_builder(
                features=dp.convert_examples_to_features(
                    examples=test_example,
                    label2id_map=label2id_map,
                    tokenizer=tokenizer
                ),
                is_training=False
            )
        )

    best_ckpt_dir = os.path.join(output_dir, "best_checkpoint")
    tf.gfile.MakeDirs(best_ckpt_dir)

    best_ckpt_exporter = best_exporter.BestCheckpointsExporter(
        serving_input_receiver_fn=best_exporter.serving_fn,
        best_checkpoint_path=best_ckpt_dir,
        compare_fn=best_exporter.loss_smaller
    )

    train_spec = tf.estimator.TrainSpec(
        input_fn=train_input_fn,
        max_steps=all_train_steps
    )

    eval_spec = tf.estimator.EvalSpec(
        input_fn=test_input_fn_list[0],
        steps=None,
        start_delay_secs=60,
        throttle_secs=60,
        exporters=best_ckpt_exporter,
        name="%s_eval" % current_task
    )

    tf.estimator.train_and_evaluate(
        estimator=estimator,
        train_spec=train_spec,
        eval_spec=eval_spec
    )

    with tf.gfile.GFile(os.path.join(best_ckpt_dir, "best_checkpoint.txt"), "r") as fin:
        best_checkpoint = json.loads(fin.readline())["best_checkpoint_path"]
    fin.close()

    tf.logging.debug("***** Evaluate on the best checkpoint: %s *****" % best_checkpoint)

    best_estimator = construct_estimator(
        bert_config=bert_config,
        init_checkpoint=best_checkpoint,
        learning_rate=learning_rate,
        all_tasks=all_tasks,
        current_task=current_task,
        all_train_steps=all_train_steps,
        num_warmup_steps=num_warmup_steps,
        save_checkpoints_steps=save_checkpoints_steps,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        output_dir=best_ckpt_dir
    )

    for idx, test_input_fn in enumerate(test_input_fn_list):
        result = best_estimator.predict(test_input_fn, yield_single_examples=True)
        evaluate_metrics = task_evaluate(
            task_name=current_task,
            task_type=all_tasks[current_task].task_type,
            id2label=id2label_map,
            idx=idx,
            output_dir=output_dir,
            result=result,
            tokenizer=tokenizer
        )

        summary_writer.write("%s\n" % json.dumps(evaluate_metrics))


def evaluate(bert_config_file, init_checkpoint, output_dir, all_tasks, current_task, tokenizer,
             train_batch_size, eval_batch_size, summary_writer, is_eng=False):
    """

    Args:
        bert_config_file: str, Bert 相关配置
        init_checkpoint: str, 初始化模型文件路径
        output_dir: str, 训练输出路径
        all_tasks: dict(task_name, BaseTask), task name 到 BaseTask 的映射
        current_task: str, 当前任务名称
        tokenizer: FullTokenizer, Bert 源码中的切词器
        train_batch_size: int, 训练时的 batch size
        eval_batch_size: int, 评测时的 batch size
        summary_writer: FileObj, 写入评测结果
        is_eng: bool, 是否为英文，若为英文将按照英文进行分词

    Returns:

    """
    model_fn = model_fn_builder(
        bert_config=bert_config_file,
        init_checkpoint=init_checkpoint,
        max_seq_length=all_tasks[current_task].max_seq_length,
        tasks=all_tasks
    )

    dp = BaseProcessor(
        task_type=all_tasks[current_task].task_type,
        max_seq_length=all_tasks[current_task].max_seq_length,
        is_eng=is_eng
    )

    run_config = RunConfig(model_dir=output_dir)

    tf.logging.debug("Use normal Estimator")
    estimator = Estimator(
        model_fn=model_fn,
        params={
            "current_task": current_task,
            "train_batch_size": train_batch_size,
            "eval_batch_size": eval_batch_size
        },
        config=run_config
    )

    labels = all_tasks[current_task].get_labels()
    label2id_map = {}
    id2label_map = {}
    for idx, label in enumerate(labels):
        label2id_map[label] = idx
        id2label_map[idx] = label

    all_tasks[current_task].set_label2idx_map(label2idx_map=label2id_map)

    _, test_examples = all_tasks[current_task].get_data()
    if len(test_examples) > 0:
        for idx, test_example in enumerate(test_examples):
            test_input_fn = dp.input_fn_builder(
                features=dp.convert_examples_to_features(
                    examples=test_example,
                    label2id_map=label2id_map,
                    tokenizer=tokenizer
                ),
                is_training=False
            )
            tf.logging.debug("***** Running evaluation on %s-%d test set *****" % (current_task, idx + 1))
            tf.logging.debug("  Num examples = %d", len(test_example))
            result = estimator.predict(input_fn=test_input_fn)
            summary_content = task_evaluate(
                idx=idx,
                result=result,
                id2label=id2label_map,
                task_name=current_task,
                task_type=all_tasks[current_task].task_type,
                output_dir=output_dir,
                tokenizer=tokenizer
            )
            summary_writer.write("%s\n" % json.dumps(summary_content))
    else:
        tf.logging.debug("***** Skip evaluation, dev set is empty *****")


def distill(bert_config, init_checkpoint, output_dir, all_tasks, current_task, tokenizer,
            train_batch_size, eval_batch_size, num_train_epoch, warmup_proportion, save_checkpoints_steps,
            learning_rate, summary_writer, is_eng=False, teacher_exist=True,
            student_init_checkpoint=None):
    """
        执行下游任务训练
        Args:
            bert_config: BertConfig, Bert 相关配置
            init_checkpoint: str, 初始化模型文件路径
            output_dir: str, 训练输出路径
            all_tasks: dict(task_name, BaseTask), task name 到 BaseTask 的映射
            current_task: str, 当前任务名称
            tokenizer: FullTokenizer, Bert 源码中的切词器
            train_batch_size: int, 训练时的 batch size
            eval_batch_size: int, 评测时的 batch size
            num_train_epoch: int, 训练所需 epoch 数量
            warmup_proportion: float, 训练步数预热比例
            save_checkpoints_steps: int, 模型保存间隔
            learning_rate: float, 学习速率
            summary_writer: FileWriter, 评测结果写入文件
            is_eng: bool, 是否为英文，若为英文将按照英文进行分词
            teacher_exist: bool, checkpoint中是否已存在teacher分支，若没有teacher分支，student将从主干初始化
            student_init_checkpoint:

        Returns:

        """

    if current_task in all_tasks:
        tf.logging.debug("Available tasks: %s, current task: %s"
                         % (" ".join(all_tasks.keys()), current_task))
    else:
        raise ValueError("Current task: %s, is not in the available tasks: %s"
                         % (current_task, " ".join(all_tasks.keys())))

    tf.gfile.MakeDirs(output_dir)

    labels = all_tasks[current_task].get_labels()

    label2id_map = {}
    id2label_map = {}
    for idx, label in enumerate(labels):
        label2id_map[label] = idx
        id2label_map[idx] = label

    all_tasks[current_task].set_label2idx_map(label2idx_map=label2id_map)

    tf.logging.debug("Labels: %s" % " ".join(labels))

    tf.logging.debug("Use estimator config")

    train_example, test_examples = all_tasks[current_task].get_data()
    num_train_steps = int(len(train_example) / train_batch_size)
    all_train_steps = int(num_train_steps * num_train_epoch)
    num_warmup_steps = int(num_train_steps * warmup_proportion)

    dp = BaseProcessor(
        task_type=all_tasks[current_task].task_type,
        max_seq_length=all_tasks[current_task].max_seq_length,
        is_eng=is_eng
    )

    # all_train_steps = max(all_train_steps, 100000)

    estimator = construct_estimator(
        bert_config=bert_config,
        init_checkpoint=init_checkpoint,
        learning_rate=learning_rate,
        all_tasks=all_tasks,
        current_task=current_task,
        all_train_steps=all_train_steps,
        num_warmup_steps=num_warmup_steps,
        save_checkpoints_steps=save_checkpoints_steps,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        output_dir=output_dir,
        use_distill=True,
        teacher_exist=teacher_exist,
        student_init_checkpoint=student_init_checkpoint
    )

    tf.logging.debug("***** Running training *****")
    tf.logging.debug("  Num examples = %d", len(train_example))
    tf.logging.debug("  Batch size = %d", train_batch_size)
    tf.logging.debug("  Num steps = %d", all_train_steps)

    train_file = os.path.join(output_dir, "train.tf_record")
    dp.file_based_convert_examples_to_features(
        examples=train_example,
        label2id_map=label2id_map,
        tokenizer=tokenizer,
        output_file=train_file)

    train_input_fn = dp.file_based_input_fn_builder(
        input_file=train_file,
        is_training=True
    )

    test_input_fn_list = []

    for test_example in test_examples:
        test_input_fn_list.append(
            dp.input_fn_builder(
                features=dp.convert_examples_to_features(
                    examples=test_example,
                    label2id_map=label2id_map,
                    tokenizer=tokenizer
                ),
                is_training=False
            )
        )
    best_ckpt_dir = os.path.join(output_dir, "best_checkpoint")
    tf.gfile.MakeDirs(best_ckpt_dir)

    best_ckpt_exporter = best_exporter.BestCheckpointsExporter(
        serving_input_receiver_fn=best_exporter.serving_fn,
        best_checkpoint_path=best_ckpt_dir,
        compare_fn=best_exporter.loss_smaller
    )

    train_spec = tf.estimator.TrainSpec(
        input_fn=train_input_fn,
        max_steps=all_train_steps
    )

    eval_spec = tf.estimator.EvalSpec(
        input_fn=test_input_fn_list[0],
        steps=None,
        start_delay_secs=60,
        throttle_secs=60,
        exporters=best_ckpt_exporter,
        name="%s_eval" % current_task
    )

    tf.estimator.train_and_evaluate(
        estimator=estimator,
        train_spec=train_spec,
        eval_spec=eval_spec
    )

    best_checkpoint = None
    if os.path.exists(os.path.join(best_ckpt_dir, "best_checkpoint.txt")):
        with tf.gfile.GFile(os.path.join(best_ckpt_dir, "best_checkpoint.txt"), "r") as fin:
            best_checkpoint = json.loads(fin.readline())["best_checkpoint_path"]
        fin.close()

    tf.logging.debug("***** Evaluate on the best checkpoint: %s *****" % best_checkpoint)

    if best_checkpoint is None:
        best_checkpoint = init_checkpoint

    best_estimator = construct_estimator(
        bert_config=bert_config,
        init_checkpoint=best_checkpoint,
        learning_rate=learning_rate,
        all_tasks=all_tasks,
        current_task=current_task,
        all_train_steps=all_train_steps,
        num_warmup_steps=num_warmup_steps,
        save_checkpoints_steps=save_checkpoints_steps,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        output_dir=best_ckpt_dir,
        use_distill=True
    )

    for idx, test_input_fn in enumerate(test_input_fn_list):
        result = best_estimator.predict(test_input_fn, yield_single_examples=True)
        teacher_evaluate_metrics = task_evaluate(
            task_name=current_task,
            task_type=all_tasks[current_task].task_type,
            id2label=id2label_map,
            idx=idx,
            output_dir=output_dir,
            result=result,
            tokenizer=tokenizer,
            is_student=False
        )
        result = best_estimator.predict(test_input_fn, yield_single_examples=True)
        student_evaluate_metrics = task_evaluate(
            task_name=current_task,
            task_type=all_tasks[current_task].task_type,
            id2label=id2label_map,
            idx=idx,
            output_dir=output_dir,
            result=result,
            tokenizer=tokenizer,
            is_student=True
        )
        summary_writer.write("%s\n" % json.dumps({
            "teacher": teacher_evaluate_metrics,
            "student": student_evaluate_metrics
        }))


def load_available_task(available_tasks, task_config):
    """
    加载可用的下游任务
    Args:
        available_tasks: list(str), 下游任务名称列表
        task_config: str, 下游任务配置文件

    Returns:
        all_tasks: dict(task_name, BaseTask), task name 到 BaseTask 的映射

    """
    config = configparser.ConfigParser()
    config.read(task_config)
    all_tasks = {}
    for task_name in available_tasks:
        conf_section = "%s_conf" % task_name
        task_processor = BaseTask(
            task_name=task_name,
            task_type=config.get(conf_section, "task_type"),
            input_files=config.get(conf_section, "input_file"),
            max_seq_length=config.getint(conf_section, "max_seq_length"),
            output_method=config.get(conf_section, "output_method"),
            is_eng=config.getboolean(conf_section, "is_eng")
        )

        all_tasks[task_name] = task_processor
        labels = task_processor.get_labels()
        task_processor.set_output_num(len(labels))
    return all_tasks
