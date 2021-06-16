#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2021/1/28 下午4:19
# @Author  : qijianwei
# @File    : model_builder.py
# @Usage: 创建计算图


import tensorflow as tf

try:
    import custom_checkpoint_utils
    import modeling
    import optimization
except ImportError:
    from . import custom_checkpoint_utils
    from . import modeling
    from . import optimization


def model_fn_builder(bert_config, init_checkpoint, tasks, max_seq_length,
                     num_train_steps=100, num_warmup_steps=0.01, learning_rate=0.00002,
                     floatx='float32', do_export=False,
                     update_checkpoint=None, update_scope=None, use_distill=False, update_from_distill=False,
                     teacher_exist=True, student_init_checkpoint=None):
    def model_fn(features, labels, mode, params):
        if labels is None:
            pass

        if "current_task" in params.keys():
            current_task = params["current_task"]
        else:
            current_task = None

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        if "label_ids" in features.keys():
            label_ids = features["label_ids"]
        else:
            label_ids = None

        model, input_ids, input_mask, lengths = build_bert_base_model(
            bert_config, features, is_training, max_seq_length, do_export, floatx)

        encoder_layers = [model.get_embedding_output()]

        encoder_layers.extend(model.get_all_encoder_layers())

        attention_mask = modeling.create_attention_mask_from_input_mask(input_ids, input_mask, floatx=floatx)

        fine_tuning_encoder_layers = downstream_task_fine_tuning_outputs(
            bert_config=bert_config,
            attention_mask=attention_mask,
            input_tensor_list=encoder_layers,
            tasks=tasks,
            floatx=floatx,
            is_training=is_training,
            is_student=False
        )

        outputs, predictions = downstream_task_outputs(
            bert_config=bert_config,
            public_input_tensor=encoder_layers,
            custom_input_tensor=fine_tuning_encoder_layers,
            lengths=lengths,
            label_ids=label_ids,
            is_training=is_training,
            tasks=tasks,
            do_export=do_export,
            current_task=current_task,
            is_student=False,
            floatx=floatx,
            attention_mask=attention_mask
        )
        distill_loss, student_outputs, student_predictions = None, None, None
        if use_distill:
            student_fine_tuning_encoder_layers = downstream_task_fine_tuning_outputs(
                bert_config=bert_config,
                attention_mask=attention_mask,
                input_tensor_list=encoder_layers,
                tasks=tasks,
                floatx=floatx,
                is_training=is_training,
                is_student=True
            )

            student_outputs, student_predictions = downstream_task_outputs(
                bert_config=bert_config,
                public_input_tensor=encoder_layers,
                custom_input_tensor=student_fine_tuning_encoder_layers,
                lengths=lengths,
                label_ids=label_ids,
                is_training=is_training,
                tasks=tasks,
                do_export=do_export,
                current_task=current_task,
                is_student=True,
                floatx=floatx
            )

            distill_loss = knowledge_distill(
                hard_loss=student_outputs[current_task].loss,
                student_logits=student_outputs[current_task].logits,
                teacher_logits=outputs[current_task].logits
            )

        tvars = tf.trainable_variables()

        initialized_variable_names = {}
        update_variable_names = {}
        student_variable_names = {}
        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint, tasks,
                                                             use_distill=use_distill, teacher_exist=teacher_exist)

            custom_checkpoint_utils.init_from_checkpoint(init_checkpoint, assignment_map)

            if update_checkpoint:
                if update_scope is None:
                    raise ValueError("update_scope have to be specified when update_checkpoint is not None")
                update_assignment_map, update_variable_names = modeling.get_assignment_map_for_update(
                    tvars=tvars,
                    init_checkpoint=update_checkpoint,
                    update_scope=update_scope,
                    update_from_distill=update_from_distill
                )
                custom_checkpoint_utils.init_from_checkpoint(update_checkpoint, update_assignment_map)

            if student_init_checkpoint:
                student_init_assignment_map, student_variable_names = modeling.get_assignment_map_for_student_init(
                    tvars=tvars,
                    init_checkpoint=student_init_checkpoint,
                    tasks=tasks
                )
                custom_checkpoint_utils.init_from_checkpoint(student_init_checkpoint, student_init_assignment_map)

        tf.logging.info("**** All Variables ****")
        for var in tvars:
            init_string = ""
            update_string = ""
            stu_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            if var.name in update_variable_names:
                update_string = ", *UPDATE_FROM_CKPT*"
            if var.name in student_variable_names:
                stu_string = ", *STU_INIT*"
            tf.logging.info("  name = %s, shape = %s%s%s%s",
                            var.name, var.shape, init_string, update_string, stu_string)

        if mode == tf.estimator.ModeKeys.TRAIN:
            assert current_task is not None
            if use_distill:
                current_loss = distill_loss
            else:
                current_loss = outputs[current_task].loss
            current_loss = tf.identity(current_loss, name="loss")

            train_op = optimization.create_optimizer(
                loss=current_loss,
                init_lr=learning_rate,
                num_train_steps=num_train_steps,
                num_warmup_steps=num_warmup_steps,
                current_task_name=current_task,
                use_distill=use_distill
            )
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=current_loss,
                train_op=train_op,
                scaffold=None
            )

        elif mode == tf.estimator.ModeKeys.EVAL:
            if current_task is not None:
                if use_distill:
                    current_loss = distill_loss
                    current_per_example_loss = student_outputs[current_task].per_example_loss
                else:
                    current_loss = outputs[current_task].loss
                    current_per_example_loss = outputs[current_task].per_example_loss

                current_loss = tf.identity(current_loss, name="loss")

                def metric_fn(per_loss):
                    eval_label_loss = tf.metrics.mean(values=per_loss, name="current_loss")
                    return {
                        "eval_loss": eval_label_loss
                    }

                eval_metrics = (metric_fn, [current_per_example_loss])
                output_spec = tf.estimator.EstimatorSpec(
                    mode=mode,
                    loss=current_loss,
                    eval_metric_ops=eval_metrics[0](*eval_metrics[1])
                )
            else:
                all_loss = []
                for task_name, output in outputs.items():
                    all_loss.append(output.loss)

                loss = tf.reduce_mean(tf.convert_to_tensor(all_loss), axis=0, keep_dims=False, name="output_loss")
                output_spec = tf.estimator.EstimatorSpec(
                    mode=mode,
                    loss=loss
                )
        else:
            if do_export:
                tasks_info = []
                for task_name, task in tasks.items():
                    task_type = task.task_type
                    if task.task_type == "tagging" and task.output_method == "crf":
                        task_type = "tagging_crf"
                    tasks_info.append("%s;%s;%s" % (task_name, task_type, task.num_output))

                tasks_info_tensor = tf.cast(tasks_info, dtype=tf.string)
                tasks_info_tensor = tf.expand_dims(input=tasks_info_tensor, axis=0)
                batch_size = modeling.get_shape_list(input_ids, name="batch_size")[0]
                tasks_info_tensor = tf.tile(tasks_info_tensor, multiples=[batch_size, 1])

                predictions["tasks_info"] = tasks_info_tensor
            else:
                if label_ids is not None:
                    predictions["%s_truth_label" % current_task] = label_ids
                predictions["%s_input_ids" % current_task] = input_ids
                predictions["%s_input_mask" % current_task] = input_mask

                if use_distill:
                    if tasks[current_task].task_type == "tagging":
                        predictions["student/%s" % current_task] = student_predictions[current_task]
                    else:
                        predictions["student/%s_label" % current_task] = student_predictions["%s_label" % current_task]

            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions
            )
        return output_spec

    return model_fn


def build_bert_base_model(bert_config, features, is_training, max_seq_length, do_export, floatx):
    tf.logging.debug("*** Features ***")
    for name in sorted(features.keys()):
        tf.logging.debug("  name = %s, shape = %s" % (name, features[name].shape))

    input_ids = features["input_ids"]
    segment_ids = features["segment_ids"]
    lengths = features["length"]

    # 训练、预测过程中，所有query会被pad到统一长度，即max_seq_length，
    # 某一个batch里所有query的实际长度可能均小于max_seq_length，因此需要指定sequence_mask时的maxlen;
    # 在模型服务时，接收一个batch的query，会被统一pad到该batch中最长query的长度，因此不需要指定maxlen，动态创建mask;
    if do_export:
        input_mask = tf.sequence_mask(lengths, dtype=tf.int32)
    else:
        input_mask = tf.sequence_mask(lengths, maxlen=max_seq_length, dtype=tf.int32)

    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=False,
        floatx=floatx
    )

    return model, input_ids, input_mask, lengths


def downstream_task_fine_tuning_outputs(
        bert_config, attention_mask, input_tensor_list, tasks, floatx, is_training, is_student):
    outputs = {}
    for task_name, task in tasks.items():
        outputs[task_name] = task.get_fine_tuning_outputs(
            bert_config=bert_config,
            attention_mask=attention_mask,
            input_tensor_list=input_tensor_list,
            is_training=is_training,
            is_student=is_student,
            floatx=floatx
        )
    return outputs


def downstream_task_outputs(bert_config, public_input_tensor, custom_input_tensor,
                            lengths, label_ids, is_training, tasks,
                            do_export=False, current_task=None, is_student=False, floatx="float32",
                            attention_mask=None):
    outputs = {}
    predictions = {}
    for task_name, task in tasks.items():
        if len(task.fine_tuning_layers) == 0:
            input_tensor = public_input_tensor
        else:
            if is_student:
                input_tensor = public_input_tensor[:task.student_fine_tuning_layers[0]] + custom_input_tensor[task_name]
            else:
                input_tensor = public_input_tensor[:task.fine_tuning_layers[0]] + custom_input_tensor[task_name]

        if current_task is not None and task_name != current_task:
            outputs[task_name] = task.get_task_outputs(
                bert_config=bert_config,
                input_tensor_list=input_tensor,
                lengths=lengths,
                label_ids=None,
                is_training=is_training,
                is_student=is_student,
                floatx=floatx,
                attention_mask=attention_mask
            )
        else:
            outputs[task_name] = task.get_task_outputs(
                bert_config=bert_config,
                input_tensor_list=input_tensor,
                lengths=lengths,
                label_ids=label_ids,
                is_training=is_training,
                is_student=is_student,
                floatx=floatx,
                attention_mask=attention_mask
            )

        if do_export:
            label_lst = tf.cast(task.get_labels(), dtype=tf.string)
            if task.task_type == "tagging":
                if task.output_method == "crf":
                    if floatx == "float32":
                        predictions[task_name] = outputs[task_name].logits
                    else:
                        predictions[task_name] = tf.cast(outputs[task_name].logits, dtype=tf.float32)
                else:
                    predictions[task_name] = tf.gather(label_lst, outputs[task_name].label_pred)
            else:
                if floatx == "float32":
                    prob = outputs[task_name].label_prob
                else:
                    prob = tf.cast(outputs[task_name].label_prob, dtype=tf.float32)
                batch_size = modeling.get_shape_list(input_tensor[0], name="batch_size")[0]
                label_lst = tf.expand_dims(input=label_lst, axis=0)
                label_lst = tf.tile(label_lst, multiples=[batch_size, 1])
                predictions["%s_label" % task_name] = label_lst
                predictions[task_name] = prob

        else:
            if task.task_type == "tagging":
                predictions[task_name] = outputs[task_name].label_pred
            else:
                if floatx == "float32":
                    predictions[task_name] = outputs[task_name].label_prob
                else:
                    predictions[task_name] = tf.cast(outputs[task_name].label_prob, dtype=tf.float32)

                predictions["%s_label" % task_name] = outputs[task_name].label_pred

    return outputs, predictions


def knowledge_distill(hard_loss, student_logits, teacher_logits, temperature=2):
    with tf.variable_scope("knowledge_distill"):
        temperature = tf.constant(temperature, dtype=tf.float32, name="temperature")

        teacher_soft_target = tf.nn.softmax(tf.divide(teacher_logits, temperature, name="teacher_soft_target"))

        soft_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=tf.divide(student_logits, temperature),
                labels=teacher_soft_target,
                name="soft_loss"
            )
        )

        loss = tf.square(temperature) * soft_loss + hard_loss

    tf.summary.scalar("soft_loss", soft_loss)
    tf.summary.scalar("hard_loss", hard_loss)

    return loss
