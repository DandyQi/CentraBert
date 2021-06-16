#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2019/12/30 上午11:55
# @Author  : qijianwei
# @File    : base_task.py
# @Usage: bert multitask基础下游任务，目前提供序列标注与分类任务

import tensorflow as tf
import copy
import json
import random

try:
    import tokenization
    import modeling
    from custom_crf_utils import crf_log_likelihood
    from masked_crf import MaskedCRF
except ImportError:
    from . import tokenization
    from . import modeling
    from .custom_crf_utils import crf_log_likelihood
    from .masked_crf import MaskedCRF


class BaseExample(object):
    def __init__(self, query, label, another_query=None):
        """
        bert multitask下游任务的基础输入结构
        Args:
            query: str，原始query
            label: str，原始label
        """
        self.text = query
        self.another_text = another_query
        self.label = label


class BaseOutput(object):
    def __init__(self, loss, per_example_loss, label_prob, label_pred, logits=None):
        """
        bert multitask下游任务的基础输出结构
        Args:
            loss: float tensor，当前任务的loss
            per_example_loss: float tensor，[batch_size]，当前任务每条样本的loss
            label_prob: float tensor， [batch_size, output_length]，当前任务预测label对应的probability
            label_pred: int tensor， [batch_size, output_length]，当前任务预测的label
            logits: float tensor，[batch_size, output_length]，采用crf任务输出的logits
        """
        self.loss = loss
        self.per_example_loss = per_example_loss
        self.label_prob = label_prob
        self.label_pred = label_pred
        self.logits = logits


class BaseTask(object):
    def __init__(self, task_name, task_type, input_files, max_seq_length,
                 output_method, fine_tuning_layers=None, gather_method="last", is_eng=False):
        """
        bert multitask下游任务的基础操作，包含读取数据与标签，数据预处理，构建input_fn，构建下游网络等方法
        Args:
            task_name: str，任务名称
            task_type: str，任务类型，目前支持tagging, classification
            input_files: str，使用逗号分隔的多个文件名，默认第一个文件名为训练集，后续的依次为测试集
            fine_tuning_layers: str，使用逗号分隔的下游任务所需要fine tuning的transformer层数
            max_seq_length: int，query最大长度
            output_method: str，输出层方法，tagging: softmax or crf, classification: cnn or cls
            gather_method: str，聚合transformer layers的方式，
                若为combine，则将所有freeze layers与fine tuning layers进行线性组合得到最终特征
                若为last，则直接采用当前最后一层layer作为最终特征
            is_eng: bool，是否为英文
        """
        self.task_name = task_name
        self.task_type = task_type
        self.input_files = input_files.split(",")
        if fine_tuning_layers:
            self.fine_tuning_layers = [int(x) for x in fine_tuning_layers.split(",")]
        else:
            self.fine_tuning_layers = []
        self.max_seq_length = max_seq_length
        self.output_method = output_method
        self.gather_method = gather_method

        self.student_fine_tuning_layers = None
        self.label2idx_map = None

        self.num_output = 0
        self.is_eng = is_eng

    def set_output_num(self, num):
        self.num_output = num

    def set_fine_tuning_layers(self, layers):
        self.fine_tuning_layers = layers

    def set_student_fine_tuning_layers(self, layers):
        self.student_fine_tuning_layers = layers

    def set_label2idx_map(self, label2idx_map):
        self.label2idx_map = label2idx_map

    def describe(self):
        labels = self.get_labels()
        label_num = len(labels)

        train_example, test_examples = self.get_data()
        all_examples = train_example
        for test_example in test_examples:
            all_examples += test_example

        max_length = 0
        total_length = 0
        for example in all_examples:
            if len(example.text) > max_length:
                max_length = len(example.text)
            total_length += len(example.text)

        if self.task_type == "classification":
            label_count = {}
            for example in all_examples:
                for label in example.label:
                    if label in label_count:
                        label_count[label] += 1
                    else:
                        label_count[label] = 1

            label_info = ""
            for label, num in label_count.items():
                label_info += "Label: %s\tCount: %d\n" % (label, num)
        else:
            slot_appear = {}
            slot_lengths = {}
            for label in labels:
                if label.startswith("B-") or label.startswith("I-"):
                    slot = label.split("-")[-1]
                    if slot not in slot_appear:
                        slot_appear[slot] = 0
                        slot_lengths[slot] = 0
            slot_num = len(slot_appear)

            for example in all_examples:
                for label in example.label:
                    if label.startswith("B-") or label.startswith("I-"):
                        slot = label.split("-")[-1]
                        slot_lengths[slot] += 1

                    if label.startswith("B-"):
                        slot = label.split("-")[-1]
                        slot_appear[slot] += 1

            label_info = "Slot num: %d\n" % slot_num
            total_slot_length = 0
            for key in slot_appear.keys():
                avg_length = float(slot_lengths[key]) / slot_appear[key]
                label_info += "Slot: %s\tCount: %d\tAverage length: %.2f\n" % (key, slot_appear[key], avg_length)
                total_slot_length += slot_lengths[key]

            label_info += "Slot length / token length: %.2f\n" % (total_slot_length / total_length)

        info = ""
        info += "Task name: %s.\nData size: %d / %s.\nMax sequence length: %d.\n" \
                "Average sequence length: %d\nLabel num: %d\n" % \
                (self.task_name, len(train_example), " / ".join([str(len(x)) for x in test_examples]),
                 max_length, (total_length / (len(all_examples))), label_num)

        info += label_info

        return info

    def get_data(self):
        """
        获取该下游任务的训练集与测试集
        当只有一个输入文件时，按照valid_size将数据集划分为训练集与测试集
        当有多个输入文件时，默认第一个文件为训练集，其余文件为测试集
        Returns:
            train_example: list(BaseExample)，训练集
            test_example: list(list(BaseExample))，测试集
        """
        test_example = []
        if len(self.input_files) > 1:
            examples = self.get_example(self.input_files[0], True)
            train_example = examples

            for f in self.input_files[1:]:
                test_example.append(self.get_example(f, False))
        else:
            raise Exception("Input files have to contain at least train & dev set")

        return train_example, test_example

    def get_example(self, input_file, shuffle):
        """
        从文件中读取数据，默认输入文件是jsonl格式
        序列标注任务数据每行的json结构：{"query": "给小明发消息", "tags": ["O", "B-username", "I-username", "O", "O", "O"]}
        分类任务数据每行的json结构：{"query": "给小明发消息", "intent": "send_message"}
        截断判别任务数据每行的json结构：{"query": "天嘉定的天气如何", "label_begin": "1", "label_end": "0"}
            其中label_begin为1代表开头存在截断，label_end同理
        Args:
            input_file: list(str)，输入文件列表
            shuffle: bool，是否随机打乱顺序

        Returns:
            examples: list(BaseExample)，数据序列
        """
        examples = []
        with tf.gfile.GFile(input_file, "r") as f:
            for line in f.readlines():
                try:
                    line = line.strip()
                    data = json.loads(line)
                    data = dict(data)
                    query = tokenization.convert_to_unicode(data["query"])
                    another_query = None
                    if "another_query" in data.keys():
                        another_query = tokenization.convert_to_unicode(data["another_query"]).lower()
                    if self.task_type == "tagging":
                        if "tags" in data:
                            label = data["tags"]
                            if len(query) == len(label):
                                examples.append(
                                    BaseExample(
                                        query=query.lower(),
                                        label=label,
                                        another_query=another_query
                                    )
                                )
                    elif self.task_type == "classification":
                        if "intent" in data:
                            label = [tokenization.convert_to_unicode(data["intent"])]
                            examples.append(
                                BaseExample(
                                    query=query.lower(),
                                    label=label,
                                    another_query=another_query
                                )
                            )
                    elif self.task_type == "truncated":
                        if "label_begin" in data and "label_end" in data:
                            label = [data["label_begin"], data["label_end"]]
                            examples.append(BaseExample(query.lower(), label))
                    else:
                        raise ValueError("Unsupported task type")
                except Exception as e:
                    print(line)
                    print(e)
                    continue

        if shuffle:
            random.shuffle(examples)
        return examples

    def get_labels(self):
        """
        获取该下游任务的label列表，对于序列标注任务，将加入cls与sep位置的标签
        为保证训练、测试与服务过程中label列表中顺序保持一致，会对列表进行排序
        为在计算评价指标时，忽略标签为"O"的槽位或类别，会将列表里的"O"置于首位，即id为0
        Returns:
            labels，list(str)，label列表
        """
        if self.task_type == "truncated":
            return ["0", "1"]

        label_set = set()
        for fin in self.input_files:
            tf.logging.info("Reading input from %s for label stats" % fin)
            with tf.gfile.GFile(fin, "r") as f:
                for line in f.readlines():
                    line = line.strip()
                    data = json.loads(line)
                    if self.task_type == "tagging":
                        label_set.add("[CLS]")
                        label_set.add("[SEP]")
                        if "tags" in data:
                            labels = data["tags"]
                            for label in labels:
                                if label not in label_set:
                                    label_set.add(label)
                    else:
                        if "intent" in data:
                            label = tokenization.convert_to_unicode(data["intent"])
                            if label not in label_set:
                                label_set.add(label)
        labels = list(label_set)
        labels.sort()
        if "O" in labels:
            labels.remove("O")
            labels.insert(0, "O")

        tf.logging.info("Labels: %s, stats in %s" % (" ".join(labels), " ".join(self.input_files)))

        return labels

    def get_fine_tuning_outputs(
            self, bert_config, attention_mask, input_tensor_list, is_training=True,
            is_student=False, floatx='float32'):
        """
        当前下游任务需要fine tuning的transformer网络，一般而言，任务越简单，需要fine tuning的层数越少
        Args:
            bert_config: BertConfig，bert配置文件
            attention_mask: Tensor，根据句子实际长度生成的mask矩阵
            input_tensor_list: list(tensor) [13, batch_size, max_seq_length, hidden_size]
                原有bert模型输出的0~13层向量（第0层为embedding）
            is_training: bool，当前是否在训练状态，决定是否dropout
            is_student: bool，当前是否是student branch
            floatx: str，浮点数精度

        Returns:
            final_outputs: list(tensor) [len(task.fine_tuning_layers), batch_size, max_seq_length, hidden_size]
            当前下游任务fine tuning过的n层transformer的输出
        """
        if is_student:
            fine_tuning_layers = self.student_fine_tuning_layers
        else:
            fine_tuning_layers = self.fine_tuning_layers

        # 当前任务不需要fine_tuning原有网络
        if len(fine_tuning_layers) == 0:
            return []
        bert_config = copy.deepcopy(bert_config)
        if not is_training:
            bert_config.hidden_dropout_prob = 0.0
            bert_config.attention_probs_dropout_prob = 0.0

        # 为使下游任务之间的网络不互相影响，使用task name与task type来区分不同任务的计算图
        scope_name = "%s/bert/encoder" % self.task_name
        if is_student:
            scope_name = "student/" + scope_name

        with tf.variable_scope(scope_name):

            # 使用不需要fine tuning的最后一层输出作为当前的输入
            input_tensor_list = input_tensor_list[:fine_tuning_layers[0]]
            last_layer_output = input_tensor_list[-1]
            final_outputs = modeling.transformer_model(
                input_tensor=last_layer_output,
                attention_mask=attention_mask,
                hidden_size=bert_config.hidden_size,
                num_hidden_layers=fine_tuning_layers,
                num_attention_heads=bert_config.num_attention_heads,
                intermediate_size=bert_config.intermediate_size,
                intermediate_act_fn=modeling.get_activation(bert_config.hidden_act),
                hidden_dropout_prob=bert_config.hidden_dropout_prob,
                attention_probs_dropout_prob=bert_config.attention_probs_dropout_prob,
                initializer_range=bert_config.initializer_range,
                do_return_all_layers=True,
                floatx=floatx
            )
            return final_outputs

    def get_task_outputs(self, bert_config, input_tensor_list, lengths, label_ids,
                         is_training=True, is_student=False, floatx='float32', attention_mask=None):
        """
        当前下游任务的输出层，目前提供用于序列标注的softmax输出，序列标注的crf输出，分类的cnn输出
        Args:
            bert_config: BertConfig，bert配置文件
            input_tensor_list: list(tensor) [n, batch_size, max_seq_length, hidden_size]
                原有bert模型输出的0~n层向量（第0层为embedding），此处n = 未fine tuning的层数 + fine tuning的层数
            lengths: tensor，[batch_size]，input id的实际长度
            label_ids: tensor，[batch_size, max_seq_length] 或 [batch_size]，真实label id
            is_training: bool，当前是否在训练状态，决定是否dropout
            is_student: bool，是否为student branch
            floatx: str，浮点数精度
            attention_mask:

        Returns:
            BaseOutput，包含当前任务的loss，预测label，预测label对应的probability信息
        """
        num_used_hidden_layers = len(input_tensor_list)
        hidden_size = bert_config.hidden_size
        batch_size = modeling.get_shape_list(input_tensor_list[0], name="batch_size")[0]
        seq_length = modeling.get_shape_list(input_tensor_list[0], name="seq_length")[1]

        if self.task_type == "truncated":
            input_tensor_list = self.gather_index(input_tensor_list, lengths, batch_size, seq_length)

        input_tensor = tf.convert_to_tensor(input_tensor_list, name="encoder_output")

        if label_ids is None:
            if self.task_type == "tagging":
                label_ids = tf.zeros(shape=[batch_size, seq_length], dtype=tf.int32)
            elif self.task_type == "classification":
                label_ids = tf.zeros(shape=[batch_size], dtype=tf.int32)
            else:
                raise ValueError("Unsupported task type")

        scope_name = self.task_name
        if is_student:
            scope_name = "student/" + scope_name

        with tf.variable_scope(scope_name):
            if self.gather_method == "combine":
                with tf.variable_scope("combine"):
                    combine_weights = tf.get_variable(
                        "combine_weights",
                        shape=[1, num_used_hidden_layers],
                        initializer=tf.truncated_normal_initializer(stddev=0.02),
                        dtype=floatx
                    )
                    if is_training:
                        input_tensor = tf.nn.dropout(input_tensor, keep_prob=0.8)

                    norm_combine_weights = tf.nn.softmax(combine_weights)
                    input_tensor = tf.reshape(input_tensor, [num_used_hidden_layers, -1])
                    input_tensor = tf.matmul(norm_combine_weights, input_tensor)

                    input_tensor = tf.reshape(input_tensor, shape=[batch_size, seq_length, hidden_size])

                    input_tensor = modeling.transformer_model(
                        input_tensor=input_tensor,
                        attention_mask=attention_mask,
                        hidden_size=bert_config.hidden_size,
                        num_hidden_layers=3,
                        num_attention_heads=bert_config.num_attention_heads,
                        intermediate_size=bert_config.intermediate_size,
                        intermediate_act_fn=modeling.get_activation(bert_config.hidden_act),
                        hidden_dropout_prob=bert_config.hidden_dropout_prob,
                        attention_probs_dropout_prob=bert_config.attention_probs_dropout_prob,
                        initializer_range=bert_config.initializer_range,
                        do_return_all_layers=False,
                        floatx=floatx
                    )

            elif self.gather_method == "last":
                input_tensor = input_tensor_list[-1]
            else:
                raise ValueError("No support gather method: %s" % self.gather_method)

            input_tensor = tf.reshape(input_tensor, [-1, hidden_size])

            if self.output_method == "softmax":
                return self.get_softmax_output(
                    input_tensor, hidden_size, self.num_output, label_ids, seq_length, floatx=floatx)
            elif self.output_method == "crf":
                return self.get_crf_output(
                    input_tensor, lengths, hidden_size, self.num_output, label_ids, seq_length, floatx=floatx)
            elif self.output_method == "cnn":
                return self.get_cnn_output(
                    input_tensor, hidden_size, self.num_output, label_ids, seq_length, floatx=floatx)
            elif self.output_method == "cls":
                return self.get_cls_output(
                    input_tensor, hidden_size, seq_length, self.num_output, label_ids, floatx=floatx
                )
            else:
                raise ValueError("Do not support!")

    def get_softmax_output(self, input_tensor, hidden_size, num_output, label_ids, seq_length, floatx='float32'):
        """
        序列标注任务或截断判别任务的softmax输出
        Args:
            input_tensor: tensor，对于序列标注任务输入形状为[batch_size * max_seq_length, hidden_size]
                对于截断判别任务输入形状为[batch_size * 2, hidden_size]
            hidden_size: int，bert模型配置时的隐藏层大小
            num_output: int，label数量
            label_ids: tensor，对于序列标注任务label形状为[batch_size, max_seq_length]，对于截断判别任务label形状为[batch_size, 2]
            seq_length: int，序列长度
            floatx: str，浮点数精度

        Returns:
            BaseOutput，包含当前任务的loss，预测label，预测label对应的probability信息
        """
        with tf.variable_scope("softmax"):
            project_weights = tf.get_variable(
                "project_output_weights",
                shape=[hidden_size, num_output],
                initializer=tf.truncated_normal_initializer(stddev=0.02),
                dtype=floatx
            )
            project_bias = tf.get_variable(
                "project_output_bias",
                shape=[num_output],
                initializer=tf.zeros_initializer(),
                dtype=floatx
            )

            logits = tf.matmul(input_tensor, project_weights)
            logits = tf.nn.bias_add(logits, project_bias)
            if self.task_type == "truncated":
                logits = tf.reshape(logits, [-1, 2, num_output])
            else:
                logits = tf.reshape(logits, [-1, seq_length, num_output])
            prob = tf.nn.softmax(logits, axis=-1)
            log_probs = tf.nn.log_softmax(logits, axis=-1)

            one_hot_labels = tf.one_hot(label_ids, depth=num_output, dtype=floatx)

            per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)

            loss = tf.reduce_mean(per_example_loss)

            label_pred = tf.argmax(logits, axis=2)
            label_pred = tf.cast(label_pred, dtype=tf.int32)

            return BaseOutput(
                loss=loss,
                per_example_loss=per_example_loss,
                label_prob=prob,
                label_pred=label_pred,
                logits=logits
            )

    def get_crf_output(self, input_tensor, lengths, hidden_size, num_output, label_ids, seq_length, floatx='float32'):
        """
        序列标注任务的crf输出
        Args:
            input_tensor: tensor，[batch_size * max_seq_length, hidden_size]，bert输出的特征向量
            lengths: tensor，[batch_size]，input id的实际长度
            hidden_size: int，bert模型配置时的隐藏层大小
            num_output: int，label数量
            label_ids: tensor，[batch_size, max_seq_length]，真实label id
            seq_length: int，序列长度
            floatx: str，浮点数精度

        Returns:
            BaseOutput，包含当前任务的loss，预测label，预测label对应的probability信息
        """
        masked_crf = MaskedCRF(
            num_output=num_output,
            use_mask=True,
            label2idx_map=self.label2idx_map,
            floatx=floatx
        )

        with tf.variable_scope("crf"):
            project_weights = tf.get_variable(
                "project_output_weights",
                shape=[hidden_size, num_output],
                initializer=tf.truncated_normal_initializer(stddev=0.02),
                dtype=floatx
            )
            project_bias = tf.get_variable(
                "project_output_bias",
                shape=[num_output],
                initializer=tf.zeros_initializer(),
                dtype=floatx
            )
            crf_project = tf.tanh(tf.nn.xw_plus_b(input_tensor, project_weights, project_bias))
            logits = tf.reshape(crf_project, shape=[-1, seq_length, num_output])
            loss, per_example_loss, label_pred = masked_crf.decode(
                logits=logits,
                label_ids=label_ids,
                lengths=lengths
            )

            return BaseOutput(
                loss=loss,
                per_example_loss=per_example_loss,
                label_prob=None,
                label_pred=label_pred,
                logits=logits
            )

    @staticmethod
    def get_cls_output(input_tensor, hidden_size, seq_length, num_output, label_ids, floatx='float32'):
        """
        分类任务使用CLS位置的输出
        Args:
            input_tensor: tensor，[batch_size * max_seq_length, hidden_size]，bert输出的特征向量
            hidden_size: int，bert模型配置时的隐藏层大小
            seq_length: int，序列长度
            num_output: int，label数量
            label_ids: tensor，[batch_size]，真实label id
            floatx: str，浮点数精度

        Returns:
            BaseOutput，包含当前任务的loss，预测label，预测label对应的probability信息
        """
        input_tensor = tf.reshape(input_tensor, shape=[-1, seq_length, hidden_size])
        with tf.variable_scope("bert"):
            with tf.variable_scope("pooler"):
                first_token_tensor = tf.squeeze(input_tensor[:, 0:1, :], axis=1)
                pooled_output = tf.layers.dense(
                    first_token_tensor,
                    hidden_size,
                    activation=tf.tanh,
                    kernel_initializer=tf.truncated_normal_initializer(stddev=0.02)
                )

        project_weights = tf.get_variable(
            "output_weights",
            shape=[num_output, hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=0.02),
            dtype=floatx
        )
        project_bias = tf.get_variable(
            "output_bias",
            shape=[num_output],
            initializer=tf.zeros_initializer(),
            dtype=floatx
        )
        # [batch_size, num_output]
        logits = tf.matmul(pooled_output, project_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, project_bias)
        prob = tf.nn.softmax(logits, axis=-1)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        one_hot_labels = tf.one_hot(label_ids, depth=num_output, dtype=floatx)

        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)

        loss = tf.reduce_mean(per_example_loss)

        label_pred = tf.argmax(logits, axis=1, output_type=tf.int32)

        return BaseOutput(
            loss=loss,
            per_example_loss=per_example_loss,
            label_prob=prob,
            label_pred=label_pred,
            logits=logits
        )

    @staticmethod
    def get_regression_output(input_tensor, hidden_size, seq_length, num_output, label_ids, floatx='float32'):
        input_tensor = tf.reshape(input_tensor, shape=[-1, seq_length, hidden_size])
        with tf.variable_scope("bert"):
            with tf.variable_scope("pooler"):
                first_token_tensor = tf.squeeze(input_tensor[:, 0:1, :], axis=1)
                pooled_output = tf.layers.dense(
                    first_token_tensor,
                    hidden_size,
                    activation=tf.tanh,
                    kernel_initializer=tf.truncated_normal_initializer(stddev=0.02)
                )

        project_weights = tf.get_variable(
            "output_weights",
            shape=[num_output, hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=0.02),
            dtype=floatx
        )
        project_bias = tf.get_variable(
            "output_bias",
            shape=[num_output],
            initializer=tf.zeros_initializer(),
            dtype=floatx
        )
        # [batch_size, num_output]
        logits = tf.matmul(pooled_output, project_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, project_bias)

        sigmoid_scores = tf.nn.sigmoid(logits)
        sigmoid_scores = tf.reshape(sigmoid_scores, shape=[-1, 1])
        reshape_scores = tf.reshape(label_ids, shape=[-1, 1])

        per_example_loss = tf.square(sigmoid_scores - reshape_scores)
        loss = tf.losses.mean_squared_error(reshape_scores, sigmoid_scores)

        return BaseOutput(
            loss=loss,
            per_example_loss=per_example_loss,
            label_prob=None,
            label_pred=sigmoid_scores,
            logits=logits
        )

    @staticmethod
    def get_cnn_output(input_tensor, hidden_size, num_output, label_ids, seq_length, floatx='float32'):
        """
        分类任务的cnn输出
        Args:
            input_tensor: tensor，[batch_size * max_seq_length, hidden_size]，bert输出的特征向量
            hidden_size: int，bert模型配置时的隐藏层大小
            num_output: int，label数量
            label_ids: tensor，[batch_size]，真实label id
            seq_length: int，序列长度
            floatx: str，浮点数精度

        Returns:
            BaseOutput，包含当前任务的loss，预测label，预测label对应的probability信息
        """
        with tf.variable_scope("cnn"):
            output_weights = tf.get_variable(
                "project_output_weights",
                shape=[num_output, 192],
                initializer=tf.truncated_normal_initializer(stddev=0.02),
                dtype=floatx
            )
            output_bias = tf.get_variable(
                "project_output_bias",
                shape=[num_output],
                initializer=tf.zeros_initializer(),
                dtype=floatx
            )

            input_tensor = tf.reshape(input_tensor, shape=[-1, seq_length, hidden_size])

            conv_outputs = []
            for filter_size in [2, 3, 4]:
                conv = tf.layers.conv1d(
                    inputs=input_tensor,
                    filters=64,
                    kernel_size=filter_size
                )
                conv = tf.nn.relu(conv)
                conv = tf.reduce_max(conv, axis=1)
                conv = tf.reshape(conv, [-1, 64])
                conv_outputs.append(conv)

            conv_output = tf.concat(conv_outputs, axis=-1)

            logits = tf.matmul(conv_output, output_weights, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)
            prob = tf.nn.softmax(logits, axis=-1)

            log_probs = tf.nn.log_softmax(logits, axis=-1)

            one_hot_labels = tf.one_hot(label_ids, depth=num_output, dtype=floatx)

            per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)

            loss = tf.reduce_mean(per_example_loss)

            label_pred = tf.argmax(logits, axis=1)

            label_pred = tf.cast(label_pred, dtype=tf.int32)

            return BaseOutput(
                loss=loss,
                per_example_loss=per_example_loss,
                label_prob=prob,
                label_pred=label_pred,
                logits=logits
            )

    @staticmethod
    def gather_index(input_tensor, lengths, batch_size, seq_length):
        """
        截断判别任务，取出input中cls与sep位置对应的向量进行分类
        Args:
            input_tensor: list(tensor) [n, batch_size, max_seq_length, hidden_size]
            lengths: tensor，[batch_size]，input id的实际长度
            batch_size: int，batch size
            seq_length: int，序列长度

        Returns:
            new_input_tensor: list(tensor) [n, batch_size, 2, hidden_size]
        """
        new_input_tensor = []
        starts = tf.reshape(tf.zeros_like(lengths), shape=[-1, 1])
        ends = tf.reshape(lengths - tf.ones_like(lengths), shape=[-1, 1])
        positions = tf.concat([starts, ends], axis=1)

        flat_offsets = tf.reshape(tf.range(0, batch_size, dtype=tf.int32) * seq_length, shape=[-1, 1])
        flat_positions = tf.reshape(positions + flat_offsets, [-1])

        for layer_tensor in input_tensor:
            flat_layer_tensor = tf.reshape(layer_tensor, shape=[batch_size * seq_length, -1])
            new_input_tensor.append(tf.reshape(tf.gather(flat_layer_tensor, flat_positions), shape=[batch_size, 2, -1]))

        return new_input_tensor


def get_input_features(text, tokenizer):
    tokens = [token for token in text]
    tokens = ["[CLS]"] + tokens + ["[SEP]"]
    length = len(tokens)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    return {
        "input_ids": input_ids,
        "length": length
    }
