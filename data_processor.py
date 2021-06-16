#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2021/2/24 下午7:32
# @Author  : qijianwei
# @File    : data_processor.py
# @Usage: 数据预处理

import collections
import tensorflow as tf

try:
    import tokenization
except ImportError:
    from . import tokenization

try:
    from minlptokenizer.tokenizer import MiNLPTokenizer
    segments_tokenizer = MiNLPTokenizer()
except ImportError:
    MiNLPTokenizer = None
    segments_tokenizer = None
    print("Please install mi nlp tokenizer by: pip install minlp-tokenizer")


CLS = [101]
SEP = [102]
CHAR_VOCAB_SIZE = 8107

error_count = 0
illegal_count = 0


class BaseFeature(object):
    def __init__(self, input_ids, label_ids, length, segment_ids=None):
        """
        bert multitask下游任务的基础特征结构
        Args:
            input_ids: list(int)，token对应的id序列，长度为max_seq_length
            label_ids: list(int)，label对应的id序列，长度为output_length
                （序列标注任务output_length=max_seq_length；分类任务output_length=1）
            length: int，token序列实际长度（未padding部分）
        """
        self.input_ids = input_ids
        self.label_ids = label_ids
        self.length = length
        self.segment_ids = segment_ids


class BaseProcessor(object):
    def __init__(self, max_seq_length, task_type, is_eng=False):
        self.max_seq_length = max_seq_length
        self.task_type = task_type
        self.is_eng = is_eng

    def file_based_convert_examples_to_features(
            self, examples, label2id_map, tokenizer, output_file):
        """
        将训练或预测数据转换为特征并写入文件，在数据量较大时，建议写入文件，避免消耗过多内存
        Args:
            examples: list(BaseExample)，数据列表
            label2id_map: dict<str, int>，label到id的映射
            tokenizer: FullTokenizer，分词器，此处仅使用它将token转换为id
            output_file: str，输出的特征文件
        Returns:
            None
        """
        writer = tf.python_io.TFRecordWriter(output_file)

        for (ex_idx, example) in enumerate(examples):
            if ex_idx % 10000 == 0:
                tf.logging.info("Writing examples %d of %d." % (ex_idx, len(examples)))

            feature = self.convert_single_example(ex_idx, example, tokenizer, label2id_map)
            if feature is None:
                continue

            def create_int_feature(values):
                f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
                return f

            features = collections.OrderedDict()
            features["input_ids"] = create_int_feature(feature.input_ids)
            features["segment_ids"] = create_int_feature(feature.segment_ids)
            features["label_ids"] = create_int_feature(feature.label_ids)
            features["length"] = create_int_feature([feature.length])

            tf_example = tf.train.Example(features=tf.train.Features(feature=features))
            writer.write(tf_example.SerializeToString())

    def convert_examples_to_features(self, examples, label2id_map, tokenizer):
        """
        将训练或预测数据转换为特征，在数据量较小时，可使用这种方式，更加简便
        Args:
            examples: list(BaseExample)，数据列表
            label2id_map: dict<str, int>，label到id的映射
            tokenizer: FullTokenizer，分词器，此处仅使用它将token转换为id

        Returns:
            list(BaseFeature)
        """
        features = []

        for ex_idx, example in enumerate(examples):
            feature = self.convert_single_example(ex_idx, example, tokenizer, label2id_map)
            if feature is None:
                continue

            features.append(feature)

        return features

    def convert_single_example(self, ex_idx, example, tokenizer, label_map):
        """
        根据任务类型不同，将单条数据转换为特征
        Args:
            ex_idx: int，数据编号，仅用于打印日志
            example: BaseExample，数据类
            tokenizer: FullTokenizer，分词器，此处仅使用它将token转换为id
            label_map: dict<str, int>，label到id的映射

        Returns:
            BaseFeature
        """
        # query在添加首尾标记后，需截长补短到预设长度，length代表input id中query的实际长度
        if self.is_eng:
            tokens = tokenizer.tokenize(example.text)
        else:
            tokens = [token for token in example.text]
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if example.another_text:
            if self.is_eng:
                another_tokens = tokenizer.tokenize(example.another_text)
            else:
                another_tokens = [token for token in example.another_text]
            tokens = tokens + another_tokens + ["[SEP]"]
            segment_ids = segment_ids + [1] * (len(another_tokens) + 1)
        length = len(tokens)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        if length < self.max_seq_length:
            input_ids += [0] * (self.max_seq_length - length)
            segment_ids += [0] * (self.max_seq_length - length)
        else:
            input_ids = input_ids[:self.max_seq_length]
            segment_ids = segment_ids[:self.max_seq_length]
            length = self.max_seq_length
        try:
            assert len(input_ids) == self.max_seq_length
            assert len(segment_ids) == self.max_seq_length
        except AssertionError:
            tf.logging.error("Input ids length: %d" % len(input_ids))
            tf.logging.error("Segment ids length: %d" % len(segment_ids))

        # label根据执行任务不同，进行不同处理
        labels = example.label
        label_ids = None
        if labels is not None:
            # 若为tagging任务，则与query处理方式类似，截长补短到对应长度
            if self.task_type == "tagging":
                labels = ["[CLS]"] + labels + ["[SEP]"]
                label_ids = [label_map[label] for label in labels]

                if len(labels) < self.max_seq_length:
                    label_ids += [label_map["O"]] * (self.max_seq_length - length)
                else:
                    label_ids = label_ids[:self.max_seq_length]
                assert len(label_ids) == self.max_seq_length
            # 若为classification任务，则直接将label转换为对应的id
            elif self.task_type == "classification":
                label_ids = [label_map[label] for label in labels]
            # 若为truncated任务，则添加首尾截断标签对应id
            elif self.task_type == "truncated":
                label_ids = [label_map[label] for label in labels]
            else:
                raise ValueError("Unsupported task type")

        if ex_idx < 3:
            tf.logging.info("*** Example ***")
            tf.logging.info("tokens: %s" % " ".join(
                [tokenization.printable_text(x) for x in tokenizer.convert_ids_to_tokens(input_ids)]))
            tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            tf.logging.info("actual length: %s" % length)
            if labels is not None:
                tf.logging.info("labels: %s" % " ".join(labels))
                tf.logging.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))

        feature = BaseFeature(
            input_ids=input_ids,
            label_ids=label_ids,
            length=length,
            segment_ids=segment_ids
        )
        return feature

    def file_based_input_fn_builder(self, input_file, is_training):
        """
        基于文件的构建input_fn方法
        Args:
            input_file: str，由file_based_convert_examples_to_features写入的数据文件
            is_training: bool，当前是否在训练状态，决定当前是否shuffle与batch大小
        Returns:
            供estimator调用的input_fn
        """
        if self.task_type == "classification":
            name_to_features = {
                "input_ids": tf.FixedLenFeature([self.max_seq_length], tf.int64),
                "label_ids": tf.FixedLenFeature([], tf.int64),
                "segment_ids": tf.FixedLenFeature([self.max_seq_length], tf.int64),
                "length": tf.FixedLenFeature([], tf.int64)
            }
        elif self.task_type == "tagging":
            name_to_features = {
                "input_ids": tf.FixedLenFeature([self.max_seq_length], tf.int64),
                "label_ids": tf.FixedLenFeature([self.max_seq_length], tf.int64),
                "length": tf.FixedLenFeature([], tf.int64)
            }
        elif self.task_type == "truncated":
            name_to_features = {
                "input_ids": tf.FixedLenFeature([self.max_seq_length], tf.int64),
                "label_ids": tf.FixedLenFeature([2], tf.int64),
                "length": tf.FixedLenFeature([], tf.int64)
            }
        else:
            raise ValueError("Unsupported task type")

        def _decode_record(record):
            """Decodes a record to a TensorFlow example."""
            example = tf.parse_single_example(record, name_to_features)

            # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
            # So cast all int64 to int32.
            for name in list(example.keys()):
                t = example[name]
                if t.dtype == tf.int64:
                    t = tf.to_int32(t)
                example[name] = t
            return example

        def input_fn(params):
            """The actual input function."""
            batch_size = params["eval_batch_size"]
            d = tf.data.TFRecordDataset(input_file)
            drop_remainder = False
            if is_training:
                batch_size = params["train_batch_size"]
                d = d.repeat()
                d = d.shuffle(buffer_size=100)
                drop_remainder = True
            d = d.apply(
                tf.contrib.data.map_and_batch(
                    lambda record: _decode_record(record),
                    batch_size=batch_size,
                    drop_remainder=drop_remainder))
            return d

        return input_fn

    def input_fn_builder(self, features, is_training):
        """
        直接解析feature的input_fn构建方法
        Args:
            features: list(BaseFeature)，由convert_examples_to_features输出的特征列表
            is_training: bool，当前是否在训练状态，决定当前batch大小

        Returns:
            供estimator调用的input_fn
        """
        all_input_ids = []
        all_label_ids = []
        all_segment_ids = []
        all_lengths = []
        for feature in features:
            all_input_ids.append(feature.input_ids)
            all_segment_ids.append(feature.segment_ids)
            all_label_ids.append(feature.label_ids)
            all_lengths.append(feature.length)
        max_seq_length = self.max_seq_length

        def input_fn(params):
            batch_size = params["eval_batch_size"] if is_training else params["train_batch_size"]
            num_examples = len(features)
            if self.task_type == "classification":
                feature_map = {
                    "input_ids": tf.constant(all_input_ids, shape=[num_examples, max_seq_length], dtype=tf.int32),
                    "segment_ids": tf.constant(all_segment_ids, shape=[num_examples, max_seq_length], dtype=tf.int32),
                    "length": tf.constant(all_lengths, shape=[num_examples], dtype=tf.int32)
                }
                if feature.label_ids is not None:
                    feature_map["label_ids"] = tf.constant(all_label_ids, shape=[num_examples], dtype=tf.int32)

            elif self.task_type == "tagging":
                feature_map = {
                    "input_ids": tf.constant(all_input_ids, shape=[num_examples, max_seq_length], dtype=tf.int32),
                    "length": tf.constant(all_lengths, shape=[num_examples], dtype=tf.int32)
                }
                if feature.label_ids is not None:
                    feature_map["label_ids"] = tf.constant(all_label_ids, shape=[num_examples, max_seq_length],
                                                           dtype=tf.int32)

            else:
                raise ValueError("Unsupported task type")

            d = tf.data.Dataset.from_tensor_slices(
                feature_map
            )
            d = d.batch(batch_size=batch_size, drop_remainder=False)
            return d

        return input_fn
