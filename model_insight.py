#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2021/6/17 下午4:12
# @Author  : qijianwei
# @File    : model_insight.py
# @Usage: list tensors in model


import json
import argparse

from tensorflow.python import pywrap_tensorflow

args = argparse.ArgumentParser()
args.add_argument("--ckpt", type=str)
args.add_argument("--output", type=str)


def write_tensor_name_to_file(ckpt, output_file):
    try:
        reader = pywrap_tensorflow.NewCheckpointReader(ckpt)
        with open(output_file, "w") as fout:
            tensor_shape_map = reader.get_variable_to_shape_map()
            for key in sorted(tensor_shape_map):
                content = {"tensor_name": key, "shape": tensor_shape_map[key]}
                fout.write(json.dumps(content) + "\n")
        fout.close()

    except Exception as e:
        print(e)


if __name__ == '__main__':
    a = args.parse_args()
    write_tensor_name_to_file(
        ckpt=a.ckpt,
        output_file=a.output
    )
