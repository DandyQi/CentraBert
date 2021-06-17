#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2021/6/17 下午5:57
# @Author  : qijianwei
# @File    : convert_data_format.py
# @Usage: Transfer the data format for centra-bert


import os
import json
import argparse


args = argparse.ArgumentParser()

args.add_argument("--task", type=str)
args.add_argument("--input_dir", type=str)
args.add_argument("--output_dir", type=str)


def convert_data_format(task, input_dir, output_dir):
    output_dir = os.path.join(output_dir, task)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for file_name in ["train", "dev", "test"]:
        input_file = os.path.join(input_dir, task, "%s.txt" % file_name)
        output_file = os.path.join(output_dir, "%s_json_format.txt" % file_name)

        with open(input_file, "r") as fin, open(output_file, "w") as fout:
            for i, line in enumerate(fin.readlines()):
                if task == "cola":
                    if file_name == "test" and i == 0:
                        continue

                    if file_name == "test":
                        text = line.strip().split("\t")[1]
                        label = "0"
                    else:
                        text = line.strip().split("\t")[3]
                        label = line.strip().split("\t")[1]

                    fout.write("%s\n" % json.dumps({"query": text, "intent": label}, ensure_ascii=False))

                elif task in ["qnli", "rte"]:
                    if i == 0:
                        continue
                    text_a = line.strip().split("\t")[1]
                    text_b = line.strip().split("\t")[2]
                    if file_name == "test":
                        label = "not_entailment"
                    else:
                        label = line.strip().split("\t")[-1]

                    fout.write("%s\n" % json.dumps({"query": text_a, "another_query": text_b, "intent": label},
                                                   ensure_ascii=False))

                elif task in ["qqp"]:
                    if i == 0:
                        continue
                    if file_name == "test":
                        label = "0"
                        text_a = line.strip().split("\t")[1]
                        text_b = line.strip().split("\t")[2]
                    else:
                        if len(line.strip().split("\t")) != 6:
                            continue
                        text_a = line.strip().split("\t")[3]
                        text_b = line.strip().split("\t")[4]
                        label = line.strip().split("\t")[5]

                    fout.write("%s\n" % json.dumps({"query": text_a, "another_query": text_b, "intent": label},
                                                   ensure_ascii=False))

                elif task in ["mnli"]:
                    if i == 0:
                        continue
                    text_a = line.strip().split("\t")[8]
                    text_b = line.strip().split("\t")[9]
                    if file_name == "test_matched":
                        label = "contradiction"
                    else:
                        label = line.strip().split("\t")[-1]

                    fout.write("%s\n" % json.dumps({"query": text_a, "another_query": text_b, "intent": label},
                                                   ensure_ascii=False))

                elif task in ["sst"]:
                    if i == 0:
                        continue
                    if file_name == "test":
                        text_a = line.strip().split("\t")[1]
                        label = "0"
                    else:
                        text_a = line.strip().split("\t")[0]
                        label = line.strip().split("\t")[1]

                    fout.write("%s\n" % json.dumps({"query": text_a, "intent": label},
                                                   ensure_ascii=False))

                elif task in ["mrpc"]:
                    if i == 0:
                        continue
                    text_a = line.strip().split("\t")[3]
                    text_b = line.strip().split("\t")[4]
                    if file_name == "test":
                        label = "0"
                    else:
                        label = line.strip().split("\t")[0]

                    fout.write("%s\n" % json.dumps({"query": text_a, "another_query": text_b, "intent": label},
                                                   ensure_ascii=False))

                elif task in ["stsb"]:
                    if i == 0:
                        continue
                    text_a = line.strip().split("\t")[7]
                    text_b = line.strip().split("\t")[8]
                    if file_name == "test":
                        label = 0.
                    else:
                        label = line.strip().split("\t")[0]

                    fout.write("%s\n" % json.dumps({"query": text_a, "another_query": text_b, "intent": label},
                                                   ensure_ascii=False))

                else:
                    raise ValueError("No support this task: %s" % task)

        fin.close(), fout.close()


if __name__ == '__main__':
    a = args.parse_args()
    convert_data_format(
        task=a.task,
        input_dir=a.input_dir,
        output_dir=a.output_dir
    )
