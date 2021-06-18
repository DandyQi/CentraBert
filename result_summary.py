#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2021/6/16 下午4:31
# @Author  : qijianwei
# @File    : result_summary.py
# @Usage: Summary


import os
import json
import argparse

import matplotlib.pyplot as plt

args = argparse.ArgumentParser()
args.add_argument("--job", type=str, default="stats")
args.add_argument("--output_dir", type=str)
args.add_argument("--task", type=str)
args.add_argument("--learning_rate", type=str)
args.add_argument("--keep_layers", type=str)
args.add_argument("--key_param", type=str, default="learning_rate")
args.add_argument("--fine_tuning_layers", type=str)
args.add_argument("--exam_num", type=int)
args.add_argument("--dev", type=bool, default=True)
args.add_argument("--version", type=str, default="teacher")


def load_metrics(result_file, dev=True, version="teacher"):
    metrics = {
        "teacher": 0.0,
        "student": 0.0
    }
    with open(result_file, "r") as fin:
        lines = fin.readlines()
        if dev:
            line = lines[0]
        else:
            line = lines[1]

        content = json.loads(line.strip())
        if version == "teacher":
            metrics["teacher"] = content["accuracy"] if "accuracy" in content.keys() else content["f1"]
        else:
            metrics["teacher"] = content["teacher"]["accuracy"] if "accuracy" in content["teacher"].keys() \
                else content["teacher"]["f1"]
            metrics["student"] = content["student"]["accuracy"] if "accuracy" in content["student"].keys() \
                else content["student"]["f1"]
    fin.close()

    return metrics


def load_best_ckpt(result_path):
    best_ckpt_record_file = os.path.join(result_path, "best_checkpoint", "best_checkpoint.txt")
    with open(best_ckpt_record_file, "r") as fin:
        line = fin.readline().strip()
        content = json.loads(line)
        best_ckpt = content["best_checkpoint_path"]
    fin.close()
    return best_ckpt


def get_best_metrics_and_ckpt(output_dir, task, learning_rate, fine_tuning_layers, exam_num,
                              dev=True, version="teacher", keep_layers=None):
    best_metrics = 0.0
    best_ckpt = None
    for lr in learning_rate.split(","):
        if lr == "1e-4":
            lr = "0.0001"
        elif lr == "2e-5":
            lr = "2e-05"
        for layers in fine_tuning_layers.split(","):
            for i in range(exam_num):
                if version == "teacher":
                    result_path = os.path.join(
                        output_dir,
                        task,
                        "Lr-%s-Layers-%s" % (lr, layers),
                        "ex-%s" % str(i + 1)
                    )
                else:
                    result_path = os.path.join(
                        output_dir,
                        task,
                        "Lr-%s-Layers-%s-%s" % (lr, keep_layers, layers),
                        "ex-%s" % str(i + 1)
                    )
                result_file = os.path.join(
                    result_path,
                    "result_summary.txt"
                )
                metrics = load_metrics(result_file=result_file, dev=dev, version=version)
                if metrics[version] > best_metrics:
                    best_metrics = metrics[version]
                    best_ckpt = load_best_ckpt(result_path=result_path)

    summary_file = os.path.join(output_dir, task, "summary.txt")
    with open(summary_file, "w") as fout:
        fout.write("Best metrics: %.2f, best checkpoint: %s" % (best_metrics, best_ckpt))


def plot_metrics(output_dir, task, key_param, learning_rate, fine_tuning_layers, exam_num,
                 dev=True, version="teacher", keep_layers=None):
    if key_param != "learning_rate" and key_param != "fine_tuning_layers":
        raise ValueError("Key param only supports 'learning_rate' or 'fine_tuning_layers'")
    key_metrics = {}
    for lr in learning_rate.split(","):
        if lr == "1e-4":
            lr = "0.0001"
        elif lr == "2e-5":
            lr = "2e-05"
        for layers in fine_tuning_layers.split(","):
            for i in range(exam_num):
                if version == "teacher":
                    result_path = os.path.join(
                        output_dir,
                        task,
                        "Lr-%s-Layers-%s" % (lr, layers),
                        "ex-%s" % str(i + 1)
                    )
                else:
                    result_path = os.path.join(
                        output_dir,
                        task,
                        "Lr-%s-Layers-%s-%s" % (lr, keep_layers, layers),
                        "ex-%s" % str(i + 1)
                    )
                result_file = os.path.join(
                    result_path,
                    "result_summary.txt"
                )
                metrics = load_metrics(result_file=result_file, dev=dev, version=version)[version]
                if key_param == "learning_rate":
                    if lr not in key_metrics.keys():
                        key_metrics[lr] = []
                    key_metrics[lr].append(metrics)
                else:
                    if layers not in key_metrics.keys():
                        key_metrics[layers] = []
                    key_metrics[layers].append(metrics)
    all_metrics, x_labels = [], []

    for key, metrics_lst in key_metrics.items():
        all_metrics.append(metrics_lst)
        x_labels.append(key)

    plt.boxplot(
        x=all_metrics,
        patch_artist=True,
        widths=0.5,
        medianprops={"linewidth": 2.5, "color": "yellow"},
        boxprops={"facecolor": "cyan"}
    )

    plt.xticks(ticks=list(range(1, len(x_labels)+1)), labels=x_labels, fontsize=16)
    plt.xlabel(key_param, fontsize=12)
    plt.title(task, fontsize=20)

    plt.savefig(os.path.join(output_dir, task, "metrics_analysis.png"))


if __name__ == '__main__':
    a = args.parse_args()
    if a.job == "stats":
        get_best_metrics_and_ckpt(
            output_dir=a.output_dir,
            task=a.task,
            learning_rate=a.learning_rate,
            fine_tuning_layers=a.fine_tuning_layers,
            exam_num=a.exam_num,
            dev=a.dev,
            version=a.version,
            keep_layers=a.keep_layers
        )
    elif a.job == "plot":
        plot_metrics(
            output_dir=a.output_dir,
            task=a.task,
            key_param=a.key_param,
            learning_rate=a.learning_rate,
            fine_tuning_layers=a.fine_tuning_layers,
            exam_num=a.exam_num,
            dev=a.dev,
            version=a.version,
            keep_layers=a.keep_layers
        )
