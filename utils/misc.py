import copy
import os
import re
from typing import List, Union

import pandas as pd
import yaml


def count_files_in_directory(directory):
    count = 0
    for root, dirs, files in os.walk(directory):
        count += len(files)
    return count


def check_synthetic_dir_valid(synthetic_dir):

    if not os.path.exists(synthetic_dir):
        raise FileNotFoundError(f"Directory '{synthetic_dir}' does not exist.")

    total_files = count_files_in_directory(synthetic_dir)
    if total_files > 100:
        print(f"Directory '{synthetic_dir}' is valid with {total_files} files.")
    else:
        raise ValueError(
            f"Directory '{synthetic_dir}' contains less than 100 files, which is insufficient."
        )


def parse_finetuned_ckpt(finetuned_ckpt):
    lora_path = None
    embed_path = None
    for file in os.listdir(finetuned_ckpt):
        if "pytorch_lora_weights" in file:
            lora_path = os.path.join(finetuned_ckpt, file)
        elif "learned_embeds-steps-last" in file:
            embed_path = os.path.join(finetuned_ckpt, file)
    return lora_path, embed_path


def checked_has_run(exp_dir, args):
    parent_dir = os.path.abspath(os.path.join(exp_dir, os.pardir))
    current_args = copy.deepcopy(args)
    current_args.pop("gpu", None)
    current_args.pop("note", None)
    current_args.pop("target_class_num", None)

    for dirpath, dirnames, filenames in os.walk(parent_dir):
        for dirname in dirnames:
            config_file = os.path.join(dirpath, dirname, "config.yaml")
            if os.path.exists(config_file):
                with open(config_file, "r") as file:
                    saved_args = yaml.load(file, Loader=yaml.FullLoader)

                if (
                    current_args["syndata_dir"] is None
                    or "aug" in current_args["syndata_dir"]
                    or "gen" in current_args["syndata_dir"]
                ):
                    current_args.pop("gamma", None)
                    saved_args.pop("gamma", None)
                saved_args.pop("gpu", None)
                saved_args.pop("note", None)
                saved_args.pop("target_class_num", None)
                if saved_args == current_args:
                    print(
                        f"This program has already been run in directory: {dirpath}/{dirname}"
                    )
                    return True
    return False


def parse_result(target_dir, extra_column=[]):
    results = []
    for file in os.listdir(target_dir):
        config_file = os.path.join(target_dir, file, "config.yaml")
        config = yaml.safe_load(open(config_file, "r"))
        if isinstance(config["syndata_dir"], list):
            syndata_dir = config["syndata_dir"][0]
        else:
            syndata_dir = config["syndata_dir"]

        if syndata_dir is None:
            strategy = "baseline"
            strength = 0
        else:
            match = re.match(r"([a-zA-Z]+)([0-9.]*).*", syndata_dir)
            if match:
                strategy = match.group(1)
                strength = match.group(2)
            else:
                continue
        for basefile in os.listdir(os.path.join(target_dir, file)):
            if "acc_eval" in basefile:
                acc = float(basefile.split("_")[-1])
                results.append(
                    (
                        config["dir"],
                        config["res_mode"],
                        config["lr"],
                        strategy,
                        strength,
                        config["gamma"],
                        config["seed"],
                        *[str(config.pop(key, "False")) for key in extra_column],
                        acc,
                    )
                )
                break

    df = pd.DataFrame(
        results,
        columns=[
            "dataset",
            "resolution",
            "lr",
            "strategy",
            "strength",
            "soft power",
            "seed",
            *extra_column,
            "acc",
        ],
    )
    df["acc"] = df["acc"].astype(float)
    result_seed = (
        df.groupby(
            [
                "dataset",
                "resolution",
                "lr",
                "strength",
                "strategy",
                "soft power",
                *extra_column,
            ]
        )
        .agg({"acc": ["mean", "var"]})
        .reset_index()
    )
    result_sorted = result_seed.sort_values(
        by=["dataset", "resolution", "lr", "strategy", "strength", *extra_column]
    )
    result_seed.columns = ["_".join(col).strip() for col in result_seed.columns.values]

    return result_sorted
