import os
import glob
from omegaconf import OmegaConf
import json
import dill as pickle
from pathlib import Path
from typing import Union, List, Dict


def load_json(json_file):
    with open(json_file, "r", encoding="utf8") as fin:
        data = json.load(fin)
    print(f"load {len(data)} from {json_file}")
    return data


def write2json(data, data_path, data_name="data", ensure_ascii=False, indent=2):
    with open(data_path, "w", encoding="utf-8") as fout:
        fout.write(json.dumps(data, ensure_ascii=ensure_ascii, indent=indent))
    print(f"{data_name}({len(data)}) saved into {data_path}")


def load_json_by_line(data_path):
    """
    load jsonline file
    """
    data = []
    with open(data_path, "r", encoding="utf8") as f:
        reader = f.readlines()
        for line in reader:
            # print(line)
            sample = json.loads(line.strip())
            data.append(sample)
    print(f"load {len(data)} from {data_path}")
    return data


def write2json_by_line(data: Union[List, Dict], data_path, data_name='data', ensure_ascii=False, indent=2):
    """
    write data to jsonline file
    """
    with open(data_path, "w", encoding="utf-8") as fout:
        if isinstance(data, dict):
            for k, v in data.items():
                fout.write(json.dumps({"{}".format(k): v}, ensure_ascii=ensure_ascii, indent=indent))
                fout.write("\n")
        elif isinstance(data, list):
            for line in data:
                fout.write(json.dumps(line, ensure_ascii=ensure_ascii, indent=indent))
                fout.write("\n")
        print(f"{data_name}({len(data)}) saved into {data_path}")


def walk_dir(dir_path, suffix=".jpg", recursive=True):
    """
    walk all files in the dir_path with suffix
    """
    file_list = glob.glob(os.path.join(dir_path, "*" + suffix), recursive=recursive)
    print(f"{len(file_list)} files found in {dir_path}")
    return file_list


def read_file_by_line_lambda(file_path, line_lambda=lambda x: x, condition=lambda x: x):
    """
    read file by line and apply line_lambda

    Args:
        file_path: file path
        line_lambda: lambda function to transform each line
        condition: lambda function to filter each line

    Returns:
        list of transformed lines

    Example:

    ```python
    >>> def transform_line(line):
    >>>     return line.capitalize()
    >>> data = read_file_by_line_lambda('data.txt', line_lambda=transform_line)
    ```
    """
    data = []
    with open(file_path, "r", encoding="utf8") as fin:
        for line in fin:
            line = line.strip()
            if not condition(line):
                continue
            data.append(line_lambda(line))
    print(f"load {len(data)} from {file_path}")
    return data


def load_pickle(file_path):
    """
    load pickle file
    """
    with open(file_path, "rb") as fin:
        data = pickle.load(fin)
    print(f"load {len(data)} from {file_path}")
    return data


def write2pickle(data, file_path, data_name="data"):
    """
    write data to pickle file
    """
    with open(file_path, "wb") as fout:
        pickle.dump(data, fout)
    print(f"{data_name}({len(data)}) saved into {file_path}")
    return data


def load_ner_char_file(file_path, sep="\t"):
    """
    load ner char file
    """
    ner_data = []
    with open(file_path, "r", encoding="utf8") as fin:
        sent = []
        for line in fin:
            line = line.strip()
            if not line:
                if len(sent) > 0:
                    ner_data.append(sent)
                continue
            splits = line.split(sep)
            if len(splits) != 2:
                continue
            char, tag = splits
            if tag[0] == "M":
                tag = "I" + tag[1:]
            if tag[0] == "E":
                tag = "I" + tag[1:]
            sent.append((char, tag))

    print(f"load {len(ner_data)} from {file_path}")
    return ner_data


def _get_config_from_cli():
    cfg_cli = OmegaConf.from_cli()
    cli_keys = list(cfg_cli.keys())
    for cli_key in cli_keys:
        if "--" in cli_key:
            cfg_cli[cli_key.replace("--", "")] = cfg_cli[cli_key]
            del cfg_cli[cli_key]

    return cfg_cli


def get_config_from_yaml(default_conf_file: str = "./configs/default.yaml"):
    cfg = OmegaConf.load(default_conf_file)

    cfg_cli = _get_config_from_cli()
    if "config" in cfg_cli:
        cfg_cli_config = OmegaConf.load(cfg_cli.config)
        cfg = OmegaConf.merge(cfg, cfg_cli_config)
        del cfg_cli["config"]

    cfg = OmegaConf.merge(cfg, cfg_cli)

    def _check_config(cfg):
        # TODO
        pass

    def _update_config(cfg):
        cfg.save_weight_dir = os.path.join(cfg.workspace, "checkpoints")
        cfg.tensorboard_dir = os.path.join(cfg.workspace, "tensorboard_logs")
        # set per-gpu batch size
        try:
            import torch
            num_devices = torch.cuda.device_count()
            for mode in ["train", "val"]:
                new_batch_size = cfg[mode].batch_size // num_devices
                cfg[mode].batch_size = new_batch_size
        except Exception as e:
            print(e)
            pass

    _check_config(cfg)
    _update_config(cfg)

    return cfg