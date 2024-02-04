import glob
import json
import os
from pathlib import Path
from typing import Dict, List, Union

import dill as pickle
from loguru import logger
from omegaconf import OmegaConf


class JsonFactory:
    @staticmethod
    def load_json(json_file):
        with open(json_file, "r", encoding="utf8") as fin:
            data = json.load(fin)
        logger.info(f"load {len(data)} from {json_file}")
        return data

    @staticmethod
    def write_json(data, data_path, data_name="data", ensure_ascii=False, indent=2):
        with open(data_path, "w", encoding="utf-8") as fout:
            fout.write(json.dumps(data, ensure_ascii=ensure_ascii, indent=indent))
        logger.info(f"{data_name}({len(data)}) saved into {data_path}")

    @staticmethod
    def load_jsonl(file_path: str, type: str = "list", gzip: bool = False):
        """
        load jsonline file
        """
        data = [] if type == "list" else {}

        def _read_jsonl(f):
            reader = f.readlines()
            for line in reader:
                # logger.info(line)
                sample = json.loads(line.strip())
                if type == "list":
                    data.append(sample)
                else:
                    data.update(sample)

        if gzip:
            import gzip

            file_path = file_path if file_path.endswith(".gz") else file_path + ".gz"
            with gzip.open(file_path, "rt", encoding="utf8") as f:
                _read_jsonl(f)
        else:
            with Path(file_path).open("r", encoding="utf8") as f:
                _read_jsonl(f)

        logger.info(f"load {len(data)} from {file_path}")
        return data

    @staticmethod
    def write_jsonl(
        data: Union[List, Dict],
        data_path,
        data_name="data",
        ensure_ascii=False,
        gzip: bool = False,
    ):
        """
        write data to jsonline file
        """
        Path(data_path).parent.mkdir(parents=True, exist_ok=True)

        def _write(f):
            if isinstance(data, dict):
                for k, v in data.items():
                    f.write(json.dumps({"{}".format(k): v}, ensure_ascii=ensure_ascii))
                    f.write("\n")
            elif isinstance(data, list):
                for line in data:
                    f.write(json.dumps(line, ensure_ascii=ensure_ascii))
                    f.write("\n")
            logger.info(f"{data_name}({len(data)}) saved into {data_path}")

        if gzip:
            import gzip

            data_path = data_path if data_path.endswith(".gz") else data_path + ".gz"
            with gzip.open(data_path, "wt", encoding="utf-8") as fout:
                _write(fout)
        else:
            with Path(data_path).open("w", encoding="utf-8") as fout:
                _write(fout)

    @staticmethod
    def load_jsonl_gzip(file_path: str, type: str = "list"):
        return JsonFactory.load_jsonl(file_path, type, gzip=True)

    @staticmethod
    def write_jsonl_gzip(data: Union[List, Dict], data_path, data_name="data", ensure_ascii=False):
        return JsonFactory.write_jsonl(data, data_path, data_name, ensure_ascii, gzip=True)


load_json = JsonFactory.load_json
write2json = JsonFactory.write_json
load_json_by_line = JsonFactory.load_jsonl
write2json_by_line = JsonFactory.write_jsonl


def walk_dir(dir_path, suffix=".jpg", recursive=True):
    """
    walk all files in the dir_path with suffix
    """
    if recursive:
        file_list = glob.glob(os.path.join(dir_path, "**", f"*{suffix}"), recursive=True)
    else:
        file_list = glob.glob(os.path.join(dir_path, f"*{suffix}"), recursive=False)
    logger.info(f"{len(file_list)} files found in {dir_path}")
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
    logger.info(f"load {len(data)} from {file_path}")
    return data


def load_pickle(file_path):
    """
    load pickle file
    """
    with open(file_path, "rb") as fin:
        data = pickle.load(fin)
    logger.info(f"load {len(data)} from {file_path}")
    return data


def write2pickle(data, file_path, data_name="data"):
    """
    write data to pickle file
    """
    with open(file_path, "wb") as fout:
        pickle.dump(data, fout)
    logger.info(f"{data_name}({len(data)}) saved into {file_path}")
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

    logger.info(f"load {len(ner_data)} from {file_path}")
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
            logger.info(e)
            pass

    _check_config(cfg)
    _update_config(cfg)

    return cfg


def split_file_to_dir(
    files: List[Path],
    output_dir: Union[Path, str],
    max_files_per_folder=1000,
    sort_files=True,
):
    """
    split files to the output directory, each folder contains max_files_per_folder files\\
    each folder name is like 1_1000, 1001_2000, etc.

    Args:
        files: list of files
        output_dir: output dir
        max_files_per_folder: max files per folder

    Example:

        ```python
        >>> from pathlib import Path
        >>> files = Path('data').glob('*.txt')
        >>> split_file_to_dir(files, 'output_dir', max_files_per_folder=1000)
        ```
    """
    if sort_files:
        files = sorted(list(files))
    logger.info(f"Splite {len(files)} files to {output_dir}")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    folder_idx = 0
    folder_name_prefix = 1
    folder_name_suffix = max_files_per_folder
    for idx in range(0, len(files), max_files_per_folder):
        folder_name = f"{folder_name_prefix}_{folder_name_suffix}"
        folder_name_suffix += max_files_per_folder
        folder_idx += 1
        Path(output_dir, folder_name).mkdir(parents=True, exist_ok=True)
        for file in files[idx : idx + max_files_per_folder]:
            file.rename(Path(output_dir, folder_name, file.name))
    logger.info(f"Split {len(files)} files to {folder_idx} folders")
