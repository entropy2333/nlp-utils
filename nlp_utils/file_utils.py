import os
import json
# from loguru import logger


def load_json(json_file):
    with open(json_file, "r", encoding="utf8") as fin:
        data = json.load(fin)
    print(f"load {len(data)} from {json_file}")
    # logger.info(f"load {len(data)} from {json_file}")
    return data


def write2json(data_list, data_path, data_name="data"):
    with open(data_path, "w", encoding="utf-8") as fout:
        fout.write(json.dumps(data_list, ensure_ascii=False, indent=2))
    print(f"{data_name}({len(data_list)}) saved into {data_path}")
    # logger.info(f"{data_name}({len(data_list)}) saved into {data_path}")


def load_json_by_line(data_path):
    data = []
    with open(data_path, "r", encoding="utf8") as f:
        reader = f.readlines()
        for line in reader:
            # print(line)
            sample = json.loads(line.strip())
            data.append(sample)
    print(f"load {len(data)} from {data_path}")
    return data


def write2json_by_line(data_list, data_path, data_name='data'):
    with open(data_path, "w", encoding="utf-8") as fout:
        if isinstance(data_list, dict):
            for k, v in data_list.items():
                fout.write(json.dumps({"{}".format(k): v}, ensure_ascii=False))
                fout.write("\n")
        elif isinstance(data_list, list):
            for line in data_list:
                fout.write(json.dumps(line, ensure_ascii=False) + "\n")
        print(f"{data_name}({len(data_list)}) saved into {data_path}")


def walk_dir(dir_path, suffix=".jpg"):
    """
    walk all files in the dir_path with suffix
    """
    file_list = []
    for root, dirs, files in os.walk(dir_path):
        for file_ in files:
            if file_.endswith(suffix):
                file_list.append(os.path.join(root, file_))
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