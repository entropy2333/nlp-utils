import os
import json


def load_json(json_file):
    with open(json_file, "r", encoding="utf8") as fin:
        data = json.load(fin)
    print(f"load {len(data)} from {json_file}")
    return data


def write2json(data_list, data_path, data_name="data"):
    with open(data_path, "w", encoding="utf-8") as fout:
        fout.write(json.dumps(data_list, ensure_ascii=False, indent=2))
        print(f"{data_name}({len(data_list)}) saved into {data_path}")


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


def write2json_by_line(data_list, data_path, data_name):
    with open(data_path, "w", encoding="utf-8") as fout:
        if isinstance(data_list, dict):
            for k, v in data_list.items():
                fout.write(json.dumps({"{}".format(k): v}, ensure_ascii=False))
                fout.write("\n")
        elif isinstance(data_list, list):
            for line in data_list:
                fout.write(json.dumps(line, ensure_ascii=False) + "\n")
        print(f"{data_name}({len(data_list)}) saved into {data_path}")
