import os
import random

import numpy as np
import torch


def seed_everything(seed=2022):
    """
    Set random seed for all modules.

    Args:
        seed: random seed
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def setup_device(cuda=True):
    """
    Setup device.

    Args:
        cuda: if True, use cuda, else use cpu
    """
    if cuda:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    return device


def save_model(model, model_path, only_state_dict=True):
    """
    Save model to a file.

    Args:
        model: model to save
        model_path: path to save model
        only_state_dict: if True, only save model's state_dict, else save model itself
    """
    model_to_save = model.module if hasattr(model, "module") else model
    if only_state_dict:
        torch.save(model_to_save.state_dict(), model_path)
    else:
        torch.save(model_to_save, model_path)


def load_model(model, model_path):
    """
    Load model from a file.

    Args:
        model: model to load
        model_path: path to load model
    """
    model_to_load = model.module if hasattr(model, "module") else model
    model_to_load.load_state_dict(torch.load(model_path, map_location="cpu"), strict=False)
    return model
