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


def memory_report():
    """
    Usage:
        python -c "from nlp_utils.train_utils import memory_report; memory_report()"

    Output:
        CPU Mem Usage: 25.1 GB / 67.3 GB
        GPU 0 Mem Usage: 481.0MB / 11448.0MB | Util  1%
        GPU 1 Mem Usage: 453.0MB / 11448.0MB | Util  1%
    """
    import psutil
    from humanize import naturalsize

    print(
        f"CPU Mem Usage: {naturalsize(psutil.virtual_memory().used)} / {naturalsize(psutil.virtual_memory().total)} |"
        f" Util {psutil.virtual_memory().percent:2.2f}%"
    )
    import GPUtil

    gpus = GPUtil.getGPUs()
    for i, gpu in enumerate(gpus):
        print(f"GPU {i} Mem Usage: {gpu.memoryFree}MB / {gpu.memoryTotal}MB | Util {gpu.memoryUtil:2.2f}%")


def model_summary(model, max_depth: int = 1, example_input_array=None):
    """
    Wrapper of pytorch_lightning.utilities.model_summary.summarize.

    Example Usage:
        >>> model.eval()
        >>> model_summary(model, example_input_array={
            'pixel_values': torch.zeros(1, 3, 960, 1280).float(),
            'decoder_input_ids': torch.ones(1, 16).long()
        })
    """
    import pytorch_lightning as pl
    from pytorch_lightning.utilities.model_summary import summarize

    class DummyModule(pl.LightningModule):
        def __init__(self, net):
            super().__init__()
            self.net = net
            if example_input_array is not None:
                self.example_input_array = example_input_array

        def forward(self, *args, **kwargs):
            return self.net(*args, **kwargs)

    return summarize(DummyModule(model), max_depth=max_depth)
