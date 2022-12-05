import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from transformers.optimization import (
    AdamW,
    get_constant_schedule,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)


torch.cuda.empty_cache()
pl.seed_everything(42, workers=True)


class BaseLightningModel(pl.LightningModule):
    """Base PyTorch Lightning Model class"""

    def __init__(
        self,
        model,
        output_dir: str = "outputs",
        learning_rate: float = 1e-4,
        warmup_ratio: float = 0.1,
        scheduler_type: str = "linear",
        num_training_steps: int = 1000,
        use_fgm: bool = False,
        **kwargs,
    ):
        """
        initiates a PyTorch Lightning Model
        Args:
            model : PreTrainedModel
            output_dir (str, optional): output directory to save model checkpoints. Defaults to "outputs".
            learning_rate (float, optional): learning rate. Defaults to 1e-4.
            warmup_ratio (float, optional): warmup ratio. Defaults to 0.1.
            scheduler_type (str, optional): scheduler type. Defaults to "linear".
            num_training_steps (int, optional): number of training steps. Defaults to 1000.
            use_fgm (bool, optional): whether to use fast gradient method. Defaults to False.
        """
        super().__init__()
        self.model = model
        self.output_dir = output_dir
        self.average_training_loss = None
        self.average_validation_loss = None
        self.average_validation_acc = None
        self.learning_rate = learning_rate
        self.scheduler_type = scheduler_type
        self.warmup_ratio = warmup_ratio
        self.num_training_steps = num_training_steps
        if use_fgm:
            from ..models.fgm import FGM

            self.fgm = FGM(self.model, epsilon=0.5)
        else:
            self.fgm = None
        self.automatic_optimization = False if use_fgm else True

    def configure_optimizers(self):
        """configure optimizers"""
        optimizer = self._configure_optimizer(self.model, self.learning_rate)
        num_warmup_steps = int(self.num_training_steps * self.warmup_ratio)
        scheduler = self._configure_scheduler(
            self.scheduler_type,
            optimizer,
            num_warmup_steps,
            self.num_training_steps,
        )
        return [optimizer], [scheduler]

    @rank_zero_only
    def on_save_checkpoint(self, checkpoint):
        raise NotImplementedError

    @staticmethod
    def _configure_optimizer(model, learning_rate):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
        return optimizer

    @staticmethod
    def _configure_scheduler(scheduler_type, optimizer, num_warmup_steps, num_training_steps):
        if scheduler_type == "linear":
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
            )
        elif scheduler_type == "cosine":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
            )
        elif scheduler_type == "constant":
            scheduler = get_constant_schedule(optimizer)
        elif scheduler_type == "constant_with_warmup":
            scheduler = get_constant_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
            )
        scheduler = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
        }
        return scheduler
