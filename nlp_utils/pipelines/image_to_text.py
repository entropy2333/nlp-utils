"""
reference:
    https://github.com/NielsRogge/Transformers-Tutorials/blob/master/Donut/DocVQA/Fine_tune_Donut_on_DocVQA.ipynb
"""

from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    TQDMProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities import rank_zero_only
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoTokenizer,
    BertTokenizerFast,
    VisionEncoderDecoderConfig,
    VisionEncoderDecoderModel,
)

from .base_model import BaseLightningModel


class PyTorchDataModule(Dataset):
    """PyTorch Dataset class"""

    def __init__(
        self,
        data: pd.DataFrame,
        feature_extractor: AutoFeatureExtractor,
        tokenizer: AutoTokenizer,
        image_dir: str = ".",
        image_transform: Optional[Callable] = None,
        target_max_token_len: int = 512,
    ):
        """
        initiates a PyTorch Dataset Module for input data
        Args:
            data (pd.DataFrame): input pandas dataframe. Dataframe must have 2 column --> "source_text" and "target_text"
            tokenizer (PreTrainedTokenizer): a PreTrainedTokenizer
            source_max_token_len (int, optional): max token length of source text. Defaults to 512.
            target_max_token_len (int, optional): max token length of target text. Defaults to 512.
        """
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.data = data
        self.image_dir = image_dir
        self.image_transform = image_transform
        self.target_max_token_len = target_max_token_len
        self.ignore_id = -100

    def __len__(self):
        """returns length of data"""
        return len(self.data)

    def __getitem__(self, index: int):
        """returns dictionary of input tensors to feed into OCR model"""

        data_row = self.data.iloc[index]
        image_path = data_row["image_path"]

        image = cv2.imread(str(Path(self.image_dir) / image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # apply image transform if provided
        if self.image_transform is not None:
            image = self.image_transform(image=image)["image"]

        pixel_values = self.feature_extractor(
            image,
            return_tensors="pt",
        ).pixel_values

        target_text_encoding = self.tokenizer(
            data_row["target_text"],
            max_length=self.target_max_token_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=False,
            add_special_tokens=True,
            return_tensors="pt",
        )

        labels = target_text_encoding["input_ids"].clone()
        # to make sure we have correct labels for text generation
        labels[labels == self.tokenizer.pad_token_id] = self.ignore_id

        return dict(
            pixel_values=pixel_values.squeeze(),
            labels=labels.flatten(),
        )


class LightningDataModule(pl.LightningDataModule):
    """PyTorch Lightning data class"""

    def __init__(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        feature_extractor: AutoFeatureExtractor,
        tokenizer: AutoTokenizer,
        image_transform: Optional[Callable] = None,
        image_dir: str = ".",
        batch_size: int = 4,
        target_max_token_len: int = 512,
        num_workers: int = 2,
    ):
        """
        initiates a PyTorch Lightning Data Module
        Args:
            train_df (pd.DataFrame): training dataframe. Dataframe must contain 2 columns --> "source_text" & "target_text"
            test_df (pd.DataFrame): validation dataframe. Dataframe must contain 2 columns --> "source_text" & "target_text"
            tokenizer (PreTrainedTokenizer): PreTrainedTokenizer
            batch_size (int, optional): batch size. Defaults to 4.
            source_max_token_len (int, optional): max token length of source text. Defaults to 512.
            target_max_token_len (int, optional): max token length of target text. Defaults to 512.
        """
        super().__init__()

        self.train_df = train_df
        self.test_df = test_df
        self.batch_size = batch_size
        self.image_transform = image_transform
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.image_dir = image_dir
        self.target_max_token_len = target_max_token_len
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_dataset = PyTorchDataModule(
            data=self.train_df,
            feature_extractor=self.feature_extractor,
            tokenizer=self.tokenizer,
            image_dir=self.image_dir,
            image_transform=self.image_transform,
            target_max_token_len=self.target_max_token_len,
        )
        self.test_dataset = PyTorchDataModule(
            data=self.test_df,
            feature_extractor=self.feature_extractor,
            tokenizer=self.tokenizer,
            image_dir=self.image_dir,
            image_transform=None,  # no need to augment validation data
            target_max_token_len=self.target_max_token_len,
        )

    def train_dataloader(self):
        """training dataloader"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        """test dataloader"""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        """validation dataloader"""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


class LightningModel(BaseLightningModel):
    """PyTorch Lightning Model class"""

    def __init__(
        self,
        feature_extractor,
        tokenizer,
        model,
        output_dir: str = "outputs",
        learning_rate: float = 1e-4,
        warmup_ratio: float = 0.1,
        num_training_steps: int = 1000,
        use_fgm: bool = False,
    ):
        """
        initiates a PyTorch Lightning Model
        Args:
            feature_extractor: AutoFeatureExtractor
            tokenizer : PreTrainedTokenizer
            model : PreTrainedModel
            output_dir (str, optional): output directory to save model checkpoints. Defaults to "outputs".
        """
        super().__init__()
        self.model = model
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.average_training_loss = None
        self.average_validation_loss = None
        self.average_validation_acc = None
        self.learning_rate = learning_rate
        self.warmup_ratio = warmup_ratio
        self.num_training_steps = num_training_steps

        if use_fgm:
            from ..models.fgm import FGM

            self.fgm = FGM(self.model, epsilon=0.5)
        else:
            self.fgm = None
        self.automatic_optimization = False if use_fgm else True

    def forward(self, pixel_values, decoder_input_ids, decoder_attention_mask, labels=None):
        """forward step"""
        output = self.model(
            pixel_values=pixel_values,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
        )

        return output.loss, output.logits

    def training_step(self, batch, batch_size):
        """training step"""
        pixel_values = batch["pixel_values"]
        decoder_input_ids = batch.get("decoder_input_ids", None)
        decoder_attention_mask = batch.get("decoder_attention_mask", None)
        labels = batch["labels"]

        if self.fgm is not None:
            opt = self.optimizers()
            scheduler = self.lr_schedulers()
            opt.zero_grad()

        loss, outputs = self(
            pixel_values=pixel_values,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
        )

        self.log("train_loss", loss, prog_bar=True, logger=True, sync_dist=True)

        if self.fgm is not None:
            self.manual_backward(loss)
            self.fgm.attack()
            loss_adv, outputs = self(
                pixel_values=pixel_values,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                labels=labels,
            )
            self.fgm.restore()
            self.manual_backward(loss_adv)
            opt.step()
            scheduler.step()

        return loss

    def validation_step(self, batch, batch_size):
        """validation step"""
        pixel_values = batch["pixel_values"]
        decoder_input_ids = batch.get("decoder_input_ids", None)
        decoder_attention_mask = batch.get("decoder_attention_mask", None)
        labels = batch["labels"]

        loss, outputs = self(
            pixel_values=pixel_values,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
        )

        # calculate accuracy
        generated_ids = outputs.argmax(-1)
        labels_mask = labels != -100
        accuracy = (generated_ids * labels_mask).eq(labels).sum() / labels_mask.sum()

        return {"val_loss": loss, "val_acc": accuracy}

    def test_step(self, batch, batch_size):
        """test step"""
        pixel_values = batch["pixel_values"]
        decoder_input_ids = batch["decoder_input_ids"]
        decoder_attention_mask = batch["decoder_attention_mask"]
        labels = batch["labels"]

        loss, outputs = self(
            pixel_values=pixel_values,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
        )

        self.log("test_loss", loss, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        """configure optimizers"""
        super().configure_optimizers()

    @rank_zero_only
    def on_save_checkpoint(self, checkpoint):
        path = (
            f"{self.output_dir}/simpleocr-epoch-{self.current_epoch}-"
            + f"train-loss-{str(self.average_training_loss)}-"
            + f"val-loss-{str(self.average_validation_loss)}-"
            + f"val-acc-{str(self.average_validation_acc)}"
        )
        self.tokenizer.save_pretrained(path)
        self.feature_extractor.save_pretrained(path)
        self.model.save_pretrained(path)

    def training_epoch_end(self, training_step_outputs):
        """save tokenizer and model on epoch end"""
        self.average_training_loss = np.round(
            torch.mean(torch.stack([x["loss"] for x in training_step_outputs])).item(),
            4,
        )

    def validation_epoch_end(self, validation_step_outputs):
        _loss = [x["val_loss"].cpu() for x in validation_step_outputs]
        _acc = [x["val_acc"].cpu() for x in validation_step_outputs]
        self.average_validation_loss = np.round(
            torch.mean(torch.stack(_loss)).item(),
            4,
        )
        self.average_validation_acc = np.round(
            torch.mean(torch.stack(_acc)).item(),
            4,
        )
        self.log("val_loss", self.average_validation_loss, prog_bar=True, logger=True, sync_dist=True)
        self.log("val_acc", self.average_validation_acc, prog_bar=True, logger=True, sync_dist=True)


class SimpleOCR:
    """Custom SimpleOCR class"""

    def __init__(self) -> None:
        """initiates SimpleOCR class"""
        self.model = None
        self.feature_extractor = None
        self.tokenizer = None
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

    def from_encoder_decoder_pretrained(
        self,
        encoder_type: str = "swin",
        encoder_name: str = "microsoft/swin-base-patch4-window7-224-in22k",
        decoder_type: str = "bart",
        decoder_name: str = "fnlp/bart-base-chinese",
    ):
        """
        load pretrained encoder and decoder
        Args:
            encoder_type (str, optional): encoder type. Defaults to "swin".
            encoder_name (str, optional): encoder name. Defaults to "microsoft/swin-base-patch4-window7-224-in22k".
            decoder_type (str, optional): decoder type. Defaults to "bart".
            decoder_name (str, optional): decoder name. Defaults to "fnlp/bart-base-chinese".
        """
        self.model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(encoder_name, decoder_name)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(encoder_name)
        if decoder_type in ["bart", "mbart", "bert"]:
            self.tokenizer = BertTokenizerFast.from_pretrained(decoder_name)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(decoder_name)
        self.model.config.decoder_start_token_id = self.tokenizer.cls_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.vocab_size = self.model.config.decoder.vocab_size
        self.model.decoder.resize_token_embeddings(len(self.tokenizer))

    def from_encoder_decoder_configs(
        self,
        encoder_config: AutoConfig,
        decoder_config: AutoConfig,
        feature_extractor: AutoFeatureExtractor = None,
        tokenizer: AutoTokenizer = None,
    ):
        """
        load pretrained encoder and decoder
        Args:
            encoder_config (str, optional): encoder config.
            decoder_config (str, optional): decoder config.
            feature_extractor (optional): feature extractor.
            tokenizer (optional): tokenizer.
        """
        if feature_extractor is not None:
            self.feature_extractor = feature_extractor
        else:
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(
                encoder_config.model_name_or_path, config=encoder_config
            )
            self.feature_extractor.size = encoder_config.image_size

        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = BertTokenizerFast.from_pretrained(decoder_config.model_name_or_path)

        config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(encoder_config, decoder_config)
        self.model = VisionEncoderDecoderModel(config=config)
        self.model.config.decoder_start_token_id = self.tokenizer.bos_token_id
        self.model.config.bos_token_id = self.tokenizer.bos_token_id
        self.model.config.eos_token_id = self.tokenizer.eos_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.vocab_size = self.model.config.decoder.vocab_size
        self.model.decoder.resize_token_embeddings(len(self.tokenizer))

    def from_pretrained(
        self,
        model_type: str = "donut",
        model_name: str = "naver-clova-ix/donut-base",
        image_height: int = 1280,
        image_width: int = 960,
        target_max_length: int = 128,
        **kwargs,
    ) -> None:
        """
        loads OCR Model model for training/finetuning
        Args:
            model_type (str, optional): "donut" or "other" . Defaults to "donut".
            model_name (str, optional): exact model architecture name. Defaults to "naver-clova-ix/donut-base".
        """
        if model_type == "donut":
            from transformers import DonutProcessor

            config = VisionEncoderDecoderConfig.from_pretrained(f"{model_name}")
            config.encoder.image_size = (image_width, image_height)
            config.decoder.max_length = target_max_length
            self.model = VisionEncoderDecoderModel.from_pretrained(f"{model_name}", config=config)
            processor = DonutProcessor.from_pretrained(f"{model_name}")
            processor.feature_extractor.size = (image_height, image_width)
            self.feature_extractor = processor.feature_extractor
            self.tokenizer = processor.tokenizer

        special_tokens = kwargs.pop("special_tokens", [])
        newly_added_num = self.tokenizer.add_tokens(special_tokens)
        if newly_added_num > 0:
            self.model.decoder.resize_token_embeddings(len(self.tokenizer))

    def train(
        self,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
        image_dir: str = ".",
        image_transform: Optional[Callable] = None,
        target_max_token_len: int = 512,
        batch_size: int = 8,
        max_epochs: int = 5,
        use_gpu: bool = True,
        output_dir: str = "outputs",
        early_stopping_patience_epochs: int = 0,  # 0 to disable early stopping feature
        precision=32,
        logger="default",
        dataloader_num_workers: int = 2,
        learning_rate: float = 1e-4,
        gradient_clip_algorithm: str = "norm",
        gradient_clip_val: float = 1.0,
        warmup_ratio: float = 0.1,
        use_fgm: bool = False,
        **kwargs,
    ):
        """
        trains OCR model on custom dataset
        Args:
            train_df (pd.DataFrame): training datarame. Dataframe must have 2 column --> "image_path" and "target_text"
            valid_df ([type], optional): validation datarame. Dataframe must have 2 column --> "image_path" and "target_text"
            image_dir (str, optional): image directory. Defaults to ".".
            target_max_token_len (int, optional): max token length of target text. Defaults to 512.
            batch_size (int, optional): batch size. Defaults to 8.
            max_epochs (int, optional): max number of epochs. Defaults to 5.
            use_gpu (bool, optional): if True, model uses gpu for training. Defaults to True.
            output_dir (str, optional): output directory to save model checkpoints. Defaults to "outputs".
            early_stopping_patience_epochs (int, optional): monitors val_loss on epoch end and stops training, if val_loss does not improve after the specied number of epochs. set 0 to disable early stopping. Defaults to 0 (disabled)
            precision (int, optional): sets precision training - Double precision (64), full precision (32) or half precision (16). Defaults to 32.
            logger (pytorch_lightning.loggers) : any logger supported by PyTorch Lightning. Defaults to "default". If "default", pytorch lightning default logger is used.
            dataloader_num_workers (int, optional): number of workers in train/test/val dataloader
        """
        self.data_module = LightningDataModule(
            train_df,
            valid_df,
            image_transform=image_transform,
            feature_extractor=self.feature_extractor,
            tokenizer=self.tokenizer,
            batch_size=batch_size,
            image_dir=image_dir,
            target_max_token_len=target_max_token_len,
            num_workers=dataloader_num_workers,
        )

        self.pl_module = LightningModel(
            feature_extractor=self.feature_extractor,
            tokenizer=self.tokenizer,
            model=self.model,
            output_dir=output_dir,
            learning_rate=learning_rate,
            warmup_ratio=warmup_ratio,
            num_training_steps=max_epochs * len(train_df) // batch_size,
            use_fgm=use_fgm,
        )

        # add callbacks
        callbacks = [
            TQDMProgressBar(refresh_rate=50),
            LearningRateMonitor(logging_interval="step"),
            RichModelSummary(max_depth=3),
        ]

        if early_stopping_patience_epochs > 0:
            early_stop_callback = EarlyStopping(
                monitor="val_loss",
                min_delta=0.00,
                patience=early_stopping_patience_epochs,
                verbose=True,
                mode="min",
            )
            callbacks.append(early_stop_callback)

        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            save_top_k=1,
            save_last=False,
            mode="min",
        )

        callbacks.append(checkpoint_callback)

        # add gpu support
        gpus = torch.cuda.device_count() if use_gpu else 0

        # add logger
        loggers = True if logger == "default" else logger

        accumulate_grad_batches = kwargs.pop("accumulate_grad_batches", 1)
        val_check_interval = kwargs.pop("val_check_interval", 1.0)
        log_every_n_steps = kwargs.pop("log_every_n_steps", 50)

        # prepare trainer
        trainer = pl.Trainer(
            logger=loggers,
            callbacks=callbacks,
            max_epochs=max_epochs,
            gpus=gpus,
            precision=precision,
            # log_every_n_steps=1,
            strategy=DDPStrategy(find_unused_parameters=False) if gpus > 1 else None,
            accumulate_grad_batches=accumulate_grad_batches,
            gradient_clip_algorithm=gradient_clip_algorithm,
            gradient_clip_val=gradient_clip_val,
            val_check_interval=val_check_interval,
            log_every_n_steps=log_every_n_steps,
            # track_grad_norm=2,
        )

        # fit trainer
        trainer.fit(self.pl_module, self.data_module)

    def load_model(
        self,
        model_type: str = "other",
        model_dir: str = "outputs",
        use_gpu: bool = False,
        special_tokens: list = [],
        **kwargs
    ):
        """
        loads a checkpoint for inferencing/prediction
        Args:
            model_type (str, optional): "donut" or "other". Defaults to "other".
            model_dir (str, optional): path to model directory. Defaults to "outputs".
            use_gpu (bool, optional): if True, model uses gpu for inferencing/prediction. Defaults to True.
        """
        if model_type == "donut":
            # TODO: add support for donut
            from transformers import DonutProcessor

        else:
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_dir)
            self.tokenizer = BertTokenizerFast.from_pretrained(model_dir)
            self.model = VisionEncoderDecoderModel.from_pretrained(model_dir)

        newly_added_num = self.tokenizer.add_tokens(special_tokens)
        if newly_added_num > 0:
            self.model.decoder.resize_token_embeddings(len(self.tokenizer))

        if use_gpu:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                raise "exception ---> no gpu found. set use_gpu=False, to use CPU"
        else:
            self.device = torch.device("cpu")

        self.model = self.model.to(self.device)

    def predict(
        self,
        source_image: Union[str, Path, np.ndarray],
        max_length: int = 512,
        num_return_sequences: int = 1,
        num_beams: int = 2,
        top_k: int = 50,
        top_p: float = 0.95,
        do_sample: bool = True,
        repetition_penalty: float = 2.5,
        length_penalty: float = 1.0,
        early_stopping: bool = True,
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True,
    ):
        """
        generates prediction for OCR model
        Args:
            source_text (str): any text for generating predictions
            max_length (int, optional): max token length of prediction. Defaults to 512.
            num_return_sequences (int, optional): number of predictions to be returned. Defaults to 1.
            num_beams (int, optional): number of beams. Defaults to 2.
            top_k (int, optional): Defaults to 50.
            top_p (float, optional): Defaults to 0.95.
            do_sample (bool, optional): Defaults to True.
            repetition_penalty (float, optional): Defaults to 2.5.
            length_penalty (float, optional): Defaults to 1.0.
            early_stopping (bool, optional): Defaults to True.
            skip_special_tokens (bool, optional): Defaults to True.
            clean_up_tokenization_spaces (bool, optional): Defaults to True.
        Returns:
            list[str]: returns predictions
        """
        self.model.eval()
        if isinstance(source_image, (str, Path)):
            source_image = cv2.cvtColor(cv2.imread(str(source_image)), cv2.COLOR_BGR2RGB)

        pixel_values = self.feature_extractor(source_image, return_tensors="pt").pixel_values

        pixel_values = pixel_values.to(self.device)

        generated_ids = self.model.generate(
            pixel_values=pixel_values,
            num_beams=num_beams,
            max_length=max_length,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            early_stopping=early_stopping,
            top_p=top_p,
            top_k=top_k,
            do_sample=do_sample,
            num_return_sequences=num_return_sequences,
        )
        preds = [
            self.tokenizer.decode(
                g,
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            )
            for g in generated_ids
        ]
        return preds

    def batch_predict(
        self,
        source_images: List[Union[str, Path, np.ndarray]],
        batch_size: int = 32,
        max_length: int = 512,
        num_return_sequences: int = 1,
        num_beams: int = 2,
        top_k: int = 50,
        top_p: float = 0.95,
        do_sample: bool = True,
        repetition_penalty: float = 2.5,
        length_penalty: float = 1.0,
        early_stopping: bool = True,
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True,
    ):
        """
        generates prediction for SwinOCR model
        Args:
            source_text (str): any text for generating predictions
            max_length (int, optional): max token length of prediction. Defaults to 512.
            num_return_sequences (int, optional): number of predictions to be returned. Defaults to 1.
            num_beams (int, optional): number of beams. Defaults to 2.
            top_k (int, optional): Defaults to 50.
            top_p (float, optional): Defaults to 0.95.
            do_sample (bool, optional): Defaults to True.
            repetition_penalty (float, optional): Defaults to 2.5.
            length_penalty (float, optional): Defaults to 1.0.
            early_stopping (bool, optional): Defaults to True.
            skip_special_tokens (bool, optional): Defaults to True.
            clean_up_tokenization_spaces (bool, optional): Defaults to True.
        Returns:
            list[str]: returns predictions
        """
        self.model.eval()
        if not isinstance(source_images, list) or len(source_images) == 0:
            return []

        if isinstance(source_images[0], (str, Path)):
            source_images = [str(p) for p in source_images]
            source_images = [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) for p in source_images]

        pixel_values = self.feature_extractor(source_images, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)

        generated_ids = self.model.generate(
            pixel_values=pixel_values,
            num_beams=num_beams,
            max_length=max_length,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            early_stopping=early_stopping,
            top_p=top_p,
            top_k=top_k,
            do_sample=do_sample,
            num_return_sequences=num_return_sequences,
        )
        preds = self.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
        )
        return preds


def predict_main(args, **kwargs):
    """
    main function for prediction
    """
    from pathlib import Path

    import pandas as pd

    from ..file_utils import walk_dir, write2json, write2json_by_line
    from ..log_utils import logger
    from ..tqdm import disable_progress_bar, tqdm

    if Path(args.image_dir_or_path).is_dir():
        image_list = walk_dir(args.image_dir_or_path, suffix=args.image_suffix)
    else:
        image_list = [args.image_dir_or_path]

    model = SimpleOCR()
    model.load_model(model_dir=args.model_name_or_path, use_gpu=args.use_gpu)

    logger.info(f"Load model from {args.model_name_or_path}")

    if args.disable_tqdm:
        disable_progress_bar()

    prediction_result = []
    batch_size = min(args.batch_size, len(image_list))
    logger.info(f"Predicting with batch_size={batch_size}")
    for i in tqdm(range(0, len(image_list), batch_size), desc="Predicting"):
        batch_image_list = image_list[i : i + batch_size]
        batch_prediction = model.batch_predict(
            source_images=batch_image_list,
            max_length=args.max_length,
            num_beams=args.num_beams,
            **kwargs,
        )
        batch_prediction = [x.replace(" ", "") for x in batch_prediction]
        for image_path, prediction in zip(batch_image_list, batch_prediction):
            prediction_result.append({"image_path": image_path, "prediction": prediction})

    output_file_name = f"prediction.{args.output_file_type}"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file_path = output_dir / output_file_name
    if args.output_file_type == "csv":
        pd.DataFrame(prediction_result).to_csv(str(output_file_path), index=False, sep="\t")
    elif args.output_file_type == "jsonl":
        write2json_by_line(prediction_result, output_file_path, output_file_name)
    elif args.output_file_type == "json":
        write2json(prediction_result, output_file_path, output_file_name)
    else:
        raise ValueError(f"Unknown output_file_type: {args.output_file_type}")
    logger.info(f"Prediction result saved to {output_file_path}")


if __name__ == "__main__":
    """
    Usage:
        export CUDA_VISIBLE_DEVICES=0
        python -m nlp_utils.pipelines.image_to_text \
            --image_dir_or_path /path/to/image_dir_or_path \
            --model_name_or_path /path/to/model_dir \
            --output_dir /path/to/output_dir \
            --batch_size 32 --use_gpu --disable_tqdm
    """
    import argparse

    parser = argparse.ArgumentParser()

    # input args
    parser.add_argument("--image_dir_or_path", "-i", type=str, required=True)
    parser.add_argument("--image_suffix", type=str, default=".jpg")
    parser.add_argument("--ground_truth_file", type=str, required=False)

    # model args
    parser.add_argument("--model_name_or_path", "-m", type=str, required=True)

    # output args
    parser.add_argument("--output_dir", "-o", type=str, required=True)
    parser.add_argument("--output_file_type", type=str, default="csv")

    # predict args
    parser.add_argument("--max_length", type=int, default=32)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--use_gpu", default=True, action="store_true")
    parser.add_argument("--disable_tqdm", default=False, action="store_true")

    args, left_argv = parser.parse_known_args()
    # TODO: parse left_argv
    predict_main(args)
