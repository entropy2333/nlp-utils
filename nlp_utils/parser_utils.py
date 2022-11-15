import argparse

from pytorch_lightning.trainer import Trainer


def add_pl_trainer_args(parser):
    parser = Trainer.add_argparse_args(parser)
    return parser


def add_jupyter_args(parser):
    parser.add_argument("-f", dest="j_cfile", help="jupyter config file", default="file.json", type=str)
    return parser
