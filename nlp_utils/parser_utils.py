import argparse


def init_parser():
    parser = argparse.ArgumentParser(description="Dummy parser")
    basic_args(parser)
    return parser


def basic_args(parser):
    # ========================== Basic Configs ==========================
    parser.add_argument("-f", dest='j_cfile', help="jupyter config file", default="file.json", type=str)
    parser.add_argument("--seed", type=int, default=2022, help="random seed.")


def model_args(parser):
    # ========================== BERT Configs =============================
    parser.add_argument('--model_type', type=str, default='vlbert', help='model type')
    parser.add_argument('--bert_dir', type=str, default='hfl/chinese-roberta-wwm-ext')
    parser.add_argument('--bert_cache', type=str, default='data/cache')
    parser.add_argument('--bert_seq_length', type=int, default=256)
    parser.add_argument('--bert_learning_rate', type=float, default=3e-5)
    parser.add_argument('--bert_warmup_steps', type=int, default=5000)
    parser.add_argument('--bert_max_steps', type=int, default=30000)
    parser.add_argument("--bert_hidden_dropout_prob", type=float, default=0.1)
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout ratio')


def train_args(parser):
    # ========================= Data Configs ==========================
    parser.add_argument('--train_dir', type=str, default='data/annotations/labeled.json')
    parser.add_argument('--val_ratio', default=0.1, type=float, help='validation ratio')
    parser.add_argument('--batch_size', default=64, type=int, help="use for training duration per worker")
    parser.add_argument('--val_batch_size', default=256, type=int, help="use for validation duration per worker")
    parser.add_argument('--prefetch', default=16, type=int, help="use for training duration per worker")
    parser.add_argument('--num_workers', default=4, type=int, help="num_workers for dataloaders")

    # ======================== SavedModel Configs =========================
    parser.add_argument('--save_model_path', type=str, default='save/v1', help="Path to save model")
    parser.add_argument('--best_score', default=0.5, type=float, help='save checkpoint if mean_f1 > best_score')

    # ========================== Training Tricks ==========================
    parser.add_argument("--grad_clip", default=-1.0, type=float, help="Gradient Clipping Norm")
    parser.add_argument("--use_ema", default=False, action='store_true', help="Use EMA for training")
    parser.add_argument('--ema_start', type=bool, default=False)
    parser.add_argument('--ema_start_step', type=int, default=25 * 8)
    parser.add_argument("--ema_decay", default=0.95, type=float, help="Exponential Moving Average decay")
    parser.add_argument("--use_swa", default=False, action='store_true', help="Use SWA for training")
    parser.add_argument("--use_fgm", default=False, action='store_true', help="Use FGM for training")
    parser.add_argument("--fast_mode", default=False, action='store_true', help="Fast training mode")
    parser.add_argument("--fgm_epsilon", default=0.5, type=float, help="FGM epsilon")
    parser.add_argument("--use_lookahead", default=False, action='store_true', help="Use Lookhead for training")
    parser.add_argument("--num_folds", default=5, type=int, help="Number of folds for k-fold cross validation")


def optimize_args(parser):
    # ========================= Learning Configs ==========================
    parser.add_argument('--learning_rate', default=5e-5, type=float, help='initial learning rate')
    parser.add_argument('--max_epochs', type=int, default=5, help='How many epochs')
    parser.add_argument('--max_steps', default=50000, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--print_steps', type=int, default=20, help="Number of steps to log training metrics.")
    parser.add_argument('--warmup_steps', default=1000, type=int, help="warm ups for parameters not in bert or vit")
    parser.add_argument('--warmup_ratio', default=0.1, type=float, help="warm ups for parameters not in bert or vit")
    parser.add_argument('--minimum_lr', default=0., type=float, help='minimum learning rate')
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float, help="Epsilon for Adam optimizer.")


def finetune_args(parser):
    model_args(parser)
    optimize_args(parser)
    train_args(parser)


def inference_args(parser):
    model_args(parser)
    train_args(parser)
    parser.add_argument('--test_dir', type=str, default='data/annotations/test_a.json')
    parser.add_argument('--test_batch_size', default=64, type=int, help="use for inference duration per worker")
    parser.add_argument('--test_output_csv', type=str, default='data/result.csv')
    parser.add_argument('--test_batch_size', default=256, type=int, help="use for testing duration per worker")
    parser.add_argument('--load_model_path', type=str, default='save/v1', help="Path to load model")