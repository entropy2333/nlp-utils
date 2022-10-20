# Image-to-Text

Image-to-Text model with huggingface VisionEncoderDecoder

## Quick Start

```python
# import
from nlp_utils.pipelines import SimpleOCR

# instantiate
model = SimpleOCR()

# load encoder and decoder
model.from_encoder_decoder_pretrained(
    "swin",
    "microsoft/swin-base-patch4-window7-224-in22k",
    "bert",
    "fnlp/bart-base-chinese",
)

# load train dataset
import pandas as pd
df_train = pd.read_csv("dataset/train.txt", sep="\t", header=None)
df_train.columns = ["image_path", "target_text"]
df_train['target_text'] = df_train['target_text'].astype(str)

# train
model.train(
    train_df=train_df, # pd.DataFrame with 2 columns: image_path & target_text
    eval_df=eval_df, # pd.DataFrame with 2 columns: image_path & target_text
    image_dir="dataset/images",
    target_max_token_len = 128,
    batch_size = 16,
    max_epochs = 5,
    use_gpu = True,
    output_dir = "outputs",
    early_stopping_patience_epochs = 0,
    precision = 32,
    accumulate_grad_batches = 1,
    learning_rate = 2e-5,
    dataloader_num_workers = 0,
    use_fgm = False,
    gradient_clip_algorithm = None,
    gradient_clip_val = None,
)

# load trained T5 model
model.load_model("other", checkpoint_dir, use_gpu=True)

# predict
model.predict("dataset/images/1.jpg")

# batch predict
model.batch_predict(["dataset/images/1.jpg", "dataset/images/2.jpg"])
```