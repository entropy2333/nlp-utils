# Text Generation

modified simpleT5

## Quick Start

```python
# import
from nlp_utils.pipelines import SimpleT5

# instantiate
model = SimpleT5()

# load (supports t5, mt5, byT5 models)
model.from_pretrained("t5", "t5-base")

# train
model.train(
    train_df=train_df, # pd.DataFrame with 2 columns: source_text & target_text
    eval_df=eval_df, # pd.DataFrame with 2 columns: source_text & target_text
    source_max_token_len = 128,
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
model.load_model("t5", checkpoint_dir, use_gpu=True)

# predict
model.predict("input text for prediction")

# batch predict
model.batch_predict(["input text1 for prediction", "input text2 for prediction"])
```

## Supported Models

specified with `model_type`

- t5
- mt5
- byt5
- bart
- cpt

## Generation Options

reference: [Utilities for Generation](https://huggingface.co/docs/transformers/main/en/internal/generation_utils)

example:

```bash
kwargs = dict(
    max_length=100,
    num_beams=10,
    do_sample=False,
    top_k=50,
    top_p=1.0,
    early_stopping=False,
    repetition_penalty=2.5,
)
model.predict(input_text, **kwargs)
```

## Acknowledgements

- [Pytorch Lightning](https://www.pytorchlightning.ai/)
- [simpleT5](https://github.com/Shivanandroy/simpleT5)