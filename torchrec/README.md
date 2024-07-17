# Purpuse

I primarily used TensorFlow for large-scale recommendation tasks when in big company, but PyTorch could be more efficient for smaller tasks in a smaller company.

This directory aims to train a Click-Through Rate (CTR) model using PyTorch. It's a simple example, seeking to keep everything minimal. While the model is straightforward, the data preprocessing pipeline is more complex due to a variety of inputs.

Supported features include:

* Both numerical and categorical input features
  * Categorical: automatic vocabulary extraction, low-frequency filtering, dynamic embedding, hash embedding
  * Numerical: standard or 0-1 normalization, automatic discretization, automatic update of statistical number for standard or 0-1 normalization if new data is fed in
* Variable-length sequence feature support, if there's order in the sequence, please put the latest data before the oldest data as it may pads at the end of the sequence
* Sequence features support weights by setting the weight column
* Implemented [DataFrameDataset, IterableDataFrameDataset](./core/dataset.py) for straightforward training with pandas DataFrame in PyTorch
* Implemented a common [Trainer](./core/trainer.py) for training pytorch models, and save/load the results
* Basic FastAPI for model serving

Not supported:

- Distribution training, as target of this tool is for small companies

# Example

[./train_amazon.ipynb](./train_amazon.ipynb)

Implement the model by inherit from nn.Module but with some extra member methods,

- required:
  - training_step
  - validation_step
- optional:
  - configure_optimizers
  - configure_lr_scheduler

# API Serving

[core/serve.py](./core/serve.py)

- Launch the service by given service name and model path

```
cd $torchrec_root
python -m core.serve --name [name] --path [path/to/model or path/to/ckpt]
```

- Test the service: reference `test_predict` in `core/serve.py`

# Related Dataset

- random generated
- https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html
- https://tianchi.aliyun.com/dataset/56?spm=a2c22.12282016.0.0.27934197fn5Vdv
