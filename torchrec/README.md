# Purpuse


I primarily used TensorFlow for large-scale recommendation tasks when in big company, but PyTorch could be more efficient for smaller tasks in a smaller company.

This directory aims to train a Click-Through Rate (CTR) model using PyTorch. It's a simple example, seeking to keep everything minimal. While the model is straightforward, the data preprocessing pipeline is more complex due to a variety of inputs.

Supported features include:

* Both numerical and categorical input features
  * Categorical: automatic vocabulary extraction, hash embedding, low-frequency filtering
  * Numerical: standard or 0-1 normalization, automatic discretization, automatic update of statistical number for standard or 0-1 normalization if new data is fed in
* Variable-length sequence feature support
* Sequence features with weights by setting the weight column, for example, 'k1:v1,k2:v2,k3:v3'
* Implemented [DataFrameDataset, IterableDataFrameDataset](./core/dataset.py) for straightforward training with pandas DataFrame in PyTorch
* Implemented a common [Trainer](./core/trainer.py) for training pytorch models, and save/load the results

Not supported:

- Distribution training, as target of this tool is for small companies

Todo:

- Generate a simple model service API

# [Example]

[./train_amazon.ipynb](./train_amazon.ipynb)

Implement the model by inherit from nn.Module but with some extra member methods,

- required:
    - training_step
    - validation_step
- optional:
    - configure_optimizers
    - configure_lr_scheduler

# Related Dataset

- random generated
- https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html
- https://tianchi.aliyun.com/dataset/56?spm=a2c22.12282016.0.0.27934197fn5Vdv
