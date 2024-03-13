# Purpuse

I mainly used TF for recommendation task in big company, but it should be more efficient of using pytorch for small recommendation tasks in small company.

This dir is trying to train a CTR model using torch, it's just a toy example and want to make everything minimum. Model is easy but the data preprocess pipeline is more complecate for various inputs.

Supported:

- Both numerical and categorical input features
- Both nomalization and discretization for numerical features
- Varlen sequence features by padding to the max length when training
- Hash for categorical features

TODOs:

- [ ] support seq features with weights, for example: 'k1:v1,k2:v2,k3:v3'

# Models

- [X] DNN
- [ ] DIN

# Related Dataset

- random generated
- https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html
- https://tianchi.aliyun.com/dataset/56?spm=a2c22.12282016.0.0.27934197fn5Vdv
