# Purpuse

I mainly used TF for recommendation task in big company, but it should be more efficient of using pytorch for small recommendation tasks in small company.

This dir is trying to train a CTR model using torch, it's just a toy example. Model is easy but the data preprocess pipeline is more complecate for various inputs.

Supported:

- Both numerical and categorical input features
- Both nomalization and discretization for numerical features
- Varlen sequence features by padding to the max length when training
- Hash for categorical features

# Models

- [X] DNN
- [ ] DIN

# Test Dataset

- random generated
- https://tianchi.aliyun.com/dataset/56?spm=a2c22.12282016.0.0.27934197fn5Vdv
