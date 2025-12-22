# FeedForwardAR

## Description

**FeedForwardAR** is a fully connected neural network (Feed-Forward Neural Network) for predicting investment indicators across regions of Russia based on historical time series.  
The model operates in an autoregressive (AR) manner, using values from several previous time steps to predict future values.

---

## Architecture

Key characteristics of the model:

- **Input data**:  
  The input tensor `x` has shape `[batch_size, seq_len, n_features]`, where:
  - `seq_len` is the length of the time window (number of past steps used for analysis),
  - `n_features` is the number of features per step.

- **Regional embeddings (optional)**:  
  If `region_emb_size > 0` is specified, an `nn.Embedding` layer is used to encode regions into fixed-size vectors.  
  These embeddings are concatenated with the time features before being fed into the network.

- **Fully connected layers (Feed-Forward)**:  
  The network consists of a sequence of `Linear -> ReLU -> Dropout` layers. Hidden layer sizes are defined via the `hidden_sizes` parameter.

- **Output layer**:  
  A linear layer that produces predictions for `horizon` future steps.

- **Weight initialization**:  
  **Xavier Uniform** initialization is used for weights, and biases are initialized to zeros.

---

## Usage

Example of creating the model and checking the number of parameters:

```python
import torch
from torch import nn
from torch.nn import init

# Create the model
test_model = FeedForwardAR(n_features=78, seq_len=4, horizon=3)
print(test_model)

# Count parameters
total_params = sum(p.numel() for p in test_model.parameters())
trainable_params = sum(p.numel() for p in test_model.parameters() if p.requires_grad)

print(f"Total parameters: {total_params}")
print(f"Trainable parameters: {trainable_params}")
```
---

## Hyperparameter Settings

The model supports the following main parameters:

| Parameter | Description |
|-----------|-------------|
| `n_features` | Number of features per step |
| `seq_len` | Length of the time window |
| `horizon` | Number of steps to predict |
| `hidden_sizes` | Sizes of hidden layers |
| `dropout` | Dropout rate |
| `region_emb_size` | Size of the regional embedding |

---

## Training

Model training is implemented using a standard PyTorch training loop, with configurable options:

- Optimizer (`Adam`, `AdamW`, `RMSProp`)
- Learning rate (`lr`)
- L2 regularization (`weight_decay`)
- Dropout
- Gradient clipping (`grad_clip`)
- Loss function type: `MSE`, `MAE`, `Huber`

---

## Features

- Simple and fast-to-train architecture.
- Suitable for small-time series.
- Support for regional embeddings.
- Easily integrates with automated hyperparameter tuning procedures (e.g., Optuna).

---
