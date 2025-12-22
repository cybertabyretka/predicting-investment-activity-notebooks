# ARLSTMForecaster

## Description

**ARLSTMForecaster** is a recurrent neural network based on LSTM with an attention mechanism for forecasting investment indicators across Russian regions several years ahead.  
The model operates in an autoregressive (AR) manner, using past-time series values to predict future ones. It supports regional embeddings, input projection, and multiple attention variants.

---

## Architecture

Main components of the model:

- **Input data**:  
  The input tensor `x` has shape `[batch_size, seq_len, n_features]`, where:
  - `seq_len` — length of the time window (number of past steps for analysis),
  - `n_features` — number of features per step.

- **Regional embeddings (optional)**:  
  If `region_emb_size > 0`, an `nn.Embedding` layer is used to encode regions into a fixed-size vector.  
  These embeddings are concatenated with time features before being fed into the LSTM encoder.

- **Input projection (optional)**:  
  An `nn.Linear` layer can be used to reduce input dimensionality before passing data to the encoder.

- **Encoder (LSTM)**:  
  A multi-layer LSTM processes the input sequence and forms a context vector `enc_ctx`.  
  Context pooling is performed by concatenating the last hidden state and the mean over the time dimension.

- **Decoder (LSTM)**:  
  The decoder receives previous predictions and hidden states to generate the next step.  
  Teacher forcing is supported during training using known target values.

- **Attention mechanism**:  
  Dot-product or scaled dot-product attention is used to weight encoder outputs during each decoder step.  
  Attention weights can be regularized with Dropout.

- **Output layer**:  
  A linear layer combines the decoder output and encoder context to predict a single time step.

- **Weight initialization**:  
  - Linear layers: Xavier Uniform  
  - LSTM: `weight_ih` — Xavier, `weight_hh` — orthogonal (with fallback to Xavier), bias — zeros, with ones for the forget gate

---

## Usage

Example of model creation and parameter count check:

```python
import torch
from torch import nn

# Create model
test_model = ARLSTMForecaster(n_features=78, horizon=3)
print(test_model)

# Count parameters
total_params = sum(p.numel() for p in test_model.parameters())
trainable_params = sum(p.numel() for p in test_model.parameters() if p.requires_grad)

print(f"Total parameters: {total_params}")
print(f"Trainable parameters: {trainable_params}")
```
---

## Main Hyperparameters

| Parameter | Description |
|----------|-------------|
| `n_features` | Number of features per time step |
| `horizon` | Number of steps to forecast |
| `encoder_hidden` | Encoder hidden state size |
| `decoder_hidden` | Decoder hidden state size (if None, uses encoder_hidden) |
| `num_layers` | Number of layers in encoder and decoder |
| `dropout` | Dropout rate inside LSTM |
| `input_proj_dim` | Input projection dimensionality (optional) |
| `region_emb_size` | Regional embedding size |
| `attention_type` | Attention type: `"dot"` or `"scaled_dot"` |
| `attention_dropout` | Dropout for attention weights |
| `input_noise_std` | Standard deviation of Gaussian input noise for regularization |

---

## Training

Model training is implemented using a standard PyTorch loop with configurable options:

- Optimizers: `Adam`, `AdamW`, `RMSProp`
- Learning rate (`lr`) and L2 regularization (`weight_decay`)
- Dropout inside LSTM and on decoder outputs
- Gradient clipping (`grad_clip`)
- Loss function type: `MSE`, `MAE`, `Huber`
- Teacher forcing with a specified probability (`teacher_forcing_prob`)
- LR scheduler support: `StepLR` or `CosineAnnealingLR`

---

## Features

- LSTM with attention enables capturing long-term temporal dependencies.
- Auto-regressive mode support for sequence generation.
- Optional regional embeddings allow the model to account for region-specific characteristics.
- Flexible encoder and decoder configuration.
- Compatible with automated hyperparameter search (Optuna).
- Input noise regularization support to improve generalization.

---
