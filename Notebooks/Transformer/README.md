# ARTransformerForecaster

## Description

**ARTransformerForecaster** is an autoregressive Transformer Encoder-based model for forecasting investment indicators across Russian regions for multiple years ahead.  
The model uses a Transformer architecture to extract contextual representations of time series and a separate autoregressive decoder that generates forecasts step by step over a given horizon.

The model supports:
- sinusoidal and learnable positional encodings,
- regional embeddings,
- multiple context pooling strategies,
- teacher forcing,
- robust training with Huber loss, dropout, input noise, and gradient clipping,
- integration with Optuna for hyperparameter tuning.

---

## Architecture

### Input Data

The input tensor `x` has shape:

- `[batch_size, seq_len, n_features]`

where:
- `seq_len` — length of the historical window,
- `n_features` — number of features per time step.

### Regional Embeddings (optional)

If `region_emb_size > 0`, an `nn.Embedding` layer is used to encode the region into a dense vector.  
The regional embedding is concatenated with input features **at each time step**.

This allows the model to account for stable structural differences between regions (economy, demographics, infrastructure, etc.).

---

### Input Projection

After combining features, a linear projection is applied:

- `Linear(n_features + region_emb_size → d_model)`

It maps the input to the `d_model` dimensionality used inside the Transformer.

Additionally, the following may be applied:
- input dropout (`input_dropout`),
- Gaussian noise (`input_noise_std`) during training.

---

### Positional Encoding

The `PositionalEncodingFlexible` module supports two modes:

- `sin` — classical sinusoidal positional encoding,
- `learned` — learnable positional embeddings.

Positional encoding is added to input embeddings before feeding into the Transformer Encoder.

---

### Transformer Encoder

The encoder is built on `nn.TransformerEncoder` and consists of:

- `num_layers` encoder blocks,
- multi-head self-attention (`nhead`),
- feed-forward layers of size `dim_feedforward`,
- residual connections and LayerNorm.

The encoder processes the entire sequence and returns a tensor:

- `[batch_size, seq_len, d_model]`

---

### Context Pooling

To convert the sequence into a fixed-size context vector, one of the pooling options is used:

#### `last_mean`
Concatenates:
- the last time step of the encoder,
- the mean across the time axis.

#### `attn`
Learnable attention pooling:
- attention weights are computed across time steps,
- a weighted sum of encoder outputs is formed,
- the result is concatenated with the last step.

The resulting context has dimension:
- `ctx_dim = 2 * d_model`

LayerNorm can be optionally applied.

---

## Autoregressive Decoder

The decoder is implemented as a **stepwise MLP**, not a Transformer Decoder.

### Initialization

The first value (`prev`) is determined:
- either via `start_proj(ctx)`,
- or using `y_last` if the last real value is provided.

---

### Forecast Step Embedding

For each step of the horizon, a step embedding is used:

- `nn.Embedding(horizon, step_emb_dim)`

This allows the decoder to distinguish between near and distant forecasts.

---

### Single Decoder Step

At each step `t`, the input is formed as:

- `[ctx, prev, step_embedding(t)]`

Then applied:
- Linear → GELU → Dropout → LayerNorm → Linear

The output is a single scalar forecast.

---

### Auto-regressive Loop

Forecasts are generated sequentially:
- the output of step `t` is used as input `prev` for step `t+1`,
- teacher forcing is supported with probability `teacher_forcing_prob`.

Model output:
- `[batch_size, horizon]`

---

## Weight Initialization

A unified approach is used:

- `Linear` — Xavier Uniform
- `LayerNorm` — weight = 1, bias = 0
- `Embedding` — Normal(mean=0, std=0.02)

---

## Key Hyperparameters

| Parameter | Description |
|--------|---------|
| n_features | Number of input features |
| horizon | Forecast horizon length |
| d_model | Dimensionality of internal representations |
| nhead | Number of self-attention heads |
| num_layers | Number of Transformer Encoder layers |
| dim_feedforward | Size of FFN in Encoder |
| dropout | Dropout in Encoder and Decoder |
| pos_type | Positional encoding type (`sin`, `learned`) |
| pool_type | Context pooling type |
| region_emb_size | Size of regional embedding |
| input_dropout | Input dropout |
| input_noise_std | Input noise standard deviation |
| teacher_forcing_prob | Probability of teacher forcing |
| grad_clip | Gradient clipping |

---

## Training

The model supports training with:

- Optimizers: Adam, AdamW, RMSProp
- Loss functions: MSE, MAE, Huber
- LR Scheduler: StepLR, CosineAnnealing, OneCycle
- Gradient clipping
- Mixed Precision (AMP)
- Early stopping
- Full Optuna integration

Validation is performed over the entire forecast horizon.

---

## Model Features

- Transformer effectively captures long-term dependencies.
- Separate AR-decoder is more stable and simpler than a classic Transformer Decoder.
- Regional embeddings improve spatial forecast quality.
- Step embeddings help differentiate near and far forecasts.
- Architecture is suitable for long-term economic forecasting.
- Scales flexibly to different dataset sizes and horizons.

---
