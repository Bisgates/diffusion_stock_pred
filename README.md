# Stock Price Distribution Prediction with Diffusion Models

A PyTorch implementation of conditional diffusion models for predicting stock price distributions.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Quick test (limited data, 3 epochs)
python scripts/train.py --quick

# Full training on MacBook
python scripts/train.py --config configs/default.yaml

# Evaluate trained model
python scripts/evaluate.py --checkpoint outputs/checkpoints/best.pt
```

## Project Structure

```
diffusion_stock_pred/
├── src/
│   ├── data/           # Data loading and preprocessing
│   ├── models/         # Diffusion model components
│   ├── training/       # Training infrastructure
│   ├── inference/      # Sampling and prediction
│   └── utils/          # Config and metrics
├── configs/            # YAML configurations
├── scripts/            # Training and evaluation scripts
├── data_1s/            # Raw 1-second bar data
└── outputs/            # Checkpoints and logs
```

## Model Architecture

- **Condition Encoder**: Lightweight Transformer (64d, 2 layers)
- **Diffusion Process**: 500 steps, cosine schedule
- **Denoising Network**: Conditional MLP with FiLM (128d, 4 blocks)
- **Total Parameters**: ~163K-520K (MacBook friendly)

## Configuration

Edit `configs/default.yaml` to customize:

```yaml
model:
  encoder_dim: 64
  diffusion_steps: 500
  denoising_hidden_dim: 128

training:
  epochs: 50
  batch_size: 128
  learning_rate: 0.0001
  device: mps  # Use 'cuda' for GPU
```

## Data Format

CSV files with columns: `symbol, bob, eob, close, high, low, volume, amount`

- `bob`: Bar open begin timestamp (UTC)
- Market hours filter: 13:30-20:00 UTC (9:30-16:00 ET)

## Evaluation Metrics

- **CRPS**: Continuous Ranked Probability Score
- **Calibration**: Coverage accuracy at different quantiles
- **Direction Accuracy**: Correct up/down predictions
- **MAE/RMSE**: Point prediction errors
