# Stock Price Distribution Prediction with Diffusion Models

## Project Plan v1.0

---

## 1. Project Overview

### 1.1 Objective

Develop a Diffusion Model to predict the **price distribution** of stocks 30 seconds into the future, given a historical price sequence.

### 1.2 Problem Formulation

- **Input**: A sequence of stock prices for the past N seconds (e.g., N=60, 120, or 300)
- **Output**: The probability distribution of the stock price at T+30 seconds
- **Key Insight**: We predict a *distribution*, not a single point estimate, which captures market uncertainty

### 1.3 Data Summary

| Metric | Value |
|--------|-------|
| Data Granularity | 1 second |
| Trading Days | 40 days (2025-09-24 to 2025-11-18) |
| Symbols per Day | ~54 stocks |
| Fields | symbol, bob, eob, close, high, low, volume, amount |
| Records per Symbol/Day | ~31,000 (includes pre/post market) |

---

## 2. Technical Approach

### 2.1 Why Diffusion Models?

Diffusion models are ideal for this task because:

1. **Distribution Modeling**: They naturally model complex probability distributions
2. **Uncertainty Quantification**: Generate multiple samples to estimate confidence intervals
3. **Conditional Generation**: Can condition on historical price sequences
4. **High-Quality Outputs**: State-of-the-art generative performance

### 2.2 Model Architecture: Conditional Denoising Diffusion

```
                    ┌─────────────────────────────┐
                    │   Historical Price Sequence  │
                    │      (Past N seconds)        │
                    └──────────────┬──────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────┐
                    │    Condition Encoder         │
                    │  (Transformer / 1D-CNN)      │
                    └──────────────┬──────────────┘
                                   │
                                   ▼
┌──────────────┐    ┌─────────────────────────────┐
│  Noise z_T   │───▶│   Conditional U-Net          │
│  (Gaussian)  │    │   Denoising Network          │
└──────────────┘    └──────────────┬──────────────┘
                                   │
                                   │ Iterative Denoising
                                   ▼
                    ┌─────────────────────────────┐
                    │   Predicted Price at T+30   │
                    │      (Distribution)          │
                    └─────────────────────────────┘
```

### 2.3 Two-Stage Approach

**Stage 1: Price Return Prediction**
- Predict the *return* (percentage change) rather than absolute price
- More stable across different price levels
- Target: `r = (P_{t+30} - P_t) / P_t`

**Stage 2: Distribution Sampling**
- Run inference multiple times (e.g., 100-1000 samples)
- Aggregate samples to form empirical distribution
- Extract statistics: mean, std, percentiles, VaR

---

## 3. Data Processing Pipeline

### 3.1 Data Preprocessing Steps

```
Raw CSV Data
     │
     ▼
┌────────────────────────────────────────┐
│ 1. Load & Parse                        │
│    - Read all CSV files                │
│    - Parse timestamps to datetime      │
│    - Handle timezone (UTC)             │
└────────────────────────────────────────┘
     │
     ▼
┌────────────────────────────────────────┐
│ 2. Resample to Regular Intervals       │
│    - Forward fill missing seconds      │
│    - Handle gaps (mark as invalid)     │
│    - Focus on market hours (9:30-16:00)│
└────────────────────────────────────────┘
     │
     ▼
┌────────────────────────────────────────┐
│ 3. Feature Engineering                 │
│    - Log returns                       │
│    - Moving averages                   │
│    - Volatility features               │
│    - Volume normalization              │
└────────────────────────────────────────┘
     │
     ▼
┌────────────────────────────────────────┐
│ 4. Create Training Samples             │
│    - Sliding window approach           │
│    - Input: [t-N, t] sequence          │
│    - Target: price at t+30             │
└────────────────────────────────────────┘
     │
     ▼
┌────────────────────────────────────────┐
│ 5. Train/Val/Test Split                │
│    - By date (temporal split)          │
│    - Train: first 30 days              │
│    - Val: next 5 days                  │
│    - Test: last 5 days                 │
└────────────────────────────────────────┘
```

### 3.2 Sample Statistics Estimation

| Metric | Estimate |
|--------|----------|
| Samples per symbol per day | ~23,400 (regular market) |
| Total samples per day | ~1,260,000 (54 symbols) |
| Total training samples | ~38M (30 days) |

---

## 4. Model Implementation Details

### 4.1 Condition Encoder

**Option A: Transformer Encoder**
```python
class ConditionEncoder(nn.Module):
    def __init__(self, seq_len, d_model=128, nhead=4, num_layers=3):
        # Positional encoding
        # Multi-head self-attention layers
        # Output: condition embedding [batch, d_model]
```

**Option B: 1D CNN + LSTM**
```python
class ConditionEncoder(nn.Module):
    def __init__(self, seq_len, hidden_dim=128):
        # Conv1d layers for local patterns
        # LSTM for sequential dependencies
        # Output: condition embedding [batch, hidden_dim]
```

### 4.2 Diffusion Model

**Denoising Network Architecture**:
```python
class DenoisingNetwork(nn.Module):
    def __init__(self, cond_dim, time_dim=64):
        # Time embedding (sinusoidal)
        # Condition integration (cross-attention or FiLM)
        # MLP backbone or small U-Net
        # Output: predicted noise
```

**Training Objective**:
- Standard denoising score matching
- L2 loss between predicted and actual noise
- Optional: velocity prediction formulation

### 4.3 Key Hyperparameters

| Parameter | Suggested Value |
|-----------|-----------------|
| Input sequence length (N) | 120 seconds |
| Prediction horizon | 30 seconds |
| Diffusion timesteps (T) | 1000 |
| Condition embedding dim | 128 |
| Batch size | 256-512 |
| Learning rate | 1e-4 |
| Noise schedule | Cosine |

---

## 5. Project Directory Structure

```
diffusion_stock_pred/
├── data_1s/                    # Raw data (existing)
│   ├── 20250924/
│   │   ├── AMD.csv
│   │   └── ...
│   └── ...
│
├── src/                        # Source code
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py          # PyTorch Dataset
│   │   ├── preprocessing.py    # Data cleaning & feature eng
│   │   └── dataloader.py       # DataLoader utilities
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── condition_encoder.py    # Sequence encoder
│   │   ├── diffusion.py            # Diffusion process
│   │   ├── denoising_net.py        # Denoising network
│   │   └── model.py                # Full model wrapper
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py          # Training loop
│   │   ├── losses.py           # Loss functions
│   │   └── scheduler.py        # LR scheduler
│   │
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── sampler.py          # Diffusion sampling
│   │   └── distribution.py     # Distribution analysis
│   │
│   └── utils/
│       ├── __init__.py
│       ├── config.py           # Configuration management
│       ├── metrics.py          # Evaluation metrics
│       └── visualization.py    # Plotting utilities
│
├── configs/                    # Configuration files
│   ├── default.yaml
│   ├── experiment_1.yaml
│   └── ...
│
├── scripts/                    # Executable scripts
│   ├── preprocess_data.py
│   ├── train.py
│   ├── evaluate.py
│   └── inference.py
│
├── notebooks/                  # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_analysis.ipynb
│   ├── 03_model_experiments.ipynb
│   └── 04_results_visualization.ipynb
│
├── tests/                      # Unit tests
│   ├── test_data.py
│   ├── test_models.py
│   └── ...
│
├── outputs/                    # Training outputs
│   ├── checkpoints/
│   ├── logs/
│   └── predictions/
│
├── requirements.txt
├── setup.py
├── PROJECT_PLAN.md            # This file
└── README.md
```

---

## 6. Implementation Milestones

### Phase 1: Data Infrastructure

- [ ] **1.1** Data loading utilities
  - CSV parsing and validation
  - Handle multiple symbols and dates

- [ ] **1.2** Data preprocessing pipeline
  - Resample to regular 1-second intervals
  - Handle missing data and gaps
  - Filter to market hours

- [ ] **1.3** Feature engineering
  - Compute log returns
  - Add technical indicators
  - Normalize features

- [ ] **1.4** Dataset and DataLoader
  - Implement PyTorch Dataset
  - Sliding window sample generation
  - Train/val/test split by date

### Phase 2: Model Development

- [ ] **2.1** Condition encoder
  - Implement Transformer-based encoder
  - Test with simple LSTM baseline

- [ ] **2.2** Diffusion model core
  - Forward diffusion process
  - Noise scheduling (linear/cosine)
  - Denoising network architecture

- [ ] **2.3** Conditional generation
  - Integrate condition into denoising
  - Implement classifier-free guidance (optional)

### Phase 3: Training Pipeline

- [ ] **3.1** Training infrastructure
  - Training loop with logging
  - Checkpoint management
  - Learning rate scheduling

- [ ] **3.2** Experiment tracking
  - Integrate with Weights & Biases or TensorBoard
  - Log metrics and visualizations

- [ ] **3.3** Hyperparameter tuning
  - Sequence length experiments
  - Model size experiments
  - Diffusion timestep experiments

### Phase 4: Evaluation & Inference

- [ ] **4.1** Sampling pipeline
  - DDPM sampling
  - DDIM accelerated sampling (optional)
  - Batch sampling for distribution estimation

- [ ] **4.2** Evaluation metrics
  - Distribution metrics (KL divergence, Wasserstein)
  - Point prediction metrics (MAE, RMSE)
  - Calibration metrics

- [ ] **4.3** Visualization
  - Predicted vs actual distributions
  - Sample trajectories
  - Error analysis by market condition

### Phase 5: Optimization & Production

- [ ] **5.1** Performance optimization
  - Mixed precision training
  - Gradient accumulation
  - Data loading optimization

- [ ] **5.2** Model compression (if needed)
  - Knowledge distillation
  - Fewer diffusion steps

- [ ] **5.3** Inference API
  - Real-time prediction interface
  - Batched prediction service

---

## 7. Technology Stack

| Component | Technology |
|-----------|------------|
| Deep Learning Framework | PyTorch 2.0+ |
| Data Processing | Pandas, NumPy |
| Configuration | Hydra / OmegaConf |
| Experiment Tracking | Weights & Biases |
| Visualization | Matplotlib, Seaborn, Plotly |
| Testing | pytest |
| Code Quality | black, isort, mypy |

### 7.1 Key Dependencies

```txt
torch>=2.0.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
hydra-core>=1.3.0
wandb>=0.15.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
pytest>=7.3.0
```

---

## 8. Evaluation Strategy

### 8.1 Distribution Metrics

1. **Continuous Ranked Probability Score (CRPS)**
   - Measures how well predicted distribution matches observed value
   - Lower is better

2. **Calibration**
   - Check if X% prediction intervals contain X% of actual values
   - Plot reliability diagram

3. **Sharpness**
   - Width of prediction intervals
   - Narrower is better (while maintaining calibration)

### 8.2 Point Prediction Metrics

1. **Mean Absolute Error (MAE)**
2. **Root Mean Squared Error (RMSE)**
3. **Directional Accuracy**
   - Percentage of correct up/down predictions

### 8.3 Financial Metrics

1. **Profit & Loss in Simulation**
   - Backtest simple trading strategies
2. **Value at Risk (VaR) Accuracy**
   - How well does the model predict tail risk?

---

## 9. Risk Analysis & Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Data quality issues | Medium | High | Thorough EDA, robust preprocessing |
| Model overfitting | High | High | Temporal CV, regularization, early stopping |
| Computational cost | Medium | Medium | Mixed precision, gradient checkpointing |
| Distribution shift | High | High | Rolling window training, domain adaptation |
| Slow inference | Medium | Medium | DDIM sampling, model distillation |

---

## 10. Success Criteria

### Minimum Viable Product (MVP)
- [ ] Model successfully generates price distributions
- [ ] CRPS better than naive baseline (historical volatility)
- [ ] Calibrated predictions (within 10% deviation)

### Target Performance
- [ ] Directional accuracy > 52%
- [ ] Sharpe ratio > 1.0 in simulated trading
- [ ] Inference time < 100ms for single prediction

---

## 11. Next Steps

1. **Immediate**: Set up project structure and install dependencies
2. **Short-term**: Complete data exploration and preprocessing pipeline
3. **Medium-term**: Implement and train baseline diffusion model
4. **Long-term**: Iterate on architecture and hyperparameters

---

## Appendix A: Alternative Approaches to Consider

### A.1 Score-Based Models
- Noise Conditional Score Networks (NCSN)
- May provide better gradient estimation

### A.2 Flow-Based Models
- Normalizing Flows as alternative generative approach
- Exact likelihood computation

### A.3 Hybrid Approaches
- Combine diffusion with autoregressive components
- Use diffusion for residual/volatility modeling

### A.4 Multi-Step Prediction
- Predict entire future trajectory, not just T+30
- Useful for path-dependent applications

---

## Appendix B: Data Field Descriptions

| Field | Description |
|-------|-------------|
| symbol | Stock ticker symbol |
| bob | Bar open begin timestamp (UTC) |
| eob | Bar open end timestamp (UTC) |
| close | Closing price of the 1-second bar |
| high | Highest price during the bar |
| low | Lowest price during the bar |
| volume | Number of shares traded |
| amount | Total dollar amount traded |

---

*Document created: 2026-01-02*
*Last updated: 2026-01-02*
