# Stock Price Movement Prediction with LSTMs

**College project** — predict whether an S&P 500 stock goes **UP or DOWN** over the
next 5 trading days, using 10 years of Yahoo Finance (yfinance) OHLCV data plus
technical indicators.

This is a **sequence-to-one binary classification** problem built with proper
financial-ML methodology.

## Stack
- **PyTorch** — LSTM, GRU, Transformer models
- **TA-Lib** — technical indicators (RSI, MACD, Bollinger Bands, volume momentum)
- **Weights & Biases (W&B)** — experiment tracking & dashboard
- **Walk-forward (rolling) validation** — to avoid look-ahead bias
- **Monte Carlo Dropout** — uncertainty quantification
- **Backtesting** — simulate returns from model signals

## Deliverables
- [ ] Feature engineering: RSI, MACD, Bollinger Bands, volume momentum
- [ ] Walk-forward validation (prevents look-ahead bias)
- [ ] LSTM vs GRU vs Transformer comparison experiment
- [ ] Uncertainty quantification via Monte Carlo Dropout
- [ ] Backtesting framework (returns if you followed the signals)
- [ ] W&B dashboard of all experiments

## Full plan
See **`PROJECT_PLAN.md`** — a detailed, code-free build roadmap covering every
deliverable, the directory structure, build milestones, and the methodological
pitfalls to avoid. No implementation code is included; build it yourself as coursework.

## Quick start (UV)
This project uses **UV** for dependency + environment management.
```bash
# 1. Install UV (if needed)
pip install uv

# 2. Create a venv + install deps (reads pyproject.toml)
uv venv
uv sync                 # installs deps; `uv sync --extra dev` adds pytest

# 3. Install TA-Lib's C library ONCE (required before the `TA-Lib` wheel):
#    Windows:  conda install -c conda-forge ta-lib
#              (or use the prebuilt wheel from Gohlke)
#    macOS:    brew install ta-lib
#    Linux:    apt-get install ta-lib   # or build from source

# 4. Run anything inside the env:
uv run python -m src.data.download
uv run wandb login
```

## Project layout
```
configs/      default + sweep configs
data/         raw (yfinance), processed, walk-forward splits
src/          data, models, training, backtesting, utils
notebooks/    EDA, feature analysis, results
scripts/      run_experiment, run_sweep, run_backtest, run_mc_dropout
outputs/      checkpoints, predictions, figures
tests/        unit tests for features, labels, splits
```
Repo: https://github.com/ZoroX78/stock-price-prediction-lstm
