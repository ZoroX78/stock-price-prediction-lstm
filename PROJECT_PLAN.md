# Stock Price Movement Prediction with LSTMs — Build Plan & Roadmap

> **Project Goal:** Predict whether an S&P 500 stock will go **UP** or **DOWN** over the next 5 trading days, using 10 years of historical OHLCV data from Yahoo Finance, plus technical indicators. Sequence-to-one binary classification with proper financial-ML methodology.

---

## Table of Contents

1. [Hardware, Environment & Library Setup](#1-hardware-environment--library-setup)
2. [Recommended Project Directory Structure](#2-recommended-project-directory-structure)
3. [Build Order & Milestones](#3-build-order--milestones)
4. [Section 1 — Feature Engineering](#4-feature-engineering)
5. [Section 2 — Walk-Forward (Rolling) Validation](#5-walk-forward-rolling-validation)
6. [Section 3 — Model Comparison Experiment (LSTM vs GRU vs Transformer)](#6-model-comparison-experiment)
7. [Section 4 — Uncertainty Quantification (Monte Carlo Dropout)](#7-uncertainty-quantification-monte-carlo-dropout)
8. [Section 5 — Backtesting Framework](#8-backtesting-framework)
9. [Section 6 — Weights & Biases (W&B) Dashboard](#9-weights--biases-wb-dashboard)
10. [Methodological Pitfalls to Avoid](#10-methodological-pitfalls-to-avoid)
11. [Final Deliverables Checklist](#11-final-deliverables-checklist)

---

## 1. Hardware, Environment & Library Setup

### Hardware

| Component | Minimum | Recommended |
|---|---|---|
| GPU | None (CPU is fine for prototyping) | NVIDIA GPU with ≥ 6 GB VRAM (e.g., RTX 3060) or Google Colab Pro |
| RAM | 8 GB | 16 GB+ |
| Storage | 2 GB free | 10 GB free (for data, checkpoints, W&B logs) |

> [!TIP]
> LSTMs/GRUs on daily OHLCV data are small models. You can train on CPU in minutes. A GPU mainly helps for Transformer experiments and hyperparameter sweeps.

### Software & Libraries

| Library | Purpose | Version Guidance |
|---|---|---|
| **Python** | Runtime | 3.10 or 3.11 (avoid 3.13 for TA-Lib compatibility) |
| **PyTorch** | Model building, training | ≥ 2.0 (use `torch.compile` for speed) |
| **yfinance** | Download OHLCV data from Yahoo Finance | Latest |
| **TA-Lib** | Compute technical indicators (RSI, MACD, Bollinger Bands, etc.) | Install the C library first, then `pip install TA-Lib` |
| **pandas / numpy** | Data manipulation | Latest |
| **scikit-learn** | Scaling, utility metrics | Latest |
| **wandb** | Experiment tracking, sweeps, dashboards | Latest |
| **matplotlib / seaborn** | Local plots, EDA | Latest |
| **tqdm** | Progress bars | Latest |

### Environment Setup Steps

1. Create a **conda** or **venv** virtual environment dedicated to this project.
2. Install TA-Lib's underlying C library first (platform-specific — on Windows use the precompiled wheel from Christoph Gohlke's site or `conda install -c conda-forge ta-lib`; on Linux/macOS use `brew install ta-lib` or build from source).
3. Install PyTorch with the correct CUDA version for your GPU (or CPU-only if no GPU). Use the selector at [pytorch.org](https://pytorch.org/get-started/locally/).
4. `pip install yfinance pandas numpy scikit-learn wandb matplotlib seaborn tqdm`.
5. Run `wandb login` and authenticate with your free W&B account.
6. Verify everything imports without errors in a scratch script.

> [!IMPORTANT]
> Pin your library versions in a `requirements.txt` or `environment.yml` from day one. Reproducibility matters for a thesis project.

---

## 2. Recommended Project Directory Structure

```
stock-prediction/
├── README.md                       # Project overview, how to run
├── requirements.txt                # Pinned dependencies
├── configs/
│   ├── default.yaml                # Default hyperparameters
│   └── sweep.yaml                  # W&B sweep configuration
├── data/
│   ├── raw/                        # Raw OHLCV CSVs from yfinance
│   ├── processed/                  # Feature-engineered DataFrames (parquet)
│   └── splits/                     # Walk-forward split index files
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── download.py             # yfinance download script
│   │   ├── features.py             # Feature engineering (TA-Lib)
│   │   ├── labels.py               # Target label construction
│   │   ├── dataset.py              # PyTorch Dataset class
│   │   └── walk_forward.py         # Walk-forward split generator
│   ├── models/
│   │   ├── __init__.py
│   │   ├── lstm.py                 # LSTM classifier
│   │   ├── gru.py                  # GRU classifier
│   │   └── transformer.py          # Transformer classifier
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py              # Training loop (shared for all models)
│   │   ├── evaluate.py             # Evaluation metrics
│   │   └── mc_dropout.py           # Monte Carlo Dropout inference
│   ├── backtesting/
│   │   ├── __init__.py
│   │   ├── simulator.py            # Trade simulation engine
│   │   └── metrics.py              # Sharpe, drawdown, hit rate, etc.
│   └── utils/
│       ├── __init__.py
│       ├── config.py               # YAML config loader
│       └── wandb_utils.py          # W&B logging helpers
├── notebooks/
│   ├── 01_eda.ipynb                # Exploratory data analysis
│   ├── 02_feature_analysis.ipynb   # Feature correlation, distributions
│   └── 03_results_analysis.ipynb   # Final results, tables, plots
├── scripts/
│   ├── run_experiment.py           # Main entry point: train + evaluate
│   ├── run_sweep.py                # Launch W&B hyperparameter sweep
│   ├── run_backtest.py             # Run backtesting on saved predictions
│   └── run_mc_dropout.py           # MC Dropout uncertainty analysis
├── outputs/
│   ├── checkpoints/                # Saved model weights (.pt)
│   ├── predictions/                # Saved prediction CSVs per fold
│   └── figures/                    # Generated plots
└── tests/
    ├── test_features.py            # Unit tests for feature engineering
    ├── test_walk_forward.py        # Unit tests for split logic
    └── test_labels.py              # Unit tests for label construction
```

> [!TIP]
> Keep raw data separate from processed data. Never overwrite raw files. This makes bugs traceable.

---

## 3. Build Order & Milestones

Follow this order. Each milestone builds on the previous one. Don't skip ahead.

### Phase 1: Data Pipeline (Week 1–2)

| # | Task | Deliverable | Done? |
|---|---|---|---|
| 1.1 | Download 10 years of daily OHLCV data for 1–3 S&P 500 stocks using yfinance | `data/raw/*.csv` | ☐ |
| 1.2 | Exploratory Data Analysis (EDA): plot prices, volume, check for missing days, stock splits, dividends | `notebooks/01_eda.ipynb` | ☐ |
| 1.3 | Construct the **target label**: binary UP/DOWN based on price change over next 5 trading days | `src/data/labels.py` | ☐ |
| 1.4 | Implement all technical-indicator features with TA-Lib | `src/data/features.py` | ☐ |
| 1.5 | Analyze feature distributions, correlations, class balance | `notebooks/02_feature_analysis.ipynb` | ☐ |
| 1.6 | Build PyTorch `Dataset` that creates sliding-window sequences | `src/data/dataset.py` | ☐ |
| 1.7 | Implement walk-forward split generator | `src/data/walk_forward.py` | ☐ |
| 1.8 | Write unit tests for labels, features, and splits | `tests/` | ☐ |

### Phase 2: Baseline Model (Week 3–4)

| # | Task | Deliverable | Done? |
|---|---|---|---|
| 2.1 | Build the LSTM classifier in PyTorch | `src/models/lstm.py` | ☐ |
| 2.2 | Build the shared training loop with W&B logging | `src/training/trainer.py` | ☐ |
| 2.3 | Train LSTM across all walk-forward folds, log to W&B | W&B runs | ☐ |
| 2.4 | Implement evaluation metrics (accuracy, F1, AUC-ROC, precision, recall) | `src/training/evaluate.py` | ☐ |
| 2.5 | Sanity-check: does the model beat a random/majority-class baseline? | Documented comparison | ☐ |

### Phase 3: Model Comparison (Week 5–6)

| # | Task | Deliverable | Done? |
|---|---|---|---|
| 3.1 | Build GRU classifier | `src/models/gru.py` | ☐ |
| 3.2 | Build Transformer classifier | `src/models/transformer.py` | ☐ |
| 3.3 | Run all three models under identical walk-forward splits, log to W&B | W&B comparative dashboard | ☐ |
| 3.4 | Run W&B sweeps for hyperparameter tuning | `configs/sweep.yaml` | ☐ |

### Phase 4: Uncertainty & Backtesting (Week 7–8)

| # | Task | Deliverable | Done? |
|---|---|---|---|
| 4.1 | Implement Monte Carlo Dropout inference | `src/training/mc_dropout.py` | ☐ |
| 4.2 | Analyze uncertainty: do high-confidence predictions have better accuracy? | Plots + analysis | ☐ |
| 4.3 | Build backtesting simulator with transaction costs | `src/backtesting/simulator.py` | ☐ |
| 4.4 | Compute backtesting metrics (Sharpe, drawdown, hit rate, cumulative return) | `src/backtesting/metrics.py` | ☐ |
| 4.5 | Compare strategy returns vs buy-and-hold benchmark | Final backtest report | ☐ |

### Phase 5: Write-Up & Polish (Week 9–10)

| # | Task | Deliverable | Done? |
|---|---|---|---|
| 5.1 | Create final results notebook with all tables and figures | `notebooks/03_results_analysis.ipynb` | ☐ |
| 5.2 | Organize W&B dashboard for presentation | Shareable W&B link | ☐ |
| 5.3 | Write thesis / report | Document | ☐ |
| 5.4 | Clean up code, add docstrings, finalize README | Clean repo | ☐ |

---

## 4. Feature Engineering

### 4.1. The Target Variable (Label)

**What you're predicting:** For each trading day *t*, compute the percentage change in closing price from day *t* to day *t+5*:

```
pct_change = (Close[t+5] - Close[t]) / Close[t]
```

- If `pct_change > 0` → Label = **1 (UP)**
- If `pct_change ≤ 0` → Label = **0 (DOWN)**

**Why 5 days?** A 1-day horizon is extremely noisy and essentially random. 5 days smooths out daily noise while remaining short enough to be actionable. It also gives you enough label variation to train on.

> [!WARNING]
> The label uses **future** prices. These 5 future days must NEVER appear in your input features. When you build sliding windows, the window of input features must end at day *t*, and the label must be derived from days *t+1* through *t+5*. This is the single most common source of look-ahead bias.

### 4.2. Raw OHLCV Features

| Feature | Description | Why it helps |
|---|---|---|
| **Open** | Opening price of the day | Captures overnight gaps |
| **High** | Highest intraday price | Measures intraday bullish pressure |
| **Low** | Lowest intraday price | Measures intraday bearish pressure |
| **Close** | Closing price | The anchor for most indicators |
| **Volume** | Number of shares traded | Measures conviction behind price moves |

> [!IMPORTANT]
> **Never use raw prices as features.** Raw prices are non-stationary (they trend upward over decades). Instead, convert them to **returns** or **percentage changes**:
> - `daily_return = (Close[t] - Close[t-1]) / Close[t-1]`
> - Or use log returns: `log_return = ln(Close[t] / Close[t-1])`
>
> This makes the data stationary and scale-invariant across stocks and time periods.

### 4.3. Technical Indicators

Each indicator below captures a different aspect of market behavior. Implement them using TA-Lib.

#### A. Relative Strength Index (RSI) — Momentum

- **What it is:** Measures the speed and magnitude of recent price changes on a scale of 0–100. Computed as `100 - 100/(1 + RS)` where RS = average gain over *n* periods ÷ average loss over *n* periods.
- **Standard period:** 14 days.
- **Why it helps:** RSI identifies overbought (>70) and oversold (<30) conditions. Stocks at extreme RSI levels tend to mean-revert, which is a predictive signal. It transforms raw price momentum into a bounded, comparable metric.
- **Suggested variants:** Compute RSI at 7, 14, and 21-day periods to capture short, medium, and long-term momentum.

#### B. MACD (Moving Average Convergence/Divergence) — Trend

- **What it is:** The difference between a fast EMA (12-period) and a slow EMA (26-period) of the closing price. A "signal line" is a 9-period EMA of the MACD itself. The MACD histogram = MACD − signal line.
- **Why it helps:** MACD captures trend direction and trend strength. When the MACD crosses above the signal line, it indicates bullish momentum, and vice versa. The histogram's magnitude shows how strongly the trend is accelerating.
- **Features to extract (3 features):** MACD value, signal line value, histogram value.

#### C. Bollinger Bands — Volatility

- **What it is:** A 20-period simple moving average (SMA) with upper/lower bands at ±2 standard deviations.
- **Why it helps:** Prices touching the upper band suggest overextension; prices at the lower band suggest potential reversal. Band width measures volatility — narrow bands (a "squeeze") often precede large moves.
- **Features to extract (3 features):**
  - `bb_position = (Close - Lower Band) / (Upper Band - Lower Band)` — where the price sits within the bands (0 to 1).
  - `bb_width = (Upper Band - Lower Band) / Middle Band` — normalized band width (volatility measure).
  - `bb_signal` = whether price is above/below the middle band.

#### D. Volume Momentum

- **What it is:** Volume relative to its own recent history.
- **Why it helps:** A price move on high volume is more meaningful than one on low volume. Volume spikes often precede trend continuations or reversals.
- **Features to extract:**
  - `volume_sma_ratio = Volume[t] / SMA(Volume, 20)` — today's volume as a multiple of its 20-day average.
  - `volume_change = (Volume[t] - Volume[t-1]) / Volume[t-1]` — day-over-day volume change.
  - `OBV` (On-Balance Volume) — cumulative volume that adds volume on up days and subtracts on down days. Captures accumulation vs distribution.

#### E. Additional Recommended Indicators

| Indicator | Category | What & Why |
|---|---|---|
| **ATR (Average True Range, 14-day)** | Volatility | Measures average daily price range. Helps the model understand the "noise level" of a stock. Normalize by dividing by Close to make it scale-invariant (`ATR / Close`). |
| **Stochastic Oscillator (%K, %D)** | Momentum | Compares closing price to the high-low range over 14 days. Complements RSI with a different momentum perspective. Bounded 0–100. |
| **Rate of Change (ROC)** | Momentum | Simple percentage change over *n* days: `(Close[t] - Close[t-n]) / Close[t-n]`. Use multiple periods (5, 10, 20 days) for multi-scale momentum. |
| **Moving Average Slopes** | Trend | Compute 10-day and 50-day SMAs, then their slopes (e.g., `SMA[t] - SMA[t-5]`). Captures whether short and long trends are accelerating. |
| **Day of Week** | Calendar | One-hot encode or cyclical-encode (sin/cos) the day of the week. There are well-documented day-of-week effects in markets (e.g., "Monday effect"). |
| **Realized Volatility** | Volatility | Standard deviation of log returns over a rolling window (e.g., 20 days). Captures the regime: low-vol vs high-vol environments behave differently. |

### 4.4. Feature Preprocessing

1. **Handle NaN values:** Technical indicators require a "warm-up" period (e.g., RSI-14 needs 14 days). The first ~50 rows of your dataset will have NaNs. **Drop these rows** from the beginning of your dataset; do NOT fill them with zeros or forward-fill, as that introduces false signals.

2. **Normalization / Scaling:**
   - Use **per-window z-score normalization**: for each walk-forward training window, compute the mean and standard deviation of each feature on the training set only, then apply that same mean/std to normalize the corresponding test set.
   - **Never fit the scaler on the test set or on the full dataset.** This is a subtle but critical source of leakage.

3. **Feature count:** You'll end up with roughly **20–30 features**. This is a good range — enough to be expressive, not so many that you overfit on small data.

### 4.5. Avoiding Leakage in Feature Engineering

| Leakage Source | How It Happens | How to Prevent It |
|---|---|---|
| **Future prices in labels** | Label uses Close[t+5] but you accidentally include Close[t+1] as a feature | Labels and features must be computed in separate, audited functions. Unit-test that no feature at time *t* uses data beyond time *t*. |
| **Global normalization** | Computing mean/std of a feature across the entire 10-year dataset including test data | Always fit scaler on training window only, then transform test window with those parameters. |
| **Indicator warm-up** | An indicator like EMA implicitly depends on all prior data. If you compute EMA on the full dataset then split, the EMA in the test period "knows" about training data | This is generally acceptable for indicators (they only use past data). But document it and understand it. For extra rigor, recompute indicators within each walk-forward window. |
| **Target encoding / feature selection on full data** | Selecting the "best 10 features" using correlation with the target across the whole dataset | Feature selection must happen inside each walk-forward fold, using only training data. |

---

## 5. Walk-Forward (Rolling) Validation

### 5.1. Why Walk-Forward Is Critical

Standard cross-validation (k-fold) **randomly shuffles** data into folds. For time-series data, this is catastrophic because:

- **Temporal ordering matters:** Financial data has autocorrelation, regime changes, and trends. A random split lets the model see 2024 data in training and then "predict" 2022 data in the test fold — which is trivially easy because it's already seen what comes after.
- **Look-ahead bias:** Any random split allows future information to leak into training. The model learns patterns that only exist because it peeked at the future.
- **Unrealistic evaluation:** In production, you can only ever train on past data and predict forward. Your evaluation must mimic this.

Even a simple **single train/test split** (e.g., train on 2014–2021, test on 2022–2024) is better than k-fold but still problematic: it gives you only one evaluation point, which could be unrepresentative (e.g., 2022 was a bear market — your results would look very different than if your test set were a bull market).

**Walk-forward validation** solves both problems by repeatedly training on a window of past data and testing on the next block of future data, then sliding the window forward.

### 5.2. Concrete Rolling-Window Scheme

For 10 years of daily data (~2,520 trading days, roughly 252 trading days/year):

```
Total data: Jan 2014 – Dec 2024 (~2,520 trading days)

Scheme: Anchored Expanding Window (recommended for this project)

Fold 1:  Train [Jan 2014 – Dec 2019]  (6 years, ~1,512 days)
          Test  [Jan 2020 – Jun 2020]  (6 months, ~126 days)

Fold 2:  Train [Jan 2014 – Jun 2020]  (6.5 years)
          Test  [Jul 2020 – Dec 2020]  (6 months)

Fold 3:  Train [Jan 2014 – Dec 2020]  (7 years)
          Test  [Jan 2021 – Jun 2021]  (6 months)

   ...

Fold 9:  Train [Jan 2014 – Jun 2024]  (10.5 years)
          Test  [Jul 2024 – Dec 2024]  (6 months)
```

This gives you **~9 test folds**, each covering a 6-month out-of-sample window. The training window **expands** with each fold (you always train on all available past data), which is realistic — in production, you'd retrain periodically using all data you've accumulated.

**Alternative: Fixed Sliding Window.** Instead of expanding, keep the training window fixed (e.g., always the most recent 4 years). This is useful if you believe very old data is no longer relevant (market regime changes). Both approaches are valid; document your choice and rationale.

### 5.3. Implementation Details

```
Walk-Forward Split Generator — Pseudocode Outline

Input: full DataFrame with datetime index
Parameters:
  - initial_train_size:  1512 days (6 years)
  - test_size:           126 days (6 months)
  - step_size:           126 days (how far to slide the window)
  - expanding:           True (anchored) or False (sliding)

Output: list of (train_indices, test_indices) tuples

Logic:
  train_start = 0 (or sliding: current - initial_train_size)
  train_end = initial_train_size
  For each fold:
    train_indices = [train_start : train_end]
    test_indices  = [train_end : train_end + test_size]
    Yield (train_indices, test_indices)
    train_end += step_size
    If expanding: train_start stays at 0
    If sliding:   train_start += step_size
```

> [!IMPORTANT]
> **Gap between train and test:** Since your label looks 5 days into the future, you must leave a **gap of 5 trading days** between the end of the training set and the start of the test set. Otherwise, the last training samples' labels overlap with the first test samples' feature windows, causing leakage. Your split generator should enforce: `test_start = train_end + label_horizon`.

### 5.4. Per-Fold Training Protocol

For each walk-forward fold:
1. Fit the scaler (mean/std) on the training data only.
2. Transform both train and test data with that scaler.
3. Create sliding-window sequences (input length, e.g., 60 days) from the scaled data.
4. Train the model on the training sequences.
5. Evaluate on the test sequences.
6. Save predictions, metrics, and model checkpoint.
7. Move to the next fold.

Final reported metrics are the **average** (and standard deviation) across all folds.

---

## 6. Model Comparison Experiment

### 6.1. Architecture Specifications

All three models solve the same task: take a sequence of *T* feature vectors (each of dimension *F*) and output a single binary prediction (UP/DOWN).

#### A. LSTM Classifier

| Component | Details |
|---|---|
| **Input** | Sequence of shape `(batch, T, F)` where `T` = sequence length (e.g., 60), `F` = number of features |
| **LSTM layers** | 2 stacked LSTM layers, hidden size 128. Use `batch_first=True`. |
| **Dropout** | Apply dropout (p=0.3) between LSTM layers AND crucially inside the LSTM (`dropout` parameter). Also add a Dropout layer before the final classifier. |
| **Classifier head** | Take the **last hidden state** `h_T` → Linear(128, 64) → ReLU → Dropout(0.3) → Linear(64, 1) → Sigmoid |
| **Output** | Probability of UP (0 to 1). Threshold at 0.5 for classification. |

#### B. GRU Classifier

| Component | Details |
|---|---|
| **Architecture** | Identical to LSTM but replace `nn.LSTM` with `nn.GRU`. |
| **Why compare?** | GRUs have fewer parameters (no separate cell state) and often train faster. They may perform comparably or even better on smaller datasets. This lets you empirically test whether the LSTM's extra gating mechanism is worth the cost. |

#### C. Transformer Classifier

| Component | Details |
|---|---|
| **Input embedding** | Linear(F, d_model) where `d_model` = 128. This projects each timestep's features into the Transformer's internal dimension. |
| **Positional encoding** | Add **learnable** positional embeddings (not sinusoidal) of shape `(T, d_model)`. For sequences of fixed max length 60, this works well and is simpler. |
| **Transformer Encoder** | Use `nn.TransformerEncoder` with 2 layers, 4 attention heads, feedforward dim = 256, dropout = 0.3. |
| **Causal masking** | Apply a **causal (triangular) attention mask** so each position can only attend to itself and earlier positions. This prevents temporal leakage within the sequence — critical for time-series. Use `nn.Transformer.generate_square_subsequent_mask()`. |
| **Aggregation** | Take the output at the **last position** (analogous to LSTM's last hidden state) OR apply mean pooling across all positions. Try both; last-position is more common for causal models. |
| **Classifier head** | Same as LSTM: Linear → ReLU → Dropout → Linear → Sigmoid |

### 6.2. Shared Training Configuration

To ensure a fair comparison, **all three models must share:**

| Setting | Value | Rationale |
|---|---|---|
| **Sequence length** | 60 trading days (~3 months) | Long enough to capture medium-term patterns |
| **Batch size** | 64 | Standard; adjust if GPU memory is tight |
| **Optimizer** | AdamW | Better generalization than plain Adam |
| **Learning rate** | 1e-3 (with cosine or reduce-on-plateau scheduler) | Start here; sweep later |
| **Weight decay** | 1e-5 | Light regularization |
| **Loss function** | `BCEWithLogitsLoss` | Numerically stable binary cross-entropy. Remove the final Sigmoid from the model and let the loss handle it. |
| **Class weighting** | Apply `pos_weight` in the loss if classes are imbalanced | If 55% of labels are UP, set `pos_weight = n_neg / n_pos` |
| **Max epochs** | 100 with early stopping (patience = 10) | Prevents overfitting |
| **Early stopping** | Monitor validation loss (on a held-out validation chunk from the training window — e.g., last 20% of training data) | Stop training when validation loss hasn't improved for 10 epochs |
| **Random seeds** | Fix PyTorch, NumPy, and Python random seeds. Run each model with **3 different seeds** and report mean ± std. | Controls for lucky/unlucky initialization |

### 6.3. Metrics to Compare

| Metric | What it measures | Why it matters for finance |
|---|---|---|
| **Accuracy** | % of correct predictions | Simple but misleading if classes are imbalanced |
| **F1 Score** | Harmonic mean of precision and recall | Better than accuracy for imbalanced classes |
| **AUC-ROC** | Discriminative ability across all thresholds | Tells you if the model ranks UP days higher than DOWN days, regardless of threshold |
| **Precision (for UP class)** | Of all predicted UPs, how many were correct? | If you're buying on UP signals, precision = your hit rate |
| **Log Loss** | Quality of probability estimates | Are the predicted probabilities calibrated? A model that says "70% UP" should be right ~70% of the time. |
| **Training time** | Wall-clock time per fold | Practical consideration for retraining frequency |
| **Parameter count** | Total trainable parameters | Helps explain overfitting differences |

### 6.4. Running as W&B Experiments

Structure your experiments with consistent W&B organization:

- **W&B Project name:** `stock-prediction`
- **Group:** one group per walk-forward fold (e.g., `fold-1`, `fold-2`, ...) OR one group per model type (e.g., `lstm`, `gru`, `transformer`). Choose one and be consistent; grouping by model type is usually more useful for comparison.
- **Run name:** `{model_type}_fold{fold_id}_seed{seed}` (e.g., `lstm_fold3_seed42`)
- **Tags:** `["lstm", "fold-3", "seed-42", "baseline"]`

Log the following to every run's `wandb.config`:
- `model_type`, `hidden_size`, `num_layers`, `dropout`, `seq_length`
- `lr`, `batch_size`, `optimizer`, `weight_decay`
- `train_start_date`, `train_end_date`, `test_start_date`, `test_end_date`
- `stock_ticker`, `label_horizon` (5), `num_features`

---

## 7. Uncertainty Quantification (Monte Carlo Dropout)

### 7.1. The Concept

In standard inference, you set the model to `eval()` mode, which **disables dropout**. You get a single deterministic prediction.

**Monte Carlo (MC) Dropout** keeps dropout **enabled** at inference time and runs the same input through the model *N* times (e.g., N = 50–100). Each forward pass randomly drops different neurons, effectively sampling from a distribution of slightly different sub-networks. This approximates **Bayesian inference** — you're sampling from an approximate posterior over model weights.

The result: instead of one prediction per sample, you get *N* predictions. From these, you extract:

- **Mean prediction:** average of all *N* predicted probabilities → your best estimate.
- **Prediction variance (or standard deviation):** measures how much the predictions disagree → your **uncertainty** estimate.

### 7.2. Implementation Steps

1. **Ensure dropout is in your model:** You already have `nn.Dropout(p=0.3)` layers. Good.

2. **Create an MC Dropout inference function:**
   - Set the model to `train()` mode (this enables dropout), BUT call `torch.no_grad()` to skip gradient computation (you don't want to update weights).
   - For each input sample, run the model *N* times (e.g., `N = 100`).
   - Collect all *N* output probabilities.
   - Compute `mean_prob = mean(all N probabilities)` and `std_prob = std(all N probabilities)`.
   - The final prediction is `1 if mean_prob > 0.5 else 0`.
   - The uncertainty is `std_prob`.

3. **Practical tip:** Batch the *N* forward passes. Repeat each input *N* times along the batch dimension, run a single forward pass on the enlarged batch, then reshape and compute statistics. This is much faster than *N* sequential forward passes.

### 7.3. Using Uncertainty for Trading

This is where MC Dropout becomes genuinely useful for your project:

- **Confidence filter:** Only trade when `std_prob < threshold` (e.g., 0.10). High uncertainty means the model is "unsure" — skip those days.
- **Position sizing:** Allocate more capital to high-confidence predictions and less to uncertain ones. For example: `position_size = base_size * (1 - std_prob / max_std)`.
- **Calibration analysis:** Plot a reliability diagram — bin predictions by `mean_prob` and check if the actual UP rate matches. Well-calibrated models are more trustworthy.

**Analysis to include in your thesis:**

1. Compute accuracy **separately** for high-confidence predictions (low uncertainty) vs low-confidence ones. If MC Dropout is working, high-confidence predictions should have notably higher accuracy.
2. Compare backtesting returns when:
   - (a) you trade on ALL signals vs
   - (b) you only trade on signals where uncertainty < threshold.
   Strategy (b) should have a higher Sharpe ratio even if it trades less frequently.

### 7.4. Important Caveats

- MC Dropout is an *approximation* of Bayesian inference — it's not exact. State this in your thesis.
- The dropout rate matters: too low → all *N* passes are nearly identical → uncertainty is artificially low; too high → too much noise. Your training dropout of 0.3 is a reasonable starting point.
- MC Dropout captures **model uncertainty** (epistemic), not **data noise** (aleatoric). Markets are inherently noisy; even a "certain" prediction can be wrong because of irreducible randomness.

---

## 8. Backtesting Framework

### 8.1. Overview

Backtesting simulates how a trading strategy **would have** performed historically. You take the model's predictions on the walk-forward test sets and simulate trading with them.

### 8.2. Signal Generation

Convert model outputs into trading signals:

| Signal | Condition | Action |
|---|---|---|
| **BUY** | `predicted_prob > 0.5 + threshold` (e.g., > 0.6) | Enter a long position (buy the stock) |
| **SELL / CLOSE** | `predicted_prob < 0.5 - threshold` (e.g., < 0.4) | Close any open long position (or go short if you model that) |
| **HOLD** | `0.4 ≤ predicted_prob ≤ 0.6` | Do nothing; maintain current position |

The threshold (0.1 above) creates a "dead zone" that filters out low-conviction signals. This is sensible because predictions near 0.5 are essentially coin flips.

Optionally, integrate MC Dropout uncertainty: only execute BUY/SELL signals when `uncertainty < max_uncertainty_threshold`.

### 8.3. Simulation Rules

Design your simulator with these rules:

1. **Starting capital:** $100,000 (or any round number — it just sets the scale).
2. **Position sizing:** Invest a fixed fraction of current portfolio value per trade (e.g., 100% for simplicity, or 10% for diversified). For a single-stock model, 100% is common.
3. **Execution price:** Assume you trade at the **next day's open** after the signal is generated (you can't trade at today's close because the signal is generated after the close). This is critical for realism.
4. **Commission:** $0 if simulating commission-free brokers (Robinhood, Interactive Brokers Lite), or ~$5–$10 per trade for traditional brokers. Document your assumption.
5. **Slippage:** The difference between expected price and actual execution price. Model as 0.05%–0.1% of the trade value per trade. This accounts for market impact, spread, and delay.
6. **No short selling** (simplify to long-only unless you want the complexity).
7. **No margin / leverage.**
8. **Positions are binary:** you're either fully invested or fully in cash. (Or you can implement fractional — but start simple.)

### 8.4. The Simulation Loop

```
Pseudocode for the backtesting loop:

Initialize: cash = $100,000, position = 0 shares, portfolio_values = []

For each day t in the test period:
  1. Record portfolio value: cash + position * Open[t]
  2. Get signal for day t (generated from yesterday's Close features)
  3. If signal == BUY and position == 0:
       shares = (cash * (1 - slippage)) / Open[t]
       cash -= shares * Open[t] + commission
       position = shares
  4. If signal == SELL and position > 0:
       cash += position * Open[t] * (1 - slippage) - commission
       position = 0
  5. Append portfolio value to list

Final portfolio value = cash + position * Close[last_day]
```

### 8.5. Backtesting Metrics

| Metric | Formula / Description | What "good" looks like |
|---|---|---|
| **Cumulative Return** | `(final_value - initial_value) / initial_value * 100%` | Positive; higher than buy-and-hold |
| **Annualized Return** | `(1 + cumulative_return)^(252/n_days) - 1` | Contextualizes returns for different test periods |
| **Sharpe Ratio** | `mean(daily_returns) / std(daily_returns) * sqrt(252)` | > 1.0 is decent; > 2.0 is excellent; < 0 means you lost money |
| **Max Drawdown** | Largest peak-to-trough decline in portfolio value | Smaller is better; > 30% is painful |
| **Hit Rate** | `# profitable trades / # total trades` | > 50% is necessary; > 55% is good for daily |
| **Profit Factor** | `sum(gains from winning trades) / sum(losses from losing trades)` | > 1.0 means profitable overall; > 1.5 is solid |
| **Number of Trades** | Count of round-trip buy→sell pairs | Too few = not enough data to be statistical; too many = overtrading |
| **Buy-and-Hold Benchmark** | What you'd earn just buying on day 1 and holding | Your strategy must beat this to justify its complexity |

> [!CAUTION]
> **If your strategy underperforms buy-and-hold, that's a valid and publishable result.** Most ML-based trading strategies underperform simple benchmarks after costs. Honest reporting of this is valued in academic work — it shows you understand the difficulty of the problem.

### 8.6. Visualization

Produce the following plots for your thesis:
1. **Equity curve:** portfolio value over time for your strategy vs buy-and-hold, on the same axes.
2. **Drawdown chart:** drawdown percentage over time.
3. **Trade markers:** overlay buy/sell points on the stock's price chart.
4. **Monthly/quarterly return heatmap:** color-coded table of returns by month and year.

---

## 9. Weights & Biases (W&B) Dashboard

### 9.1. What to Log

#### A. Configuration (logged once per run via `wandb.config`)

Log everything needed to reproduce the run:

```
wandb.config items:
  model_type: "lstm" | "gru" | "transformer"
  hidden_size: 128
  num_layers: 2
  dropout: 0.3
  d_model: 128          # Transformer only
  num_heads: 4          # Transformer only
  seq_length: 60
  batch_size: 64
  lr: 0.001
  optimizer: "adamw"
  weight_decay: 1e-5
  scheduler: "cosine"
  epochs: 100
  early_stop_patience: 10
  label_horizon: 5
  num_features: 25
  stock_ticker: "AAPL"
  fold_id: 3
  seed: 42
  train_start: "2014-01-02"
  train_end: "2021-06-30"
  test_start: "2021-07-06"
  test_end: "2021-12-31"
```

#### B. Training Metrics (logged per epoch via `wandb.log`)

| Metric | Frequency |
|---|---|
| `train/loss` | Every epoch |
| `train/accuracy` | Every epoch |
| `val/loss` | Every epoch (validation = last 20% of training window) |
| `val/accuracy` | Every epoch |
| `val/auc_roc` | Every epoch |
| `val/f1` | Every epoch |
| `learning_rate` | Every epoch |
| `epoch` | Every epoch |

#### C. Test Metrics (logged once at end of fold)

| Metric | Description |
|---|---|
| `test/accuracy` | Final test accuracy for this fold |
| `test/f1` | Final test F1 |
| `test/auc_roc` | Final test AUC-ROC |
| `test/precision` | Final test precision |
| `test/recall` | Final test recall |
| `test/log_loss` | Calibration quality |

#### D. Backtesting Metrics (logged once per fold)

| Metric | Description |
|---|---|
| `backtest/cumulative_return` | Strategy total return |
| `backtest/sharpe_ratio` | Risk-adjusted return |
| `backtest/max_drawdown` | Worst peak-to-trough loss |
| `backtest/hit_rate` | Trade win rate |
| `backtest/num_trades` | Total trades executed |
| `backtest/buy_hold_return` | Benchmark return |

#### E. Tables and Artifacts

- **Prediction table:** Log a `wandb.Table` with columns `[date, actual_label, predicted_prob, predicted_label, uncertainty]` for each test fold. This lets you inspect individual predictions in the W&B UI.
- **Confusion matrix:** Log via `wandb.plot.confusion_matrix()`.
- **ROC curve:** Log via `wandb.plot.roc_curve()`.
- **Model checkpoints:** Save `.pt` files as `wandb.Artifact` for reproducibility.

#### F. Sweep Configuration

For hyperparameter tuning, create a sweep config (YAML):

```yaml
# configs/sweep.yaml
program: scripts/run_experiment.py
method: bayes        # Bayesian optimization (better than random/grid for small budgets)
metric:
  name: test/auc_roc
  goal: maximize
parameters:
  hidden_size:
    values: [64, 128, 256]
  num_layers:
    values: [1, 2, 3]
  dropout:
    min: 0.1
    max: 0.5
  lr:
    min: 0.0001
    max: 0.01
    distribution: log_uniform_values
  seq_length:
    values: [30, 60, 90]
  batch_size:
    values: [32, 64, 128]
```

Run the sweep with `wandb sweep configs/sweep.yaml`, then launch agents with `wandb agent <sweep_id>`.

### 9.2. Dashboard Organization

Set up your W&B project as follows:

| Element | Configuration |
|---|---|
| **Project** | `stock-prediction` |
| **Groups** | Group runs by `model_type` (lstm, gru, transformer). This lets you compare models side-by-side. |
| **Tags** | Use tags for `fold-{id}`, `seed-{value}`, `sweep`, `baseline`, `mc-dropout`, `final` |
| **Workspace panels** | Create panels for: (1) training loss curves by model, (2) test metrics comparison bar chart, (3) backtesting metrics table, (4) sweep parallel coordinates plot |
| **Reports** | Create a W&B Report summarizing final results — this can be shared with your supervisor or linked from your thesis |

---

## 10. Methodological Pitfalls to Avoid

These are the most common mistakes that will invalidate your results. Treat this section as a pre-flight checklist.

### Pitfall 1: Look-Ahead Bias / Data Leakage

| What can go wrong | How to prevent it |
|---|---|
| Using future data in features (e.g., normalized with future mean/std) | **Scaler is fit on training data only**, then applied to test data. Never call `fit_transform` on the full dataset. |
| Label leakage (features computed from the same window as the label) | Enforce a **gap of `label_horizon` days** between feature window end and the first label day. |
| Feature selection on full data | All feature selection / correlation analysis must happen inside each walk-forward fold, using only training data. |
| Inadvertently using `Close[t+1]` or later in a feature at time *t* | **Unit test** your feature pipeline: for a known date, manually verify that no feature uses data from after that date. |

### Pitfall 2: Shuffling Time-Series Data

- **Never use `shuffle=True` in your DataLoader for time-series.** Shuffling destroys temporal order.
- Exception: within a **training** set, shuffling the already-constructed sliding windows is acceptable (each window is self-contained). But **never shuffle across train/test boundaries**.

### Pitfall 3: Survivorship Bias

- The S&P 500 today contains stocks that have survived and thrived. Stocks that went bankrupt, were delisted, or were removed from the index are absent from today's constituent list.
- If you only test on today's S&P 500 members and apply 10 years of history, you're implicitly selecting for winners.
- **Mitigation for a student project:** Acknowledge this limitation in your thesis. If you want to go further, use historical constituent lists (available from some data providers) or test on the full index (SPY ETF) instead of individual stocks.

### Pitfall 4: Overfitting to Noise

- Daily stock returns are approximately 50/50 UP/DOWN. Any accuracy above ~55% on out-of-sample data is actually impressive. If you're seeing 70%+ on test data, you have a bug (almost certainly leakage).
- Use early stopping, dropout, and weight decay.
- Report results across **multiple folds** and **multiple seeds** to show robustness.
- Compare against baselines: (1) always predict UP, (2) random 50/50 classifier, (3) previous-day direction. Your model should beat all three.

### Pitfall 5: Overfitting to the Validation Metric via Hyperparameter Tuning

- If you tune 50 hyperparameter combinations and pick the best one, you've selected for the lucky configuration that happened to work on your test data.
- **Mitigation:** Use a separate validation set within the training window for early stopping and hyperparameter selection. Report final metrics on the test set, which was not used for any decision.
- Alternatively, report results for a small number of well-motivated hyperparameter choices rather than an exhaustive search.

### Pitfall 6: Ignoring Transaction Costs

- A strategy that generates 5% annual return but executes 500 trades/year might actually lose money after commissions and slippage.
- **Always** report backtesting results with and without costs, so the reader can see the impact.

### Pitfall 7: P-Hacking / Selective Reporting

- Don't try 20 different stock tickers, 10 different feature sets, and 5 different label horizons, then report only the one combination that worked.
- Pre-register your experimental design (model types, stocks, features, label horizon) and report all results, including negative ones.

---

## 11. Final Deliverables Checklist

Use this checklist to ensure your project is complete before submission.

### Code & Data
- [ ] Clean, documented Python codebase with the directory structure above
- [ ] `requirements.txt` or `environment.yml` with pinned versions
- [ ] README with setup instructions, how to reproduce results
- [ ] Raw data download script (not the raw data itself — it can be re-downloaded)
- [ ] Unit tests for feature engineering, labels, and walk-forward splits

### Experiments
- [ ] LSTM, GRU, and Transformer trained across all walk-forward folds
- [ ] Each model run with 3 random seeds
- [ ] W&B dashboard with all runs, organized by model group
- [ ] Hyperparameter sweep results (at least for the best-performing model)
- [ ] MC Dropout uncertainty analysis with confidence-accuracy plots

### Results & Analysis
- [ ] Summary table: model × metric (accuracy, F1, AUC-ROC, Sharpe, drawdown)
- [ ] Statistical significance: mean ± std across folds and seeds
- [ ] Equity curves for strategy vs buy-and-hold
- [ ] Uncertainty-filtered backtest comparison
- [ ] Discussion of what worked, what didn't, and why

### Thesis / Report
- [ ] Literature review of ML for stock prediction
- [ ] Methodology section covering all 6 main sections above
- [ ] Results section with tables and figures
- [ ] Discussion of limitations (survivorship bias, market efficiency, noise)
- [ ] Conclusion with honest assessment of model utility

> [!NOTE]
> **A reminder on expectations:** Predicting stock prices is an extremely hard problem. The Efficient Market Hypothesis suggests that publicly available information (like historical OHLCV and technical indicators) is already priced in. An honest thesis that demonstrates proper methodology, thorough experimentation, and insightful analysis of *why* the models struggle is more valuable than one that claims unrealistic performance. Focus on the **process and methodology** — that's what your examiners will evaluate.
