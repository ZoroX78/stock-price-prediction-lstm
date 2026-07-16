from pathlib import Path
import shutil

import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, QuantileLoss
import pandas as pd

# Full float32 matmul precision (no TF32). Prefer numerical fidelity over Ampere matmul speed.
torch.set_float32_matmul_precision("highest")

CHECKPOINT_DIR = Path("outputs/checkpoints")
BEST_CKPT_NAME = "tft_best.ckpt"
FINAL_STATE_DICT_NAME = "tft_stock_model.pt"


def run_tft_pipeline():
    # 1. Load the master dataset
    df = pd.read_csv("data/tft_master_dataset.csv")

    # 2. Define TimeSeries Parameters
    max_prediction_length = 5   # We want to forecast out 5 trading days
    max_encoder_length = 60      # Lookback window of 60 historical days

    # Cutoff point for chronological train/validation split
    training_cutoff = df["time_idx"].max() - max_prediction_length

    # 3. Create the specialized TFT Dataset wrapper
    training_dataset = TimeSeriesDataSet(
        df[df["time_idx"] <= training_cutoff],
        time_idx="time_idx",
        target="Close_Target",
        group_ids=["Ticker"],
        min_encoder_length=max_encoder_length // 2,
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=max_prediction_length,
        static_categoricals=["Ticker"],               # Static feature
        time_varying_known_reals=["time_idx"],        # Known future inputs
        time_varying_unknown_reals=[                 # Historical market variables
            "Close", "Volume", "RSI_14",
            "MACD_12_26_9", "BBL_20", "EMA_9"
        ],
        allow_missing_timesteps=True
    )

    # Create validation dataset using the exact same scaling parameters
    validation_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset, df, predict=True, stop_randomization=True
    )

    # Convert datasets to PyTorch DataLoaders
    batch_size = 64
    train_dataloader = training_dataset.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_dataloader = validation_dataset.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=0)

    # 4. Initialize Temporal Fusion Transformer Model
    # It uses QuantileLoss to give us standard trading prediction boundaries (e.g., 10th, 50th, 90th percentiles)
    tft = TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=0.03,
        hidden_size=32,          # Size of attention layers
        attention_head_size=2,   # Multi-head attention structure
        dropout=0.2,
        loss=QuantileLoss(),
        reduce_on_plateau_patience=4
    )
    num_params = sum(p.numel() for p in tft.parameters())
    print(f"🧠 TFT Network initialized with {num_params / 1e3:.1f}k active parameters.")

    # 5. Configure PyTorch Lightning Trainer Engine
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=1e-4, patience=5, verbose=False, mode="min"
    )
    # Keep the best checkpoint by validation loss (and always keep the last epoch)
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(CHECKPOINT_DIR),
        filename="tft-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )

    trainer = pl.Trainer(
        max_epochs=30,
        accelerator="auto",  # Automatically detects and utilizes GPU (CUDA) if available
        callbacks=[early_stop_callback, checkpoint_callback],
        enable_model_summary=True,
        default_root_dir=str(CHECKPOINT_DIR),
    )

    # Fit Model
    print("🚀 Commencing Temporal Fusion Transformer attention training loop...")
    trainer.fit(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    # 6. Persist the best model for later inference / evaluation
    best_model_path = checkpoint_callback.best_model_path
    if best_model_path:
        stable_ckpt = CHECKPOINT_DIR / BEST_CKPT_NAME
        state_dict_path = CHECKPOINT_DIR / FINAL_STATE_DICT_NAME

        # Fixed-name copy of the best Lightning checkpoint (full model + hparams)
        shutil.copy2(best_model_path, stable_ckpt)

        # Weights-only .pt for lighter loading (same pattern as lstm_stock_model.pt)
        best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
        torch.save(best_tft.state_dict(), state_dict_path)

        print(f"💾 Best checkpoint: {best_model_path}")
        print(f"💾 Stable checkpoint copy: {stable_ckpt}")
        print(f"💾 State dict saved to: {state_dict_path}")
        print(f"   best val_loss={checkpoint_callback.best_model_score}")
    else:
        # Fallback if no val metric checkpoint was produced (e.g. training aborted early)
        fallback_path = CHECKPOINT_DIR / FINAL_STATE_DICT_NAME
        torch.save(tft.state_dict(), fallback_path)
        print(f"⚠️ No best checkpoint found; saved final weights to {fallback_path}")


if __name__ == "__main__":
    run_tft_pipeline()
