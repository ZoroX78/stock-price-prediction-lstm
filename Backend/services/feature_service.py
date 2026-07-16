import torch
import numpy as np
from app.services.model_service import model_service
from app.services.data_service import data_service

class FeatureService:
    """Permutation importance for the LSTM model."""

    async def compute_importance(self, symbol: str) -> list[dict]:
        df = await data_service.get_ohlcv(symbol)
        feats = model_service._build_features(df)
        seq = feats.tail(model_service.lstm_seq_len).values
        scaled = model_service.scaler.transform(seq)
        x = torch.tensor(scaled, dtype=torch.float32) \
              .unsqueeze(0).to(model_service.device)

        with torch.no_grad():
            base = model_service.lstm(x).cpu().numpy().flatten()

        importances = []
        for i, col in enumerate(model_service.feature_cols):
            x_perm = x.clone()
            x_perm[0, :, i] = x_perm[0, :, i][torch.randperm(x.size(1))]
            with torch.no_grad():
                perm = model_service.lstm(x_perm).cpu().numpy().flatten()
            imp = float(np.mean(np.abs(base - perm)))
            importances.append({"feature": col, "importance": round(imp, 4)})

        # Normalize
        total = sum(i["importance"] for i in importances) or 1
        for i in importances:
            i["importance_pct"] = round(i["importance"] / total * 100, 2)
        return sorted(importances, key=lambda x: -x["importance"])

feature_service = FeatureService()