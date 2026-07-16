import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=128,
                 num_layers=2, output_dim=5, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]      # take last timestep
        return self.fc(out)