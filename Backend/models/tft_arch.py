import torch
import torch.nn as nn

class TFTModel(nn.Module):
    """Simplified Temporal Fusion Transformer."""
    def __init__(self, input_dim=8, hidden_dim=64,
                 num_heads=4, output_dim=5, seq_len=30):
        super().__init__()
        self.seq_len = seq_len
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Temporal attention
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads,
                                               batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        
        # LSTM for local context
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Quantile output
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        x = self.input_proj(x)
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        lstm_out, _ = self.lstm(x)
        x = self.norm2(x + lstm_out)
        return self.fc(lstm_out[:, -1, :])