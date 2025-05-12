# Core neural network building blocks (nn.Module) 

import torch
import torch.nn as nn

class LSTMBackbone(nn.Module):
    """Core LSTM network structure (nn.Module)."""
    def __init__(self, n_features: int, n_targets: int, hidden_units: int = 50, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_units,
            num_layers=num_layers,
            batch_first=True # Input shape: (batch, seq_len, features)
        )
        self.linear = nn.Linear(hidden_units, n_targets)

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        # Use the output of the last time step for prediction
        # lstm_out shape: (batch, seq_len, hidden_units)
        last_time_step_out = lstm_out[:, -1, :]
        # last_time_step_out shape: (batch, hidden_units)
        prediction = self.linear(last_time_step_out)
        # prediction shape: (batch, n_targets)
        return prediction 