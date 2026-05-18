import torch
import torch.nn as nn


class LSTMAutoencoder(nn.Module):
    """LSTM Autoencoder for anomaly detection in multivariate time series."""

    def __init__(self, seq_len: int = 144, num_features: int = 7):
        super().__init__()
        self.seq_len = seq_len
        self.num_features = num_features

        # Encoder
        self.encoder_lstm1 = nn.LSTM(
            input_size=num_features, hidden_size=64, batch_first=True
        )
        self.encoder_lstm2 = nn.LSTM(input_size=64, hidden_size=32, batch_first=True)
        self.encoder_lstm3 = nn.LSTM(input_size=32, hidden_size=16, batch_first=True)

        # Decoder
        self.decoder_lstm1 = nn.LSTM(input_size=16, hidden_size=32, batch_first=True)
        self.decoder_lstm2 = nn.LSTM(input_size=32, hidden_size=64, batch_first=True)
        self.decoder_linear = nn.Linear(64, num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Input tensor of shape (batch_size, seq_len, num_features)
        Returns:
            Reconstructed tensor of shape (batch_size, seq_len, num_features)
        """
        # --- Encoder ---
        out, _ = self.encoder_lstm1(x)
        out, _ = self.encoder_lstm2(out)
        out, (hn, _) = self.encoder_lstm3(out)

        # hn shape: (1, batch_size, 16) -> extract the final hidden state
        bottleneck = hn[-1]  # shape: (batch_size, 16)

        # --- Decoder ---
        # Repeat the bottleneck vector seq_len times
        # shape: (batch_size, seq_len, 16)
        repeated = bottleneck.unsqueeze(1).repeat(1, x.size(1), 1)

        out, _ = self.decoder_lstm1(repeated)
        out, _ = self.decoder_lstm2(out)

        # TimeDistributed Linear Layer: apply linear to all timesteps
        out = self.decoder_linear(out)

        return out
