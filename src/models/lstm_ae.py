import torch
import torch.nn as nn


class LSTMAutoencoder(nn.Module):
    """LSTM Autoencoder for anomaly detection in multivariate time series."""

    def __init__(
        self,
        seq_len: int = 144,
        num_features: int = 7,
        latent_dim: int = 16,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.num_features = num_features

        # Encoder
        self.encoder_lstm1 = nn.LSTM(
            input_size=num_features, hidden_size=128, batch_first=True
        )
        self.encoder_lstm2 = nn.LSTM(input_size=128, hidden_size=64, batch_first=True)
        self.encoder_lstm3 = nn.LSTM(
            input_size=64, hidden_size=latent_dim, batch_first=True
        )
        self.encoder_dropout = nn.Dropout(dropout)

        # Decoder — symmetric depth with encoder (3 LSTM layers)
        self.decoder_lstm1 = nn.LSTM(
            input_size=latent_dim, hidden_size=latent_dim, batch_first=True
        )
        self.decoder_lstm2 = nn.LSTM(
            input_size=latent_dim, hidden_size=64, batch_first=True
        )
        self.decoder_lstm3 = nn.LSTM(input_size=64, hidden_size=128, batch_first=True)
        self.decoder_dropout = nn.Dropout(dropout)
        self.decoder_linear = nn.Linear(128, 3)  # outputs continuous features only

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
        out = self.encoder_dropout(out)
        out, _ = self.encoder_lstm2(out)
        out = self.encoder_dropout(out)
        out, (hn, _) = self.encoder_lstm3(out)

        # hn shape: (1, batch_size, latent_dim) -> extract the final hidden state
        bottleneck = hn[-1]  # shape: (batch_size, latent_dim)

        # --- Decoder ---
        # Repeat the bottleneck vector seq_len times
        # shape: (batch_size, seq_len, latent_dim)
        repeated = bottleneck.unsqueeze(1).repeat(1, x.size(1), 1)

        out, _ = self.decoder_lstm1(repeated)
        out = self.decoder_dropout(out)
        out, _ = self.decoder_lstm2(out)
        out = self.decoder_dropout(out)
        out, _ = self.decoder_lstm3(out)

        out = self.decoder_linear(out)

        return out
