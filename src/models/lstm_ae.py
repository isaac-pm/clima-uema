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


class LSTMAutoencoderImproved(nn.Module):
    """LSTM Autoencoder with symmetric encoder/decoder (improved variant)."""

    def __init__(
        self,
        seq_len: int = 144,
        num_features: int = 7,
        latent_dim: int = 64,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.num_features = num_features

        self.encoder_lstm1 = nn.LSTM(num_features, 128, batch_first=True)
        self.encoder_lstm2 = nn.LSTM(128, 64, batch_first=True)
        self.encoder_lstm3 = nn.LSTM(64, latent_dim, batch_first=True)

        self.decoder_lstm1 = nn.LSTM(latent_dim, 64, batch_first=True)
        self.decoder_lstm2 = nn.LSTM(64, 128, batch_first=True)
        self.decoder_linear = nn.Linear(128, num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc1, _ = self.encoder_lstm1(x)
        enc2, _ = self.encoder_lstm2(enc1)
        enc3, (hn, _) = self.encoder_lstm3(enc2)

        latent = hn[-1].unsqueeze(1).repeat(1, x.size(1), 1)

        dec1, _ = self.decoder_lstm1(latent)
        dec2, _ = self.decoder_lstm2(dec1)
        return self.decoder_linear(dec2)


class StationAwareLSTMAutoencoder(nn.Module):
    """LSTM Autoencoder with station embeddings for global models."""

    def __init__(
        self,
        num_stations: int,
        station_embed_dim: int = 8,
        seq_len: int = 144,
        num_features: int = 7,
        latent_dim: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.station_embedding = nn.Embedding(num_stations, station_embed_dim)
        self.autoencoder = LSTMAutoencoder(
            seq_len=seq_len,
            num_features=num_features + station_embed_dim,
            latent_dim=latent_dim,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor, station_ids: torch.Tensor) -> torch.Tensor:
        station_ids = station_ids.long()
        emb = self.station_embedding(station_ids)
        emb = emb.unsqueeze(1).expand(-1, x.size(1), -1)
        x_with_emb = torch.cat([x, emb], dim=-1)
        return self.autoencoder(x_with_emb)
