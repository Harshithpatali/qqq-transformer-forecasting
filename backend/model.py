import math

import torch
import torch.nn as nn

from config import (
    INPUT_DIM,
    D_MODEL,
    NHEAD,
    NUM_LAYERS,
    DROPOUT
)


# ==========================================
# POSITIONAL ENCODING
# ==========================================

class PositionalEncoding(nn.Module):

    def __init__(
        self,
        d_model,
        max_len=5000
    ):

        super().__init__()

        pe = torch.zeros(
            max_len,
            d_model
        )

        position = torch.arange(
            0,
            max_len,
            dtype=torch.float
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(
                0,
                d_model,
                2
            ).float()
            * (
                -math.log(10000.0)
                / d_model
            )
        )

        pe[:, 0::2] = torch.sin(
            position * div_term
        )

        pe[:, 1::2] = torch.cos(
            position * div_term
        )

        pe = pe.unsqueeze(0)

        self.register_buffer(
            "pe",
            pe
        )

    def forward(self, x):

        x = x + self.pe[
            :,
            :x.size(1)
        ]

        return x


# ==========================================
# TRANSFORMER FORECASTER
# ==========================================

class TransformerForecaster(nn.Module):

    def __init__(self):

        super().__init__()

        # ==================================
        # INPUT PROJECTION
        # ==================================

        self.input_projection = nn.Linear(
            INPUT_DIM,
            D_MODEL
        )

        # ==================================
        # POSITIONAL ENCODING
        # ==================================

        self.positional_encoding = (
            PositionalEncoding(
                D_MODEL
            )
        )

        # ==================================
        # TRANSFORMER ENCODER LAYER
        # ==================================

        encoder_layer = (
            nn.TransformerEncoderLayer(
                d_model=D_MODEL,
                nhead=NHEAD,
                dropout=DROPOUT,
                batch_first=True,
                dim_feedforward=256,
                activation="gelu"
            )
        )

        # ==================================
        # TRANSFORMER ENCODER
        # ==================================

        self.transformer_encoder = (
            nn.TransformerEncoder(
                encoder_layer,
                num_layers=NUM_LAYERS
            )
        )

        # ==================================
        # DROPOUT
        # ==================================

        self.dropout = nn.Dropout(
            DROPOUT
        )

        # ==================================
        # REGRESSION HEAD
        # ==================================

        self.regressor = nn.Sequential(

            nn.Linear(
                D_MODEL,
                128
            ),

            nn.ReLU(),

            nn.Dropout(
                DROPOUT
            ),

            nn.Linear(
                128,
                64
            ),

            nn.ReLU(),

            nn.Dropout(
                DROPOUT
            ),

            nn.Linear(
                64,
                1
            )
        )

    def forward(self, x):

        # ==================================
        # INPUT SHAPE
        # ==================================
        # (batch, seq_len, features)
        # ==================================

        x = self.input_projection(x)

        # ==================================
        # ADD POSITIONAL ENCODING
        # ==================================

        x = self.positional_encoding(x)

        # ==================================
        # TRANSFORMER ENCODER
        # ==================================

        x = self.transformer_encoder(x)

        # ==================================
        # USE LAST TIME STEP
        # ==================================

        x = x[:, -1, :]

        # ==================================
        # DROPOUT
        # ==================================

        x = self.dropout(x)

        # ==================================
        # REGRESSION OUTPUT
        # ==================================

        output = self.regressor(x)

        return output.squeeze()


# ==========================================
# MODEL TEST
# ==========================================

if __name__ == "__main__":

    print("Testing Transformer Model...")

    model = TransformerForecaster()

    dummy_input = torch.randn(
        32,
        60,
        INPUT_DIM
    )

    output = model(dummy_input)

    print("\nInput Shape:")
    print(dummy_input.shape)

    print("\nOutput Shape:")
    print(output.shape)

    print("\nModel Architecture:\n")

    print(model)