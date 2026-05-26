"""Decoder-only Informer model."""

import torch.nn as nn
import torch.nn.functional as F

from models.components.transformer import DecoderOnly, DecoderOnlyLayer
from models.components.self_attention import AttentionLayer, ProbAttention
from models.components.embedding import DataEmbedding

class DInformer(nn.Module):
    """
    Informer for Stock Prediction (12->1 forecasting)
    """

    def __init__(self, args):
        super(DInformer, self).__init__()

        # Embedding
        self.embedding = DataEmbedding(args.feature_dim, args.d_model, args.embed, args.freq, args.dropout)

        # Decoder
        self.decoder = DecoderOnly(
            [
                DecoderOnlyLayer(
                    AttentionLayer(
                        ProbAttention(True, args.factor, attention_dropout=args.dropout, output_attention=False),
                        args.d_model, args.n_heads
                    ),
                    args.d_model,
                    args.d_ff,
                    dropout=args.dropout,
                    activation=args.activation
                ) for _ in range(args.d_layers)
            ],
            norm_layer=nn.LayerNorm(args.d_model),
            projection=nn.Linear(args.d_model, args.c_out, bias=True)
        )


    def forward(self, x, x_mark):
        """
        Forward pass for 12->1 stock prediction with timing logs
        """
        
        # ===== Encoding =====
        embedding = self.embedding(x, x_mark) # [B, T, d_model]
        dec_out = self.decoder(embedding, x_mask=None) # [B, T, c_out]

        # ===== Output projection =====
        x = dec_out[:, -1, :]
        v = - F.softplus(x[:, 0])
        e = v - F.softplus(x[:, 1])
        return v, e



__all__ = ["DInformer"]
