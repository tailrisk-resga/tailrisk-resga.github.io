"""Encoder-only Informer model."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.components.transformer import ConvLayer, Encoder, EncoderLayer
from models.components.self_attention import AttentionLayer, ProbAttention
from models.components.embedding import DataEmbedding

class EInformer(nn.Module):
    """
    Informer for Stock Prediction (12->1 forecasting)
    """

    def __init__(self, args):
        super(EInformer, self).__init__()

        # Embedding
        self.enc_embedding = DataEmbedding(args.feature_dim, args.d_model, args.embed, args.freq, args.dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        ProbAttention(False, args.factor, attention_dropout=args.dropout, output_attention=False),
                        args.d_model, args.n_heads
                    ),
                    args.d_model,
                    args.d_ff,
                    dropout=args.dropout,
                    activation=args.activation
                ) for _ in range(args.e_layers)
            ],
            [
                ConvLayer(args.d_model)
                for _ in range(args.e_layers - 1)
            ] if args.distil else None,
            norm_layer=nn.LayerNorm(args.d_model)
        )

        self.proj = nn.Linear(args.d_model, args.c_out)

        # Learnable mask token
        self.mask_token = nn.Parameter(torch.randn(1, 1, args.d_model))


    def forward(self, x_enc, x_mark_enc):
        """
        Forward pass for 12->1 stock prediction with timing logs
        """
        
        # ===== Encoding =====
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        mask_token = self.mask_token.expand(enc_out.size(0), 1, -1)
        enc_out = torch.cat([enc_out, mask_token], dim=1)

        enc_out, _ = self.encoder(enc_out, attn_mask=None)
        enc_out = enc_out[:, -1, :]
        x = self.proj(enc_out)

        # ===== Output projection =====
        v = - F.softplus(x[:, 0])
        e = v - F.softplus(x[:, 1])
        return v, e



__all__ = ["EInformer"]
