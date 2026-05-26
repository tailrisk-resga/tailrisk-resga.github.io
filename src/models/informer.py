"""Informer model."""

import torch.nn as nn
import torch.nn.functional as F

from models.components.transformer import ConvLayer, Decoder, DecoderLayer, Encoder, EncoderLayer
from models.components.self_attention import AttentionLayer, ProbAttention
from models.components.embedding import DataEmbedding

class Informer(nn.Module):
    """
    Informer for Stock Prediction (12->1 forecasting)
    """

    def __init__(self, args):
        super(Informer, self).__init__()

        # Embedding
        self.enc_embedding = DataEmbedding(args.feature_dim, args.d_model, args.embed, args.freq, args.dropout)
        self.dec_embedding = DataEmbedding(args.feature_dim, args.d_model, args.embed, args.freq, args.dropout)

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

        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        ProbAttention(True, args.factor, attention_dropout=args.dropout, output_attention=False),
                        args.d_model, args.n_heads
                    ),
                    AttentionLayer(
                        ProbAttention(False, args.factor, attention_dropout=args.dropout, output_attention=False),
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


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        Forward pass for 12->1 stock prediction with timing logs
        """
        

        # ===== Encoding =====
        
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)
        
        # ===== Decoding =====
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)

        # ===== Output projection =====
        x = dec_out[:, -1, :]
        v = - F.softplus(x[:, 0])
        e = v - F.softplus(x[:, 1])
        return v, e



__all__ = ["Informer"]
