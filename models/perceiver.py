import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Tuple
from collections import OrderedDict
from einops import rearrange
from models.layers_3d import SinusoidalPosEmb
from models.attention import SelfAttentionBlock, CrossAttentionLayer


class FeaturePerceiver(nn.Module):

    def __init__(
        self,
        transition_dim,
        condition_dim,
        time_emb_dim,
        encoder_q_input_channels=512,
        encoder_kv_input_channels=256,
        encoder_num_heads=8,
        encoder_widening_factor=1,
        encoder_dropout=0.1,
        encoder_residual_dropout=0.0,
        encoder_self_attn_num_layers=2,
        decoder_q_input_channels=256,
        decoder_kv_input_channels=512,
        decoder_num_heads=8,
        decoder_widening_factor=1,
        decoder_dropout=0.1,
        decoder_residual_dropout=0.0,
    ) -> None:
        super().__init__()

        self.encoder_q_input_channels = encoder_q_input_channels
        self.encoder_kv_input_channels = encoder_kv_input_channels
        self.encoder_num_heads = encoder_num_heads
        self.encoder_widening_factor = encoder_widening_factor
        self.encoder_dropout = encoder_dropout
        self.encoder_residual_dropout = encoder_residual_dropout
        self.encoder_self_attn_num_layers = encoder_self_attn_num_layers

        self.decoder_q_input_channels = decoder_q_input_channels
        self.decoder_kv_input_channels = decoder_kv_input_channels
        self.decoder_num_heads = decoder_num_heads
        self.decoder_widening_factor = decoder_widening_factor
        self.decoder_dropout = decoder_dropout
        self.decoder_residual_dropout = decoder_residual_dropout

        self.condition_adapter = nn.Linear(
            condition_dim, self.encoder_q_input_channels, bias=True
        )
        
        if time_emb_dim > 0:
            self.time_embedding_adapter = nn.Linear(
                time_emb_dim, self.encoder_q_input_channels, bias=True
            )
        else:
            self.time_embedding_adapter = None

        self.encoder_adapter = nn.Linear(
            transition_dim,
            self.encoder_kv_input_channels,
            bias=True,
        )
        self.decoder_adapter = nn.Linear(
            self.encoder_kv_input_channels, self.decoder_q_input_channels, bias=True
        )

        self.encoder_cross_attn = CrossAttentionLayer(
            num_heads=self.encoder_num_heads,
            num_q_input_channels=self.encoder_q_input_channels,
            num_kv_input_channels=self.encoder_kv_input_channels,
            widening_factor=self.encoder_widening_factor,
            dropout=self.encoder_dropout,
            residual_dropout=self.encoder_residual_dropout,
        )

        self.encoder_self_attn = SelfAttentionBlock(
            num_layers=self.encoder_self_attn_num_layers,
            num_heads=self.encoder_num_heads,
            num_channels=self.encoder_q_input_channels,
            widening_factor=self.encoder_widening_factor,
            dropout=self.encoder_dropout,
            residual_dropout=self.encoder_residual_dropout,
        )

        self.decoder_cross_attn = CrossAttentionLayer(
            num_heads=self.decoder_num_heads,
            num_q_input_channels=self.decoder_q_input_channels,
            num_kv_input_channels=self.decoder_kv_input_channels,
            widening_factor=self.decoder_widening_factor,
            dropout=self.decoder_dropout,
            residual_dropout=self.decoder_residual_dropout,
        )
        self.last_dim = self.decoder_q_input_channels
        
        
    def forward(
        self,
        x,
        condition_feat,
        time_embedding=None,
    ):
        """Forward pass of the ContactMLP.

        Args:
            x: input contact map, [bs, num_points, transition_dim]
            condition_feat: [bs, 1, condition_dim]
            time_embedding: [bs, 1, time_embedding_dim]

        Returns:
            Output contact map, [bs, num_points, contact_dim]
        """

        # encoder
        enc_kv = self.encoder_adapter(x)  # [bs, num_points, enc_kv_dim]
        cond_feat = self.condition_adapter(condition_feat)  # [bs, 1, enc_q_dim]
        if time_embedding is not None and self.time_embedding_adapter is not None:
            time_embedding = self.time_embedding_adapter(
                time_embedding
            )  # [bs, 1, enc_q_dim]

            enc_q = torch.cat([cond_feat, time_embedding], dim=1)  # [bs, 1 + 1, enc_q_dim]
        else:
            enc_q = cond_feat

        enc_q = self.encoder_cross_attn(enc_q, enc_kv).last_hidden_state
        enc_q = self.encoder_self_attn(enc_q).last_hidden_state

        # decoder
        dec_kv = enc_q
        dec_q = self.decoder_adapter(enc_kv)  # [bs, num_points, dec_q_dim]
        dec_q = self.decoder_cross_attn(
            dec_q, dec_kv
        ).last_hidden_state  # [bs, num_points, dec_q_dim]

        return dec_q


if __name__ == "__main__":
    print("Testing ContactPerceiver")
    feat_preceiver = FeaturePreceiver(
        transition_dim=64 + 3,
        condition_dim=512,
        time_emb_dim=64,
    )
    x = torch.randn(2, 80, 64 + 3)
    cond = torch.randn(2, 1, 512)
    time = torch.randn(2, 1, 64)
    feat = feat_preceiver(x, cond, time)
    print(feat.shape)
