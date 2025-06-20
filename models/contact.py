import pytorch_lightning as pl
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import numpy as np

from models.layers_2d import ResNet50Encoder, ResNet50Decoder, PositionalEmbeddingV2
import einops
import torchvision.transforms as transforms
from diffuser_utils.dataset_utils import compute_model_size
from models.perceiver import FeaturePerceiver


class ContactPredictor(pl.LightningModule):
    def __init__(
        self,
        in_channels=3,
        out_channels=2,
        use_skip=True,
        encode_action=False,
        use_min_loss=False,
        **kwargs,
    ):
        super(ContactPredictor, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_skip = use_skip
        self.encode_action = encode_action
        self.use_min_loss = use_min_loss
        self.transform = transforms.Compose(
            [
                transforms.Normalize([0.485, 0.456, 0.406], [
                                     0.229, 0.224, 0.225]),
            ]
        )
        self.visual = ResNet50Encoder(input_channels=in_channels)
        self.decoder = ResNet50Decoder(
            output_channels=out_channels, use_skip=use_skip)
        self.bottleneck_feature_key = self.decoder.bottleneck_feature_key
        self.latent_dim = self.decoder.latent_dim
        self.visual_proj = nn.Linear(self.latent_dim, 512)
        # self.proj = nn.Linear(self.latent_dim, 512)

        if self.encode_action:
            self.action_proj = nn.Linear(512, 512)
            self.action_fuser = FeaturePerceiver(
                transition_dim=512, condition_dim=512, time_emb_dim=0
            )
            self.final_proj = nn.Linear(self.action_fuser.last_dim, 512)
        else:
            print("No action encoding")

        self.fuser = nn.Sequential(
            nn.TransformerEncoderLayer(
                d_model=512, nhead=4, dim_feedforward=512, batch_first=True
            ),
            nn.Linear(512, self.latent_dim),
        )
        self.positional_embedding = PositionalEmbeddingV2(
            d_model=512, max_len=400)

    def forward(self, data_batch, training=False):
        outputs = {}
        object_color_key = "object_color"
        object_depth_key = "object_depth"
        if training:
            object_color_key += "_aug"

        inputs = self.transform(data_batch[object_color_key])
        if self.in_channels == 4:
            object_depth = data_batch[object_depth_key][:, None]
            inputs = torch.cat([inputs, object_depth], dim=1)
        features = self.visual(inputs)
        features = self.forward_latent(data_batch, features)
        pred = self.decoder(features)

        # Post-process the prediction
        pred_final = []
        pred_vfs = pred[:, : self.out_channels - 1]  # [B, 8, H, W]
        pred_mask = pred[:, self.out_channels - 1:]  # [B, 1, H, W]
        for hi in range(0, self.out_channels - 1, 2):
            pred_vf = pred_vfs[:, hi: hi + 2]  # [B, 2, H, W]
            pred_vf = F.normalize(pred_vf, p=2, dim=1)  # [-1, 1]
            pred_vf = pred_vf.clamp(-1, 1)
            pred_final.append(pred_vf)
        pred_final.append(pred_mask)
        pred_final = torch.cat(pred_final, dim=1)
        outputs["pred"] = pred_final  # [B, 8+1, H, W]
        return outputs

    def forward_latent(self, data_batch, features):
        latent = features[self.bottleneck_feature_key]
        h, w = latent.shape[-2:]
        latent = einops.rearrange(latent, "b c h w -> b (h w) c")
        latent = self.visual_proj(latent)
        # latent = self.proj(latent)
        if self.encode_action:
            action_feature = data_batch["action_feature"][:, None]
            action_feature = self.action_proj(action_feature)
            latent = self.action_fuser(latent, action_feature)
            latent = self.final_proj(latent)

        latent = self.positional_embedding(latent)
        latent = self.fuser(latent)
        latent = einops.rearrange(latent, "b (h w) c -> b c h w", h=h, w=w)
        features[self.bottleneck_feature_key] = latent
        return features
