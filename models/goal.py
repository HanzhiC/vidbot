import pytorch_lightning as pl
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import numpy as np
from models.layers_2d import Decoder, Encoder
from models.helpers import FocalLoss
from models.clip import clip, tokenize

import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.ops import roi_align, roi_pool
from copy import deepcopy
from models.layers_2d import load_clip
from models.perceiver import FeaturePerceiver


class GoalPredictor(pl.LightningModule):
    def __init__(
        self,
        in_channels=4,
        out_channels=3,
        resolution=[256, 448],
        channel_multiplier=[1, 2, 4, 8, 16],
        bbox_feature_dim=64,
        visual_feature_dim=512,
        encode_action=False,
        encode_bbox=False,
        encode_object=False,
        num_heads_attention=4,
        num_layers_attention=2,
        object_encode_mode="roi_pool",
        **kwargs,
    ):
        super().__init__()

        # self.gpt = gpt
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.resolution = resolution

        self.encode_action = encode_action
        self.encode_bbox = encode_bbox
        self.encode_object = encode_object

        self.visual_feature_dim = visual_feature_dim
        self.bbox_feature_dim = bbox_feature_dim
        self.object_encode_mode = object_encode_mode

        # ch_mult=[1, 2, 4, 8, 16]
        self.channel_multiplier = channel_multiplier

        self.downscale_factor = 2 ** (len(self.channel_multiplier) - 1)
        attn_resolutions = (
            resolution[0] // self.downscale_factor,
            resolution[1] // self.downscale_factor,
        )

        # Decide the model architecture
        self.visual = Encoder(
            ch=64,
            ch_mult=self.channel_multiplier,
            num_res_blocks=2,
            attn_resolutions=attn_resolutions,
            in_channels=in_channels,
            out_ch=out_channels,
            resolution=resolution,
            double_z=False,
            z_channels=self.visual_feature_dim,
        )

        self.decoder = Decoder(
            ch=64,
            ch_mult=self.channel_multiplier,
            num_res_blocks=2,
            attn_resolutions=attn_resolutions,
            in_channels=in_channels,
            out_ch=out_channels,
            resolution=resolution,
            double_z=False,
            z_channels=self.visual_feature_dim,
        )

        self.transform = transforms.Compose(
            [
                transforms.Normalize([0.485, 0.456, 0.406], [
                                     0.229, 0.224, 0.225]),
            ]
        )

        if self.encode_object:
            if self.object_encode_mode == "vlm":
                obj_dim = self.vlm_dim
            else:
                obj_dim = self.visual_feature_dim

            self.object_encode_module = nn.Linear(
                obj_dim, self.visual_feature_dim)

        if self.encode_action:

            self.action_encode_module = nn.Linear(
                self.visual_feature_dim, self.visual_feature_dim
            )

        if self.encode_bbox:

            self.bbox_encode_module = nn.Linear(4, bbox_feature_dim)

        fuser_dim = 0
        if self.encode_action:
            fuser_dim += self.visual_feature_dim
        if self.encode_object:
            fuser_dim += self.visual_feature_dim
        if self.encode_bbox:
            fuser_dim += self.bbox_feature_dim

        if self.encode_action or self.encode_object or self.encode_bbox:

            self.fuser = FeaturePerceiver(
                transition_dim=self.visual_feature_dim,
                condition_dim=fuser_dim,
                time_emb_dim=0,
            )
            self.proj = nn.Linear(self.fuser.last_dim, self.visual_feature_dim)

        else:
            self.fuser = None

        self.depth_fuser = FeaturePerceiver(
            transition_dim=self.visual_feature_dim,
            condition_dim=self.visual_feature_dim,
            time_emb_dim=0,  # No time embedding
        )
        self.depth_proj = nn.Sequential(
            nn.Linear(self.depth_fuser.last_dim, self.visual_feature_dim),
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=self.visual_feature_dim,
                    dim_feedforward=512,
                    nhead=4,
                    batch_first=True,
                ),
                num_layers=3,
            ),
            nn.Linear(self.visual_feature_dim, 1),
        )

    def forward(self, data_batch, training=False):
        target_key = "vfd"
        color_key = "color"
        object_color_key = "object_color"

        if training:
            color_key += "_aug"
            object_color_key += "_aug"

        inputs = self.transform(data_batch[color_key])
        depth = data_batch["depth"][:, None]  # [B, 1, H, W]

        if self.in_channels == 4:
            start_pos_depth = data_batch["start_pos_depth"][:, None]
            start_pos_depth_res = depth - start_pos_depth
            inputs = torch.cat([inputs, start_pos_depth_res], dim=1)

        # Shape information
        batch_size = inputs.shape[0]
        h_in, w_in = inputs.shape[-2:]
        h_out, w_out = h_in // self.downscale_factor, w_in // self.downscale_factor

        # Box information
        bbox = data_batch["bbox"]  # [B, 4]
        bbox_batch_id = torch.arange(
            batch_size, device=inputs.device
        )  # Only one box per sample
        bbox = torch.cat([bbox_batch_id[:, None], bbox], dim=1)  # [B, 5]

        # Extract visual features
        context_feature = self.visual(inputs)
        context_feature = einops.rearrange(
            context_feature.clone(), "b c h w -> b (h w) c", h=h_out, w=w_out
        )
        feature = context_feature

        # Acquire bbox features
        condition_feature = []
        if self.encode_object:
            if self.object_encode_mode == "vlm":

                print("... Try another way to encode object")
            elif self.object_encode_mode in ["roi_pool", "roi_align"]:
                roi_res = 6

                roi_method = eval(self.object_encode_mode)
                context_feature_obj = einops.rearrange(
                    context_feature, "b (h w) c -> b c h w", h=h_out, w=w_out
                )
                spatial_scale = context_feature_obj.shape[-1] / \
                    inputs.shape[-1]
                assert spatial_scale == context_feature_obj.shape[-2] / \
                    inputs.shape[-2]

                context_feature_obj = roi_method(
                    context_feature_obj,
                    bbox,
                    spatial_scale=spatial_scale,
                    output_size=(roi_res, roi_res),
                )  # [B, c, roi_res, roi_res]
                object_feature = einops.rearrange(
                    context_feature_obj, "b c h w -> b (h w) c"
                )
                object_feature = object_feature.mean(
                    dim=1)[:, None]  # [B, 1, c]
            else:
                raise NotImplementedError(
                    "Object encode mode {} not implemented".format(
                        self.object_encode_mode
                    )
                )
            object_feature = self.object_encode_module(object_feature)

            condition_feature.append(object_feature)

        if self.encode_action:

            action_feature = data_batch["action_feature"][:, None]
            action_feature = self.action_encode_module(
                action_feature)  # [B, 1, c]

            condition_feature.append(action_feature)

        if self.encode_bbox:
            bbox_norm = bbox[:, 1:].clone()
            bbox_norm[:, [0, 2]] = bbox_norm[:, [0, 2]] / inputs.shape[-1]
            bbox_norm[:, [1, 3]] = bbox_norm[:, [1, 3]] / inputs.shape[-2]
            bbox_feature = self.bbox_encode_module(bbox_norm)  # [B, 64]

            bbox_feature = bbox_feature[:, None]  # [B, 1, 64]
            condition_feature.append(bbox_feature)

        condition_feature = torch.cat(
            condition_feature, dim=-1)  # [B, 1, 2c+64)

        if self.fuser is not None and len(condition_feature) > 0:
            feature = self.fuser(feature, condition_feature)
            feature = self.proj(feature)

        depth_feature = feature
        verb_feature = data_batch["verb_feature"][:, None]

        depth_feature = self.depth_fuser(depth_feature, verb_feature)
        pred_depth = self.depth_proj(depth_feature)

        feature = einops.rearrange(
            feature, "b (h w) c -> b c h w", h=h_out, w=w_out)
        pred_depth = einops.rearrange(
            pred_depth, "b (h w) c -> b c h w", h=h_out, w=w_out
        )

        pred = self.decoder(feature)
        pred_depth = F.interpolate(
            pred_depth, size=(
                self.resolution[0], self.resolution[1]), mode="bilinear"
        )
        # Postprocess the output
        if pred is not None:
            pred_vf, pred_dres, pred_heatmap = pred[:,
                                                    :2], pred_depth, pred[:, -1:]
            pred_vf = F.normalize(pred_vf, p=2, dim=1)  # [-1, 1]
            pred_vf = pred_vf.clamp(-1, 1)
            if "d_res_scale" in data_batch:
                print(
                    "... Rescale the depth with d_res_scale: ",
                    data_batch["d_res_scale"],
                )
                pred_dres = pred_dres * data_batch["d_res_scale"]
            pred_d_final = start_pos_depth - pred_dres  # [B, 1, H, W]
            pred = torch.cat(
                [pred_vf, pred_d_final, pred_heatmap], dim=1
            )  # [B, 4, H, W]

        outputs = {"pred": pred}

        return outputs
