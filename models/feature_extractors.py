import einops
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from models.layers_2d import (
    BackprojectDepth,
    load_clip,
    Project3D,
)
from torchvision.ops import FeaturePyramidNetwork
import torchvision
from models.helpers import TSDFVolume, get_view_frustum
from models.layers_3d import VoxelGridEncoder

from typing import Union, List, Tuple

from models.perceiver import FeaturePerceiver


class MultiScaleImageFeatureExtractor(nn.Module):
    _RESNET_MEAN = [0.485, 0.456, 0.406]
    _RESNET_STD = [0.229, 0.224, 0.225]

    def __init__(
        self,
        modelname: str = "dino_vits16",
        freeze: bool = False,
        scale_factors: list = [1, 1 / 2, 1 / 3],
        embedding_dim: int = None,
    ):
        super().__init__()
        self.freeze = freeze
        self.scale_factors = scale_factors
        self.embedding_dim = embedding_dim

        if "res" in modelname:
            self._net = getattr(torchvision.models, modelname)(pretrained=True)
            self._output_dim = self._net.fc.weight.shape[1]
            self._net.fc = nn.Identity()
        elif "dinov2" in modelname:
            self._net = torch.hub.load("facebookresearch/dinov2", modelname)
            self._output_dim = self._net.norm.weight.shape[0]
        elif "dino" in modelname:
            self._net = torch.hub.load("facebookresearch/dino:main", modelname)
            self._output_dim = self._net.norm.weight.shape[0]
        else:
            raise ValueError(f"Unknown model name {modelname}")

        for name, value in (
            ("_resnet_mean", self._RESNET_MEAN),
            ("_resnet_std", self._RESNET_STD),
        ):
            self.register_buffer(
                name, torch.FloatTensor(value).view(1, 3, 1, 1), persistent=False
            )

        if self.freeze:
            for param in self.parameters():
                param.requires_grad = False

        if self.embedding_dim is not None:
            self._last_layer = nn.Linear(self._output_dim, self.embedding_dim)
            self._output_dim = self.embedding_dim
        else:
            self._last_layer = nn.Identity()

    def get_output_dim(self):
        return self._output_dim

    def forward(self, image_rgb: torch.Tensor):
        img_normed = self._resnet_normalize_image(image_rgb)
        features = self._compute_multiscale_features(img_normed)
        return features

    def _resnet_normalize_image(self, img: torch.Tensor):
        return (img - self._resnet_mean) / self._resnet_std

    def _compute_multiscale_features(self, img_normed: torch.Tensor):
        multiscale_features = None

        if len(self.scale_factors) <= 0:
            raise ValueError(
                f"Wrong format of self.scale_factors: {self.scale_factors}"
            )

        for scale_factor in self.scale_factors:
            if scale_factor == 1:
                inp = img_normed
            else:
                inp = self._resize_image(img_normed, scale_factor)

            if multiscale_features is None:
                multiscale_features = self._net(inp)
            else:
                multiscale_features += self._net(inp)

        averaged_features = multiscale_features / len(self.scale_factors)
        averaged_features = self._last_layer(averaged_features)
        return averaged_features

    @staticmethod
    def _resize_image(image: torch.Tensor, scale_factor: float):
        return nn.functional.interpolate(
            image, scale_factor=scale_factor, mode="bilinear", align_corners=False
        )

        
class TSDFMapFeatureExtractor(nn.Module):
    def __init__(
        self,
        input_image_shape,
        voxel_resolution=32,
        voxel_feature_dim=64,
        vlm_feature_attn_dim=256,
        # use_feature_decoder=True,
    ):
        super().__init__()
        self.input_image_shape = input_image_shape
        self.voxel_resolution = voxel_resolution
        self.embedding_dim = voxel_feature_dim
        self.backproject = BackprojectDepth(input_image_shape[0], input_image_shape[1])
        self.project_3d = Project3D()

        # Load pretrained backbone
        # self.vlm, self.vlm_transform = clip.load("ViT-B/16", jit=False)
        self.vlm, self.vlm_transform = load_clip()

        self.vlm.float()
        for p in self.vlm.parameters():
            p.requires_grad = False

        # Load 3D Unet
        self.tsdf_net = VoxelGridEncoder(
            self.voxel_resolution, c_dim=self.embedding_dim
        )

        self.feature_pyramid = FeaturePyramidNetwork(
            [64, 256, 512, 1024, 2048], voxel_feature_dim
        )
        
        self.feature_map_pyramid_keys = ["res1", "res2", "res3"]

        # Cross Attention Layer
        # self.action_proj = nn.Linear(vlm_feature_attn_dim, self.embedding_dim, bias=True)
        self.vlm_preceiver_pyramid = nn.ModuleList()
        self.vlm_proj_pyramid = nn.ModuleList()
        vlm_preceiver = FeaturePerceiver(
            transition_dim=self.embedding_dim, 
            condition_dim=vlm_feature_attn_dim,
            time_emb_dim=0,
        )
        vlm_proj = nn.Linear(vlm_preceiver.last_dim, self.embedding_dim, bias=True)
        for _ in range(len(self.feature_map_pyramid_keys)):

            self.vlm_preceiver_pyramid.append(vlm_preceiver)
            self.vlm_proj_pyramid.append(vlm_proj)        
        # Feature projection layer
        proj_dim_in = voxel_feature_dim * (1 + len(self.feature_map_pyramid_keys))
        self.proj = nn.Linear(proj_dim_in, self.embedding_dim, bias=True)
        
    def compute_tsdf_volume(self, color, depth, intrinsics, verbose=False):
        cam_pose = np.eye(4)
        tsdf_grid_batch = []
        tsdf_bounds_batch = []
        tsdf_color_batch = []
        mesh_batch = []
        for i in range(len(depth)):
            d_np = depth[i].cpu().numpy()[0]
            c_np = (
                color[i].cpu().numpy().transpose(1, 2, 0)
            )  # [H, W, 3], requested by TSDFVolume
            K_np = intrinsics[i].cpu().numpy()
            view_frust_pts = get_view_frustum(d_np, K_np, cam_pose)
            vol_bnds = np.zeros((3, 2))
            vol_bnds[:, 0] = np.minimum(
                vol_bnds[:, 0], np.amin(view_frust_pts, axis=1)
            ).min()
            vol_bnds[:, 1] = np.maximum(
                vol_bnds[:, 1], np.amax(view_frust_pts, axis=1)
            ).max()
            tsdf = TSDFVolume(vol_bnds, voxel_dim=self.voxel_resolution)
            tsdf.integrate(c_np * 255, d_np, K_np, cam_pose)
            tsdf_grid = torch.from_numpy(tsdf.get_tsdf_volume())
            tsdf_grid_batch.append(tsdf_grid)
            tsdf_bounds_batch.append(torch.from_numpy(vol_bnds[0]))
            if verbose:
                mesh = tsdf.get_mesh()
                color_grid = torch.from_numpy(tsdf.get_color_volume()) / 255.0
                mesh_batch.append(mesh)
                tsdf_color_batch.append(color_grid)
        tsdf_bounds_batch = (
            torch.stack(tsdf_bounds_batch, dim=0).to(depth.device).float()
        )
        tsdf_grid_batch = torch.stack(tsdf_grid_batch, dim=0).to(depth.device).float()
        if verbose:
            tsdf_color_batch = (
                torch.stack(tsdf_color_batch, dim=0).to(depth.device).float()
            )
            return tsdf_grid_batch, tsdf_color_batch, tsdf_bounds_batch, mesh_batch
        return tsdf_grid_batch

    def compute_context_features(
        self, color, depth, intrinsics, tsdf=None, action_features=None
    ):
        if tsdf is None:
            tsdf = self.compute_tsdf_volume(color, depth, intrinsics)

        h_in, w_in = color.shape[-2:]

        color = self.vlm_transform(color)
        color_features = self.vlm(color)  # [B, N, C]
        color_features = self.feature_pyramid(color_features) # [B, N, C]
        
        # Action grounding
        if action_features is not None:
            for i, k in enumerate(self.feature_map_pyramid_keys):
                color_feature_i = color_features[k] # 
                h, w = color_feature_i.shape[-2:]
                color_feature_i = einops.rearrange(color_feature_i, "B C H W-> B (H W) C")
                color_feature_i = self.vlm_preceiver_pyramid[i](color_feature_i, action_features[:, None])
                color_feature_i = self.vlm_proj_pyramid[i](color_feature_i)
                color_feature_i = einops.rearrange(color_feature_i, "B (H W) C -> B C H W", H=h, W=w)
                color_features[k] = color_feature_i

        color_features_pyramid = []
        for i, k in enumerate(self.feature_map_pyramid_keys):
            color_feature_i = color_features[k]
            color_feature_i = F.interpolate(
                color_feature_i, size=(h_in, w_in), mode="bilinear"
            )
            color_features_pyramid.append(color_feature_i)     
        points_map_pyramid = [tsdf] * len(color_features_pyramid) # [B, D, H, W]
        points_pe_pyramid = [self.tsdf_net(tsdf)] * len(color_features_pyramid) # [B, P, D, H, W]

        # color_features = einops.rearrange(
        #     color_features, "B (H W) C -> B C H W", H=h_out, W=w_out
        # )
        # import pdb; pdb.set_trace()
        # if self.use_feature_decoder:
        #     color_features = self.feature_decoder(color_features)
        
        # color_features = F.interpolate(
        #     color_features, size=tuple(self.input_image_shape), mode="bilinear"
        # )


        # # Compute the point features pyramid
        # for i, k in enumerate(self.feature_map_pyramid_keys):
        #     color_feature_i = color_features[k]
        #     color_feature_i = F.interpolate(
        #         color_feature_i, size=tuple(self.input_image_shape), mode="bilinear"
        #     )

        #     color_features_pyramid.append(color_feature_i)
        return color_features_pyramid, points_map_pyramid, points_pe_pyramid

    @staticmethod
    def interpolate_voxel_grid_features(voxel_grid, query_points, voxel_bounds):
        """
        Parameters
        ----------
        voxel_grid : torch.Tensor
            with shape [B, C, D, H, W]
        query_points : torch.Tensor
            _with shape [B, N, 3]
        voxel_bounds: torch.Tensor
            _with shape [B, 2]
        """
        voxel_bounds = voxel_bounds.unsqueeze(-1).repeat(1, 1, 3)  # [B, 2, 3]
        query_points = (query_points - voxel_bounds[:, 0:1]) / (
            voxel_bounds[:, 1:2] - voxel_bounds[:, 0:1]
        )
        query_grids = (
            query_points * 2 - 1
        )  # Normalize the query points from [0, 1] to [-1, 1]
        query_grids = query_grids[
            ..., [2, 1, 0]
        ]  # Convert to the voxel grid coordinate system
        query_grids = query_grids[:, :, None, None]  # [B, N, 1, 1, 3]
        query_features = F.grid_sample(
            voxel_grid, query_grids, mode="bilinear", align_corners=True
        )  # [B, C, N, 1, 1]
        query_features = query_features.squeeze(-1).squeeze(-1)  # [B, C, N]
        return query_features

    def interpolate_image_grid_features(self, image_grid, query_points, intrinsics):
        """
        Parameters
        ----------
        image_grid : torch.Tensor
            with shape [B, C, H, W]
        query_points : torch.Tensor
            _with shape [B, N, 3]
        """
        batch_size, _, height, width = image_grid.shape
        query_grids = self.project_3d(query_points, intrinsics)  # [B, 2, N]
        query_grids[:, 0] = (query_grids[:, 0] / (width - 1)) * 2 - 1
        query_grids[:, 1] = (query_grids[:, 1] / (height - 1)) * 2 - 1
        query_grids = query_grids.permute(0, 2, 1)[:, :, None]  # [B, N, 1, 2]
        query_featurs = F.grid_sample(
            image_grid, query_grids, mode="bilinear", align_corners=True
        )  # [B, C, N, 1]
        query_featurs = query_featurs.squeeze(-1)
        return query_featurs

    def forward(
        self,
        color_features_pyramid,
        points_map_pyramid,
        points_pe_pyramid,
        query_points,
        intrinsics,
        voxel_bounds,
        **kwargs,
    ):
        """_summary_

        Parameters
        ----------
        color_features_pyramid :  list of torch.Tensor
            with shape [[B, C, H, W]]
        points_map_pyramid : list of torch.Tensor for TSDF volume
            [[B, D, H, W]]
        points_pe_pyramid :  list of torch.Tensor for TSDF volume feature
            [[B, P, D, H, W]]
        query_points : query points
            [B, N, 3]
        intrinsics : torch.Tensor or np.ndarray
            [3, 3]
        voxel_bounds : _type_
            [B, 2]

        Returns
        -------
        torch.Tensor
            shape of [B, N, C*4]
        """
        assert len(color_features_pyramid) == len(points_map_pyramid)
        assert len(color_features_pyramid) == len(points_pe_pyramid)
        batch_size, num_query_points, _ = query_points.shape
        features = []

        for i in range(len(color_features_pyramid)):
            # Re-arrange to feature maps
            color_feature_i = color_features_pyramid[i]  # [B, C, H, W]
            points_pe_i = points_pe_pyramid[i]  # [B, P, D, H, W]
            points_map_i = points_map_pyramid[i][:, None]  # [B, 1, D, H, W]

            if i == 0:
                # Interpolate the voxel grid features
                feat_occ = self.interpolate_voxel_grid_features(
                    points_map_i, query_points, voxel_bounds
                )

                # Interpolate the voxel grid features
                feat_3d = self.interpolate_voxel_grid_features(
                    points_pe_i, query_points, voxel_bounds
                )
                features.append(feat_3d)  # [B, C, N]

            # Interpolate the 2D feature maps
            feat_2d = self.interpolate_image_grid_features(
                color_feature_i, query_points, intrinsics
            )
            features.append(feat_2d)  # [B, C, N]
        features = torch.cat(features, dim=1).permute(0, 2, 1)  # [B, N, C*3]
        features = self.proj(features) # [B, N, C]
        return features

class TSDFMapGeometryExtractor(nn.Module):
    def __init__(
        self,
        input_image_shape,
        voxel_resolution=64,
        voxel_feature_dim=64,
    ):
        super().__init__()
        self.input_image_shape = input_image_shape
        self.voxel_resolution = voxel_resolution
        self.embedding_dim = voxel_feature_dim
        self.backproject = BackprojectDepth(input_image_shape[0], input_image_shape[1])
        self.project_3d = Project3D()

        # Load 3D Unet
        self.tsdf_net = VoxelGridEncoder(
            self.voxel_resolution, c_dim=self.embedding_dim
        )
        
    def compute_tsdf_volume(self, color, depth, intrinsics, verbose=False):
        cam_pose = np.eye(4)
        tsdf_grid_batch = []
        tsdf_bounds_batch = []
        tsdf_color_batch = []
        mesh_batch = []
        for i in range(len(depth)):
            d_np = depth[i].cpu().numpy()[0]
            c_np = (
                color[i].cpu().numpy().transpose(1, 2, 0)
            )  # [H, W, 3], requested by TSDFVolume
            K_np = intrinsics[i].cpu().numpy()
            view_frust_pts = get_view_frustum(d_np, K_np, cam_pose)
            vol_bnds = np.zeros((3, 2))
            vol_bnds[:, 0] = np.minimum(
                vol_bnds[:, 0], np.amin(view_frust_pts, axis=1)
            ).min()
            vol_bnds[:, 1] = np.maximum(
                vol_bnds[:, 1], np.amax(view_frust_pts, axis=1)
            ).max()
            tsdf = TSDFVolume(vol_bnds, voxel_dim=self.voxel_resolution)
            tsdf.integrate(c_np * 255, d_np, K_np, cam_pose)
            tsdf_grid = torch.from_numpy(tsdf.get_tsdf_volume())
            tsdf_grid_batch.append(tsdf_grid)
            tsdf_bounds_batch.append(torch.from_numpy(vol_bnds[0]))
            if verbose:
                mesh = tsdf.get_mesh()
                color_grid = torch.from_numpy(tsdf.get_color_volume()) / 255.0
                mesh_batch.append(mesh)
                tsdf_color_batch.append(color_grid)
        tsdf_bounds_batch = (
            torch.stack(tsdf_bounds_batch, dim=0).to(depth.device).float()
        )
        tsdf_grid_batch = torch.stack(tsdf_grid_batch, dim=0).to(depth.device).float()
        if verbose:
            tsdf_color_batch = (
                torch.stack(tsdf_color_batch, dim=0).to(depth.device).float()
            )
            return tsdf_grid_batch, tsdf_color_batch, tsdf_bounds_batch, mesh_batch
        return tsdf_grid_batch

    def compute_context_features(
        self, color, depth, intrinsics, tsdf=None, action_featurs=None
    ):
        if tsdf is None:
            tsdf = self.compute_tsdf_volume(color, depth, intrinsics)


        color_features_pyramid = [None]  # [B, C, H, W]
        points_map_pyramid = [tsdf]  # [B, D, H, W]
        points_pe_pyramid = [self.tsdf_net(tsdf)]  # [B, P, D, H, W]

        return color_features_pyramid, points_map_pyramid, points_pe_pyramid

    @staticmethod
    def interpolate_voxel_grid_features(voxel_grid, query_points, voxel_bounds):
        """
        Parameters
        ----------
        voxel_grid : torch.Tensor
            with shape [B, C, D, H, W]
        query_points : torch.Tensor
            _with shape [B, N, 3]
        voxel_bounds: torch.Tensor
            _with shape [B, 2]
        """
        voxel_bounds = voxel_bounds.unsqueeze(-1).repeat(1, 1, 3)  # [B, 2, 3]
        query_points = (query_points - voxel_bounds[:, 0:1]) / (
            voxel_bounds[:, 1:2] - voxel_bounds[:, 0:1]
        )
        query_grids = (
            query_points * 2 - 1
        )  # Normalize the query points from [0, 1] to [-1, 1]
        query_grids = query_grids[
            ..., [2, 1, 0]
        ]  # Convert to the voxel grid coordinate system
        query_grids = query_grids[:, :, None, None]  # [B, N, 1, 1, 3]
        query_features = F.grid_sample(
            voxel_grid, query_grids, mode="bilinear", align_corners=True
        )  # [B, C, N, 1, 1]
        query_features = query_features.squeeze(-1).squeeze(-1)  # [B, C, N]
        return query_features

    def interpolate_image_grid_features(self, image_grid, query_points, intrinsics):
        """
        Parameters
        ----------
        image_grid : torch.Tensor
            with shape [B, C, H, W]
        query_points : torch.Tensor
            _with shape [B, N, 3]
        """
        batch_size, _, height, width = image_grid.shape
        query_grids = self.project_3d(query_points, intrinsics)  # [B, 2, N]
        query_grids[:, 0] = (query_grids[:, 0] / (width - 1)) * 2 - 1
        query_grids[:, 1] = (query_grids[:, 1] / (height - 1)) * 2 - 1
        query_grids = query_grids.permute(0, 2, 1)[:, :, None]  # [B, N, 1, 2]
        query_featurs = F.grid_sample(
            image_grid, query_grids, mode="bilinear", align_corners=True
        )  # [B, C, N, 1]
        query_featurs = query_featurs.squeeze(-1)
        return query_featurs

    def forward(
        self,
        color_features_pyramid,
        points_map_pyramid,
        points_pe_pyramid,
        query_points,
        intrinsics,
        voxel_bounds,
        **kwargs,
    ):
        """_summary_

        Parameters
        ----------
        color_features_pyramid :  list of torch.Tensor
            with shape [[B, C, H, W]]
        points_map_pyramid : list of torch.Tensor for TSDF volume
            [[B, D, H, W]]
        points_pe_pyramid :  list of torch.Tensor for TSDF volume feature
            [[B, P, D, H, W]]
        query_points : query points
            [B, N, 3]
        intrinsics : torch.Tensor or np.ndarray
            [3, 3]
        voxel_bounds : _type_
            [B, 2]

        Returns
        -------
        torch.Tensor
            shape of [B, N, C*4]
        """
        assert len(color_features_pyramid) == len(points_map_pyramid)
        assert len(color_features_pyramid) == len(points_pe_pyramid)
        batch_size, num_query_points, _ = query_points.shape
        features = []

        for i in range(len(color_features_pyramid)):
            # Re-arrange to feature maps
            color_feature_i = color_features_pyramid[i]  # [B, C, H, W]
            points_pe_i = points_pe_pyramid[i]  # [B, P, D, H, W]
            points_map_i = points_map_pyramid[i][:, None]  # [B, 1, D, H, W]

            if i == 0:
                # Interpolate the voxel grid features
                feat_occ = self.interpolate_voxel_grid_features(
                    points_map_i, query_points, voxel_bounds
                )

                # Interpolate the voxel grid features
                feat_3d = self.interpolate_voxel_grid_features(
                    points_pe_i, query_points, voxel_bounds
                )
                features.append(feat_3d)  # [B, C, N]

            # # Interpolate the 2D feature maps
            # feat_2d = self.interpolate_image_grid_features(
            #     color_feature_i, query_points, intrinsics
            # )
            # features.append(feat_2d)  # [B, C, N]
        features = torch.cat(features, dim=1).permute(0, 2, 1)  # [B, N, C*3]
        return features
 