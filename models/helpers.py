import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from numba import njit, prange
from skimage import measure
import open3d as o3d
import time
import torch
from math import ceil
from models.clip import clip, tokenize


def compute_null_text_embeddings(vlm, batch_size=1, device="cuda"):
    action_tokens_null = tokenize("")
    action_tokens_null = action_tokens_null.repeat(batch_size, 1)
    action_tokens_null = action_tokens_null.to(device)
    action_feature_null = vlm.encode_text(action_tokens_null).float()
    return action_feature_null


def fourier_positional_encoding(input, L):  # [B,...,C]
    shape = input.shape
    freq = 2 ** torch.arange(L, dtype=torch.float32, device=input.device) * np.pi  # [L]
    spectrum = input[..., None] * freq  # [B,...,C,L]
    sin, cos = spectrum.sin(), spectrum.cos()  # [B,...,C,L]
    input_enc = torch.stack([sin, cos], dim=-2)  # [B,...,C,2,L]
    input_enc = input_enc.view(*shape[:-1], -1)  # [B,...,2CL]
    return input_enc


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def round_up_multiple(num, mult):
    return ceil(num / mult) * mult


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def cosine_beta_schedule(timesteps, s=0.008, dtype=torch.float32):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas_clipped = np.clip(betas, a_min=0, a_max=0.999)
    return torch.tensor(betas_clipped, dtype=dtype)


# -----------------------------------------------------------------------------#
# ---------------------------------- losses -----------------------------------#
# -----------------------------------------------------------------------------#


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 0, size_average: bool = True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, input, target):
        logpt = F.log_softmax(input, dim=-1)
        logpt = logpt.gather(1, target.view(-1, 1)).view(-1)
        pt = logpt.exp()

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class WeightedLoss(nn.Module):

    def __init__(self, weights):
        super().__init__()
        self.register_buffer("weights", weights)

    def forward(self, pred, targ):
        """
        pred, targ : tensor
            [ batch_size x horizon x transition_dim ]
        """
        loss = self._loss(pred, targ)
        weighted_loss = (loss * self.weights).mean()
        return weighted_loss


class WeightedL1(WeightedLoss):

    def _loss(self, pred, targ):
        return torch.abs(pred - targ)


class WeightedL2(WeightedLoss):

    def _loss(self, pred, targ):
        return F.mse_loss(pred, targ, reduction="none")


Losses = {
    "l1": WeightedL1,
    "l2": WeightedL2,
}


class EMA:
    """
    empirical moving average
    """

    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        with torch.no_grad():
            ema_state_dict = ma_model.state_dict()
            for key, value in current_model.state_dict().items():
                ema_value = ema_state_dict[key]
                ema_value.copy_(self.beta * ema_value + (1.0 - self.beta) * value)


# -----------------------------------------------------------------------------#
# ---------------------------------- TSDF -----------------------------------#
# -----------------------------------------------------------------------------#


def get_view_frustum(depth_im, cam_intr, cam_pose):
    """Get corners of 3D camera view frustum of depth image"""
    im_h = depth_im.shape[0]
    im_w = depth_im.shape[1]
    max_depth = np.max(depth_im)
    view_frust_pts = np.array(
        [
            (np.array([0, 0, 0, im_w, im_w]) - cam_intr[0, 2])
            * np.array([0, max_depth, max_depth, max_depth, max_depth])
            / cam_intr[0, 0],
            (np.array([0, 0, im_h, 0, im_h]) - cam_intr[1, 2])
            * np.array([0, max_depth, max_depth, max_depth, max_depth])
            / cam_intr[1, 1],
            np.array([0, max_depth, max_depth, max_depth, max_depth]),
        ]
    )
    view_frust_pts = view_frust_pts.T @ cam_pose[:3, :3].T + cam_pose[:3, 3]
    return view_frust_pts.T


# try:
#     import pycuda.driver as cuda
#     import pycuda.autoinit
#     from pycuda.compiler import SourceModule

#     FUSION_GPU_MODE = 1
# except Exception as err:
#     print("Warning: {}".format(err))
#     print("Failed to import PyCUDA. Running fusion in CPU mode.")
#     FUSION_GPU_MODE = 0


class TSDFVolume:
    """
    Volumetric with TSDF representation
    """

    def __init__(
        self,
        vol_bounds: np.ndarray,
        voxel_dim: float,
        use_gpu: bool = False,
        verbose: bool = False,
        num_margin: float = 5.0,
        enable_color=True,
        unknown_free=True,
    ):
        """
        Constructor

        :param vol_bounds: An ndarray is shape (3,2), define the min & max bounds of voxels.
        :param voxel_size: Voxel size in meters.
        :param use_gpu: Use GPU for voxel update.
        :param verbose: Print verbose message or not.
        """

        vol_bounds = np.asarray(vol_bounds)
        assert vol_bounds.shape == (3, 2), "vol_bounds should be of shape (3,2)"

        self._verbose = verbose
        self._use_gpu = use_gpu
        self._vol_bounds = vol_bounds

        self._vol_dim = [voxel_dim, voxel_dim, voxel_dim]
        self._vox_size = float(
            (self._vol_bounds[0, 1] - self._vol_bounds[0, 0]) / voxel_dim
        )
        self._trunc_margin = num_margin * self._vox_size  # truncation on SDF

        # Check GPU
        if self._use_gpu:
            if torch.cuda.is_available():
                if self._verbose:
                    print("# Using GPU mode")
                self._device = torch.device("cuda:0")
            else:
                if self._verbose:
                    print("# Not available CUDA device, using CPU mode")
                self._device = torch.device("cpu")
        else:
            if self._verbose:
                print("# Using CPU mode")
            self._device = torch.device("cpu")

        # Coordinate origin of the volume, set as the min value of volume bounds
        self._vol_origin = torch.tensor(
            self._vol_bounds[:, 0].copy(order="C"), device=self._device
        ).float()

        # Grid coordinates of voxels
        xx, yy, zz = torch.meshgrid(
            torch.arange(self._vol_dim[0]),
            torch.arange(self._vol_dim[1]),
            torch.arange(self._vol_dim[2]),
            indexing="ij",
        )
        self._vox_coords = (
            torch.cat([xx.reshape(1, -1), yy.reshape(1, -1), zz.reshape(1, -1)], dim=0)
            .int()
            .T
        )
        if self._use_gpu:
            self._vox_coords = self._vox_coords.cuda()

        # World coordinates of voxel centers
        self._world_coords = self.vox2world(
            self._vol_origin, self._vox_coords, self._vox_size
        )
        self.enable_color = enable_color

        # TSDF & weights
        self._tsdf_vol = torch.ones(
            size=self._vol_dim, device=self._device, dtype=torch.float32
        )
        self._weight_vol = torch.zeros(
            size=self._vol_dim, device=self._device, dtype=torch.float32
        )
        if self.enable_color:
            self._color_vol = torch.zeros(
                size=[*self._vol_dim, 3], device=self._device, dtype=torch.float32
            )

        # Mesh paramters
        self._mesh = o3d.geometry.TriangleMesh()
        self.unknown_free = unknown_free
    @staticmethod
    def vox2world(vol_origin: torch.Tensor, vox_coords: torch.Tensor, vox_size):
        """
        Converts voxel grid coordinates to world coordinates

        :param vol_origin: Origin of the volume in world coordinates, (3,1).
        :parma vol_coords: List of all grid coordinates in the volume, (N,3).
        :param vol_size: Size of volume.
        :retrun: Grid points under world coordinates. Tensor with shape (N, 3)
        """

        cam_pts = torch.empty_like(vox_coords, dtype=torch.float32)
        cam_pts = vol_origin + (vox_size * vox_coords)

        return cam_pts

    @staticmethod
    def cam2pix(cam_pts: torch.Tensor, intrinsics: torch.Tensor):
        """
        Convert points in camera coordinate to pixel coordinates

        :param cam_pts: Points in camera coordinates, (N,3).
        :param intrinsics: Vamera intrinsics, (3,3).
        :return: Pixel coordinate (u,v) cooresponding to input points. Tensor with shape (N, 2).
        """

        cam_pts_z = cam_pts[:, 2].repeat(3, 1).T
        pix = torch.round((cam_pts @ intrinsics.T) / cam_pts_z)

        return pix

    @staticmethod
    def ridgid_transform(points: torch.Tensor, transform: torch.Tensor):
        """
        Apply rigid transform (4,4) on points

        :param points: Points, shape (N,3).
        :param transform: Tranform matrix, shape (4,4).
        :return: Points after transform.
        """

        points_h = torch.cat(
            [points, torch.ones((points.shape[0], 1), device=points.device)], 1
        )
        points_h = (transform @ points_h.T).T

        return points_h[:, :3]

    def get_tsdf_volume(self):
        return self._tsdf_vol.cpu().numpy()

    def get_color_volume(self):
        return self._color_vol.permute(3, 0, 1, 2).cpu().numpy()

    def get_mesh(self):
        """
        Get mesh.
        """
        return self._mesh

    def integrate(self, color_img, depth_img, intrinsic, cam_pose, weight: float = 1.0):
        """
        Integrate an depth image to the TSDF volume

        :param depth_img: depth image with depth value in meter.
        :param intrinsics: camera intrinsics of shape (3,3).
        :param cam_pose: camera pose, transform matrix of shape (4,4)
        :param weight: weight assign for current frame, higher value indicate higher confidence
        """

        time_begin = time.time()
        img_h, img_w = depth_img.shape
        depth_img = torch.from_numpy(depth_img).float().to(self._device)
        color_img = torch.from_numpy(color_img).float().to(self._device)  # [H, W, 3]
        cam_pose = torch.from_numpy(cam_pose).float().to(self._device)
        intrinsic = torch.from_numpy(intrinsic).float().to(self._device)

        # TODO:
        # Better way to select valid voxels.
        # - Current:
        #   -> Back project all voxels to frame pixels according to current camera pose.
        #   -> Select valid pixels within frame size.
        # - Possible:
        #   -> Project pixel to voxel coordinates
        #   -> hash voxel coordinates
        #   -> dynamically allocate voxel chunks

        # Get the world coordinates of all voxels
        # world_points = geometry.vox2world(self._vol_origin, self._vox_coords, self._vox_size)

        # Get voxel centers under camera coordinates
        world_points = self.ridgid_transform(
            self._world_coords, cam_pose.inverse()
        )  # [N^3, 3]

        # Get the pixel coordinates (u,v) of all voxels under current camere pose
        # Multiple voxels can be projected to a same (u,v)
        voxel_uv = self.cam2pix(world_points, intrinsic)  # [N^3, 3]
        voxel_u, voxel_v = voxel_uv[:, 0], voxel_uv[:, 1]  # [N^3], [N^3]
        voxel_z = world_points[:, 2]

        # Filter out voxels points that visible in current frame
        pixel_mask = torch.logical_and(
            voxel_u >= 0,
            torch.logical_and(
                voxel_u < img_w,
                torch.logical_and(
                    voxel_v >= 0, torch.logical_and(voxel_v < img_h, voxel_z > 0)
                ),
            ),
        )

        # Get depth value
        depth_value = torch.zeros(voxel_u.shape, device=self._device)
        depth_value[pixel_mask] = depth_img[
            voxel_v[pixel_mask].long(), voxel_u[pixel_mask].long()
        ]

        # Compute and Integrate TSDF
        sdf_value = depth_value - voxel_z  # Compute SDF
        if self.unknown_free:
            voxel_mask = torch.logical_and(
                depth_value > 0, sdf_value >= -self._trunc_margin
            )  # Truncate SDF
        else:
            voxel_mask = depth_value > 0  # Truncate SDF
            
        tsdf_value = torch.minimum(
            torch.ones_like(sdf_value, device=self._device),
            sdf_value / self._trunc_margin,
        )
        tsdf_value = tsdf_value[voxel_mask]
        # Get coordinates of valid voxels with valid TSDF value
        valid_vox_x = self._vox_coords[voxel_mask, 0].long()
        valid_vox_y = self._vox_coords[voxel_mask, 1].long()
        valid_vox_z = self._vox_coords[voxel_mask, 2].long()

        # Update TSDF of cooresponding voxels
        weight_old = self._weight_vol[valid_vox_x, valid_vox_y, valid_vox_z]
        tsdf_old = self._tsdf_vol[valid_vox_x, valid_vox_y, valid_vox_z]

        if self.enable_color:
            color_value = torch.zeros([voxel_u.shape[0], 3], device=self._device)
            color_value[pixel_mask] = color_img[
                voxel_v[pixel_mask].long(), voxel_u[pixel_mask].long(), :
            ]
            color_value = color_value[voxel_mask]
            color_old = self._color_vol[valid_vox_x, valid_vox_y, valid_vox_z]

        else:
            color_value = None
            color_old = None
        tsdf_new, color_new, weight_new = self.update_tsdf(
            tsdf_old, tsdf_value, color_old, color_value, weight_old, weight
        )

        self._tsdf_vol[valid_vox_x, valid_vox_y, valid_vox_z] = tsdf_new
        self._weight_vol[valid_vox_x, valid_vox_y, valid_vox_z] = weight_new

        if self.enable_color:
            self._color_vol[valid_vox_x, valid_vox_y, valid_vox_z] = color_new

        if self._verbose:
            print("# Update {} voxels.".format(len(tsdf_new)))
            print(
                "# Integration Timing: {:.5f} (second).".format(
                    time.time() - time_begin
                )
            )

    def get_mesh(self):
        """
        Extract mesh from current TSDF volume.
        """

        time_begin = time.time()

        if self._use_gpu:
            tsdf_vol = self._tsdf_vol.cpu().numpy()
            vol_origin = self._vol_origin.cpu().numpy()
            if self.enable_color:
                color_vol = self._color_vol.cpu().numpy() / 255

        else:
            tsdf_vol = self._tsdf_vol.numpy()
            vol_origin = self._vol_origin.numpy()
            if self.enable_color:
                color_vol = self._color_vol.numpy() / 255

        _vertices, triangles, _, _ = measure.marching_cubes(-tsdf_vol, 0)
        vertices_sample = (_vertices / self._vol_dim[0] - 0.5) * 2

        # interpolation to get colors
        vertices_pt = torch.from_numpy(vertices_sample).float()[
            None, None, None, :, [2, 1, 0]
        ]  # [1, 1, 1, N, 3]

        # mesh_vertices = _vertices / self._vol_dim[0] + self._vol_origin.cpu().numpy()
        mesh_vertices = vertices_sample
        self._mesh.vertices = o3d.utility.Vector3dVector(mesh_vertices.astype(float))
        self._mesh.triangles = o3d.utility.Vector3iVector(triangles.astype(np.int32))
        if self.enable_color:
            color_vol_pt = (
                torch.from_numpy(color_vol).float().permute(3, 0, 1, 2)[None]
            )  # [1, 3, H, W, D]
            vert_colors = torch.nn.functional.grid_sample(
                color_vol_pt, vertices_pt, align_corners=True
            )  # [1, 3, 1, 1, N]
            vert_colors = vert_colors.squeeze().cpu().numpy().T
            self._mesh.vertex_colors = o3d.utility.Vector3dVector(
                vert_colors.astype(float)
            )

        self._mesh.compute_vertex_normals()
        if self._verbose:
            print("# Extracting Mesh: {} Vertices".format(mesh_vertices.shape[0]))
            print("# Meshing Timing: {:.5f} (second).".format(time.time() - time_begin))
        return self._mesh

    def update_tsdf(
        self, tsdf_old, tsdf_new, color_old, color_new, weight_old, obs_weight
    ):
        """
        Update the TSDF value of given voxel
        V = (wv + WV) / w + W

        :param tsdf_old: Old TSDF values.
        :param tsdf_new: New TSDF values.
        :param weight_old: Voxels weights.
        :param obs_weight: Weight of current update.
        :return: Updated TSDF values & Updated weights.
        """

        tsdf_vol_int = torch.empty_like(
            tsdf_old, dtype=torch.float32, device=self._device
        )
        weight_new = torch.empty_like(
            weight_old, dtype=torch.float32, device=self._device
        )

        weight_new = weight_old + obs_weight
        tsdf_vol_int = (weight_old * tsdf_old + obs_weight * tsdf_new) / weight_new
        if color_old is not None:
            color_vol_int = (
                weight_old[:, None] * color_old + obs_weight * color_new
            ) / weight_new[:, None]
            return tsdf_vol_int, color_vol_int, weight_new
        else:
            color_vol_int = None

        return tsdf_vol_int, color_vol_int, weight_new


class TSDFVolume2(TSDFVolume):
    """
    Volumetric with TSDF representation
    """

    def __init__(
        self,
        vol_bounds: np.ndarray,
        voxel_size: float,
        use_gpu: bool = False,
        verbose: bool = False,
        num_margin: float = 5.0,
        enable_color=True,
    ):
        """
        Constructor

        :param vol_bounds: An ndarray is shape (3,2), define the min & max bounds of voxels.
        :param voxel_size: Voxel size in meters.
        :param use_gpu: Use GPU for voxel update.
        :param verbose: Print verbose message or not.
        """

        vol_bounds = np.asarray(vol_bounds)
        assert vol_bounds.shape == (3, 2), "vol_bounds should be of shape (3,2)"

        self._verbose = verbose
        self._use_gpu = use_gpu
        self._vol_bounds = vol_bounds
        if self._use_gpu:
            if torch.cuda.is_available():
                if self._verbose:
                    print("# Using GPU mode")
                self._device = torch.device("cuda:0")
            else:
                if self._verbose:
                    print("# Not available CUDA device, using CPU mode")
                self._device = torch.device("cpu")
        else:
            if self._verbose:
                print("# Using CPU mode")
            self._device = torch.device("cpu")
        self._voxel_size = float(voxel_size)
        self._trunc_margin = 5 * self._voxel_size  # truncation on SDF
        self._vox_size = float(voxel_size)
        self._trunc_margin = num_margin * self._vox_size  # truncation on SDF
        # Adjust volume bounds and ensure C-order contiguous
        self._vol_dim = (
            np.ceil(
                (self._vol_bounds[:, 1] - self._vol_bounds[:, 0]) / self._voxel_size
            )
            .copy(order="C")
            .astype(int)
        )

        self._vol_bounds[:, 1] = (
            self._vol_bounds[:, 0] + self._vol_dim * self._voxel_size
        )
        self._vol_dim = self._vol_dim.tolist()
        # self._vol_origin = self._vol_bounds[:, 0].copy(order="C").astype(np.float32)
        # self._vol_dim = torch.tensor(self._vol_dim, device=self._device).int()
        self._vol_origin = torch.tensor(
            self._vol_bounds[:, 0].copy(order="C"), device=self._device
        ).float()
        # Grid coordinates of voxels
        xx, yy, zz = torch.meshgrid(
            torch.arange(self._vol_dim[0]),
            torch.arange(self._vol_dim[1]),
            torch.arange(self._vol_dim[2]),
            indexing="ij",
        )
        self._vox_coords = (
            torch.cat([xx.reshape(1, -1), yy.reshape(1, -1), zz.reshape(1, -1)], dim=0)
            .int()
            .T
        )
        if self._use_gpu:
            self._vox_coords = self._vox_coords.cuda()

        # World coordinates of voxel centers
        self._world_coords = self.vox2world(
            self._vol_origin, self._vox_coords, self._vox_size
        )
        self.enable_color = enable_color

        # TSDF & weights
        self._tsdf_vol = torch.ones(
            size=self._vol_dim, device=self._device, dtype=torch.float32
        )
        self._weight_vol = torch.zeros(
            size=self._vol_dim, device=self._device, dtype=torch.float32
        )
        if self.enable_color:
            self._color_vol = torch.zeros(
                size=[*self._vol_dim, 3], device=self._device, dtype=torch.float32
            )

        # Mesh paramters
        self._mesh = o3d.geometry.TriangleMesh()
