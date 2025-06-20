import torch
import torch.nn.functional as F
from scipy.signal import savgol_filter
from models.layers_2d import BackprojectDepth, Project3D
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_rotation_6d
import numpy as np
import cv2

class TrajectoryOptimizer:
    def __init__(
        self,
        resolution=(256, 456),
        lr_scale_global=0.05,
        lr_scale=0.1,
        lr_pose=0.05,
        num_iters_scale=10,
        num_iters_pose=100,
        warp_mode="points",
        device="cuda",
    ):
        self.height, self.width = resolution[0], resolution[1]
        self.device = device
        self.lr_scale_global = lr_scale_global
        self.lr_scale = lr_scale
        self.lr_pose = lr_pose
        self.num_iters_scale = num_iters_scale
        self.num_iters_pose = num_iters_pose
        self.backproject_depth = BackprojectDepth(self.height, self.width).to(device)
        self.project_3d = Project3D().to(device)
        self.warp_mode = warp_mode

    def compute_warped_results(
        self,
        intrinsics,
        rgb_tensors,
        depth_tensors,
        mask_tensors,
        scale_tensors,
        rgb_key,
        depth_key,
        mask_key,
        scake_key,
        T_kc_tensors,
        mode,
        return_color=False,
        verbose=False,
    ):
        N, _, height, width = depth_tensors.shape
        depth_key = depth_key * scake_key[:, None, None, None]  # To Colmap space
        depth_key = depth_key.repeat(N, 1, 1, 1)
        mask_key = mask_key.repeat(N, 1, 1, 1)
        rgb_key = rgb_key.repeat(N, 1, 1, 1)

        # Prepare the depth
        depth_tensors_tmp = (
            depth_tensors * scale_tensors[:, None, None, None]
        )  # To Colmap space

        # Compute the warping flow from i to k,
        points = self.backproject_depth(depth_tensors_tmp, K=intrinsics)  # [N, 3, H*W]
        points = points.permute(0, 2, 1)  # [N, H*W, 3]
        pix_coords = self.project_3d(
            points, K=intrinsics, T=T_kc_tensors
        )  # [N, 2, H*W]
        
        # Acquire the backward pixel flow
        pix_coords[:, 0] = (pix_coords[:, 0] / (width - 1)) * 2 - 1
        pix_coords[:, 1] = (pix_coords[:, 1] / (height - 1)) * 2 - 1
        pix_coords = pix_coords.view(-1, 2, height, width)  # [N, 2, H, W]
        pix_coords = pix_coords.permute(0, 2, 3, 1)  # [N, H, W, 2]

        warped_depths_key = F.grid_sample(
            depth_key,
            pix_coords,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        ) # In frame c
        
        warped_masks_key = F.grid_sample(
            mask_key,
            pix_coords,
            mode="nearest",
            padding_mode="border",
            align_corners=True,
        ) # In frame c
        
        if mode == "points":
            # Warp the points
            points_c = points # In frame c
            points_c = points_c @ T_kc_tensors[:, :3, :3].transpose(
                -1, -2
            ) + T_kc_tensors[:, :3, 3].unsqueeze(-2) # In frame k
            points_k = self.backproject_depth(depth_key, K=intrinsics) # In frame k
            points_k = points_k.view(-1, 3, height, width) # [N, 3, H, W]
            points_k_to_c = F.grid_sample(
                points_k,
                pix_coords,
                mode="nearest",
                padding_mode="border",
                align_corners=True,
            )
            points_k_to_c = points_k_to_c.view(-1, 3, height, width) # [N, 3, H, W]
            points_k_to_c = points_k_to_c.permute(0, 2, 3, 1) # [N, H, W, 3]
            points_k_to_c = points_k_to_c.view(N, -1, 3) # [N, H*W, 3]
            points_k_to_c = points_k_to_c.clone().detach()
            
        # Warp the depth
        elif mode == "depth":
            points_k_to_c = self.backproject_depth(
                warped_depths_key, K=intrinsics
            )  # In frame c
            points_k_to_c = points_k_to_c.permute(0, 2, 1)  # In frame c
            points_c = points # In frame c
            points_c = points_c @ T_kc_tensors[:, :3, :3].transpose(
                -1, -2
            ) + T_kc_tensors[:, :3, 3].unsqueeze(-2) # In frame k
            points_k_to_c = points_k_to_c @ T_kc_tensors[:, :3, :3].transpose(
                -1, -2
            ) + T_kc_tensors[:, :3, 3].unsqueeze(-2) # In frame k

        else:
            raise ValueError("Invalid mode: {}".format(mode))

        if return_color:
            warped_rgbs_key = F.grid_sample(
                rgb_key,
                pix_coords,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )
            if verbose:
                for bi in range(len(rgb_key)):
                    warped_rgb_key_for_i = warped_rgbs_key[bi].detach().cpu()
                    rgb_i = rgb_tensors[bi].detach().cpu()
                    rgb_vis_i = torch.cat([rgb_i, warped_rgb_key_for_i], dim=-1)
                    rgb_vis_i = rgb_vis_i.permute(1, 2, 0).cpu().numpy()
                    rgb_vis_i = (rgb_vis_i * 255).astype(np.uint8)
                    rgb_vis_i = rgb_vis_i[..., ::-1]
                    cv2.imshow("rgb_{}".format(bi), rgb_vis_i)
                    cv2.waitKey(0)

            return (
                warped_rgbs_key,
                points_c,
                points_k_to_c,
                mask_tensors,
                warped_masks_key,
            )

        return (
            None,
            points_c,
            points_k_to_c,
            mask_tensors,
            warped_masks_key,
        )

    def optimize_pose(
        self,
        intr,
        rgb_tensors,
        depth_tensors,
        mask_tensors,
        scale_init_tensors,
        scale_global,
        colmap_results,
        key_idx=0,
        depth_filter_thresh=1.25,
        optimize_pose=True,
        verbose=False,
    ):
        # Prepare the tensors
        T_wc_tensors = []
        frame_ids = list(colmap_results.keys())

        for ii, fi in enumerate(frame_ids):
            T_world_ci = np.array(colmap_results[str(fi)]["T_wc"]).reshape(4, 4)
            T_wc_tensors.append(torch.from_numpy(T_world_ci).float())
        T_wc_tensors = torch.stack(T_wc_tensors).to(self.device)  # [N, 4, 4]
        key_rgb = rgb_tensors[key_idx][None]  # [1, 3, H, W]
        key_depth = depth_tensors[key_idx][None]  # [1, 1, H, W]
        key_mask = mask_tensors[key_idx][None]  # [1, 1, H, W]
        key_scale = scale_global.clone()[None]  # [1]
        T_wk = T_wc_tensors[key_idx][None]  # [1, 4, 4]
        T_kc_tensors = torch.matmul(torch.inverse(T_wk), T_wc_tensors)  # [N, 4, 4]

        # Prepare the optimization
        scale_tensors_global = torch.ones_like(scale_init_tensors) * key_scale
        delta_scale = scale_init_tensors / scale_tensors_global
        delta_translation = torch.zeros_like(T_kc_tensors[:, :3, 3]).float()
        delta_r6d = matrix_to_rotation_6d(
            torch.eye(3, device=self.device).float()
        )[None].repeat(len(T_kc_tensors), 1)
    
        delta_scale.requires_grad = True
        delta_translation.requires_grad = optimize_pose
        delta_r6d.requires_grad = optimize_pose

        optimizer = torch.optim.Adam([delta_scale], self.lr_scale, betas=(0.9, 0.9))
        if optimize_pose:
            optimizer.add_param_group(
                {"params": delta_translation, "lr": self.lr_pose, "betas": (0.9, 0.9)}
            )
            optimizer.add_param_group(
                {"params": delta_r6d, "lr": self.lr_pose, "betas": (0.9, 0.9)}
            )

        for it in range(self.num_iters_pose):
            optimizer.zero_grad()
            height, width = rgb_tensors.shape[-2:]
            scale_curr = scale_tensors_global * delta_scale
            T_kc_tensors_curr = T_kc_tensors.clone()
            delta_rot = rotation_6d_to_matrix(delta_r6d).transpose(-1, -2)
            delta_T = (
                torch.eye(4, device=self.device)
                .float()[None]
                .repeat(len(T_kc_tensors), 1, 1)
            )
            delta_T[..., :3, :3] = delta_rot
            delta_T[..., :3, 3] = delta_translation
            T_kc_tensors_curr = torch.matmul(
                delta_T,
                T_kc_tensors_curr,
            )
            
            _, points_c, points_k_to_c, masks_c, warped_masks_key = (
                self.compute_warped_results(
                    intr,
                    rgb_tensors,
                    depth_tensors,
                    mask_tensors,
                    scale_curr,
                    key_rgb,
                    key_depth,
                    key_mask,
                    key_scale,
                    T_kc_tensors_curr,
                    mode=self.warp_mode,
                    verbose=verbose,
                    return_color=verbose,
                )
            )
            points_c = points_c.view(-1, height, width, 3).permute(0, 3, 1, 2)
            points_k_to_c = points_k_to_c.view(-1, height, width, 3).permute(
                0, 3, 1, 2
            )
            points_loss = F.mse_loss(
                points_c, points_k_to_c, reduction="none"
            )  # [N, 3, H, W]

            masks_static = masks_c * warped_masks_key

            points_loss = torch.cat(
                [points_loss[:key_idx], points_loss[key_idx + 1 :]], dim=0
            )
            masks_static = torch.cat(
                [masks_static[:key_idx], masks_static[key_idx + 1 :]], dim=0
            )
            depth_filter = torch.cat(
                [depth_tensors[:key_idx], depth_tensors[key_idx + 1 :]], dim=0
            ).repeat(1, 3, 1, 1)
            depth_filter = depth_filter < depth_filter_thresh
            points_loss = points_loss * masks_static * depth_filter  # [N, 3, H, W]

            loss_geo = points_loss.mean()
            loss_scale_reg = F.l1_loss(delta_scale, torch.ones_like(delta_scale)) * 10
            loss_translation_reg = F.l1_loss(
                delta_translation, torch.zeros_like(delta_translation)
            )
            loss_rot_reg = F.l1_loss(
                delta_r6d,
                matrix_to_rotation_6d(torch.eye(3, device=self.device).float())[None].repeat(len(T_kc_tensors), 1)
            )
            loss_reg = loss_scale_reg + loss_translation_reg + loss_rot_reg
            loss = loss_geo + loss_reg
            
            # Compute the loss and backprop
            loss.backward()
            optimizer.step()
        
        T_kc_final = T_kc_tensors.clone()
        delta_rot = rotation_6d_to_matrix(delta_r6d).transpose(-1, -2)
        delta_T = (
            torch.eye(4, device=self.device)
            .float()[None]
            .repeat(len(T_kc_tensors), 1, 1)
        )
        delta_T[..., :3, :3] = delta_rot
        delta_T[..., :3, 3] = delta_translation
        T_kc_final = torch.matmul(
            delta_T,
            T_kc_tensors_curr,
        )
        scale_final = scale_tensors_global * delta_scale
        return T_kc_final, scale_final

    def optimize_global_scale(
        self,
        rgb_tensors,
        depth_tensors,
        mask_tensors,
        colmap_results,
    ):
        # Prepare the results from colmap
        scale_init_tensors = []
        metric_d_tensors, colmap_d_tensors, valid_d_tensors = [], [], []
        frame_ids = list(colmap_results.keys())
        frame_id_start = int(frame_ids[0])
        for ii, fi in enumerate(frame_ids):
            depth = depth_tensors[ii, 0].cpu().numpy()
            mask = mask_tensors[ii, 0].cpu().numpy()
            uv = np.array(colmap_results[str(fi)]["uv"]).reshape(-1, 2)
            colmap_d = np.array(colmap_results[str(fi)]["d"])
            uv_mask = np.logical_and(
                np.logical_and(uv[:, 0] >= 0, uv[:, 0] < depth.shape[1]),
                np.logical_and(uv[:, 1] >= 0, uv[:, 1] < depth.shape[0]),
            )
            uv = uv[uv_mask]
            colmap_d = colmap_d[uv_mask]
            metric_d = depth[uv[:, 1], uv[:, 0]]  # [S]
            valid_d = mask[uv[:, 1], uv[:, 0]]
            scale_init = np.median(colmap_d) / np.median(metric_d)
            scale_init_tensors.append(scale_init)
            metric_d_tensors.append(torch.from_numpy(metric_d).float())
            colmap_d_tensors.append(torch.from_numpy(colmap_d).float())
            valid_d_tensors.append(torch.from_numpy(valid_d).float())

        # Prepare the tensors
        metric_d_tensors = torch.cat(metric_d_tensors).to(self.device)  # [S]
        colmap_d_tensors = torch.cat(colmap_d_tensors).to(self.device)  # [S]
        valid_d_tensors = torch.cat(valid_d_tensors).to(self.device)  # [S]
        scale_init_tensors = (
            torch.tensor(scale_init_tensors).float().to(self.device)
        )  # [N]

        # Start the optimization
        scale_global = torch.median(scale_init_tensors)
        delta_scale_global = torch.ones_like(scale_global)
        delta_scale_global.requires_grad = True
        optimizer_scale = torch.optim.Adam([delta_scale_global], self.lr_scale_global)

        # Do the optimization
        for it in range(self.num_iters_scale):
            optimizer_scale.zero_grad()
            scale_global_curr = scale_global * delta_scale_global
            loss_d = F.mse_loss(
                metric_d_tensors * scale_global_curr,
                colmap_d_tensors,
                reduction="none",
            )  # [S]
            
            loss_d = loss_d * valid_d_tensors
            loss_d = loss_d.sum() / valid_d_tensors.sum()
            loss_d.backward()
            optimizer_scale.step()
        scale_global_final = (scale_global * delta_scale_global).detach().clone()
        
        # Compute the amount of valid landmarks
        scale_global_np = scale_global_final.detach().cpu().numpy()
        key_idx, key_valid_diff_d = 0, -np.inf
        for ii, fi in enumerate(frame_ids):
            depth = depth_tensors[ii, 0].cpu().numpy()
            mask = mask_tensors[ii, 0].cpu().numpy()
            uv = np.array(colmap_results[str(fi)]["uv"]).reshape(-1, 2)
            colmap_d = np.array(colmap_results[str(fi)]["d"])
            uv_mask = np.logical_and(
                np.logical_and(uv[:, 0] >= 0, uv[:, 0] < depth.shape[1]),
                np.logical_and(uv[:, 1] >= 0, uv[:, 1] < depth.shape[0]),
            )
            uv = uv[uv_mask]
            colmap_d = colmap_d[uv_mask]
            metric_d = depth[uv[:, 1], uv[:, 0]]  # [S]
            valid_d = mask[uv[:, 1], uv[:, 0]] # [S]
            diff_d = np.abs(colmap_d / scale_global_np  - metric_d) # [S]
            diff_d[valid_d == 0] = np.inf
            
            # Compute the amount of valid landmarks
            valid_diff_d = (diff_d < 0.07).sum()
            if valid_diff_d > key_valid_diff_d:
                key_idx = ii
                key_valid_diff_d = valid_diff_d
        return scale_init_tensors, scale_global_final, key_idx

