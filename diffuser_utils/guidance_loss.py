import torch
from torch import nn
import torch.nn.functional as F
from models.layers_2d import Project3D, BackprojectDepth
import diffuser_utils.dataset_utils as DatasetUtils
import open3d as o3d


class Guidance:
    def __init__(self, scale=1.0, valid_horizon=-1):
        self.scale = scale
        self.valid_horizon = valid_horizon
        pass

    def descale_trajectory_length(self, traj, scale):
        """
        - traj: B x N x H x 3
        """
        traj_init = traj[..., 0:1, :]  # [B, N, 1, 3]
        traj_dist = traj - traj_init  # [B, N, H, 3]
        traj = traj_init + traj_dist * scale
        return traj

    def scale_trajectory_length(self, traj, scale):
        traj_init = traj[..., 0:1, :]  # [B, N, 1, 3]
        traj_dist = traj - traj_init  # [B, N, H, 3]
        traj = traj_init + traj_dist / scale
        return traj

    def scale_trajectory(self, traj, min_bound, max_bound):
        """
        - traj: B x N x H x 3
        """
        traj = self.scale_trajectory_length(traj, self.scale)
        if len(traj.shape) == 3:
            min_bound_batch = min_bound.unsqueeze(1)  # [B, 1, 3]
            max_bound_batch = max_bound.unsqueeze(1)  # [B, 1, 3]
        elif len(traj.shape) == 4:
            min_bound_batch = min_bound.unsqueeze(
                1).unsqueeze(1)  # [B, 1, 1, 3]
            max_bound_batch = max_bound.unsqueeze(
                1).unsqueeze(1)  # [B, 1, 1, 3]
        else:
            raise ValueError("Invalid shape of the input trajectory")

        scale = max_bound_batch - min_bound_batch  # [B, 1, 3]
        traj = (traj - min_bound_batch) / (scale + 1e-5)
        traj = traj * 2 - 1
        # traj = traj.clamp(-1, 1)
        return traj

    def descale_trajectory(self, traj, min_bound, max_bound):
        """
        - traj: B x N x H x 3
        """
        if len(traj.shape) == 3:
            min_bound_batch = min_bound.unsqueeze(1)  # [B, 1, 3]
            max_bound_batch = max_bound.unsqueeze(1)  # [B, 1, 3]
        elif len(traj.shape) == 4:
            min_bound_batch = min_bound.unsqueeze(
                1).unsqueeze(1)  # [B, 1, 1, 3]
            max_bound_batch = max_bound.unsqueeze(
                1).unsqueeze(1)  # [B, 1, 1, 3]
        else:
            raise ValueError("Invalid shape of the input trajectory")

        scale = max_bound_batch - min_bound_batch
        traj = (traj + 1) / 2
        traj = traj * scale + min_bound_batch
        traj = self.descale_trajectory_length(traj, self.scale)
        return traj

    def compute_guidance_loss(self, x, t, data_batch):
        """
        Evaluates all guidance losses and total and individual values.
        - x: (B, N, H, 3) the trajectory to use to compute losses and 6 is (x, y, vel, yaw, acc, yawvel)
        - data_batch : various tensors of size (B, ...) that may be needed for loss calculations
        """
        guide_losses = dict()
        loss_tot = 0.0

        return loss_tot, guide_losses


class DiffuserGuidance:
    def __init__(
        self,
        goal_weight=0,
        noncollide_weight=0,
        contact_weight=0,
        smooth_weight=0,
        normal_weight=0,
        set_goal_infinite=False,
        scale=1.0,
        valid_horizon=-1,
        exclude_object_points=False,
    ):
        self.goal_weight = goal_weight
        self.noncollide_weight = noncollide_weight
        self.contact_weight = contact_weight
        self.smooth_weight = smooth_weight
        self.normal_weight = normal_weight
        self.set_goal_infinite = set_goal_infinite
        if self.set_goal_infinite:
            self.goal_guidance = InfiniteGoalConditionedGuidance(
                scale, valid_horizon)
        else:
            self.goal_guidance = GoalConditionedGuidance(scale, valid_horizon)
        self.contact_guidance = MapContactGuidance(scale, valid_horizon)
        self.noncollide_guidance = MapNonCollisionGuidance(
            scale, valid_horizon, exclude_object_points)
        self.smooth_guidance = TrajectorySmoothnessGuidance(
            scale, valid_horizon)
        self.normal_guidance = NormalVectorGuidance(scale, valid_horizon)

    def compute_guidance_loss(self, x, t, data_batch):
        """
        Evaluates all guidance losses and total and individual values.
        - x: (B, N, H, 3) the trajectory to use to compute losses and 6 is (x, y, vel, yaw, acc, yawvel)
        - data_batch : various tensors of size (B, ...) that may be needed for loss calculations
        """
        guide_losses = dict()
        loss_tot = 0.0

        if self.goal_weight > 0:
            goal_loss, losses_dict = self.goal_guidance.compute_guidance_loss(
                x, t, data_batch
            )
            loss_tot += goal_loss * self.goal_weight
            guide_losses.update(losses_dict)

        if self.noncollide_weight > 0:
            noncollide_loss, losses_dict = (
                self.noncollide_guidance.compute_guidance_loss(
                    x, t, data_batch)
            )
            loss_tot += noncollide_loss * self.noncollide_weight
            guide_losses.update(losses_dict)

        if self.contact_weight > 0:
            contact_loss, losses_dict = self.contact_guidance.compute_guidance_loss(
                x, t, data_batch
            )
            loss_tot += contact_loss * self.contact_weight
            guide_losses.update(losses_dict)

        if self.smooth_weight > 0:
            smooth_loss, losses_dict = self.smooth_guidance.compute_guidance_loss(
                x, t, data_batch
            )
            loss_tot += smooth_loss * self.smooth_weight
            guide_losses.update(losses_dict)

        if self.normal_weight > 0:
            normal_loss, losses_dict = self.normal_guidance.compute_guidance_loss(
                x, t, data_batch
            )
            loss_tot += normal_loss * self.normal_weight
            guide_losses.update(losses_dict)

        loss_per_traj = 0
        for k, v in guide_losses.items():
            loss_per_traj += v
        guide_losses["total_loss"] = loss_per_traj
        return loss_tot, guide_losses


class TrajectorySmoothnessGuidance(Guidance):
    def __init__(self, scale, valid_horizon=-1):
        super(TrajectorySmoothnessGuidance, self).__init__(
            scale, valid_horizon)

    def compute_guidance_loss(self, x, t, data_batch):
        guide_losses = dict()
        loss_tot = 0.0

        gt_traj_min_bound = data_batch["gt_traj_min_bound"]
        gt_traj_max_bound = data_batch["gt_traj_max_bound"]
        points = self.descale_trajectory(
            x, gt_traj_min_bound, gt_traj_max_bound
        )  # [B, N, H, 3]
        points_diff = points[:, :, 1:] - points[:, :, :-1]  # [B, N, H-1, 3]
        points_lap = points_diff[:, :, 1:] - \
            points_diff[:, :, :-1]  # [B, N, H-2, 3]
        # smoothness_loss = F.mse_loss(
        #     points_diff, torch.zeros_like(points_diff), reduction="none"
        # )  # [B, N, H-2, 3]
        smoothness_loss = F.mse_loss(
            points_lap, torch.zeros_like(points_lap), reduction="none"
        )  # [B, N, H-2, 3]

        smoothness_loss = smoothness_loss.mean(dim=-1)[
            ..., : self.valid_horizon
        ]  # [B, N, H-2]
        smoothness_loss = smoothness_loss.mean(dim=-1)  # [B, N]
        guide_losses["smoothness_loss"] = smoothness_loss  # [B, N]
        smoothness_loss = smoothness_loss.mean()
        loss_tot += smoothness_loss
        return loss_tot, guide_losses


class GoalConditionedGuidance(Guidance):
    def __init__(self, scale, valid_horizon=-1):
        super(GoalConditionedGuidance, self).__init__(scale, valid_horizon)
        self.proj = Project3D()

    def compute_guidance_loss(self, x, t, data_batch, num_goals=1,
                              strict_goal=False, include_far_goals=False):
        """
        Evaluates all guidance losses and total and individual values.
        - x: (B, N, H, 3) the trajectory to use to compute losses
        - data_batch : various tensors of size (B, ...) that may be needed for loss calculations
        """
        guide_losses = dict()
        loss_tot = 0.0

        bsize, num_samp, horizon, _ = x.size()
        intr = data_batch["intrinsics"]
        depth = data_batch["depth"]  # [B, H, W]
        inv_intr = torch.inverse(intr)

        if not strict_goal:
            # # Dummy ..
            # gt_traj = data_batch["gt_trajectory"]  # [B, H, 3]
            # gt_traj_goal = gt_traj[:, -1].unsqueeze(1)  # [B, 1, 3]
            # traj_goal_pix = self.proj(gt_traj_goal, intr).squeeze(-1)  # [B, 2]
            # traj_goal_pix = traj_goal_pix.clone().to(torch.int64)

            traj_goal_pix = data_batch["goal_pix"]

            # Sample points along the ray
            depth_max_goal_pix = data_batch["goal_pos"][:, 2] * 1.1
            depth_min_goal_pix = data_batch["goal_pos"][:, 2] * 0.9

            # Do sampling along within the depth
            samples = torch.linspace(0, 1, 100, device=depth.device)
            samples = samples[None].repeat(bsize, 1)  # [B, R]
            depth_samples = depth_min_goal_pix[:, None] + samples * (
                depth_max_goal_pix[:, None] - depth_min_goal_pix[:, None]
            )  # [B, R]

            # Backproject the depth samples
            ones = torch.ones_like(traj_goal_pix[:, 0]).unsqueeze(1)  # [B, 1]
            traj_goal_pix_hom = torch.cat(
                [traj_goal_pix, ones], dim=1
            ).float()  # [B, 3]
            traj_goal_coords = torch.matmul(
                inv_intr, traj_goal_pix_hom.unsqueeze(-1)
            )  # [B, 3, 1]
            traj_goal_coords = traj_goal_coords.repeat(1, 1, 100)  # [B, 3, R]
            traj_goal_samples = traj_goal_coords * depth_samples.unsqueeze(
                1
            )  # [B, 3, R]
            traj_goal_samples = traj_goal_samples.transpose(1, 2)  # [B, R, 3]
            if "goal_pos_samples" in data_batch:
                pred_goal_samples = data_batch["goal_pos_samples"]  # [B, R, 3]
                traj_goal_samples = torch.cat(
                    [traj_goal_samples, pred_goal_samples], dim=1
                )
            if include_far_goals:
                object_top_normal = data_batch["object_top_normal"]  # [B, 3]
                start_pos = data_batch["start_pos"]  # [B, 3]
                goal_pos_sym = []
                for sgn in [-1, 1]:
                    goal_pos_sgn = start_pos + sgn * object_top_normal * 10
                    goal_pos_sym.append(goal_pos_sgn)
                goal_pos_sym = torch.stack(goal_pos_sym, dim=1)  # [B, 2, 3]
                goal_pos_sym_dist = torch.norm(goal_pos_sym, dim=-1)  # [B, 2]
                goal_pos_id = torch.argmax(goal_pos_sym_dist, dim=-1)  # [B]
                far_goal_pos = goal_pos_sym[torch.arange(
                    bsize), goal_pos_id]  # [B, 3]
                far_goal_samples = far_goal_pos[:, None].expand(
                    -1, traj_goal_samples.shape[1], -1
                )  # [B, R, 3]
                traj_goal_samples = torch.cat(
                    [traj_goal_samples, far_goal_samples], dim=1
                )  # [B, 2R, 3]
            traj_goal_samples_scaled = self.scale_trajectory(
                traj_goal_samples,
                data_batch["gt_traj_min_bound"],
                data_batch["gt_traj_max_bound"],
            )  # [B, R, 3]

            traj_goal_samples_scaled = traj_goal_samples_scaled.unsqueeze(1).expand(
                -1, num_samp, -1, -1
            )  # [B, N, R, 3]

            # Select the number of waypoints to use for the goal
            x_goal = x[:, :, : self.valid_horizon][:, :, -num_goals:, :].unsqueeze(
                -2
            )  # [B, N, G, 1, 3]
            x_goal = x_goal.expand(
                -1, -1, -1, traj_goal_samples.shape[1], -1
            )  # [B, N, G, R, 3]
            traj_goal_samples_scaled = traj_goal_samples_scaled[:, :, None].expand(
                -1, -1, num_goals, -1, -1
            )  # [B, N, G, R, 3]

            # MSE loss provides smooth gradient for goal-based optimization
            goal_loss = F.mse_loss(
                x_goal, traj_goal_samples_scaled, reduction="none"
            )  # [B, N, G, R, 3]
            goal_loss = goal_loss.mean(dim=-1)  # [B, N, G, R]
            goal_loss, _ = goal_loss.min(dim=-1)  # [B, N, G]
            goal_loss = goal_loss.mean(dim=-1)  # [B, N]
            guide_losses["goal_loss"] = goal_loss  # [B, N]

            goal_loss = goal_loss.mean()
            loss_tot += goal_loss

        else:
            # Select the number of waypoints to use for the goal
            x_goal = x[:, :, -num_goals:, :]  # [B, N, 1, 3]
            goal_pos = data_batch["goal_pos"]  # [B, 3]
            goal_pos_scaled = self.scale_trajectory(
                goal_pos[:, None],
                data_batch["gt_traj_min_bound"],
                data_batch["gt_traj_max_bound"],
            )  # [B, 1, 3]
            goal_pos_scaled = goal_pos_scaled[:, None].expand(
                -1, num_samp, -1, -1
            )  # [B, N, 1, 3]
            goal_loss = F.mse_loss(
                x_goal, goal_pos_scaled, reduction="none"
            )  # [B, N, 1, 3]
            goal_loss = goal_loss.mean(dim=-1)  # [B, N, 1]
            goal_loss = goal_loss.mean(dim=-1)  # [B, N]
            guide_losses["goal_loss"] = goal_loss
            goal_loss = goal_loss.mean()
            loss_tot += goal_loss
        return loss_tot, guide_losses


class InfiniteGoalConditionedGuidance(GoalConditionedGuidance):
    def __init__(self, scale, valid_horizon=-1):
        super(InfiniteGoalConditionedGuidance,
              self).__init__(scale, valid_horizon)

    def compute_guidance_loss(self, x, t, data_batch):
        """
        Evaluates all guidance losses and total and individual values.
        - x: (B, N, H, 3) the trajectory to use to compute losses and 6 is (x, y, vel, yaw, acc, yawvel)
        - data_batch : various tensors of size (B, ...) that may be needed for loss calculations
        """
        guide_losses = dict()
        loss_tot = 0.0

        bsize, num_samp, horizon, _ = x.size()
        intr = data_batch["intrinsics"]
        depth = data_batch["depth"]  # [B, H, W]
        inv_intr = torch.inverse(intr)

        # # Dummy ..
        # gt_traj = data_batch["gt_trajectory"]  # [B, H, 3]
        # gt_traj_goal = gt_traj[:, -1].unsqueeze(1)  # [B, 1, 3]
        # traj_goal_pix = self.proj(gt_traj_goal, intr).squeeze(-1)  # [B, 2]
        # traj_goal_pix = traj_goal_pix.clone().to(torch.int64)
        goal_pix = data_batch["goal_pix"]  # [B, 2]
        batch_size, height, width = depth.shape
        v_index = torch.clamp(goal_pix[:, 1], 0, height - 1).long()
        u_index = torch.clamp(goal_pix[:, 0], 0, width - 1).long()

        depth_goal_pix = depth[torch.arange(bsize), v_index, u_index]
        goal_depth = torch.ones_like(depth_goal_pix) * 10  # 2

        ones = torch.ones(batch_size, 1).to(goal_pix.device)
        goal_pix_hom = torch.cat([goal_pix, ones], dim=1).float()  # [B, 3]
        goal_pos = (goal_pix_hom @ inv_intr.transpose(1, 2)) * goal_depth[
            :, None
        ]  # [B, 3]
        goal_pos = goal_pos[:, 0]

        if "object_top_normal" in data_batch:
            object_top_normal = data_batch["object_top_normal"]  # [B, 3]
            start_pos = data_batch["start_pos"]  # [B, 3]
            goal_pos_sym = []
            for sgn in [-1, 1]:
                goal_pos_sgn = start_pos + sgn * object_top_normal * 10
                goal_pos_sym.append(goal_pos_sgn)
            goal_pos_sym = torch.stack(goal_pos_sym, dim=1)  # [B, 2, 3]
            goal_pos_sym_dist = torch.norm(goal_pos_sym, dim=-1)  # [B, 2]
            goal_pos_id = torch.argmax(goal_pos_sym_dist, dim=-1)  # [B]
            goal_pos = goal_pos_sym[torch.arange(bsize), goal_pos_id]  # [B, 3]

        # Select the number of waypoints to use for the goal
        x_goal = x[:, :, : self.valid_horizon][:, :, -1:, :]  # [B, N, 1, 3]

        goal_pos_scaled = self.scale_trajectory(
            goal_pos[:, None],
            data_batch["gt_traj_min_bound"],
            data_batch["gt_traj_max_bound"],
        )  # [B, 1, 3]
        goal_pos_scaled = goal_pos_scaled[:, None].expand(
            -1, num_samp, -1, -1
        )  # [B, N, 1, 3]
        goal_loss = F.mse_loss(
            x_goal, goal_pos_scaled, reduction="none"
        )  # [B, N, 1, 3]
        goal_loss = goal_loss.mean(dim=-1)  # [B, N, 1]
        goal_loss = goal_loss.mean(dim=-1)  # [B, N]
        guide_losses["goal_loss"] = goal_loss
        goal_loss = goal_loss.mean()
        loss_tot += goal_loss

        return loss_tot, guide_losses


class MapNonCollisionGuidance(Guidance):
    def __init__(self, scale, valid_horizon=-1, exclude_object_points=False):
        super(MapNonCollisionGuidance, self).__init__(scale, valid_horizon)
        self.exclude_object_points = exclude_object_points

    def compute_guidance_loss(self, x, t, data_batch, num_noncollide=60, tsdf_threshold=0.2):
        """
        Evaluates all guidance losses and total and individual values.
        - x: (B, N, H, 3) diffusion states
        - data_batch : various tensors of size (B, ...) that may be needed for loss calculations
        """
        guide_losses = dict()
        loss_tot = 0.0
        batch_size, num_samp, horizon, _ = x.size()

        voxel_bounds = data_batch["voxel_bounds"]  # [B, 2]
        object_points = data_batch["object_points"]  # [B, O, 3]
        # [B, G, 3]
        gripper_points_in_contact = data_batch["gripper_points_in_contact"]
        if self.exclude_object_points:
            object_points = gripper_points_in_contact
        else:
            object_points = torch.cat(
                [gripper_points_in_contact, object_points], dim=1)

        num_pcdobj = object_points.size(1)
        gt_traj_min_bound = data_batch["gt_traj_min_bound"]
        gt_traj_max_bound = data_batch["gt_traj_max_bound"]
        tsdf = data_batch["tsdf_grid_fine"][:, None]  # [B, 1, D, H, W]

        # Acquire the tsdf of the trajectory points
        waypoints = self.descale_trajectory(
            x, gt_traj_min_bound, gt_traj_max_bound
        )  # [B, N, H, 3]

        # With object points
        # noncollide_after_horizon = horizon - num_noncollide
        waypoints = waypoints.unsqueeze(2).expand(
            -1, -1, object_points.size(1), -1, -1
        )[
            ..., -num_noncollide:, :
        ]  # [B, N, O, H, 3]
        waypoints_init = waypoints[:, :, :, :1]  # [B, N, O, 1, 3]
        travel_dist = waypoints - waypoints_init  # [B, N, O, H, 3]

        object_points = (
            object_points.unsqueeze(1)
            .unsqueeze(3)
            .expand(-1, num_samp, -1, num_noncollide, -1)
        )  # [B, N, O, H, 3]
        query_points = object_points + travel_dist  # [B, N, O, H, 3]
        query_points = query_points.view(
            batch_size, num_samp, -1, 3)  # [B, N, O*H, 3]

        voxel_bounds = voxel_bounds.unsqueeze(-1).repeat(1, 1, 3)  # [B, 2, 3]
        voxel_bounds_min = voxel_bounds[:, 0][:, None, None].repeat(
            1, num_samp, num_noncollide * num_pcdobj, 1
        )  # [B, N, O*H, 3]
        voxel_bounds_max = voxel_bounds[:, 1][:, None, None].repeat(
            1, num_samp, num_noncollide * num_pcdobj, 1
        )  # [B, N, O*H, 3]
        query_points = (query_points - voxel_bounds_min) / (
            voxel_bounds_max - voxel_bounds_min
        )  # [B, N, O*H, 3]
        query_grids = query_points * 2 - 1  # [B, N, H, 3]

        query_grids = query_grids[..., [2, 1, 0]]  # [B, N, H, 3]
        query_grids = query_grids.unsqueeze(-2)  # [B, N, H, 1, 3]

        query_tsdf = F.grid_sample(
            tsdf, query_grids, align_corners=True, mode="bilinear"
        )  # [B, 1, N, H, 1]
        query_tsdf = query_tsdf.squeeze(-1).squeeze(1)  # [B, N, H]
        query_tsdf = query_tsdf[:, :, : self.valid_horizon]  # [B, N, H]
        map_loss = F.relu(-(query_tsdf - tsdf_threshold)
                          ).mean(dim=-1)  # [B, N]

        guide_losses = {"collision_loss": map_loss}
        map_loss = map_loss.mean()
        loss_tot += map_loss
        return loss_tot, guide_losses


class MapContactGuidance(Guidance):
    def __init__(self, scale, valid_horizon=-1):
        super(MapContactGuidance, self).__init__(scale, valid_horizon)

    def compute_guidance_loss(self, x, t, data_batch):
        """
        Evaluates all guidance losses and total and individual values.
        - x: (B, N, H, 3) diffusion states
        - data_batch : various tensors of size (B, ...) that may be needed for loss calculations
        """
        guide_losses = dict()
        loss_tot = 0.0

        batch_size, num_samp, horizon, _ = x.size()

        voxel_bounds = data_batch["voxel_bounds"]  # [B, 2]
        gt_traj_min_bound = data_batch["gt_traj_min_bound"]
        gt_traj_max_bound = data_batch["gt_traj_max_bound"]
        tsdf = data_batch["tsdf_grid_fine"][:, None]  # [B, 1, D, H, W]

        # Acquire the tsdf of the trajectory points
        query_points = self.descale_trajectory(
            x, gt_traj_min_bound, gt_traj_max_bound
        )  # [B, N, H, 3]
        voxel_bounds = voxel_bounds.unsqueeze(-1).repeat(1, 1, 3)  # [B, 2, 3]
        voxel_bounds_min = voxel_bounds[:, 0][:, None, None].repeat(
            1, num_samp, horizon, 1
        )  # [B, N, H, 3]
        voxel_bounds_max = voxel_bounds[:, 1][:, None, None].repeat(
            1, num_samp, horizon, 1
        )  # [B, N, H, 3]
        query_points = (query_points - voxel_bounds_min) / (
            voxel_bounds_max - voxel_bounds_min
        )  # [B, N, H, 3]
        query_grids = query_points * 2 - 1  # [B, N, H, 3]
        query_grids = query_grids[..., [2, 1, 0]]  # [B, N, H, 3]
        query_grids = query_grids.unsqueeze(-2)  # [B, N, H, 1, 3]
        query_tsdf = F.grid_sample(
            tsdf, query_grids, align_corners=True
        )  # [B, 1, N, H, 1]
        query_tsdf = query_tsdf.squeeze(-1).squeeze(1)  # [B, N, H]

        # Compute the guidance loss
        query_tsdf = query_tsdf[:, :, : self.valid_horizon]  # [B, N, H']
        goal_tsdf = query_tsdf[:, :, :horizon]  # [B, N, 5]
        contact_loss = F.mse_loss(
            goal_tsdf, torch.zeros_like(goal_tsdf), reduction="none"
        ).mean(
            dim=-1
        )  # [B, N]

        # collide_loss = F.relu(-query_tsdf[:, :, :-num_goals]).mean(dim=-1)  # [B, N]
        # map_loss = contact_loss  # + collide_loss  # [B, N]

        guide_losses = {"contact_loss": contact_loss}
        contact_loss = contact_loss.mean()
        loss_tot += contact_loss
        return loss_tot, guide_losses


class NormalVectorGuidance(Guidance):
    def __init__(self, scale, valid_horizon=-1):
        super(NormalVectorGuidance, self).__init__(scale, valid_horizon)

    def compute_guidance_loss(self, x, t, data_batch):
        """
        Evaluates all guidance losses and total and individual values.
        - x: (B, N, H, 3) diffusion states
        - data_batch : various tensors of size (B, ...) that may be needed for loss calculations
        """
        guide_losses = dict()
        loss_tot = 0.0

        gt_traj_min_bound = data_batch["gt_traj_min_bound"]
        gt_traj_max_bound = data_batch["gt_traj_max_bound"]
        points = self.descale_trajectory(
            x, gt_traj_min_bound, gt_traj_max_bound
        )  # [B, N, H, 3]

        # points_start = points[:, :, 0:1, :].clone().detach()  # [B, N, 1, 3]
        # points_rest = points[:, :, 1:, :]  # [B, N, H-1, 3]
        points_normal = points[..., 1:, :] - \
            points[..., 0:1, :].clone().detach()
        # points_normal = (
        #     points[..., 1:, :] - points[..., :-1, :].detach()
        # )  # [B, N, H-1, 3]
        points_normal = F.normalize(points_normal, dim=-1)  # [B, N, H-1, 3]
        points_normal = points_normal[:, :,
                                      : self.valid_horizon]  # [B, N, H', 3]
        normal_loss_sym = []

        for sgn in [-1, 1]:
            object_normal = sgn * data_batch["object_top_normal"]  # [B, 3]
            object_normal = object_normal.unsqueeze(
                1).unsqueeze(1)  # [B, 1, 1, 3]
            object_normal = object_normal.expand(
                -1, points_normal.size(1), points_normal.size(2), -1
            )  # [B, 1, H-1, 3]
            normal_dot = torch.sum(
                points_normal * object_normal, dim=-1)  # [B, N, H-1]
            normal_loss_dot = F.mse_loss(
                normal_dot, torch.ones_like(normal_dot)
            )  # [B, N, H-1]
            normal_loss_diff = F.mse_loss(
                points_normal, object_normal, reduction="none"
            )
            normal_loss = normal_loss_diff
            normal_loss = normal_loss.mean(dim=-1)  # [B, N, H-1]
            normal_loss = normal_loss.mean(dim=-1)  # [B, N]
            normal_loss_sym.append(normal_loss)
        normal_loss_sym = torch.stack(normal_loss_sym, dim=0)  # [2, B, N]
        normal_loss = normal_loss_sym.min(dim=0)[0]  # [B, N]
        guide_losses["normal_loss"] = normal_loss  # [B, N]
        normal_loss = normal_loss.mean()
        loss_tot += normal_loss
        return loss_tot, guide_losses
