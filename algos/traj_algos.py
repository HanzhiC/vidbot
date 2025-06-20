import numpy as np
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import torch.nn.functional as F
import diffuser_utils.dataset_utils as DatasetUtils
from models.diffuser import DiffuserModel
from models.helpers import EMA
import open3d as o3d
from diffuser_utils.guidance_params import COMMON_ACTIONS
import torchvision
 

class TrajectoryDiffusionModule(pl.LightningModule):
    def __init__(self, algo_config):
        super(TrajectoryDiffusionModule, self).__init__()
        self.algo_config = algo_config
        self.nets = nn.ModuleDict()

        # Initialize the diffuser
        policy_kwargs = algo_config.model
        self.nets["policy"] = DiffuserModel(**policy_kwargs)


    @torch.no_grad()
    def encode_action(self, data_batch, clip_model, max_length=20):
        action_tokens, action_feature = DatasetUtils.encode_text_clip(
            clip_model,
            [data_batch["action_text"]],
            max_length=max_length,
            device="cuda",
        )

        action_tokens.to(self.device)
        action_feature.to(self.device)

        action_text = data_batch["action_text"]
        verb_text = action_text.split(" ")[0]
        if verb_text not in COMMON_ACTIONS:
            verb_text = "other"
        else:
            verb_text = verb_text.replace("-", "")
        verb_text = [verb_text]

        verb_tokens, verb_feature = DatasetUtils.encode_text_clip(
            clip_model,
            verb_text,
            max_length=max_length,
            device="cuda",
        )

        verb_tokens.to(self.device)
        verb_feature.to(self.device)

        data_batch.update({"action_feature": action_feature.float()})
        data_batch.update({"verb_feature": verb_feature.float()})


    def forward(
        self,
        data_batch,
        num_samp=1,
        return_diffusion=False,
        return_guidance_losses=False,   
        apply_guidance=False,   
        class_free_guide_w=0.0,
        guide_clean=False,  
    ):
        curr_policy = self.nets["policy"]
        return curr_policy(
            data_batch,
            num_samp,
            return_diffusion=return_diffusion,
            return_guidance_losses=return_guidance_losses,
            apply_guidance=apply_guidance,
            class_free_guide_w=class_free_guide_w,
            guide_clean=guide_clean,
        )

    def visualize_trajectory_by_rendering(
        self,
        data_batch,
        config_path,
        window=False,
        return_vis=False,
        draw_grippers=False,
        **kwargs
    ):
        batch_size = len(data_batch["color"])
        results = []
        for i in range(batch_size):
            vis_o3d = []
            depth = data_batch["depth"][i].cpu().numpy()
            color = data_batch["color"][i].cpu().numpy().transpose(1, 2, 0)
            intr = data_batch["intrinsics"][i].cpu().numpy()
            # gt_traj = data_batch["gt_trajectory"][i].cpu().numpy()

            # backproject
            points_scene, scene_ids = DatasetUtils.backproject(
                depth,
                intr,
                depth > 0,
                # np.logical_and(hand_mask == 0, depth > 0),
                NOCS_convention=False,
            )

            colors_scene = color.copy()[scene_ids[0], scene_ids[1]]
            pcd_scene = DatasetUtils.visualize_points(
                points_scene, colors_scene)
            # gt_traj_vis = DatasetUtils.visualize_3d_trajectory(
            #     gt_traj, size=0.02, cmap_name="viridis"
            # )

            vis_o3d = [pcd_scene]  # + gt_traj_vis
            if "pred_trajectories" in data_batch:
                print("===> Visualizing pred trajectories")
                pred_trajs = data_batch["pred_trajectories"][i].cpu().numpy()
                pred_traj_colors = DatasetUtils.random_colors(len(pred_trajs))
                for pi, pred_traj in enumerate(pred_trajs):
                    _pred_traj_vis = DatasetUtils.visualize_3d_trajectory(
                        pred_traj, size=0.01, cmap_name="plasma"
                    )
                    if len(pred_trajs) > 1:
                        _pred_traj_vis = [
                            s.paint_uniform_color(pred_traj_colors[pi])
                            for s in _pred_traj_vis
                        ]

                    pred_traj_vis = _pred_traj_vis[0]
                    for ii in range(1, len(_pred_traj_vis)):
                        pred_traj_vis += _pred_traj_vis[ii]

                    vis_o3d += [pred_traj_vis]
 
            if window:
                o3d.visualization.draw(vis_o3d)

            if return_vis:
                return vis_o3d

            render_dist = np.median(np.linalg.norm(points_scene, axis=1))
            render_img = DatasetUtils.render_offscreen(
                vis_o3d,
                config_path,
                # dist=render_dist,
                resize_factor=0.5,
            )
            render_img = torchvision.transforms.ToTensor()(render_img)
            results.append(render_img)
        results = torch.stack(results, dim=0)  # [B, C, H, W]
        data_batch.update({"pred_vis": results})
