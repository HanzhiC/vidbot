import argparse
import sys
import os
import pytorch_lightning as pl
from omegaconf import OmegaConf
import torch
import numpy as np
import cv2
from diffuser_utils.guidance_loss import DiffuserGuidance
from diffuser_utils.guidance_params import GUIDANCE_PARAMS_DICT
import diffuser_utils.dataset_utils as DatasetUtils
from models.clip import clip
import open3d as o3d
from algos.afford_algos import AffordanceInferenceEngine
from easydict import EasyDict as edict
from copy import deepcopy
from transformations import rotation_matrix

# CLIP-based action text encoder
VLM, VLM_TRANSFORM = clip.load("ViT-B/16", jit=False)
VLM.float()
VLM.eval()
VLM.cuda()
for p in VLM.parameters():
    p.requires_grad = False


def main(args):
    # Parse the instruction
    config = edict(OmegaConf.to_container(OmegaConf.load(args.config)))
    gripper_mesh = o3d.io.read_triangle_mesh(config.gripper_mesh_file)
    gripper_mesh.compute_vertex_normals()
    gripper_mesh.paint_uniform_color([255 / 255.0, 192 / 255.0, 203 / 255.0])
    gripper_mesh.rotate(rotation_matrix(
        np.pi/2, [0, 0, 1])[:3, :3], center=[0, 0, 0])

    # Parse the instruction
    frame_id = args.frame
    action = args.instruction.split(" ")[0]
    object_name = args.object
    if args.object == "" and not args.load_results:
        print("No object name provided, please provide an object name for the detector to work")
        return

    dataset_path = os.path.join(config.dataset_dir, args.dataset)
    camera_info = DatasetUtils.load_json(
        os.path.join(dataset_path, "camera_intrinsic.json")
    )
    intr = np.array(camera_info["intrinsic_matrix"]
                    ).reshape(3, 3).astype(np.float32).T

    if args.scale > 0:
        calib_scale = args.scale
    else:
        calib_scale = 1.5 * 640 / intr[0, 0]

    # Set the parameters for guidance
    if action not in GUIDANCE_PARAMS_DICT:
        print("WARNING: Action [{}] not found in guidance params, quality not guaranteed".format(
            action))
        guidance_params = GUIDANCE_PARAMS_DICT["other"]
    else:
        guidance_params = GUIDANCE_PARAMS_DICT[action]

    goal_weight = guidance_params["goal_weight"]
    noncollide_weight = guidance_params["noncollide_weight"]
    normal_weight = guidance_params["normal_weight"]
    contact_weight = guidance_params["contact_weight"]
    fine_voxel_resolution = guidance_params["fine_voxel_resolution"]
    exclude_object_points = guidance_params["exclude_object_points"]
    print("Using guidance params: ", guidance_params)

    # Read the data
    depth_file_path = os.path.join(
        dataset_path, "depth_m3d", "{:06d}.png".format(frame_id))
    if not os.path.exists(depth_file_path):
        depth_file_path = os.path.join(
            dataset_path, "depth", "{:06d}.png".format(frame_id))

    color_file_path = os.path.join(
        dataset_path, "color", "{:06d}.png".format(frame_id))
    if not os.path.exists(color_file_path):
        color_file_path = color_file_path.replace(".png", ".jpg")
    depth = cv2.imread(depth_file_path, -1)
    depth = depth / 1000.0
    color = cv2.imread(color_file_path, -1)[..., [2, 1, 0]].copy()
    depth[depth > 2] = 0
    data_batch = DatasetUtils.get_context_data_from_rgbd(
        color,
        depth,
        intr,
        voxel_resolution=32,
        fine_voxel_resolution=fine_voxel_resolution,
    )

    # Initialize the inference engine
    config_traj_path = config.config_traj
    config_goal_path = config.config_goal
    config_contact_path = config.config_contact
    cfg_traj = edict(OmegaConf.to_container(OmegaConf.load(config_traj_path)))
    cfg_goal = edict(OmegaConf.to_container(OmegaConf.load(config_goal_path)))
    cfg_contact = edict(OmegaConf.to_container(
        OmegaConf.load(config_contact_path)))
    cfg_traj.TEST.ckpt_path = config.traj_ckpt
    cfg_goal.TEST.ckpt_path = config.goal_ckpt
    cfg_contact.TEST.ckpt_path = config.contact_ckpt
    cfg_traj.TEST.num_samples = 40
    cfg_traj.TEST.class_free_guide_weight = -0.7

    use_detector, use_esam = True, True

    if args.skip_coarse_stage:
        assert (
            args.load_results
        ), "Must load results if skipping coarse affordance prediction"
        cfg_goal, cfg_contact = None, None

    if args.load_results:
        use_detector, use_esam = False, False

    diffuser_guidance = DiffuserGuidance(
        goal_weight=goal_weight,  # 100.0
        noncollide_weight=noncollide_weight,
        contact_weight=contact_weight,
        normal_weight=normal_weight,
        scale=calib_scale,
        exclude_object_points=exclude_object_points,
        valid_horizon=-1,
    )

    inference_engine = AffordanceInferenceEngine(
        traj_config=cfg_traj,
        goal_config=cfg_goal,
        contact_config=cfg_contact,
        traj_guidance=diffuser_guidance,
        use_detector=use_detector,
        use_esam=use_esam,
        use_graspnet=args.use_graspnet,
        detector_config=config.config_detector,
        detector_ckpt=config.detector_ckpt,
        esam_ckpt=config.esam_ckpt,
        graspnet_ckpt=config.graspnet_ckpt,
    )

    for k, v in data_batch.items():
        if isinstance(v, torch.Tensor):
            data_batch[k] = v.unsqueeze(0).cuda()
    _, null_embeddings = DatasetUtils.encode_text_clip(
        VLM, [""], max_length=None, device="cuda"
    )
    data_batch["action_feature_null"] = null_embeddings.cuda()
    data_batch["action_text"] = args.instruction
    data_batch["gripper_points"] = torch.from_numpy(
        np.asarray(gripper_mesh.sample_points_uniformly(
            number_of_points=2048).points).astype(np.float32)
    ).cuda()

    # Run the open-vocabulary object detection
    meta_name = "{:06d}_{}.npz".format(
        frame_id, args.instruction.replace(" ", "-"))
    meta_save_path = os.path.join(dataset_path, "scene_meta", meta_name)
    os.makedirs(os.path.dirname(meta_save_path), exist_ok=True)
    if args.load_results:
        assert os.path.exists(meta_save_path), "Results {} not found".format(
            meta_save_path
        )
        AffordanceInferenceEngine.load_results(meta_save_path, data_batch)
    else:
        inference_engine.forward_detect(data_batch, text=object_name)
        AffordanceInferenceEngine.export_results(meta_save_path, data_batch)

    num_objects = len(data_batch["bbox_all"][0])
    if num_objects == 0:
        print("No objects detected")
        return
    for n in range(num_objects):
        outputs = {}
        results_name = "{:06d}_{:06d}_{}.npz".format(
            frame_id, n, args.instruction.replace(" ", "-")
        )
        results_save_path = os.path.join(
            dataset_path, "prediction", results_name)
        if args.load_results:
            assert os.path.exists(results_save_path), "Results {} not found".format(
                results_save_path
            )
            AffordanceInferenceEngine.load_results(
                results_save_path, data_batch)
        else:
            # Set the data for the object
            label_text = data_batch["label_text_all"][n]
            data_batch["bbox"] = data_batch["bbox_all"][:, n]
            data_batch["cropped_intr"] = data_batch["cropped_intr_all"][:, n]
            data_batch["object_mask"] = data_batch["object_mask_all"][:, n]
            data_batch["object_bbox_mask"] = data_batch["object_bbox_mask_all"][:, n]

            data_batch["object_color"] = data_batch["object_color_all"][:, n]
            data_batch["object_depth"] = data_batch["object_depth_all"][:, n]
            data_batch["object_points"] = data_batch["object_points_all"][:, n]
            data_batch["resize_ratio"] = data_batch["resize_ratio_all"][:, n]
            data_batch["label_text"] = label_text

        # Inference
        inference_engine.encode_action(
            data_batch, clip_model=VLM, max_length=20)
        if not args.skip_coarse_stage:
            inference_engine.forward_contact(
                data_batch,
                outputs,
                solve_vf=False,
                update_data_batch=True,
                sample_num=100,
            )
            inference_engine.forward_goal(
                data_batch,
                outputs,
                solve_vf=False,
                update_data_batch=True,
                sample_num=100,
            )

        inference_engine.compute_object_contact_normal(
            data_batch
        )
        inference_engine.compute_object_grasp_pose(
            data_batch, collision_thresh=0.25)

        if not args.skip_fine_stage:
            inference_engine.forward_traj(
                data_batch,
                outputs,
                radii=0.65,
                scale=calib_scale,
                use_guidance=True,
                update_data_batch=True,
            )
            data_batch["pred_trajectories"] = data_batch["pred_trajectories"][:, :, :60]
            inference_engine.smooth_traj(data_batch)

        # Save trajectories and guidance losses
        if not args.no_save:
            os.makedirs(os.path.dirname(results_save_path), exist_ok=True)
            AffordanceInferenceEngine.export_results(
                results_save_path, data_batch)

        if args.visualize:
            assert not args.skip_fine_stage, "Cannot visualize fine stage if it is skipped"
            # Update the predicted trajectories
            pred_trajs = data_batch["pred_trajectories"]  # [B, N, H, 3]
            vis_o3d = inference_engine.nets["traj"].visualize_trajectory_by_rendering(
                data_batch, "configs/_render.json", window=False, return_vis=True
            )

            if "guide_losses" in data_batch:
                pred_trajs_loss = data_batch["guide_losses"]["total_loss"].detach(
                )
                traj_loss_colors = DatasetUtils.get_heatmap(
                    pred_trajs_loss.cpu().numpy(), cmap_name="turbo"
                )
                best_traj_idx = np.argmin(pred_trajs_loss.cpu().numpy())
                best_traj = pred_trajs[0,
                                       best_traj_idx].cpu().numpy().squeeze()
                for i in range(1, len(vis_o3d)):
                    vis_o3d[i].paint_uniform_color(
                        traj_loss_colors[0, i - 1, :])
            else:
                traj_loss_colors = None
            vis_o3d_trajs = vis_o3d[1]
            for _vis_traj in vis_o3d[2:]:
                vis_o3d_trajs += _vis_traj

            if "grasp_pose" in data_batch:
                gripper_colors = DatasetUtils.get_heatmap(
                    np.arange(len(best_traj))[None], cmap_name="plasma"
                )[0]
                grasp_pose = data_batch["grasp_pose"].cpu().numpy().squeeze()
                gripper_mesh_init = deepcopy(gripper_mesh)
                gripper_mesh_init.transform(grasp_pose)
                gripper_mesh_init.paint_uniform_color(gripper_colors[10])
                vis_o3d_best = gripper_mesh_init

                for hi in range(15, len(best_traj)):
                    if hi % 20 == 0:
                        gripper_mesh_hi = deepcopy(gripper_mesh)
                        gripper_mesh_hi.transform(grasp_pose)
                        gripper_mesh_hi.translate(best_traj[hi] - best_traj[0])
                        gripper_mesh_hi.paint_uniform_color(gripper_colors[hi])
                        vis_o3d_best += gripper_mesh_hi
            vis_o3d_final = [vis_o3d[0], vis_o3d_trajs, vis_o3d_best]

            o3d.visualization.draw(
                vis_o3d_final, title="Affordance Visualization for Instruction: {} (Object {})".format(
                    args.instruction, n)
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="./config/test_config.yaml",
        help="Specify the checkpoint path, config file path, and the dataset directory"
    )
    parser.add_argument("-d", "--dataset", type=str,
                        default="vidbot_data_demo", help="Dataset name")
    parser.add_argument("-f", "--frame", type=int,
                        default=0, help="Frame index")
    parser.add_argument("-i", "--instruction", type=str,
                        default="open drawer",
                        help="Instruction fed to the affordance model, should be in the format of Verb + Object, no space between the verb")
    parser.add_argument("-o", "--object", type=str, default="",
                        help="Object class name fed to the detector")
    parser.add_argument("-v", "--visualize",
                        action="store_true", help="Visualize the results")
    parser.add_argument("-s", "--scale", type=float,
                        default=-1, help="Scale of the trajectory, set this to -1 if you want to use the default scale")
    parser.add_argument("--no_save", action="store_true",
                        help="Do not save the results")
    parser.add_argument("--load_results", action="store_true",
                        help="Load the results from the file, set this to True if you don't want to install the detector")
    parser.add_argument("--skip_coarse_stage",
                        action="store_true", help="Skip the coarse stage, make sure to set --load_results to True")
    parser.add_argument("--skip_fine_stage",
                        action="store_true", help="Skip the fine stage, make sure to set --load_results to True")
    parser.add_argument("--use_graspnet", action="store_true",
                        help="Use the graspnet, use this if GraspNet is successfully installed, othewise we use a heuristic approach to acuqire the grasp pose")
    args = parser.parse_args()
    pl.seed_everything(42)

    main(args)
