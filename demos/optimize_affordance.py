import cv2
import numpy as np
import open3d as o3d
import torch
from scipy.signal import savgol_filter
from algos.traj_optimizer import TrajectoryOptimizer
from diffuser_utils.dataset_utils import visualize_3d_trajectory, load_json, visualize_points, backproject, interpolate_trajectory

def run(colmap_results, data, device, visualize=True):
    traj, vis, vis_scene = [], [], []

    # Prepare the tensors
    frame_ids = data["frame_ids"]
    intr = data["intr"]
    rgbs = data["rgbs"]
    depths = data["depths"]
    hand_masks = data["masks"]
    obj_masks = data["obj_masks"]
    hand_bboxes = data["hand_bboxes"]
    obj_bboxes = data["obj_bboxes"]
    rgb_tensors, depth_tensors, mask_tensors = [], [], []
    for ii, fi in enumerate(frame_ids):
        rgb = rgbs[ii] / 255.0
        depth = depths[ii]
        mask_hand = hand_masks[ii]
        mask_obj = obj_masks[ii]
        mask_hand = cv2.dilate(mask_hand, np.ones((5, 5), np.uint8), iterations=3)
        mask_obj = cv2.dilate(mask_obj, np.ones((5, 5), np.uint8), iterations=3)
        mask_dynamic = np.logical_or(mask_hand > 0, mask_obj > 0)
        mask_static = (1 - mask_dynamic).astype(np.float32)
        rgb_tensors.append(torch.from_numpy(rgb).permute(2, 0, 1).float())
        depth_tensors.append(torch.from_numpy(depth).unsqueeze(0).float())
        mask_tensors.append(torch.from_numpy(mask_static).unsqueeze(0).float())
    rgb_tensors = torch.stack(rgb_tensors).to(device)
    depth_tensors = torch.stack(depth_tensors).to(device)
    mask_tensors = torch.stack(mask_tensors).to(device)
    height, width = rgb_tensors.shape[-2:]
    intr = torch.from_numpy(intr).float().to(device)

    # Initialize the pose and scale optimizer
    traj_optimizer = TrajectoryOptimizer(
        resolution=(height, width),
        lr_scale_global=0.05,
        lr_scale=0.1,
        lr_pose=0.05,
        num_iters_scale=10,
        num_iters_pose=50,
        device=device,
    )

    # Optimize the global scale
    scale_init_tensors, scale_global_final, key_idx = traj_optimizer.optimize_global_scale(
        rgb_tensors,
        depth_tensors,
        mask_tensors,
        colmap_results,
    )

    # Optimize the pose and scale
    scale_init_tensors = scale_global_final * torch.ones_like(scale_init_tensors)
    T_kc_final, scale_final = traj_optimizer.optimize_pose(
        intr,
        rgb_tensors,
        depth_tensors,
        mask_tensors,
        scale_init_tensors,
        scale_global_final,
        colmap_results,
        key_idx=key_idx,
        optimize_pose=True,
        verbose=False,
    )
    
    # Acquire the optimized results
    T_kc_final = T_kc_final.detach().cpu().numpy()
    scale_final = scale_final.detach().cpu().numpy()
    intr_np = intr.clone().cpu().numpy()
    
    # Pose and scale of the first frame
    T_kc0 = T_kc_final[0]
    scale_m2c0 = scale_final[0]
    T_kc0[:3, 3] = T_kc0[:3, 3] / scale_m2c0   

    for ii, fi in enumerate(frame_ids):
        # Pose and scale of the current frame
        T_kc = T_kc_final[ii]
        scale_m2c = scale_final[ii]
        T_kc[:3, 3] = T_kc[:3, 3] / scale_m2c
        
        # Transformation from current frame to the first frame
        T_c0c = np.linalg.inv(T_kc0) @ T_kc 
    
        # Get the depth and hand bbox
        depth = depths[ii]
        hand_bbox = hand_bboxes[ii]
        hand_bbox_mask = np.zeros_like(depth)
        hand_bbox_mask[hand_bbox[1] : hand_bbox[3], hand_bbox[0] : hand_bbox[2]] = 1
        points_hand, scene_ids = backproject(depth, intr_np, hand_bbox_mask > 0, False)
        points_hand = points_hand @ T_c0c[:3, :3].T + T_c0c[:3, 3]
        wp = np.median(points_hand, axis=0)
        traj.append(wp)
        
        # Acquire the hand points in the scene
        if visualize:
            hand_seg_mask = hand_masks[ii]
            hand_seg_mask = cv2.erode(hand_seg_mask, np.ones((5, 5), np.uint8), iterations=2)
            hand_seg_mask = hand_seg_mask * (hand_bbox_mask > 0)
            points_hand_scene, scene_ids = backproject(depth, intr_np, hand_seg_mask > 0, False)
            point_colors_scene = rgbs[ii][scene_ids[0], scene_ids[1]]
            point_colors_scene = point_colors_scene / 255.0
            points_hand_scene = points_hand_scene @ T_c0c[:3, :3].T + T_c0c[:3, 3]
            pcd_scene = visualize_points(points_hand_scene, point_colors_scene)
            vis_scene.append(pcd_scene)
            
            # Acquire the background points
            if ii == 0:
                rgb_orig = rgbs[ii] / 255.0
                hand_seg_mask = hand_masks[ii]
                points_orig, scene_ids = backproject(depth, intr_np, hand_seg_mask == 0, False)
                point_colors_orig = rgb_orig[scene_ids[0], scene_ids[1]]
                pcd_orig = visualize_points(points_orig, point_colors_orig)
                
    # Build trajectory and visualize
    traj = np.array(traj)
    traj = savgol_filter(traj, len(traj) - 1, (len(traj) + 1) // 2, axis=0)
    fill_indices = frame_ids - frame_ids[0]
    traj_smooth, _ = interpolate_trajectory(fill_indices, traj)
    filter_window = min(5, len(traj_smooth) - 1)
    traj_smooth = savgol_filter(
        traj_smooth, filter_window, (filter_window + 1) // 2, axis=0
    )
    if visualize:
        _traj_vis = visualize_3d_trajectory(traj_smooth, size=0.05, cmap_name="viridis")
        traj_vis = _traj_vis[0]
        for v in _traj_vis[1:]:
            traj_vis += v
        
        hand_vis = vis_scene[0]
        for v in vis_scene[1:]:
            hand_vis += v  
        vis = [pcd_orig, traj_vis, hand_vis]
    return traj_smooth, vis

if __name__ == "__main__":
    # Load the data and prepare the tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    colmap_results_fpath = "datasets/epickitchens_traj_demo/colmap.json"
    data_fpath = "datasets/epickitchens_traj_demo/observation.npz"
    colmap_results = load_json(colmap_results_fpath)
    data = np.load(data_fpath)
    
    # Optimize the trajectory and visualize the results
    traj_smooth, vis = run(colmap_results, data, device, visualize=True)
    o3d.visualization.draw(vis)