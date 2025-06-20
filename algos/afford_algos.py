import sys
import os
import numpy as np
import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from algos.contact_algos import ContactPredictorModule
from algos.goal_algos import GoalPredictorModule
from algos.traj_algos import TrajectoryDiffusionModule
import diffuser_utils.dataset_utils as DatasetUtils
from torchvision.ops import box_convert
import open3d as o3d  # type: ignore
import cv2
from transformations import rotation_matrix
from PIL import Image
from sklearn.mixture import GaussianMixture
from scipy.signal import savgol_filter
from transformations import rotation_matrix
from copy import deepcopy
sys.path.append("./third_party/EfficientSAM")
sys.path.append("./third_party/GroundingDINO")
sys.path.append("./third_party/graspness_unofficial")


class AffordanceInferenceEngine(pl.LightningModule):
    def __init__(
        self,
        contact_config=None,
        goal_config=None,
        traj_config=None,
        traj_guidance=None,
        use_detector=False,
        use_esam=False,
        use_graspnet=False,
        detector_config="./third_party/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        detector_ckpt="./third_party/GroundingDINO/weights/groundingdino_swint_ogc.pth",
        esam_ckpt="./third_party/EfficientSAM/weights/efficient_sam_vitt.pt",
        graspnet_ckpt="./third_party/graspness_unofficial/weights/minkuresunet_kinect.tar",
    ):
        super(AffordanceInferenceEngine, self).__init__()

        # Initialize the modules
        self.nets = {}

        if use_detector:
            import groundingdino.datasets.transforms as T  # type: ignore
            from groundingdino.util.inference import load_model  # type: ignore
            self.nets["detector"] = load_model(
                detector_config, detector_ckpt
            )
            self.transform = T.Compose(
                [
                    T.RandomResize([800], max_size=1333),
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
            self.object_color_resolution = 256
            self.context_image_shape = [256, 456]
            self.default_image_shape = [256, 448]

        if use_esam:
            from efficient_sam.efficient_sam import build_efficient_sam  # type: ignore
            self.nets["esam"] = build_efficient_sam(
                encoder_patch_embed_dim=192,
                encoder_num_heads=3,
                checkpoint=esam_ckpt,
            ).eval()

        if use_graspnet:
            from graspnet_models.graspnet import GraspNet  # type: ignore
            self.nets["graspnet"] = GraspNet(
                seed_feat_dim=512, is_training=False)
            self.nets["graspnet"].load_state_dict(
                torch.load(graspnet_ckpt)['model_state_dict'])

        # Module config
        self.contact_config = contact_config
        self.goal_config = goal_config
        self.traj_config = traj_config

        # Pretrained path
        if self.contact_config is not None:
            self.contact_pretrained_path = contact_config.TEST.ckpt_path
            self.nets["contact"] = ContactPredictorModule.load_from_checkpoint(
                self.contact_pretrained_path,
                algo_config=self.contact_config.ALGORITHM,
            )

        if self.goal_config is not None:
            self.goal_pretrained_path = goal_config.TEST.ckpt_path
            self.nets["goal"] = GoalPredictorModule.load_from_checkpoint(
                self.goal_pretrained_path,
                algo_config=self.goal_config.ALGORITHM,
            )

        if self.traj_config is not None:
            self.traj_pretrained_path = traj_config.TEST.ckpt_path
            self.nets["traj"] = TrajectoryDiffusionModule.load_from_checkpoint(
                self.traj_pretrained_path,
                algo_config=self.traj_config.ALGORITHM,
            )

        if traj_guidance is not None:
            self.traj_guidance = traj_guidance
            assert "traj" in self.nets, "Trajectory module must be loaded"
            self.nets["traj"].nets["policy"].set_guidance(self.traj_guidance)

        if len(self.nets.keys()) > 0:
            for k, v in self.nets.items():
                v.eval()
                v.cuda()

    @torch.no_grad()
    def encode_action(self, data_batch, clip_model, max_length=20):
        net_keys = list(self.nets.keys())
        if "detector" in net_keys.copy():
            net_keys.remove("detector")
        if "esam" in net_keys.copy():
            net_keys.remove("esam")
        if "graspnet" in net_keys.copy():
            net_keys.remove("graspnet")
        selected_key = net_keys[0]
        if "goal" in net_keys:
            selected_key = "goal"
        self.nets[selected_key].encode_action(
            data_batch, clip_model, max_length=max_length
        )

    @staticmethod
    def get_interaction_uvs(hmaps, thres=90, sample_nums=500):
        max_pix_uv_all, sample_uv_all = [], []

        for hmap in hmaps:
            hmap_np = hmap.cpu().numpy()
            hmap_np_top = np.percentile(hmap_np, thres)
            max_pix_uv = np.unravel_index(
                np.argmax(hmap_np.copy()), hmap_np.shape)
            max_pix_uv = np.array([max_pix_uv[1], max_pix_uv[0]])
            valid_pix_uv = np.where(hmap_np > hmap_np_top)
            valid_pix_uv = np.stack(valid_pix_uv, axis=1)[:, [1, 0]]
            sample_ids = np.random.choice(valid_pix_uv.shape[0], sample_nums)
            sample_uv = valid_pix_uv[sample_ids]
            max_pix_uv_all.append(max_pix_uv)
            sample_uv_all.append(sample_uv)
        max_pix_uv_all = np.stack(max_pix_uv_all, axis=0)
        sample_uv_all = np.stack(sample_uv_all, axis=0)
        max_pix_uv_all = torch.from_numpy(max_pix_uv_all).float().cuda()
        sample_uv_all = torch.from_numpy(sample_uv_all).float().cuda()
        return max_pix_uv_all, sample_uv_all

    @staticmethod
    def rescale_bbox(bbox, resize_ratio):
        center = np.array([bbox[0] + bbox[2], bbox[1] + bbox[3]]) / 2  # (x, y)
        center = center * resize_ratio  # (y, x)
        bbox_height = (bbox[3] - bbox[1]) * resize_ratio  # in y direction
        bbox_width = (bbox[2] - bbox[0]) * resize_ratio  # in x direction
        bbox_new = np.array(
            [
                center[0] - bbox_width / 2,
                center[1] - bbox_height / 2,
                center[0] + bbox_width / 2,
                center[1] + bbox_height / 2,
            ]
        ).astype(np.int32)
        return bbox_new

    @torch.no_grad()
    def forward_grasp(self, data_batch, num_point=15000, voxel_size=0.005, collision_thresh=-1, verbose=False):
        from graspnet_utils.data_utils import CameraInfo, create_point_cloud_from_depth_image  # type: ignore
        from graspnet_utils.collision_detector import ModelFreeCollisionDetector  # type: ignore
        from graspnet_dataset.graspnet_dataset import minkowski_collate_fn  # type: ignore
        from graspnet_models.graspnet import pred_decode  # type: ignore
        from graspnetAPI.graspnet_eval import GraspGroup  # type: ignore

        def vis_grasp(gg, cloud):
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(cloud)
            gg_vis = gg.to_open3d_geometry_list()
            o3d.visualization.draw_geometries([pcd, *gg_vis])

        rot180 = rotation_matrix(np.pi, [0, 1, 0])[:3, :3]
        # Get the object depth and object depth samples
        depth_np = data_batch["depth_raw"][0].cpu().numpy()
        intr = data_batch["intrinsics_raw"][0].cpu().numpy()
        height, width = depth_np.shape
        assert height == 720 and width == 1280, "GraspNet supports 720x1280 images"

        camera = CameraInfo(
            1280.0, 720.0, intr[0][0], intr[1][1], intr[0][2], intr[1][2], 1)
        cloud = create_point_cloud_from_depth_image(
            depth_np, camera, organized=True)
        # mask = np.logical_and(depth_np > 0, obj_mask > 0)
        mask = depth_np > 0
        cloud_masked = cloud[mask]

        # sample points random
        if len(cloud_masked) >= num_point:
            idxs = np.random.choice(
                len(cloud_masked), num_point, replace=False)
        else:
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(
                len(cloud_masked), num_point - len(cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        cloud_sampled = cloud_masked[idxs]
        ret_dict = {'point_clouds': cloud_sampled.astype(np.float32),
                    'coors': cloud_sampled.astype(np.float32) / voxel_size,
                    'feats': np.ones_like(cloud_sampled).astype(np.float32),
                    }
        grasp_data_in = minkowski_collate_fn([ret_dict])
        for key in grasp_data_in:
            if 'list' in key:
                for i in range(len(grasp_data_in[key])):
                    for j in range(len(grasp_data_in[key][i])):
                        grasp_data_in[key][i][j] = grasp_data_in[key][i][j].cuda()
            else:
                grasp_data_in[key] = grasp_data_in[key].cuda()

        end_points = self.nets["graspnet"](grasp_data_in)
        grasp_preds = pred_decode(end_points)
        grasp_preds = grasp_preds[0].detach().cpu().numpy()
        gg = GraspGroup(grasp_preds)

        # Add the 180 degree flipped grasp predictions
        gg2 = deepcopy(gg)
        gg2.rotation_matrices = rot180 @ gg2.rotation_matrices
        for _g in gg2:
            gg.add(_g)

        # Filter the grasp predictions
        object_normal = data_batch["object_top_normal"].clone(
        ).cpu().numpy().squeeze()
        if data_batch["normal_sign"] > 0:
            object_normal *= -1
        gg = gg.sort_by_score()
        start_pos = data_batch["start_pos"].cpu().numpy().squeeze()  # [3]
        gg_translations = gg.translations  # [G, 3]
        gg_approchdirs = gg.rotation_matrices[:, :, 0]  # [G, 3]
        gg_dist = np.linalg.norm(
            gg_translations - start_pos[None], axis=1)  # [G]
        gg_angle = np.sum(gg_approchdirs * object_normal[None], axis=1)  # [G]
        gg_angle = np.arccos(gg_angle) * 180 / np.pi
        gg_valid = np.logical_and(gg_dist < 0.1, gg_angle < 90)
        gg = gg[gg_valid][:min(30, len(gg))]

        # Delete the grasp poses that are too close to the object
        if collision_thresh > 0:
            cloud = grasp_data_in['point_clouds'].cpu().numpy().squeeze()
            mfcdetector = ModelFreeCollisionDetector(
                cloud, voxel_size=0.01)
            collision_mask = mfcdetector.detect(
                gg, approach_dist=0.05, collision_thresh=collision_thresh)
            gg = gg[~collision_mask]

        if verbose:
            print("Visualizing grasp predictions before filtering")
            vis_grasp(gg, cloud_masked)

        grasp_poses_valid = []
        for _g in gg:
            rot, t = _g.rotation_matrix, _g.translation
            grasp_pose = np.eye(4)
            grasp_pose[:3, 0] = rot[:3, 1]
            grasp_pose[:3, 1] = rot[:3, 2]
            grasp_pose[:3, 2] = rot[:3, 0]

            # Measure the angle between the grasp normal and the object normal
            grasp_normal = grasp_pose[:3, 2]
            angle = np.dot(grasp_normal, object_normal)
            angle = np.arccos(angle) * 180 / np.pi
            if angle < 75:
                # rotate 180 degrees around the y-axis
                grasp_pose[:3, 2] = object_normal
                grasp_pose[:3, 0] = np.cross(
                    grasp_pose[:3, 1], grasp_pose[:3, 2])
                grasp_pose[:3, 1] = np.cross(
                    grasp_pose[:3, 2], grasp_pose[:3, 0])

            grasp_pose[:3, 3] = start_pos - 0.11 * grasp_pose[:3, 2]
            grasp_poses_valid.append(grasp_pose)

        if verbose and len(grasp_poses_valid) > 0:
            print("Visualizing grasp predictions after filtering")
            vis_grasp(gg, cloud_masked)

        if len(grasp_poses_valid) == 0:
            print(
                "No valid grasp poses, using dummy normal-based grasp detection strategy")
            # Get the object depth and object depth samples
            start_pos = data_batch["start_pos"]
            object_top_normal = data_batch["object_top_normal"].clone()
            if data_batch["normal_sign"] > 0:
                object_top_normal *= -1
            batch_size = start_pos.shape[0]
            grasp_poses_valid = torch.eye(4).cuda()[None]
            grasp_poses_valid = grasp_poses_valid.repeat(batch_size, 1, 1)
            grasp_poses_valid[:, :3, 3] = start_pos
            grasp_poses_valid[:, :3, 2] = object_top_normal
            grasp_poses_valid[:, :3, 1] = torch.cross(
                object_top_normal, grasp_poses_valid[:, :3, 0]).cuda()
            grasp_poses_valid[:, :3, 0] = torch.cross(
                grasp_poses_valid[:, :3, 1], grasp_poses_valid[:, :3, 2]).cuda()
            grasp_poses_valid[:, :3, 3] = grasp_poses_valid[:, :3, 3] - \
                0.11 * grasp_poses_valid[:, :3, 2]
            grasp_poses_valid = grasp_poses_valid[:, None]
        else:
            grasp_poses_valid = np.stack(
                grasp_poses_valid, axis=0)  # [G, 4, 4]
            grasp_poses_valid = torch.from_numpy(
                grasp_poses_valid).cuda()[None].float()
        return grasp_poses_valid

    @torch.no_grad()
    def forward_detect(self, data_batch, text, box_thres=0.25, text_thres=0.25):
        from groundingdino.util.inference import predict  # type: ignore
        assert len(data_batch["color_raw"]) == 1, "Only support batch size 1"
        color_np = data_batch["color_raw"][0].cpu().numpy().transpose(1, 2, 0)
        depth_np = data_batch["depth_raw"][0].cpu().numpy()
        intr = data_batch["intrinsics_raw"][0].cpu().numpy()

        # Forward pass
        color_np = (color_np * 255).astype(np.uint8)
        color_pil = Image.fromarray(color_np)
        color_inp, _ = self.transform(color_pil, None)
        boxes, logits, phrases = predict(
            model=self.nets["detector"],
            image=color_inp,
            caption=text,
            box_threshold=box_thres,
            text_threshold=text_thres,
        )

        h, w, _ = color_np.shape
        boxes = boxes * torch.Tensor([w, h, w, h])
        boxes = box_convert(boxes=boxes, in_fmt="cxcywh",
                            out_fmt="xyxy").numpy()

        # Prepare the data for the next step
        cropped_intr_all = []
        object_depth_all = []
        object_color_all = []
        object_mask_all = []
        object_bbox_mask_all = []
        label_text_all = []
        bbox_all = []
        bbox_raw_all = []
        score_all = []
        start_pos_all = []
        end_pos_all = []
        resize_ratio_all = []
        points_obj_subsampled_all = []
        gt_rot_init_all = []
        for bbox, label_text in zip(boxes, phrases):
            bbox = bbox.astype(np.int32)
            bbox_resize_ratio = (
                data_batch["color"].shape[-1] /
                data_batch["color_raw"].shape[-1]
            )
            bbox_scaled = self.rescale_bbox(
                bbox, resize_ratio=bbox_resize_ratio)
            pad_ratio = 1
            center = np.array(
                [bbox[1] + bbox[3], bbox[0] + bbox[2]]) / 2  # [h, w]
            scale = (
                max(bbox[3] - bbox[1], bbox[2] - bbox[0]) * pad_ratio
            )  # shape before resize
            resize_ratio = float(self.object_color_resolution / scale)

            object_color = DatasetUtils.crop_and_pad_image(
                color_np.copy(), center, scale, self.object_color_resolution, channel=3
            )

            object_depth = DatasetUtils.crop_and_pad_image(
                depth_np.copy(),
                center,
                scale,
                self.object_color_resolution,
                channel=1,
                interpolation=cv2.INTER_NEAREST,
            )[..., 0]

            center_offset = DatasetUtils.get_center_offset(
                center, scale, self.context_image_shape[0], self.context_image_shape[1]
            )
            cropped_intr = DatasetUtils.compute_cropped_intrinsics(
                intr.copy(),
                resize_ratio,
                center + center_offset,
                res=self.object_color_resolution,
            )

            object_color = cv2.resize(
                object_color, [self.object_color_resolution] * 2)
            object_color = object_color / 255.0
            object_color = object_color.transpose(2, 0, 1)

            # Compute start pos
            if "esam" not in self.nets:
                object_bbox_mask = np.zeros_like(depth_np)
                object_bbox_mask[bbox[1]: bbox[3], bbox[0]: bbox[2]] = 1
            else:
                input_points = np.array(
                    [[bbox[0], bbox[1]], [bbox[2], bbox[3]]])
                input_labels = np.array([2, 3])
                input_points = torch.reshape(
                    torch.tensor(input_points), (1, 1, -1, 2))
                input_labels = torch.reshape(
                    torch.tensor(input_labels), (1, 1, -1))
                predicted_logits, predicted_iou = self.nets["esam"](
                    data_batch["color_raw"],
                    input_points.cuda().float(),
                    input_labels.cuda().float(),
                )
                sorted_ids = torch.argsort(
                    predicted_iou, dim=-1, descending=True)
                predicted_iou = torch.take_along_dim(
                    predicted_iou, sorted_ids, dim=2)
                predicted_logits = torch.take_along_dim(
                    predicted_logits, sorted_ids[..., None, None], dim=2
                )
                object_bbox_mask = (
                    torch.ge(predicted_logits[0, 0, 0,
                             :, :], 0).cpu().detach().numpy()
                )
                object_bbox_mask = object_bbox_mask.astype(np.float32).copy()
                object_bbox_mask = cv2.erode(
                    object_bbox_mask, np.ones((7, 7), np.uint8)
                )

            points_obj, _ = DatasetUtils.backproject(
                depth_np,
                intr,
                np.logical_and(depth_np < 2, object_bbox_mask > 0),
                NOCS_convention=False,
            )
            object_bbox = DatasetUtils.crop_and_pad_image(
                object_bbox_mask.copy(),
                center,
                scale,
                self.object_color_resolution,
                channel=1,
                interpolation=cv2.INTER_NEAREST,
            )[..., 0]

            if len(points_obj) == 0:
                continue
            start_pos = np.median(points_obj, axis=0)

            # Compute the trajectory bounds
            pcd_obj = o3d.geometry.PointCloud()
            pcd_obj.points = o3d.utility.Vector3dVector(points_obj)
            pcd_obj_bbox = pcd_obj.get_axis_aligned_bounding_box()
            pcd_obj_bbox_corners = np.array(pcd_obj_bbox.get_box_points())
            if pcd_obj_bbox_corners.sum() == 0:
                continue

            corners_dist = np.linalg.norm(pcd_obj_bbox_corners, axis=1)
            end_pos = pcd_obj_bbox_corners[np.argmin(corners_dist)]
            points_obj_subsampled = points_obj[
                np.random.choice(points_obj.shape[0], 2048)
            ]

            scale_new = (
                max(bbox_scaled[3] - bbox_scaled[1],
                    bbox_scaled[2] - bbox_scaled[0])
                * pad_ratio
            )

            resize_ratio = float(self.object_color_resolution / scale_new)
            gt_rot_init = rotation_matrix(np.pi / 2, [0, 0, 1])[:3, :3]

            # Add to queue
            cropped_intr_all.append(cropped_intr)
            object_depth_all.append(object_depth)
            object_color_all.append(object_color)
            object_mask_all.append(object_bbox)
            object_bbox_mask_all.append(object_bbox_mask)
            label_text_all.append(label_text)
            bbox_all.append(bbox_scaled)
            bbox_raw_all.append(bbox)
            start_pos_all.append(start_pos)
            end_pos_all.append(end_pos)
            resize_ratio_all.append(resize_ratio)
            points_obj_subsampled_all.append(points_obj_subsampled)
            gt_rot_init_all.append(gt_rot_init)

        if len(object_color_all) > 0:
            cropped_intr_all = np.stack(cropped_intr_all, axis=0)
            object_depth_all = np.stack(object_depth_all, axis=0)
            object_color_all = np.stack(object_color_all, axis=0)
            object_mask_all = np.stack(object_mask_all, axis=0)
            object_bbox_mask_all = np.stack(object_bbox_mask_all, axis=0)
            bbox_all = np.stack(bbox_all, axis=0)
            bbox_raw_all = np.stack(bbox_raw_all, axis=0)
            score_all = np.array(score_all)
            start_pos_all = np.stack(start_pos_all, axis=0)
            end_pos_all = np.stack(end_pos_all, axis=0)
            resize_ratio_all = np.array(resize_ratio_all).astype(np.float32)
            points_obj_subsampled_all = np.stack(
                points_obj_subsampled_all, axis=0)
            gt_rot_init_all = np.stack(gt_rot_init_all, axis=0)

        else:
            label_text_all = [""]
            cropped_intr_all = np.zeros((0, 3, 3))
            object_depth_all = np.zeros(
                (0, self.object_color_resolution, self.object_color_resolution)
            )
            object_color_all = np.zeros(
                (0, 3, self.object_color_resolution, self.object_color_resolution)
            )
            object_mask_all = np.zeros(
                (0, self.context_image_shape[0], self.context_image_shape[1])
            )
            object_bbox_mask_all = np.zeros(
                (0, self.context_image_shape[0], self.context_image_shape[1])
            )
            bbox_all = np.zeros((0, 4))
            bbox_raw_all = np.zeros((0, 4))
            score_all = np.zeros(0)
            start_pos_all = np.zeros((0, 3))
            end_pos_all = np.zeros((0, 3))
            resize_ratio_all = np.zeros(0)
            points_obj_subsampled_all = np.zeros((0, 2048, 3))
            gt_rot_init_all = np.zeros((0, 3, 3))

        detection_results = {
            "cropped_intr_all": cropped_intr_all,
            "object_depth_all": object_depth_all,
            "object_color_all": object_color_all,
            "object_points_all": points_obj_subsampled_all,
            "object_mask_all": object_mask_all,
            "object_bbox_mask_all": object_bbox_mask_all,
            "bbox_all": bbox_all,
            "bbox_raw_all": bbox_raw_all,
            "score_all": score_all,
            "start_pos_all": start_pos_all,
            "end_pos_all": end_pos_all,
            "resize_ratio_all": resize_ratio_all,
            "gt_rot_init_all": gt_rot_init_all,
            "label_text_all": label_text_all,
        }
        for k, v in detection_results.items():
            if isinstance(v, np.ndarray):
                data_batch[k] = torch.from_numpy(v).unsqueeze(0).float().cuda()
            else:
                data_batch[k] = v

    @torch.no_grad()
    def forward_contact(
        self,
        data_batch,
        outputs,
        use_mask=False,
        solve_vf=False,
        update_data_batch=False,
        sample_num=1000,
    ):
        assert len(data_batch["object_color"]
                   ) == 1, "Only support batch size 1"
        data_batch["object_color_vis"] = data_batch["object_color"].clone()
        data_batch["object_color"] = (
            data_batch["object_color"] * data_batch["object_mask"][:, None]
        )
        batch_size, _, height, width = data_batch["color"].shape
        patch_height, patch_width = data_batch["object_color"].shape[2:]

        outputs_contact = self.nets["contact"](data_batch)
        contact_vf = outputs_contact["pred"][:, :2]
        pred_scores = outputs_contact["pred"][:, -1].sigmoid()  # [B, H, W]
        pred_mask = (pred_scores > 0).float()

        if solve_vf:
            if use_mask:
                if "object_mask" in data_batch:
                    mask = data_batch["object_mask"]
                else:
                    mask = pred_mask > 0
                ransac_num = 1000
            else:
                mask = None
                ransac_num = 100000
            contact_pix, contact_pix_sample = self.coord_from_vector_field(
                contact_vf,
                mask,
                sigma_scale=0.02,
                inlier_threshold=0.995,
                ransac_num=ransac_num,
                sample_num=sample_num,
            )  # [B, 2], [B, 1000, 2]
            contact_pix = contact_pix[:, None]

        else:
            max_uv, sample_uvs = self.get_interaction_uvs(
                pred_scores, thres=95)
            sample_uvs = sample_uvs.cpu().numpy()[0]
            gm = GaussianMixture(
                n_components=4,
                random_state=0,
                covariance_type="diag",
                init_params="k-means++",
            ).fit(sample_uvs)
            # contact_pix = torch.tensor(gm.means_).float().cuda()[None]  # [B, 2, 2]
            contact_pix = max_uv[:, None]
            contact_pix_sample = gm.sample(sample_num)[0]
            contact_pix_sample = (
                torch.from_numpy(contact_pix_sample).float().cuda()[None]
            )

        contact_pix_sample = contact_pix_sample.long()
        v_index = torch.clamp(
            contact_pix_sample[:, :, 1], 0, patch_height - 1
        )  # [B, 1000]
        u_index = torch.clamp(
            contact_pix_sample[:, :, 0], 0, patch_width - 1
        )  # [B, 1000]
        start_pos_d = data_batch["object_depth"][
            torch.arange(data_batch["object_depth"].size(0))[
                :, None], v_index, u_index
        ]  # [B, 1000]
        start_pos_d = start_pos_d.clamp(0.1, 2.0)
        start_pos_d_med = torch.median(start_pos_d, dim=1)[0]  # [B]
        start_pos_d_diff = (
            start_pos_d - start_pos_d_med[:, None]).abs()  # [B, 1000]
        start_pos_d = torch.where(
            start_pos_d_diff < 0.3, start_pos_d, start_pos_d_med[:, None]
        )  # [B, 1000]
        start_pos_d_med = start_pos_d.mean(1)  # [B]

        # Compute contact pixels on the original images
        resize_ratio = data_batch["resize_ratio"]  # [B]
        contact_pix_all = torch.cat(
            [contact_pix, contact_pix_sample], dim=1
        )  # [B, 1001, 2]

        bbox = data_batch["bbox"].clone()  # [B, 4]
        bbox_center_u = (bbox[:, 0] + bbox[:, 2]) / 2
        bbox_center_v = (bbox[:, 1] + bbox[:, 3]) / 2
        bbox_center = torch.stack(
            [bbox_center_u, bbox_center_v], dim=1)  # [B, 2]
        patch_center = torch.tensor([patch_width, patch_height]).float() / 2
        patch_center = patch_center.to(bbox_center.device)
        patch_center = patch_center[None].expand(bbox_center.shape[0], -1)
        pix_dir_vec = contact_pix_all - \
            patch_center[:, None, :]  # [B, 1001, 2]
        contact_pix_all_full = bbox_center[:, None, :] + pix_dir_vec / (
            resize_ratio[:, None, None]
        )  # [B, 1001, 2]
        contact_pix_full, contact_pix_sample_full = (
            contact_pix_all_full[:, : contact_pix.shape[1]],
            contact_pix_all_full[:, contact_pix.shape[1]:],
        )

        # Compute the start point
        inv_intr = data_batch["inv_intrinsics"]  # [B, 3, 3]
        ones = torch.ones(batch_size, 1).to(contact_pix_full.device)
        contact_pix_one = torch.cat(
            [contact_pix_full[:, 0], ones], dim=-1)  # [B, 3]
        start_point = (contact_pix_one @ inv_intr.transpose(1, 2)) * start_pos_d_med[
            :, None
        ]  # [B, 3]
        start_point = start_point[:, 0]

        ones_samples = torch.ones(batch_size, contact_pix_sample_full.shape[1], 1).to(
            contact_pix_sample_full.device
        )
        contaxt_pix_samples_one = torch.cat(
            [contact_pix_sample_full, ones_samples], dim=-1
        )  # [B, 1000, 3]
        start_point_samples = (
            contaxt_pix_samples_one @ inv_intr.transpose(1, 2)
        ) * start_pos_d[
            ..., None
        ]  # [B, 1000, 3] @ [B, 3, 3] -> [B, 1000, 3] * [B, 1000, 1]

        start_point_samples_med = torch.median(
            start_point_samples, dim=1)[0]  # [B, 3]
        start_point_dist = torch.norm(
            start_point_samples_med - start_point, dim=-1, keepdim=True
        )  # [B,1]
        start_point_dist = start_point_dist.repeat(1, 3)  # [B, 3]

        start_point = torch.where(
            start_point_dist < 0.2, start_point, start_point_samples_med
        )

        # Compute the distance
        start_pos_depth = torch.ones_like(data_batch["depth"])  # [B, H, W]
        start_pos_depth *= start_point[..., None, 2:3]  # [B, H, W]

        outputs.update(
            {
                "contact_vf": contact_vf,
                "contact_pix_patch": contact_pix,
                "contact_pix_samples_patch": contact_pix_sample,
                "contact_pix": contact_pix_full,
                "contact_pix_samples": contact_pix_sample_full,
                "start_pos": start_point,
                "start_pos_samples": start_point_samples,
                "start_pos_depth": start_pos_depth,
                "contact_mask": pred_mask,
                "contact_scores": pred_scores,
            }
        )
        if update_data_batch:
            self.update_outputs_to_databatch(data_batch, outputs)

    @torch.no_grad()
    def forward_traj(
        self,
        data_batch,
        outputs,
        radii=0.65,
        scale=1.0,
        use_guidance=True,
        update_data_batch=False,
    ):
        # Compute trajectory boundary
        assert len(data_batch["start_pos"]) == 1, "Only support batch size 1"

        minimal_traj = np.concatenate(
            [
                data_batch["start_pos"].cpu().numpy(),
                data_batch["end_pos"].cpu().numpy(),
            ],
            axis=0,
        )
        min_bound, max_bound = DatasetUtils.compute_trajectory_bounds_with_radii(
            minimal_traj, radii=radii
        )  # , enlarge_ratio=4)

        data_batch["gt_traj_min_bound"] = (
            torch.tensor(min_bound).unsqueeze(0).cuda().float()
        )
        data_batch["gt_traj_max_bound"] = (
            torch.tensor(max_bound).unsqueeze(0).cuda().float()
        )

        # Forward pass
        if use_guidance:
            assert self.traj_guidance is not None, "Guidance must be provided"
        outputs_traj = self.nets["traj"](
            data_batch,
            num_samp=self.traj_config.TEST.num_samples,
            class_free_guide_w=self.traj_config.TEST.class_free_guide_weight,
            apply_guidance=use_guidance,
            guide_clean=True,
            return_guidance_losses=use_guidance,
        )

        pred_trajs = outputs_traj["predictions"][:, :, 5:]
        gt_start_pos = data_batch["start_pos"]  # [B, 3]
        pred_start_poses = pred_trajs[:, :, 0]  # [B, N, 3]
        offsets = gt_start_pos[:, None, :] - pred_start_poses  # [B, N, 3]
        pred_trajs += offsets[:, :, None, :]  # [B, N, H, 3]
        pred_trajs = DatasetUtils.descale_trajectory_length(
            pred_trajs, scale=scale)
        outputs.update(outputs_traj)
        outputs.update({"pred_trajectories": pred_trajs})
        if update_data_batch:
            self.update_outputs_to_databatch(data_batch, outputs)

    @staticmethod
    def smooth_traj(data_batch, num_subsamples=20, window_length=20, polyorder=10):
        pred_trajs = data_batch["pred_trajectories"]
        batch_size, num_samples, num_points, _ = pred_trajs.shape
        pred_trajs_np = pred_trajs.cpu().numpy()  # [B, N, H, 3]
        smoothed_trajs = []
        for bi in range(batch_size):
            pred_trajs_np_bi = pred_trajs_np[bi]
            smoothed_trajs_bi = []
            for ni in range(num_samples):
                pred_traj = pred_trajs_np_bi[ni]
                # pred_traj = pred_traj[:50]
                pred_traj_goal = pred_traj[-1]
                pred_traj = pred_traj[
                    :: len(pred_traj) // num_subsamples
                ]  # Subsample the trajectory
                pred_traj = np.concatenate(
                    [pred_traj, pred_traj_goal[None]], axis=0
                )  # Append the goal
                smoothed_traj = savgol_filter(
                    pred_traj,
                    window_length=window_length,
                    polyorder=polyorder,
                    axis=0,
                )  # [H, 3]

                # fit the trajectory to get more waypioints
                fill_indices = np.linspace(
                    0, len(smoothed_traj) - 1, len(smoothed_traj)
                )
                smoothed_traj_x = DatasetUtils.spline_interpolation(
                    fill_indices, smoothed_traj[:, 0]
                )[0]
                smoothed_traj_y = DatasetUtils.spline_interpolation(
                    fill_indices, smoothed_traj[:, 1]
                )[0]
                smoothed_traj_z = DatasetUtils.spline_interpolation(
                    fill_indices, smoothed_traj[:, 2]
                )[0]
                smoothed_traj = np.stack(
                    [smoothed_traj_x, smoothed_traj_y, smoothed_traj_z], axis=1
                )
                smoothed_trajs_bi.append(smoothed_traj)
            smoothed_trajs_bi = np.stack(
                smoothed_trajs_bi, axis=0)  # [N, H, 3]
            smoothed_trajs.append(smoothed_trajs_bi)
        smoothed_trajs = np.stack(smoothed_trajs, axis=0)  # [B, N, H, 3]
        smoothed_trajs = torch.from_numpy(
            smoothed_trajs).float().to(pred_trajs.device)
        data_batch["pred_trajectories"] = smoothed_trajs

    @staticmethod
    def compute_object_contact_normal(data_batch, inverse_normal=False):
        # Compute the top normal
        _, points = DatasetUtils.get_normal_from_depth_in_batch(
            data_batch["object_depth"] * data_batch["object_mask"],
            data_batch["cropped_intr"],
            return_points=True,
        )  # [B, 3, H, W], [B, 3, H, W]
        batch_size, _, height, width = points.shape
        points = points.view(batch_size, 3, -1).permute(0, 2, 1)  # [B, N, 3]
        normals = []
        for bi in range(batch_size):
            points_np_bi = points[bi].cpu().numpy()
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_np_bi)
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=0.1, max_nn=30)
            )
            pcd.orient_normals_towards_camera_location(
                camera_location=[0, 0, 0])
            normals_np_bi = np.array(pcd.normals)  # [N, 3]
            normals_np_bi = normals_np_bi.reshape(height, width, 3).transpose(
                2, 0, 1
            )  # [3, H, W]
            normals.append(torch.from_numpy(normals_np_bi).float().cuda())
        normals = torch.stack(normals, dim=0)  # [B, 3, H, W]
        v_index = torch.clamp(
            data_batch["contact_pix_samples_patch"][:, :, 1], 0, 256 - 1
        ).long()  # [B, 1000]
        u_index = torch.clamp(
            data_batch["contact_pix_samples_patch"][:, :, 0], 0, 256 - 1
        ).long()

        contact_point_normals = normals[
            torch.arange(normals.size(0)), :, v_index.long(), u_index.long()
        ]

        contact_point_normals = contact_point_normals.permute(0, 2, 1).unsqueeze(
            -1
        )  # [B, 3, N, 1]
        non_zero_masks = (contact_point_normals.norm(dim=1)
                          > 0).float()  # [B, N, 1]

        cluster_top_normal, cluster_labels, cluster_centers, cluster_counts = (
            DatasetUtils.get_normal_clutters_in_batch(
                contact_point_normals, non_zero_masks, n_clusters=2
            )
        )  # [B, 3], [B, HW], [B, K, 3], [B, K]

        # # compute the angle between normals and the camera z-axis
        # camera_z = torch.tensor([0, 0, 1]).float().cuda()
        # angle = torch.acos(
        #     torch.clamp(torch.sum(cluster_top_normal *
        #                 camera_z[None, :], dim=1), -1, 1)
        # )
        # angle = angle * 180 / np.pi
        # if angle.mean() < 30:
        #     cluster_top_normal = -cluster_top_normal
        # if inverse_normal:
        #     cluster_top_normal = -cluster_top_normal
        cluster_top_normal = data_batch["normal_sign"] * cluster_top_normal

        data_batch["object_top_normal"] = cluster_top_normal  # [B, 3]

    @torch.no_grad()
    def forward_goal(
        self,
        data_batch,
        outputs,
        goal_scale=1,
        solve_vf=True,
        update_data_batch=False,
        sample_num=1000,
    ):
        # time_begin = time.time()
        outputs_goal = self.nets["goal"](data_batch)
        height, width = data_batch["color"].shape[2:]
        goal_vfd, goal_heatmap = (
            outputs_goal["pred"][:, :3],
            outputs_goal["pred"][:, 3].sigmoid(),
        )
        batch_size = goal_vfd.size(0)
        goal_vf, goal_d = goal_vfd[:, :2], goal_vfd[:, 2]

        if solve_vf:
            mask = None
            ransac_num = 100000
            goal_pix, goal_pix_samples = self.coord_from_vector_field(
                goal_vf,
                mask,
                sigma_scale=0.02,
                inlier_threshold=0.99,
                ransac_num=ransac_num,
                sample_num=sample_num,
            )  # [B, 2], [B, 1000, 2]

        else:
            _, sample_uvs = self.get_interaction_uvs(goal_heatmap, thres=95)
            sample_uvs = sample_uvs.cpu().numpy()[0]
            gm = GaussianMixture(
                n_components=1,
                random_state=0,
                covariance_type="diag",
                init_params="k-means++",
            ).fit(sample_uvs)
            goal_pix = torch.tensor(gm.means_).float().cuda()  # [B, 2]
            goal_pix_samples = gm.sample(sample_num)[0]
            goal_pix_samples = (
                torch.from_numpy(goal_pix_samples).float().cuda()[None]
            )  # [B, 1000, 2]

        # goal_pix = goal_pix
        # goal_pix = goal_pix.long()
        # v_index = torch.clamp(goal_pix[:, 1], 0, height - 1)
        # u_index = torch.clamp(goal_pix[:, 0], 0, width - 1)
        # goal_depth = goal_d[torch.arange(goal_d.size(0)), v_index, u_index]

        goal_pix_samples = goal_pix_samples.long()
        v_index = torch.clamp(
            goal_pix_samples[:, :, 1], 0, height - 1)  # [B, 1000]
        u_index = torch.clamp(
            goal_pix_samples[:, :, 0], 0, width - 1)  # [B, 1000]
        goal_depth = goal_d[
            torch.arange(goal_d.size(0))[:, None], v_index, u_index
        ]  # [B, 1000]
        start_depth_avg = data_batch["start_pos_depth"].view(
            batch_size, -1).mean(dim=1)  # [B]
        goal_depth_avg = goal_depth.mean(dim=1)  # [B]
        normal_sign = -torch.sign(goal_depth_avg - start_depth_avg)  # [B]

        goal_depth_samples = goal_depth.clone()
        goal_depth = goal_depth.mean(dim=1)  # [B]
        inv_intr = data_batch["inv_intrinsics"]  # [B, 3, 3]
        ones = torch.ones(batch_size, 1).to(goal_pix.device)
        goal_pix_one = torch.cat([goal_pix, ones], dim=1)  # [B, 3]
        goal_point = (goal_pix_one @ inv_intr.transpose(1, 2)) * goal_depth[
            :, None
        ]  # [B, 3]
        goal_point = goal_point[:, 0]

        ones_samples = torch.ones(batch_size, goal_pix_samples.shape[1], 1).to(
            goal_pix_samples.device
        )
        goal_pix_samples_one = torch.cat(
            [goal_pix_samples, ones_samples], dim=-1
        )  # [B, 1000, 3]
        goal_point_samples = (
            goal_pix_samples_one @ inv_intr.transpose(1, 2)
        ) * goal_depth_samples[
            ..., None
        ]  # [B, 1000, 3] @ [B, 3, 3] -> [B, 1000, 3] * [B, 1000, 1]
        goal_point_samples = goal_point_samples * goal_scale
        goal_point = goal_point * goal_scale

        outputs.update(
            {
                # "goal_pix_selected": goal_pix_selected,
                "goal_heatmap": goal_heatmap,
                "goal_vf": goal_vf,
                "goal_pix": goal_pix,
                "goal_depth": goal_depth,
                "goal_pix_samples": goal_pix_samples,
                "end_pos": goal_point,
                "goal_pos": goal_point,
                "goal_pos_samples": goal_point_samples,
                "normal_sign": normal_sign,
            }
        )
        if update_data_batch:
            self.update_outputs_to_databatch(data_batch, outputs)

    def compute_object_grasp_pose(self, data_batch, num_point=15000, voxel_size=0.005, collision_thresh=0.2):
        # Get the object depth and object depth samples
        start_pos = data_batch["start_pos"]
        object_top_normal = data_batch["object_top_normal"].clone()
        if data_batch["normal_sign"] > 0:
            object_top_normal *= -1
        batch_size = start_pos.shape[0]
        if "graspnet" in self.nets:
            grasp_poses_valid = self.forward_grasp(
                data_batch, num_point, voxel_size, collision_thresh)
            grasp_pose = grasp_poses_valid[:, 0]
        else:
            grasp_pose = torch.eye(4).cuda()[None]
            grasp_pose = grasp_pose.repeat(batch_size, 1, 1)
            grasp_pose[:, :3, 3] = start_pos
            grasp_pose[:, :3, 2] = object_top_normal
            grasp_pose[:, :3, 1] = torch.cross(
                object_top_normal, grasp_pose[:, :3, 0]).cuda()
            grasp_pose[:, :3, 0] = torch.cross(
                grasp_pose[:, :3, 1], grasp_pose[:, :3, 2]).cuda()
            grasp_pose[:, :3, 3] = grasp_pose[:, :3, 3] - \
                0.11 * grasp_pose[:, :3, 2]
        gripper_points_in_contact = data_batch["gripper_points"].clone()[
            None].cuda().float()
        gripper_points_in_contact = gripper_points_in_contact.repeat(
            batch_size, 1, 1)
        gripper_points_in_contact = gripper_points_in_contact @ grasp_pose[:, :3, :3].transpose(-1, -2) \
            + grasp_pose[:, :3, 3].unsqueeze(1)
        data_batch["grasp_pose"] = grasp_pose
        data_batch["gripper_points_in_contact"] = gripper_points_in_contact

    def coord_from_vector_field(
        self,
        vector_field,
        mask=None,
        sigma_scale=0.1,
        inlier_threshold=0.99,
        sample_num=1000,
        ransac_num=100000,
    ):
        h, w = vector_field.size(2), vector_field.size(3)
        # uid = data_batch["uid"]
        results_uv, results_sample = [], []
        for bi in range(vector_field.size(0)):
            vf_i = vector_field[bi].cpu().numpy().transpose(1, 2, 0)
            if mask is not None:
                mask_i = mask[bi].cpu().numpy()
                mask_flatten = mask_i.reshape(-1)  # [H*W]
                mask_torch = torch.from_numpy(mask_flatten).float()  # [HW]
            else:
                mask_torch = None

            # Do voting ...
            height, width = vf_i.shape[:2]
            grid = np.meshgrid(
                np.arange(width), np.arange(height), indexing="xy")
            # [H, W, 2], last dim is (x, y)
            grid = np.stack(grid, axis=-1) + 0.5
            grid_flatten = grid.reshape(-1, 2)  # [H*W, 2]
            vfield_flatten = vf_i[..., :2].reshape(-1, 2)  # [H*W, 2]

            # RANSAC voting to find the goal pixel
            grid_torch = torch.from_numpy(grid_flatten).float()  # [HW, 2]
            vfield_torch = torch.from_numpy(vfield_flatten).float()  # [HW, 2]
            vfield_torch = F.normalize(vfield_torch, dim=-1, p=2)  # [HW, 2]
            uv, sigma, win_hypothesis, win_hypotheses, inlier_ratio = (
                DatasetUtils.ransac_voting_layer(
                    grid_torch,
                    vfield_torch,
                    mask_torch,
                    num_samples=ransac_num,
                    inlier_threshold=inlier_threshold,
                )
            )

            # Sample from the fitted distribution and draw on the image
            sampled_uvs = np.random.multivariate_normal(
                uv, sigma * sigma_scale, sample_num
            )
            results_sample.append(torch.from_numpy(sampled_uvs).float())
            results_uv.append(uv)
        results_uv = torch.stack(results_uv, dim=0).to(vector_field.device)
        results_sample = torch.stack(
            results_sample, dim=0).to(vector_field.device)
        return results_uv, results_sample

    @staticmethod
    def update_outputs_to_databatch(data_batch, outputs, selected_keys=None):
        if selected_keys is None:
            selected_keys = outputs.keys()
        for k, v in outputs.items():
            if k in selected_keys:
                if k in data_batch:
                    # print(f"Overwriting key {k} in data_batch")
                    data_batch[k] = v
                else:
                    data_batch[k] = v

    @staticmethod
    def export_results(save_path, data_batch):
        to_write = {}
        for k, v in data_batch.items():
            if isinstance(v, torch.Tensor):
                to_write[k] = v.cpu().numpy().astype(np.float32)
            elif isinstance(v, np.ndarray):
                to_write[k] = v.astype(np.float32)
            elif isinstance(v, str):
                k += "_singlestr"
                to_write[k] = DatasetUtils.encode_text_list([v])
            elif isinstance(v, list) and isinstance(v[0], str):
                k += "_strlist"
                to_write[k] = DatasetUtils.encode_text_list(v)
            elif isinstance(v, o3d.geometry.TriangleMesh):
                mesh_vertices = np.array(v.vertices).astype(np.float32)
                mesh_faces = np.array(v.triangles).astype(np.float32)
                mesh_vertex_colors = np.array(
                    v.vertex_colors).astype(np.float32)
                to_write["mesh_vertices"] = mesh_vertices
                to_write["mesh_faces"] = mesh_faces
                to_write["mesh_vertex_colors"] = mesh_vertex_colors
            elif k == "guide_losses":
                for k2, v2 in v.items():
                    k2 = f"{k}-{k2}"
                    to_write[k2] = v2.detach().cpu().numpy().astype(np.float32)
            else:
                print(f"Skipping key {k} of type {type(v)}")

        np.savez_compressed(save_path, **to_write)

    @staticmethod
    def load_results(save_path, data_batch):
        data = np.load(save_path)
        skip_keys = []
        for k, v in data.items():
            if "_strlist" in k and k not in skip_keys:
                k = k.replace("_strlist", "")
                data_batch[k] = DatasetUtils.decode_text_list(v)
                skip_keys.append(k)
            elif "_singlestr" in k and k not in skip_keys:
                k = k.replace("_singlestr", "")
                data_batch[k] = DatasetUtils.decode_text_list(v)[0]
                skip_keys.append(k)

            elif k == "mesh_vertices" and k not in skip_keys:
                assert "mesh_faces" in data and "mesh_vertex_colors" in data
                mesh = o3d.geometry.TriangleMesh()
                mesh.vertices = o3d.utility.Vector3dVector(
                    data["mesh_vertices"])
                mesh.triangles = o3d.utility.Vector3iVector(data["mesh_faces"])
                mesh.vertex_colors = o3d.utility.Vector3dVector(
                    data["mesh_vertex_colors"]
                )
                mesh.compute_vertex_normals()
                data_batch["mesh"] = mesh
                skip_keys += ["mesh_vertices",
                              "mesh_faces", "mesh_vertex_colors"]

            # elif "guide_losses" and k not in skip_keys:
            #     guide_losses = {}
            #     for k2, v2 in v.items():
            #         k2 = k2.split("-")[-1]
            #         guide_losses[k2] = torch.from_numpy(v2)
            #     data_batch[k] = guide_losses
            #     skip_keys.append(k)

            elif isinstance(v, np.ndarray) and k not in skip_keys:
                data_batch[k] = torch.from_numpy(v).cuda()

            else:
                print(f"Skipping key {k} of type {type(v)}")

        return data_batch
