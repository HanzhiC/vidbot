import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import cv2
import colorsys
import json
from scipy.interpolate import CubicHermiteSpline, PchipInterpolator
import torch
from models.layers_2d import Project3D, BackprojectDepth
import flow_vis
import torch.nn.functional as F
from models.clip import clip
from typing import List
from sklearn.cluster import KMeans
from models.helpers import get_view_frustum, TSDFVolume
from torchvision import transforms as T
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp


def backproject(depth, intrinsics, instance_mask, NOCS_convention=True):
    intrinsics_inv = np.linalg.inv(intrinsics)
    # image_shape = depth.shape
    # width = image_shape[1]
    # height = image_shape[0]

    # x = np.arange(width)
    # y = np.arange(height)

    # non_zero_mask = np.logical_and(depth > 0, depth < 5000)
    non_zero_mask = depth > 0
    final_instance_mask = np.logical_and(instance_mask, non_zero_mask)

    idxs = np.where(final_instance_mask)
    grid = np.array([idxs[1], idxs[0]])

    # shape: height * width
    # mesh_grid = np.meshgrid(x, y) #[height, width, 2]
    # mesh_grid = np.reshape(mesh_grid, [2, -1])
    length = grid.shape[1]
    ones = np.ones([1, length])
    uv_grid = np.concatenate((grid, ones), axis=0)  # [3, num_pixel]

    xyz = intrinsics_inv @ uv_grid  # [3, num_pixel]
    xyz = np.transpose(xyz)  # [num_pixel, 3]

    z = depth[idxs[0], idxs[1]]

    # print(np.amax(z), np.amin(z))
    pts = xyz * z[:, np.newaxis] / xyz[:, -1:]
    if NOCS_convention:
        pts[:, 1] = -pts[:, 1]
        pts[:, 2] = -pts[:, 2]

    return pts, idxs


def load_json(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


def save_json(file_path, data):
    with open(file_path, "w") as f:
        json.dump(data, f)


def crop_and_pad_image(
    img,
    center,
    scale,
    res=None,
    channel=3,
    interpolation=cv2.INTER_LINEAR,
    resize=True,
):
    # Code from CDPN
    ht, wd = img.shape[0], img.shape[1]
    dtype = img.dtype
    upper = max(0, int(center[0] - scale / 2.0 + 0.5))
    left = max(0, int(center[1] - scale / 2.0 + 0.5))
    bottom = min(ht, int(center[0] - scale / 2.0 + 0.5) + int(scale))
    right = min(wd, int(center[1] - scale / 2.0 + 0.5) + int(scale))
    crop_ht = float(bottom - upper)
    crop_wd = float(right - left)

    if resize:
        if crop_ht > crop_wd:
            resize_ht = res
            resize_wd = int(res / crop_ht * crop_wd + 0.5)
        elif crop_ht < crop_wd:
            resize_wd = res
            resize_ht = int(res / crop_wd * crop_ht + 0.5)
        else:
            resize_wd = resize_ht = int(res)

    if channel <= 3:
        tmpImg = img[upper:bottom, left:right]
        if not resize:
            if channel == 3:
                outImg = np.ones(
                    (int(scale), int(scale), channel), dtype=dtype) * 0.5
            else:
                outImg = np.zeros(
                    (int(scale), int(scale), channel), dtype=dtype)
            outImg[
                int(scale / 2.0 - (bottom - upper) / 2.0 + 0.5): (
                    int(scale / 2.0 - (bottom - upper) /
                        2.0 + 0.5) + (bottom - upper)
                ),
                int(scale / 2.0 - (right - left) / 2.0 + 0.5): (
                    int(scale / 2.0 - (right - left) /
                        2.0 + 0.5) + (right - left)
                ),
                :,
            ] = tmpImg
            return outImg
        resizeImg = cv2.resize(
            tmpImg, (resize_wd, resize_ht), interpolation=interpolation
        )
        # print(tmpImg.shape, scale)
        if len(resizeImg.shape) < 3:
            resizeImg = np.expand_dims(
                resizeImg, axis=-1
            )  # for depth image, add the third dimension
        if channel == 3:
            outImg = np.ones((int(res), int(res), channel), dtype=dtype) * 125
            outImg = outImg.astype(dtype)
        else:
            outImg = np.zeros((int(res), int(res), channel), dtype=dtype)
        outImg[
            int(res / 2.0 - resize_ht / 2.0 + 0.5): (
                int(res / 2.0 - resize_ht / 2.0 + 0.5) + resize_ht
            ),
            int(res / 2.0 - resize_wd / 2.0 + 0.5): (
                int(res / 2.0 - resize_wd / 2.0 + 0.5) + resize_wd
            ),
            :,
        ] = resizeImg

    else:
        raise NotImplementedError
    return outImg


def get_center_offset(center, scale, ht, wd):
    upper = max(0, int(center[0] - scale / 2.0 + 0.5))
    left = max(0, int(center[1] - scale / 2.0 + 0.5))
    bottom = min(ht, int(center[0] - scale / 2.0 + 0.5) + int(scale))
    right = min(wd, int(center[1] - scale / 2.0 + 0.5) + int(scale))

    if upper == 0:
        h_offset = -int(center[0] - scale / 2.0 + 0.5) / 2
    elif bottom == ht:
        h_offset = -(int(center[0] - scale / 2.0 + 0.5) + int(scale) - ht) / 2
    else:
        h_offset = 0

    if left == 0:
        w_offset = -int(center[1] - scale / 2.0 + 0.5) / 2
    elif right == wd:
        w_offset = -(int(center[1] - scale / 2.0 + 0.5) + int(scale) - wd) / 2
    else:
        w_offset = 0
    center_offset = np.array([h_offset, w_offset])
    return center_offset


def compute_cropped_intrinsics(cam_K, resize, crop_center, res):
    # This implementation is tested faithfully. Results in PnP with 0.02% drop.
    K = cam_K.copy()

    # First resize from original size to the target size
    K[0, 0] = K[0, 0] * resize
    K[1, 1] = K[1, 1] * resize
    K[0, 2] = (K[0, 2] + 0.5) * resize - 0.5
    K[1, 2] = (K[1, 2] + 0.5) * resize - 0.5

    # Then crop the image --> need to modify the optical center,
    # remember that current top left is the coordinates measured in resized results
    # And its information is vu instead of uv
    top_left = crop_center * resize - res / 2
    K[0, 2] = K[0, 2] - top_left[1]
    K[1, 2] = K[1, 2] - top_left[0]
    return K


def crop_image(image, bbox, pad_ratio=1.5):
    height, width = image.shape[:2]
    x1, y1, x2, y2 = bbox
    crop_height, crop_width = y2 - y1, x2 - x1
    crop_center = (x1 + crop_width // 2, y1 + crop_height // 2)
    crop_size = int(max(crop_height, crop_width) * pad_ratio)
    x1_crop = max(0, crop_center[0] - crop_size // 2)
    y1_crop = max(0, crop_center[1] - crop_size // 2)
    x2_crop = min(width, crop_center[0] + crop_size // 2)
    y2_crop = min(height, crop_center[1] + crop_size // 2)
    cropped_image = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
    cropped_image[
        (crop_size - (y2_crop - y1_crop)): (crop_size + (y2_crop - y1_crop)),
        (crop_size - (x2_crop - x1_crop)): (crop_size + (x2_crop - x1_crop)),
    ] = image[y1_crop:y2_crop, x1_crop:x2_crop]
    return cropped_image


def center_crop_image(img, crop_height, corp_width):
    height, width = img.shape[:2]
    startx = width // 2 - (corp_width // 2)
    starty = height // 2 - (crop_height // 2)
    cropped_image = img[
        starty: starty + crop_height,
        startx: startx + corp_width,
    ]
    return cropped_image


def resize_image_keep_aspect_ratio(image, height, width):
    target_size = np.array([height, width])
    image_size = image.shape[:2]
    scale = np.min(target_size / image_size)
    _image = cv2.resize(image, (0, 0), fx=scale, fy=scale)
    _image_size = _image.shape[:2]
    size_diff = target_size - _image_size
    dim = np.argmax(size_diff)
    size_diff_dim = np.max(size_diff // 2)
    if len(image.shape) == 2:
        new_image = np.zeros((target_size[0], target_size[1]))
    else:
        new_image = np.zeros((target_size[0], target_size[1], image.shape[2]))
    new_image = new_image.astype(image.dtype)
    if dim == 0:
        new_image[size_diff_dim: size_diff_dim + _image.shape[0], :] = _image
    else:
        new_image[:, size_diff_dim: size_diff_dim + _image.shape[1]] = _image
    return new_image


def transform_points(points, T):
    points = points @ T[:3, :3].T + T[:3, 3]
    return points


def get_heatmap(values, cmap_name="turbo", invert=False):
    if invert:
        values = -values
    values = (values - values.min()) / (values.max() - values.min())
    colormaps = plt.cm.get_cmap(cmap_name)
    rgb = colormaps(values)[..., :3]  # don't need alpha channel
    return rgb


def visualize_sphere_o3d(center, color=[1, 0, 0], size=0.03):
    # center
    center_o3d = o3d.geometry.TriangleMesh.create_sphere()
    center_o3d.compute_vertex_normals()
    center_o3d.scale(size, [0, 0, 0])
    center_o3d.translate(center)
    center_o3d.paint_uniform_color(color)
    return center_o3d


def visualize_3d_trajectory(trajectory, size=0.03, cmap_name="plasma", invert=False):
    vis_o3d = []
    traj_color = get_heatmap(
        np.arange(len(trajectory)), cmap_name=cmap_name, invert=invert
    )
    for i, traj_point in enumerate(trajectory):
        vis_o3d.append(visualize_sphere_o3d(
            traj_point, color=traj_color[i], size=size))
    return vis_o3d


def visualize_points(points, colors=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    # random.shuffle(colors)
    return colors


def get_3d_bbox(size, shift=0):
    """
    Args:
        size: [3] or scalar
        shift: [3] or scalar
    Returns:
        bbox_3d: [N, 3]

    """
    bbox_3d = (
        np.array(
            [
                [+size[0] / 2, +size[1] / 2, +size[2] / 2],
                [+size[0] / 2, +size[1] / 2, -size[2] / 2],
                [-size[0] / 2, +size[1] / 2, +size[2] / 2],
                [-size[0] / 2, +size[1] / 2, -size[2] / 2],
                [+size[0] / 2, -size[1] / 2, +size[2] / 2],
                [+size[0] / 2, -size[1] / 2, -size[2] / 2],
                [-size[0] / 2, -size[1] / 2, +size[2] / 2],
                [-size[0] / 2, -size[1] / 2, -size[2] / 2],
            ]
        )
        + shift
    )
    return bbox_3d


def spline_interpolation(fill_indices, traj):
    fill_times = np.array(fill_indices, dtype=np.float32)
    fill_traj = np.array(
        [traj[ii] for ii, idx in enumerate(fill_indices)], dtype=np.float32
    )
    dt = fill_times[2:] - fill_times[:-2]
    dt = np.hstack([fill_times[1] - fill_times[0], dt,
                   fill_times[-1] - fill_times[-2]])
    dx = fill_traj[2:] - fill_traj[:-2]
    dx = np.hstack([fill_traj[1] - fill_traj[0], dx,
                   fill_traj[-1] - fill_traj[-2]])
    dxdt = dx / dt
    curve = CubicHermiteSpline(fill_times, fill_traj, dxdt)
    curve = PchipInterpolator(fill_times, fill_traj)
    step = (fill_indices[-1] - fill_indices[0]) / 80
    full_traj = curve(
        np.arange(fill_indices[0], fill_indices[-1] +
                  step, step=step, dtype=np.float32)
    )
    return full_traj, curve


def interpolate_trajectory(fill_indices, traj):
    # import pdb; pdb.set_trace()
    full_traj_x, curve_x = spline_interpolation(fill_indices, traj[:, 0])
    full_traj_y, curve_y = spline_interpolation(fill_indices, traj[:, 1])
    full_traj_z, curve_z = spline_interpolation(fill_indices, traj[:, 2])
    full_traj = np.stack([full_traj_x, full_traj_y, full_traj_z], axis=1)
    curve = (curve_x, curve_y, curve_z)
    return full_traj, curve


def compute_vector_field_from_coordinate(
    goal, height, width, return_grid=True, eps=1e-6
):
    """
    Compute the vector field from the coordinate in the image plane.
    Parameters
    ----------
    goal : [2] (x, y), np.ndarray

    Returns
    -------
    [N, 2] np.ndarray
    """

    grid = np.meshgrid(np.arange(width), np.arange(height), indexing="xy")
    grid = np.stack(grid, axis=-1) + 0.5  # [H, W, 2], last dim is (x, y)
    vfield = goal[None, None] - grid  # [H, W, 2]
    vfield = vfield / np.linalg.norm(vfield, axis=-1, keepdims=True)
    vfield = np.nan_to_num(vfield, nan=0.0)
    if return_grid:
        return vfield, grid
    return vfield


def transform_point_to_VFD(
    point, depth, intr, downsample=1, depth_min=1e-3, depth_max=2
):
    """_summary_

    Parameters
    ----------
    point : np.array
        [3]
    depth : np.array
        [H, W]
    intr : _type_
        [3, 3]
    downsample: (0, 1]

    Returns
    -------
    vfd: np.array
        [H, W, 3], between [0, 1]
    """
    assert 0 < downsample <= 1
    depth = cv2.resize(depth, (0, 0), fx=downsample, fy=downsample)
    depth = depth.clip(1e-3, 10)

    intr[0, :] *= downsample
    intr[1, :] *= downsample
    point = torch.from_numpy(point).float()[None, None]  # [1, 1, 3]
    intr = torch.from_numpy(intr).float()[None]  # [1, 3, 3]

    uv = Project3D()(point, intr)[0, :, 0].numpy()  # [1, 2, 1] => [2]
    # inv_d = 1 / point[..., 2].squeeze().numpy()
    # inv_depth = 1 / depth
    # inv_depth_min = 1 / depth_max
    # inv_depth_max = 1 / depth_min
    # inv_d = np.clip(inv_d, inv_depth_min, inv_depth_max)
    # inv_d_ratio = (inv_d - inv_depth_min) / (inv_depth_max - inv_depth_min)

    d = point[..., 2].squeeze().numpy()
    d = np.clip(d, depth_min, depth_max)
    d_ratio = (d - depth_min) / (depth_max - depth_min)

    # print("tovfd: final d", point[..., 2].squeeze().numpy())
    height, width = depth.shape

    vfield, grid = compute_vector_field_from_coordinate(uv, height, width)
    dfield = np.ones_like(depth) * d_ratio  # [0, 1]
    vfield = 0.5 * vfield + 0.5  # [0, 1]
    vfd = np.stack([vfield[..., 0], vfield[..., 1], dfield],
                   axis=-1)  # uvd, [0, 1]
    vfd = vfd.clip(0, 1)
    return vfd


def vote_hypotheses(hypotheses, pixels, vectors, threshold=0.99):
    """_summary_

    Parameters
    ----------
    hypotheses : [H, 2], torch.Tensor

    pixels : [N, 2], torch.Tensor

    vectors : [N, 2], torch.Tensor

    threshold : float, optional
        _description_, by default 0.99

    Returns
    -------
    [H]
        vote scores for each hypothesis
    """

    hp_vecs = hypotheses[None] - pixels[:, None]  # [N, H, 2]
    hp_vecs = F.normalize(hp_vecs, dim=-1, p=2)  # [N, H, 2]
    vecs = F.normalize(vectors, dim=-1, p=2)  # [N, 2]
    vecs = vecs[:, None].repeat(1, hp_vecs.shape[1], 1)  # [N, H, 2]

    dots = torch.sum(hp_vecs * vecs, dim=-1)  # [N, H]
    dots = torch.clamp(dots, -1, 1)
    votes = torch.sum(dots > threshold, dim=0)  # [H]

    return votes


def compute_final_center(hypotheses, votes, mode="mean"):
    """_summary_

    Parameters
    ----------
    hypotheses : [H, 2], torch.Tensor

    votes : [H], torch.Tensor

    Returns
    -------
    mu: [2]
    sigma: [2, 2]
    """

    if mode == "mean" and torch.sum(votes) > 0:
        weights = votes / torch.sum(votes)
        mu = torch.sum(hypotheses * weights[:, None], dim=0)  # [2]
        # hmu = (hypotheses - mu[None])[..., None]  # [H, 2, 1]
        # sigma = hmu @ hmu.transpose(1, 2)  # [H, 2, 2]
        # sigma = torch.sum(sigma * weights[:, None, None], dim=0)  # [2, 2]
        sigma = torch.cov(hypotheses.T, aweights=weights)
        sigma_corr = (sigma[0, 1] + sigma[1, 0]) / 2
        sigma[0, 1] = sigma_corr
        sigma[1, 0] = sigma_corr

    elif mode == "max":
        mu = hypotheses[votes.argmax()]
        sigma = torch.eye(2)

    return mu, sigma


def ransac_voting_layer(
    pixels,
    vectors,
    masks=None,
    num_samples=100000,
    num_hypothesis=100,
    inlier_threshold=0.999,
    max_iter=20,
    confidence=0.99,
    mode="mean",
    verbose=False,
):
    assert len(pixels) == len(vectors)
    assert len(pixels) > 0

    if len(pixels) > num_samples:
        if masks is None:
            selected_idx = np.random.choice(
                np.arange(len(pixels)), num_samples, replace=False
            )
        else:
            valid_pixels = torch.nonzero(masks).squeeze(1)  # [N, ]
            selected_idx = np.random.choice(
                np.arange(len(valid_pixels)), num_samples, replace=False
            )
            selected_idx = valid_pixels[selected_idx]
        pixels = pixels[selected_idx]
        vectors = vectors[selected_idx]

    win_ratio = -1
    win_hypothesis = torch.zeros(2)
    win_hypotheses = torch.zeros(num_hypothesis, 2)
    win_votes = torch.zeros(num_hypothesis)
    num_all_hypothesis = 0

    mu, sigma = torch.zeros(2), torch.eye(2)

    it = 0
    while True:
        # generate hypotheses
        curr_hypotheses = generate_hypotheses(
            pixels, vectors, num_hypothesis)  # [H, 2]
        curr_hypotheses = curr_hypotheses[~torch.isnan(
            curr_hypotheses.mean(axis=1))]
        if len(curr_hypotheses) == 0:
            continue

        # vote for the hypotheses
        curr_votes = vote_hypotheses(
            curr_hypotheses, pixels, vectors, threshold=inlier_threshold
        )  # [H]
        # curr_weights = curr_votes / torch.sum(curr_votes)
        curr_win_counts, curr_win_idx = torch.max(curr_votes, dim=0)
        curr_win_hypothesis = curr_hypotheses[curr_win_idx]
        curr_win_ratio = curr_win_counts / torch.sum(curr_votes)

        # update if necessary
        if curr_win_ratio > win_ratio:
            win_ratio = curr_win_ratio
            win_hypothesis = curr_win_hypothesis
            win_hypotheses = curr_hypotheses
            win_votes = curr_votes
        it += 1
        num_all_hypothesis += num_hypothesis

        if (1 - (1 - win_ratio**2) ** num_all_hypothesis) > confidence or it > max_iter:
            if verbose:
                print(
                    "Converged with inlier ratio {:.3f} at iter {}..".format(
                        win_ratio, it - 1
                    )
                )
            break

    mu, sigma = compute_final_center(win_hypotheses, win_votes, mode=mode)

    return mu, sigma, win_hypothesis, win_hypotheses, win_ratio


def generate_hypotheses(pixels, vectors, num_hypothesis=100, eps=1e-5):
    """
    Based on: https://arxiv.org/pdf/2004.01314.pdf
    Generate hypotheses of the goal pixel from the vector field.
    Currently does not support batch processing.

    Parameters
    ----------
    pixels : [N, 2] (x, y)

    vectors : [N, 2] (x, y)


    Returns
    -------
    [H, 2]

    """

    selected_pix_id = np.random.choice(
        np.arange(pixels.shape[0]), 2 * num_hypothesis, replace=False
    )  # 2*H
    vectors = F.normalize(vectors, dim=-1, p=2)  # [N, 2]

    # selected pix1 and pix2
    pix1 = pixels[selected_pix_id[:num_hypothesis]]  # [H, 2]
    pix2 = pixels[selected_pix_id[num_hypothesis:]]  # [H, 2]
    vec1 = vectors[selected_pix_id[:num_hypothesis]]  # [H, 2]
    vec2 = vectors[selected_pix_id[num_hypothesis:]]  # [H, 2]
    v1v2 = torch.sum(vec1 * vec2, dim=-1, keepdim=False)  # [H]

    # compute length of the vectors
    l1 = torch.linalg.norm(vec1, dim=-1, keepdim=False)  # [H,]
    l2 = torch.linalg.norm(vec2, dim=-1, keepdim=False)  # [H,]

    # compute dot product of the vectors
    A = (
        1 / (eps + (l1**2 * l2**2 -
             torch.sum(vec1 * vec2, dim=-1) ** 2))[:, None]
    )  # [H, 1]

    M = torch.empty((num_hypothesis, 2, 2), device=pixels.device)  # [H, 2, 2]
    M[:, 0, 0] = l1.squeeze() ** 2
    M[:, 0, 1] = v1v2
    M[:, 1, 0] = v1v2
    M[:, 1, 1] = l2.squeeze() ** 2

    b = torch.empty((num_hypothesis, 2), device=pixels.device)  # [H, 2]
    b[:, 0] = torch.sum((pix2 - pix1) * vec1, dim=-1, keepdim=False)
    b[:, 1] = torch.sum((pix1 - pix2) * vec2, dim=-1, keepdim=False)
    lx = A * torch.bmm(M, b.unsqueeze(-1)).squeeze(
        -1
    )  # [H, 2] * ([H, 2, 2] @ [H, 2, 1]) -> [H, 2]
    x = (pix1 + lx[:, :1] * vec1 + pix2 + lx[:, 1:] * vec2) / 2  # [H, 2]
    x = x[~torch.isnan(x.mean(dim=1))]
    if len(x) == 0:
        print("Warning: all hypotheses are NaN")
        return torch.zeros(0, 2, device=pixels.device)
    return x


def transform_VFD_to_point(
    vfd, depth, intr, depth_min=1e-3, depth_max=2, return_sigma=False, **kwargs
):
    """_summary_

    Parameters
    ----------
    vfd : np.array, between [0, 1]
         [H, W, 3]
    depth : np.array
        [H, W]
    intr : _type_
        [3, 3]
    downsample: (0, 1]

    Returns
    -------

    """
    height, width = vfd.shape[:2]
    vfield, d = vfd[..., :2], vfd[..., 2]
    vfield = 2 * vfield - 1
    height_d, width_d = depth.shape
    import pdb

    pdb.set_trace()
    assert height // height_d == width // width_d
    downsample = height // height_d

    # Downsample the depth if necessary
    depth = cv2.resize(depth, (0, 0), fx=downsample, fy=downsample)
    depth = depth.clip(1e-3, 10)
    intr[0, :] *= downsample
    intr[1, :] *= downsample
    intr = torch.from_numpy(intr).float()[None]

    # Generate grid
    grid = np.meshgrid(np.arange(width), np.arange(height), indexing="xy")
    grid = np.stack(grid, axis=-1) + 0.5  # [H, W, 2], last dim is (x, y)
    grid_flatten = grid.reshape(-1, 2)  # [H*W, 2]
    vfield_flatten = vfield.reshape(-1, 2)  # [H*W, 2]

    # RANSAC voting to find the goal pixel
    grid_torch = torch.from_numpy(grid_flatten).float()  # [H, W, 2]
    vfield_torch = torch.from_numpy(vfield_flatten).float()  # [H, W, 2]
    vfield_torch = F.normalize(vfield_torch, dim=-1, p=2)  # [H, W, 2]

    uv, sigma, win_hypothesis, win_hypotheses, inlier_ratio = ransac_voting_layer(
        grid_torch, vfield_torch, **kwargs
    )
    # Get the depth value
    # inv_d_ratio = d.mean()
    # inv_depth = 1 / depth
    # inv_depth_min = 1 / depth_max
    # inv_depth_max = 1 / depth_min
    # inv_d = inv_d_ratio * (inv_depth_max - inv_depth_min) + inv_depth_min
    # inv_d = np.clip(inv_d, inv_depth_min, inv_depth_max)
    # d = 1 / inv_d
    d_ratio = d.mean()
    d = d_ratio * (depth_max - depth_min) + depth_min
    d = np.clip(d, depth_min, depth_max)

    # Get the 3D point
    inv_intr = np.linalg.pinv(intr.squeeze().numpy())
    point = (np.array([uv[0], uv[1], 1]) @ inv_intr.T) * d
    # if np.isnan(point).any():
    #     import pdb

    #     pdb.set_trace()
    if return_sigma:
        return point, sigma
    return point


def visualize_vector_field(vector_field):
    flow_color = flow_vis.flow_to_color(vector_field, convert_to_bgr=False)
    return flow_color


def compute_trajectory_bounds(trajectory, enlarge_ratio=4, different_z_size=False):
    start_point = trajectory[0]
    end_point = trajectory[-1]
    size_xy = np.max(np.abs(end_point - start_point))
    size_z = np.abs(end_point[2] - start_point[2])
    size_z = max(size_z, 0.025)
    if different_z_size:
        size = np.array([size_xy, size_xy, size_z])
    else:
        size_xyz = max(size_xy, size_z)
        size = np.array([size_xyz, size_xyz, size_xyz])
    size = size * enlarge_ratio
    center = (start_point + end_point) / 2
    _min_bound = center - size / 2
    _max_bound = center + size / 2
    _bounds = np.stack([_min_bound, _max_bound], axis=0)
    min_bound, max_bound = _bounds.min(axis=0), _bounds.max(axis=0)
    return min_bound, max_bound


def compute_trajectory_bounds_with_radii(
    trajectory, radii=0.8, enlarge_ratio=1, short_z_size=False
):
    # start_point = trajectory[0]
    # end_point = trajectory[-1]

    dirvec = trajectory[-1] - trajectory[0]
    dirvec = dirvec / np.linalg.norm(dirvec)

    start_point = trajectory[0] - dirvec * radii
    end_point = trajectory[0] + dirvec * radii

    size_xy = np.max(np.abs(end_point - start_point))

    size_z = np.abs(end_point[2] - start_point[2])
    size_z = max(size_z, 0.025)

    if short_z_size:
        size = np.array([size_xy, size_xy, size_z])
    else:
        size = np.array([size_xy, size_xy, size_xy])

    size = size * enlarge_ratio
    center = (start_point + end_point) / 2
    _min_bound = center - size / 2
    _max_bound = center + size / 2
    _bounds = np.stack([_min_bound, _max_bound], axis=0)
    min_bound, max_bound = _bounds.min(axis=0), _bounds.max(axis=0)
    return min_bound, max_bound


def compute_model_size(model):
    size_model = 0
    for param in model.parameters():
        if param.data.is_floating_point():
            size_model += param.numel() * torch.finfo(param.data.dtype).bits
        else:
            size_model += param.numel() * torch.iinfo(param.data.dtype).bits

    print(f"model size: {size_model} / bit | {size_model / 8e6:.2f} / MB")


def load_and_freeze_clip_model(version: str):
    """Load CLIP model and freeze its parameters.

    Args:
        version: CLIP model version.

    Return:
        CLIP model.
    """
    clip_model, _ = clip.load(version, device="cpu", jit=False)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    return clip_model


def encode_text_clip(
    clip_model: torch.nn.Module,
    raw_text: List,
    max_length: int = 32,
    device: str = "cpu",
):
    """Encode text using CLIP model.

    Args:
        clip_model: CLIP model.
        raw_text: List of raw text.
        device: Device to use.

    Return:
        Detached encoded text.
    """
    if max_length is not None:
        default_context_length = 77
        context_length = max_length + 2
        assert context_length < default_context_length
        texts = clip.tokenize(
            raw_text, context_length=context_length, truncate=True
        ).to(device)
        zero_pad = torch.zeros(
            [texts.shape[0], default_context_length - context_length],
            dtype=texts.dtype,
            device=texts.device,
        )
        texts = torch.cat([texts, zero_pad], dim=1)
    else:
        texts = clip.tokenize(raw_text, truncate=True).to(device)

    encoded_text = clip_model.encode_text(texts).float()  # [bs, clip_dim]
    return texts.detach(), encoded_text.detach()


def descale_trajectory_length(traj, scale):
    """
    - traj: B x N x H x 3
    """
    traj_init = traj[..., 0:1, :]  # [B, N, 1, 3]
    traj_dist = traj - traj_init  # [B, N, H, 3]
    traj = traj_init + traj_dist * scale
    return traj


def scale_trajectory_length(traj, scale):
    traj_init = traj[..., 0:1, :]  # [B, N, 1, 3]
    traj_dist = traj - traj_init  # [B, N, H, 3]
    traj = traj_init + traj_dist / scale
    return traj


def get_normal_from_depth_in_batch(depth, intrinsics, return_points=False):
    batch_size, h, w = depth.shape
    device = depth.device
    # depth_np = depth.cpu().numpy()
    # depth_smooth = []
    # for depth_i in depth_np:
    #     depth_i = cv2.GaussianBlur(depth_i, (11, 11), 0)
    #     depth_smooth.append(depth_i)
    # depth = torch.tensor(depth_smooth).to(device)

    cam_points = BackprojectDepth(h, w)(depth, intrinsics)  # [B, 3, H, W]
    cam_points = cam_points.view(1, 3, h, w)
    tu = cam_points[:, :, 1:-1, 2:] - cam_points[:, :, 1:-1, :-2]
    tv = cam_points[:, :, 2:, 1:-1] - cam_points[:, :, :-2, 1:-1]
    normal = tu.cross(tv, dim=1)
    normal = torch.cat(
        [
            torch.zeros(batch_size, 3, 1, w - 2).to(device),
            normal,
            torch.zeros(batch_size, 3, 1, w - 2).to(device),
        ],
        dim=-2,
    )
    normal = torch.cat(
        [
            torch.zeros(batch_size, 3, h, 1).to(device),
            normal,
            torch.zeros(batch_size, 3, h, 1).to(device),
        ],
        dim=-1,
    )
    normal = torch.nn.functional.normalize(normal, dim=1)
    normal[:, -1] *= -1  # [B, 3, H, W]

    # normal_np = normal.cpu().numpy()
    # normal_smooth = []
    # for normal_i in normal_np:
    #     normal_i = normal_i.transpose(1, 2, 0)
    #     normal_i = cv2.GaussianBlur(normal_i, (11, 11), 0)
    #     normal_i = normal_i.transpose(2, 0, 1)
    #     normal_smooth.append(normal_i)
    # normal = torch.tensor(normal_smooth).to(device)

    if return_points:
        return normal, cam_points
    return normal


def get_normal_clutters_in_batch(normals, masks=None, n_clusters=3):
    batch_size, _, h, w = normals.shape
    if masks is not None:
        assert masks.shape == (batch_size, h, w)
    cluster_labels, cluster_centers, cluster_counts = [], [], []
    for bi in range(batch_size):
        normals_i = normals[bi]
        normals_i = normals_i.permute(1, 2, 0).reshape(-1, 3)  # [H*W, 3]
        normals_i_np = normals_i.cpu().numpy()
        if masks is not None:
            masks_i = masks[bi]
            masks_i = masks_i.view(-1)  # [H*W]
            masks_i_np = masks_i.cpu().numpy()
            normals_i_np = normals_i_np[masks_i_np > 0]
        kmeans_i = KMeans(n_clusters=n_clusters,
                          random_state=0).fit(normals_i_np)
        cluster_labels_i = kmeans_i.labels_
        cluster_centers_i = kmeans_i.cluster_centers_
        cluster_counts_i = np.bincount(cluster_labels_i)
        cluster_labels.append(cluster_labels_i)
        cluster_centers.append(cluster_centers_i)
        cluster_counts.append(cluster_counts_i)
    cluster_labels = torch.tensor(cluster_labels).to(
        normals.device)  # [B, H*W]
    cluster_centers = torch.tensor(cluster_centers).to(
        normals.device)  # [B, K, 3]
    cluster_counts = torch.tensor(cluster_counts).to(normals.device)  # [B, K]
    cluster_top_id = torch.argmax(cluster_counts, dim=1)  # [B]
    cluster_top_normal = torch.gather(
        cluster_centers, 1, cluster_top_id.unsqueeze(-1).repeat(1, 1, 3)
    )  # [B, 1, 3]
    cluster_top_normal = cluster_top_normal.squeeze(1)
    return cluster_top_normal, cluster_labels, cluster_centers, cluster_counts


def trimesh_to_o3d(mesh, scale=1.0, with_color=True):
    mesh_o3d = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(mesh.vertices * scale),
        o3d.utility.Vector3iVector(mesh.faces),
    )
    # mesh_o3d.paint_uniform_color([255 / 255, 229 / 255, 180 / 255])
    if with_color:
        mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(
            mesh.visual.vertex_colors[:, :3] / 255
        )
    mesh_o3d.compute_vertex_normals()
    return mesh_o3d


# Encoding: Convert text to a NumPy array of ASCII values
def encode_text_list(text_list):
    encoded_list = [np.array([ord(char) for char in word])
                    for word in text_list]
    return encoded_list


# Decoding: Convert NumPy array of ASCII values back to text
def decode_text_list(encoded_list):
    decoded_list = ["".join([chr(num) for num in word])
                    for word in encoded_list]
    return decoded_list


def get_context_data_from_rgbd(
    color_orig,
    depth_orig,
    intr_orig,
    voxel_resolution=32,
    fine_voxel_resolution=None,
    fine_voxel_margin=30,
    default_image_shape=(256, 456),
    context_image_shape=(256, 448),
    to_tensor=True,
    tight_bounds=False,
):
    """
    color: np.ndarray, shape [H, W, 3], within [0, 255], in RGB
    depth: np.ndarray, shape [H, W], in meters
    intr: np.ndarray, shape [3, 3], intrinsic matrix
    voxel_resolution: int, resolution of the voxel grid
    fine_voxel_resolution: int, resolution of the fine voxel grid
    """

    def data_batch_to_tensor(data_batch):
        for k, v in data_batch.items():
            if not isinstance(v, np.ndarray):
                continue
            elif "color" in k:
                data_batch[k] = T.ToTensor()(v)
            else:
                data_batch[k] = torch.from_numpy(v).float()

    data_batch = {}
    resize_scale = default_image_shape[0] / color_orig.shape[0]
    color = cv2.resize(
        color_orig, (default_image_shape[1], default_image_shape[0]))
    depth = cv2.resize(
        depth_orig,
        (default_image_shape[1], default_image_shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    intr = intr_orig.copy()
    intr[:2] *= resize_scale
    inv_intr = np.linalg.inv(intr)
    color = center_crop_image(
        color, context_image_shape[0], context_image_shape[1])
    depth = center_crop_image(
        depth, context_image_shape[0], context_image_shape[1])
    vol_bnds = np.zeros((3, 2))
    view_frust_pts = get_view_frustum(depth_orig, intr_orig, np.eye(4))
    vol_bnds[:, 0] = np.minimum(
        vol_bnds[:, 0], np.amin(view_frust_pts, axis=1)).min()
    vol_bnds[:, 1] = np.maximum(
        vol_bnds[:, 1], np.amax(view_frust_pts, axis=1)).max()
    if tight_bounds:
        vol_bnds[:, 0] *= 1.1
        vol_bnds[:, 1] *= 0.9
    tsdf = TSDFVolume(vol_bnds, voxel_dim=voxel_resolution, num_margin=30)
    tsdf.integrate(color_orig.copy(), depth_orig, intr_orig, np.eye(4))
    tsdf_grid = tsdf.get_tsdf_volume()
    data_batch.update(
        {
            "color_raw": color_orig,
            "depth_raw": depth_orig,
            "intrinsics_raw": intr_orig,
            "color": color,
            "depth": depth,
            "intrinsics": intr,
            "inv_intrinsics": inv_intr,
            "tsdf_grid": tsdf_grid,
            "voxel_bounds": vol_bnds[0],
        }
    )
    if fine_voxel_resolution is not None:
        tsdf_fine = TSDFVolume(
            vol_bnds, voxel_dim=fine_voxel_resolution, num_margin=fine_voxel_margin, unknown_free=False
        )
        tsdf_fine.integrate(color_orig.copy(), depth_orig,
                            intr_orig, np.eye(4))
        tsdf_grid_fine = tsdf_fine.get_tsdf_volume()
        data_batch["tsdf_grid_fine"] = tsdf_grid_fine
        mesh = tsdf_fine.get_mesh()
        data_batch.update(
            {
                "mesh": mesh,
            }
        )
        # o3d.visualization.draw([mesh])
    if to_tensor:
        data_batch_to_tensor(data_batch)
    return data_batch


def smooth_depth_image(depth_image, max_hole_size=10):
    mask = np.zeros(depth_image.shape, dtype=np.uint8)
    mask[depth_image == 0] = 1

    # Do not include in the mask the holes bigger than the maximum hole size
    kernel = np.ones((max_hole_size, max_hole_size), np.uint8)
    erosion = cv2.erode(mask, kernel, iterations=1)
    mask = mask - erosion

    smoothed_depth_image = cv2.inpaint(
        depth_image.astype(np.uint16), mask, max_hole_size, cv2.INPAINT_NS
    )

    return smoothed_depth_image


def smooth_rotation_matrices(rotation_matrices, smooth_factor=0.1):
    """
    Smooth a sequence of rotation matrices using SLERP.

    Args:
        rotation_matrices: A numpy array of shape (N, 3, 3), where N is the number of matrices.
        smooth_factor: A float between 0 and 1 indicating how much to smooth. Lower values mean more smoothing.

    Returns:
        A numpy array of the smoothed rotation matrices.
    """
    # Convert rotation matrices to quaternions
    # quaternions = rotation_matrices_to_quaternions(rotation_matrices)
    quaternions = R.from_matrix(rotation_matrices).as_quat()
    # Interpolate quaternions with SLERP
    smoothed_quaternions = []
    for i in range(1, len(quaternions)):
        # r1 = R.from_quat(quaternions[i - 1])
        # r2 = R.from_quat(quaternions[i])
        # import pdb; pdb.set_trace()
        slerp = Slerp([0, 1], R.from_quat(
            [quaternions[i - 1], quaternions[i]]))
        t = np.linspace(0, 1, num=int(1 / smooth_factor))
        interp_quats = slerp(t).as_quat()
        interp_quats_mid = interp_quats[len(interp_quats) // 2]
        smoothed_quaternions.append(interp_quats_mid)
    # import pdb; pdb.set_trace()
    # Flatten the list of interpolated quaternions
    smoothed_quaternions = np.stack(smoothed_quaternions, axis=0)

    # Convert smoothed quaternions back to rotation matrices
    # smoothed_rotation_matrices = quaternions_to_rotation_matrices(smoothed_quaternions)
    smoothed_rotation_matrices = R.from_quat(smoothed_quaternions).as_matrix()
    smoothed_rotation_matrices = np.concatenate(
        [rotation_matrices[:1], smoothed_rotation_matrices], axis=0
    )
    return smoothed_rotation_matrices
