"""Unprojection / projection helpers against the backbone's predicted geometry."""

import math

import numpy as np
import torch
import torch.nn.functional as F


def unproject_uv(cache, u, v, frame, device):
    """Lift pixels (u, v) of `frame` to 3D using the predicted depth and intrinsics."""
    depth = cache["predictions"]["depth"]
    K = cache["predictions"]["intrinsics"][0, frame].cpu().numpy()
    H, W = cache["H"], cache["W"]
    u_g = 2.0 * torch.from_numpy(u).float() / max(W - 1, 1) - 1.0
    v_g = 2.0 * torch.from_numpy(v).float() / max(H - 1, 1) - 1.0
    grid = torch.stack([u_g, v_g], dim=-1).unsqueeze(0).unsqueeze(0).to(device)
    z = F.grid_sample(depth[0, frame].unsqueeze(0).unsqueeze(0), grid,
                      mode="bilinear", align_corners=True).squeeze().cpu().numpy()
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    x = (u - cx) * z / max(fx, 1e-6)
    y = (v - cy) * z / max(fy, 1e-6)
    return np.stack([x, y, z], axis=-1).astype(np.float32)


def project_to_uv(pts3d, K):
    uvw = pts3d @ K.T
    return uvw[:, :2] / np.clip(uvw[:, 2:3], 1e-8, None)


def w2c_matrix(cache, frame):
    m = np.eye(4, dtype=np.float32)
    m[:3, :] = cache["predictions"]["extrinsics"][0, frame].cpu().numpy()
    return m


def to_frame0(pts, cache, frame):
    """Move points from `frame`'s camera coords into the chunk's frame-0 coords."""
    T = w2c_matrix(cache, 0) @ np.linalg.inv(w2c_matrix(cache, frame))
    h = np.hstack([pts, np.ones((len(pts), 1), dtype=np.float32)])
    return (T @ h.T).T[:, :3]


def alignment_grid(H, W, n_points):
    n_h = int(math.sqrt(n_points * H / W))
    n_w = int(n_points / n_h)
    ys = np.linspace(0, H - 1, n_h, dtype=np.float32)
    xs = np.linspace(0, W - 1, n_w, dtype=np.float32)
    gx, gy = np.meshgrid(xs, ys)
    return np.stack([gx.ravel(), gy.ravel()], axis=-1)


def unproject_depth_map(cache, frame, device=None):
    """Dense unprojection of `frame`'s predicted depth into the chunk's frame-0 coords."""
    depth = cache["predictions"]["depth"][0, frame].float().cpu().numpy()
    K = cache["predictions"]["intrinsics"][0, frame].cpu().numpy()
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    pts = np.stack([(u - cx) * depth / max(fx, 1e-6),
                    (v - cy) * depth / max(fy, 1e-6), depth], axis=-1)
    T = w2c_matrix(cache, 0) @ np.linalg.inv(w2c_matrix(cache, frame))
    return pts @ T[:3, :3].T + T[:3, 3][None, None, :]


def scene_scale(cache):
    """Mean distance-to-origin of the chunk's predicted geometry.

    Queries are expressed in units of this scale, matching how the model was
    trained; predictions are scaled back by the same factor.
    """
    norms = []
    for f in range(cache["S"]):
        pts = unproject_depth_map(cache, f).reshape(-1, 3)
        valid = np.linalg.norm(pts, axis=-1) > 1e-8
        if valid.any():
            norms.append(np.linalg.norm(pts[valid], axis=-1))
    return max(float(np.mean(np.concatenate(norms))), 1e-6) if norms else 1.0
