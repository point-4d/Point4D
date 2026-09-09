"""Long-video 4D tracking by chaining 3D queries across overlapping chunks.

A video longer than the model's context is split into overlapping chunks. Each
chunk is encoded once; every query is decoded at every frame of that chunk. At
the handoff into the next chunk the *predicted 3D point* is carried over
directly -- no reprojection into pixel space -- which is what lets a query
survive occlusion and leaving the field of view.
"""

import math
from dataclasses import dataclass

import cv2
import numpy as np
import torch

from point4d.inference.geometry import (alignment_grid, project_to_uv,
                                        scene_scale, to_frame0, unproject_uv,
                                        unproject_depth_map, w2c_matrix)
from point4d.inference.sim3 import (apply_sim3, compose_sim3, invert_sim3,
                                    umeyama_alignment)


@dataclass
class LongTrackConfig:
    chunk_size: int = 48
    overlap: int = 8
    image_hw: tuple = (294, 518)      # (H, W) fed to the model
    query_chunk_size: int = 40000     # queries per decoder call
    n_align_points: int = 1024        # grid size for chunk-to-chunk alignment
    confidence_percentile: float = 85.0
    bf16: bool = True
    overlap_strategy: str = "smooth"  # "smooth" | "prefer_last"
    geometry_stride: int = 4          # pixel stride of the returned depth clouds


# ---------------------------------------------------------------------------
# Model-facing helpers
# ---------------------------------------------------------------------------

def _preprocess(images, hw):
    """(T, H, W, 3) uint8 RGB -> (1, T, 3, h, w) float in [0, 1]."""
    H, W = hw
    out = []
    for img in images:
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
        out.append(torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1))
    return torch.stack(out, dim=0).unsqueeze(0)


def _encode_chunk(model, images, frames, cfg, device):
    """Run the backbone once for a chunk and cache its KV for the decoder."""
    imgs = _preprocess(images[frames], cfg.image_hw).to(device)
    if cfg.bf16:
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            cache = model.forward_backbone(imgs)
            cache["memory_kv_cache"] = model.compute_d4rt_kv_cache(cache)
    else:
        cache = model.forward_backbone(imgs)
        cache["memory_kv_cache"] = model.compute_d4rt_kv_cache(cache)
    # queries live in units of the chunk's own predicted scale
    cache["scale"] = scene_scale(cache)
    return cache


def _build_queries(xyz, uv, n_local):
    """One [x, y, z, p, s, t, c] query per (point, target frame)."""
    n = xyz.shape[0]
    queries, uvs = [], []
    for ell in range(n_local):
        q = np.zeros((n, 7), dtype=np.float32)
        q[:, 0:3] = xyz
        q[:, 5] = ell          # target frame within the chunk
        queries.append(q)
        uvs.append(uv.copy())
    return np.concatenate(queries, 0), np.concatenate(uvs, 0)


def _decode(model, cache, queries, uvs, device, cfg,
            cached_patch_emb=None, return_patch_emb=False):
    """Decode queries in batches; returns (points, confidence[, patch_emb])."""
    pts, confs, embs = [], [], []
    for i in range(0, queries.shape[0], cfg.query_chunk_size):
        q = torch.from_numpy(queries[i:i + cfg.query_chunk_size]).unsqueeze(0).to(device).float()
        uv = torch.from_numpy(uvs[i:i + cfg.query_chunk_size]).unsqueeze(0).to(device).float()
        ce = cached_patch_emb[:, i:i + cfg.query_chunk_size] if cached_patch_emb is not None else None
        kw = dict(d4rt_uvs=uv, cached_local_patch_emb=ce,
                  return_local_patch_emb=return_patch_emb,
                  memory_kv_cache=cache.get("memory_kv_cache"))
        if cfg.bf16:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                out = model.forward_d4rt_only(cache, q, **kw)
        else:
            out = model.forward_d4rt_only(cache, q, **kw)
        pts.append(out["d4rt_pred"][0].float().cpu().numpy())
        confs.append(out["d4rt_pred_conf"][0].float().cpu().numpy())
        if return_patch_emb:
            embs.append(out["local_patch_emb"].cpu())
    result = (np.concatenate(pts, 0), np.concatenate(confs, 0))
    return result + (torch.cat(embs, dim=1),) if return_patch_emb else result


# ---------------------------------------------------------------------------
# Chunk-to-chunk alignment
# ---------------------------------------------------------------------------

def _alignment_points(cache_k, cache_next, ell_k, n_overlap, grid, device):
    """Depth point clouds of the shared frames, each in its own chunk's frame 0."""
    pk, pn = [], []
    for j in range(n_overlap):
        a = unproject_uv(cache_k, grid[:, 0], grid[:, 1], ell_k + j, device)
        pk.append(to_frame0(a, cache_k, ell_k + j))
        b = unproject_uv(cache_next, grid[:, 0], grid[:, 1], j, device)
        pn.append(to_frame0(b, cache_next, j))
    return np.concatenate(pk, 0), np.concatenate(pn, 0)


def _chunk_sim3(pts_k, pts_next, percentile):
    """Sim(3) taking chunk k's frame-0 coords into chunk k+1's frame-0 coords."""
    valid = ((np.linalg.norm(pts_k, axis=-1) > 1e-8)
             & (np.linalg.norm(pts_next, axis=-1) > 1e-8)
             & np.isfinite(pts_k).all(axis=-1)
             & np.isfinite(pts_next).all(axis=-1))
    if valid.sum() < 10:
        return 1.0, np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32)
    src, dst = pts_k[valid], pts_next[valid]
    n_keep = max(int(len(src) * percentile / 100), 10)
    keep = np.argsort(np.linalg.norm(src - dst, axis=-1))[:n_keep]
    return umeyama_alignment(src[keep], dst[keep], with_scale=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

@torch.no_grad()
def track_long_video(model, images, query_uv, cfg=None, device=None, query_xyz=None,
                     return_geometry=False):
    """Track 2D query pixels of frame 0 through a long video, in 3D.

    Args:
        model: a loaded `D4RT_DA3`.
        images: (T, H, W, 3) uint8 RGB frames.
        query_uv: (N, 2) pixel coordinates in frame 0, in `cfg.image_hw` pixels.
        cfg: `LongTrackConfig`.
        device: torch device; defaults to the model's.
        query_xyz: (N, 3) optional 3D positions of the queries in frame 0's camera
            coordinates. Defaults to lifting `query_uv` through the predicted depth.
        return_geometry: also return the per-frame scene geometry the tracker
            already computes internally, in the same coordinate frame as the
            trajectories. Costs a dense unprojection per frame.

    Returns:
        trajectories: (T, N, 3) in frame 0's camera coordinates.
        confidence:   (T, N).
        geometry: only when `return_geometry`, a dict with
            points     (T, h, w, 3) depth clouds, `cfg.geometry_stride` subsampled
            cam_R      (T, 3, 3) and cam_t (T, 3), camera-to-world of each frame
            intrinsics (T, 3, 3) at the model's input resolution
            hw         the (h, w) of `points`
    """
    cfg = cfg or LongTrackConfig()
    device = device or next(model.parameters()).device
    images = np.asarray(images)
    T, N = len(images), len(query_uv)

    stride = cfg.chunk_size - cfg.overlap
    n_chunks = math.ceil((T - cfg.chunk_size) / stride) + 1 if T > cfg.chunk_size else 1
    chunks = [list(range(k * stride, min(k * stride + cfg.chunk_size, T)))
              for k in range(n_chunks)]
    grid = alignment_grid(*cfg.image_hw, cfg.n_align_points)

    traj = np.full((T, N, 3), np.nan, dtype=np.float32)
    conf = np.full((T, N), np.nan, dtype=np.float32)

    geom = None
    if return_geometry:
        gs = max(cfg.geometry_stride, 1)
        gh, gw = len(range(0, cfg.image_hw[0], gs)), len(range(0, cfg.image_hw[1], gs))
        geom = {"points": np.full((T, gh, gw, 3), np.nan, dtype=np.float32),
                "cam_R": np.tile(np.eye(3, dtype=np.float32), (T, 1, 1)),
                "cam_t": np.zeros((T, 3), dtype=np.float32),
                "intrinsics": np.zeros((T, 3, 3), dtype=np.float32),
                "hw": (gh, gw)}
        done = np.zeros(T, dtype=bool)

    cache = _encode_chunk(model, images, chunks[0], cfg, device)
    # queries start as frame-0 pixels lifted through the predicted depth,
    # unless the caller already knows their 3D positions
    xyz = (np.asarray(query_xyz, dtype=np.float32).copy() if query_xyz is not None
           else unproject_uv(cache, query_uv[:, 0], query_uv[:, 1], 0, device))
    uv = query_uv.astype(np.float32).copy()

    # Sim(3) mapping global (chunk-0) coords into the current chunk's coords
    c_acc, R_acc, t_acc = 1.0, np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32)
    patch_emb = None

    for k, frames in enumerate(chunks):
        n_local = len(frames)
        s = cache["scale"]
        queries, uvs = _build_queries(xyz / s, uv, n_local)

        if k == 0:
            pred, pconf, patch_emb = _decode(model, cache, queries, uvs, device, cfg,
                                             return_patch_emb=True)
            patch_emb = patch_emb[:, :N, :]
        else:
            # the appearance patch is taken once, from where the query was born
            tiled = patch_emb.repeat(1, n_local, 1).to(device)
            pred, pconf = _decode(model, cache, queries, uvs, device, cfg,
                                  cached_patch_emb=tiled)
        pred = pred.reshape(n_local, N, 3) * s
        pconf = pconf.reshape(n_local, N)

        c_inv, R_inv, t_inv = invert_sim3(c_acc, R_acc, t_acc)
        for ell in range(n_local):
            g = frames[ell]
            pts_global = apply_sim3(pred[ell], c_inv, R_inv, t_inv)
            if np.isnan(traj[g, 0, 0]):
                traj[g], conf[g] = pts_global, pconf[ell]
            elif cfg.overlap_strategy == "prefer_last":
                traj[g], conf[g] = pts_global, pconf[ell]
            else:  # blend across the overlap so the seam is not visible
                w = ell / max(cfg.overlap - 1, 1)
                traj[g] = (1 - w) * traj[g] + w * pts_global
                conf[g] = (1 - w) * conf[g] + w * pconf[ell]

            # geometry is not blended: a frame keeps the first chunk that saw it
            if return_geometry and not done[g]:
                done[g] = True
                pts = unproject_depth_map(cache, ell)[::gs, ::gs]
                flat = apply_sim3(pts.reshape(-1, 3), c_inv, R_inv, t_inv)
                geom["points"][g] = flat.reshape(gh, gw, 3)
                c2w = w2c_matrix(cache, 0) @ np.linalg.inv(w2c_matrix(cache, ell))
                geom["cam_R"][g] = R_inv @ c2w[:3, :3]
                geom["cam_t"][g] = apply_sim3(c2w[None, :3, 3], c_inv, R_inv, t_inv)[0]
                geom["intrinsics"][g] = (
                    cache["predictions"]["intrinsics"][0, ell].cpu().numpy())

        if k == n_chunks - 1:
            break

        # ---- handoff: carry the predicted 3D points into the next chunk ----
        ell_k = (k + 1) * stride - frames[0]
        cache_next = _encode_chunk(model, images, chunks[k + 1], cfg, device)
        n_overlap = min(n_local - ell_k, len(chunks[k + 1]))
        pts_k, pts_next = _alignment_points(cache, cache_next, ell_k, n_overlap, grid, device)
        c_loc, R_loc, t_loc = _chunk_sim3(pts_k, pts_next, cfg.confidence_percentile)

        xyz = apply_sim3(pred[ell_k], c_loc, R_loc, t_loc)
        K_next = cache_next["predictions"]["intrinsics"][0, 0].cpu().numpy()
        uv = project_to_uv(xyz, K_next)
        uv[:, 0] = np.clip(uv[:, 0], 0, cfg.image_hw[1] - 1)
        uv[:, 1] = np.clip(uv[:, 1], 0, cfg.image_hw[0] - 1)

        c_acc, R_acc, t_acc = compose_sim3(c_acc, R_acc, t_acc, c_loc, R_loc, t_loc)
        cache = cache_next

    return (traj, conf, geom) if return_geometry else (traj, conf)
