"""Sim(3) helpers used to stitch consecutive chunks into one coordinate frame."""

import numpy as np


def umeyama_alignment(src, dst, with_scale=True):
    """Estimate Sim(3) such that dst ~= c * R @ src + t (Umeyama 1991)."""
    assert src.shape == dst.shape and src.shape[1] == 3
    n = src.shape[0]
    mu_src, mu_dst = src.mean(axis=0), dst.mean(axis=0)
    src_c, dst_c = src - mu_src, dst - mu_dst
    var_src = np.sum(src_c ** 2) / n
    U, D, Vt = np.linalg.svd((dst_c.T @ src_c) / n)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    c = np.sum(D * np.diag(S)) / max(var_src, 1e-12) if with_scale else 1.0
    return c, R, mu_dst - c * R @ mu_src


def apply_sim3(pts, c, R, t):
    return (c * (pts @ R.T)) + t


def invert_sim3(c, R, t):
    c_inv = 1.0 / max(c, 1e-12)
    R_inv = R.T
    return c_inv, R_inv, -c_inv * R_inv @ t


def compose_sim3(c1, R1, t1, c2, R2, t2):
    return c2 * c1, R2 @ R1, c2 * R2 @ t1 + t2
