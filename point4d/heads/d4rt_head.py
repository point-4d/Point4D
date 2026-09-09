# Copyright (c) 2026 Minsik Jeon.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# The module layout and the activation dispatch follow VGGT
# (https://github.com/facebookresearch/vggt).


# Inspired by https://github.com/DepthAnything/Depth-Anything-V2


from typing import List
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageDraw
import os

class FourierEncoding(nn.Module):
    """
    SRT-style Fourier positional encoding.
 
    For each input coordinate, computes:
        [sin(2^k * π * x), cos(2^k * π * x)]  for k = start_octave, ..., start_octave + num_octaves - 1
 
    Frequencies grow exponentially (octave = 2x), with base frequency 2^start_octave * π.
    Works well for inputs in [-3, 3] or [0, 1] range (e.g., normalized coordinates, point clouds).
 
    Args:
        in_dim: Input dimensionality (2 for uv, 3 for xyz, etc.)
        num_octaves: Number of frequency bands per coordinate.
        start_octave: Starting octave index. Base frequency = 2^start_octave * π.
        out_dim: Optional linear projection to desired output dimension.
    """
 
    def __init__(self, in_dim=2, num_octaves=15, start_octave=0, out_dim=None):
        super().__init__()
        self.in_dim = in_dim
        self.num_octaves = num_octaves
        self.start_octave = start_octave
 
        feat_dim = in_dim * num_octaves * 2  # sin + cos per octave per coord
        self.feat_dim = feat_dim
        self.proj = None
        if out_dim is not None and out_dim != feat_dim:
            self.proj = nn.Linear(feat_dim, out_dim)
 
    def forward(self, x):
        """
        Args:
            x: (..., in_dim) — any batch shape, last dim = coordinate dimension.
        Returns:
            (..., feat_dim) or (..., out_dim) if projection is set.
        """
        assert x.shape[-1] == self.in_dim, (
            f"Expected last dim {self.in_dim}, got {x.shape[-1]}"
        )
        device, dtype = x.device, x.dtype
 
        octaves = torch.arange(
            self.start_octave,
            self.start_octave + self.num_octaves,
            device=device,
            dtype=dtype,
        )
        multipliers = (2.0 ** octaves) * math.pi  # (num_octaves,)
 
        scaled = x.unsqueeze(-1) * multipliers  # (..., in_dim, num_octaves)
 
        sines = torch.sin(scaled)
        cosines = torch.cos(scaled)
 
        # Flatten: (..., in_dim, num_octaves) -> (..., in_dim * num_octaves)
        prefix = scaled.shape[:-2]
        sines = sines.reshape(*prefix, self.in_dim * self.num_octaves)
        cosines = cosines.reshape(*prefix, self.in_dim * self.num_octaves)
 
        f = torch.cat([sines, cosines], dim=-1)  # (..., in_dim * num_octaves * 2)
 
        if self.proj is not None:
            f = self.proj(f)
        return f
 

def make_head(dim_in: int, output_dim: int, hidden_dim: int = 1024):
    return nn.Sequential(
        # nn.LayerNorm(dim_in),
        nn.Linear(dim_in, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, output_dim),
    )

def _project_kv(attn: nn.MultiheadAttention, m: torch.Tensor):
    """Project memory through K and V weights of an nn.MultiheadAttention.

    Returns cached_k, cached_v each shaped (B, nhead, S, head_dim) ready for
    F.scaled_dot_product_attention.
    """
    d = attn.embed_dim
    # in_proj_weight is [W_q; W_k; W_v] each (d, d)
    w_k = attn.in_proj_weight[d:2*d]
    w_v = attn.in_proj_weight[2*d:3*d]
    b_k = attn.in_proj_bias[d:2*d] if attn.in_proj_bias is not None else None
    b_v = attn.in_proj_bias[2*d:3*d] if attn.in_proj_bias is not None else None
    k = F.linear(m, w_k, b_k)  # (B, S, d)
    v = F.linear(m, w_v, b_v)  # (B, S, d)
    B, S, _ = k.shape
    head_dim = d // attn.num_heads
    k = k.reshape(B, S, attn.num_heads, head_dim).transpose(1, 2)  # (B, nhead, S, head_dim)
    v = v.reshape(B, S, attn.num_heads, head_dim).transpose(1, 2)
    return k, v


def _cross_attn_with_cache(attn: nn.MultiheadAttention, q_input: torch.Tensor,
                           cached_k: torch.Tensor, cached_v: torch.Tensor,
                           dropout_p: float = 0.0):
    """Run cross-attention using pre-cached K/V.

    q_input: (B, Q, d) — raw query (will be projected through W_q).
    cached_k, cached_v: (B, nhead, S, head_dim) — pre-projected.
    Returns: (B, Q, d) attention output.
    """
    d = attn.embed_dim
    w_q = attn.in_proj_weight[:d]
    b_q = attn.in_proj_bias[:d] if attn.in_proj_bias is not None else None
    q = F.linear(q_input, w_q, b_q)  # (B, Q, d)
    B, Q, _ = q.shape
    head_dim = d // attn.num_heads
    q = q.reshape(B, Q, attn.num_heads, head_dim).transpose(1, 2)  # (B, nhead, Q, head_dim)
    dp = dropout_p if attn.training else 0.0
    out = F.scaled_dot_product_attention(q, cached_k, cached_v, dropout_p=dp)  # (B, nhead, Q, head_dim)
    out = out.transpose(1, 2).reshape(B, Q, d)  # (B, Q, d)
    out = attn.out_proj(out)
    return out

class CrossAttentionDecoderLayer(nn.Module):
    """
    Decoder layer without tgt self-attention.
    Applies: cross-attention (tgt <- memory) + FFN.
    """
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.0,
        normalize_all: bool = True,
    ):
        super().__init__()
        self.normalize_all = normalize_all
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm_q1 = nn.LayerNorm(d_model)
        if self.normalize_all:
            self.norm_m = nn.LayerNorm(d_model)
        self.norm_q2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def compute_kv_cache(self, memory: torch.Tensor):
        """Pre-compute normalized memory and K/V projections."""
        m = self.norm_m(memory) if self.normalize_all else memory
        return _project_kv(self.cross_attn, m)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor,
                kv_cache=None) -> torch.Tensor:
        x = tgt

        q = self.norm_q1(x)
        if kv_cache is not None:
            x_attn = _cross_attn_with_cache(
                self.cross_attn, q, kv_cache[0], kv_cache[1],
                dropout_p=self.cross_attn.dropout,
            )
        else:
            m = self.norm_m(memory) if self.normalize_all else memory
            x_attn, _ = self.cross_attn(query=q, key=m, value=m, need_weights=False)
        x = x + self.dropout1(x_attn)

        y = self.norm_q2(x)
        x_ffn = self.linear2(self.dropout(self.activation(self.linear1(y))))
        x = x + self.dropout2(x_ffn)
        return x



class D4RTHead(nn.Module):
    def __init__(
        self,
        dim_in: int,
        output_dim: int = 6,
        patch_size: int = 14,
        activation: str = "inv_log",
        conf_activation: str = "expp1",
        decoder_layers: int = 8,
        decoder_num_heads: int = 8,
        decoder_mlp_ratio: float = 1.0,
        decoder_dropout: float = 0.0,
        max_t: int = 50,
        local_patch_size: int = 9,
        local_patch_extract_size: int = None,
        local_patch_ref_width: int = 252,
        ego_centric: bool = True,
        use_camtoken: bool = False,
        visualize_local_patch: bool = False,
        visualize_local_patch_max_queries: int = 100,
        visualize_local_patch_dir: str = "./test_vis/d4rt_local_patch",
        normalize_all: bool = True,
        use_memory_pos_enc: bool = False,
        time_encoding_scheme: str = None,
        aux_uv: bool = False,
        aux_vis: bool = False,
    ):
        super().__init__()
        self.dim_in = dim_in
        self.time_encoding_scheme = time_encoding_scheme
        self.output_dim = output_dim
        # aux_uv: when True, the head emits a single trailing (u, v) per query
        # (NOT per control point). Per-point stride is always 4
        # ([x, y, z, conf]); when aux_uv is on, an extra 2-channel block is
        # appended for the single (u, v). The (u, v) channels are linear (no
        # activation) and represent normalized image coords in camera-`c`'s
        # frame; predictions can fall outside [0, 1] for queries whose 3D
        # point projects out of FOV.
        #
        # aux_vis: when True, the head emits one additional trailing channel
        # for visibility logits (raw, no activation in the head — sigmoid is
        # applied by the BCE-with-logits loss). The vis channel is ALWAYS the
        # very last output channel when aux_vis is on, regardless of aux_uv.
        # Layout:
        #   no aux:    [x, y, z, conf]                       (4)
        #   aux_uv:    [x, y, z, u, v, conf]                 (6)
        #   aux_vis:   [x, y, z, conf, vis]                  (5)
        #   both:      [x, y, z, u, v, conf, vis]            (7)
        self.aux_uv = aux_uv
        self.aux_vis = aux_vis
        self._point_stride = 4
        self._aux_uv_extra = 2 if aux_uv else 0
        self._aux_vis_extra = 1 if aux_vis else 0
        self.activation = activation
        self.conf_activation = conf_activation
        self.decoder_layers = decoder_layers
        self.local_patch_size = local_patch_size
        self.local_patch_extract_size = local_patch_extract_size
        self.local_patch_ref_width = local_patch_ref_width
        self.ego_centric = ego_centric
        self.use_camtoken = use_camtoken
        self.normalize_all = normalize_all
        self.use_memory_pos_enc = use_memory_pos_enc
        self.visualize_local_patch = visualize_local_patch
        self.visualize_local_patch_max_queries = int(visualize_local_patch_max_queries)
        self.visualize_local_patch_dir = visualize_local_patch_dir
        self.norm = nn.LayerNorm(dim_in)

        if self.normalize_all:
            self.norm_xyz = nn.LayerNorm(dim_in)
            self.norm_src = nn.LayerNorm(dim_in)
            self.norm_tgt = nn.LayerNorm(dim_in)
            self.norm_local_patch = nn.LayerNorm(dim_in)
        self.memory_pos_encodings = {}

        self.xyz_embed = FourierEncoding(
            in_dim=3,
            num_octaves=15,
            start_octave=0,
            out_dim=dim_in,
        )

        # if self.use_camtoken:
        self.src_mlp = make_head(dim_in, dim_in)
        self.tgt_mlp = make_head(dim_in, dim_in)
        self.cam_pos_encodings = {}
        if self.ego_centric:
            if self.normalize_all:
                self.norm_cam = nn.LayerNorm(dim_in)
            self.cam_mlp = make_head(dim_in, dim_in)

        self.local_patch_mlp = nn.Sequential(
            nn.Linear(local_patch_size * local_patch_size * 3, dim_in),
            nn.GELU(),
            nn.Linear(dim_in, dim_in),
        )
        query_concat_factor = 5 if self.ego_centric else 4

        ff_dim = int(dim_in * decoder_mlp_ratio)
        self.decoder_blocks = nn.ModuleList(
            [
                CrossAttentionDecoderLayer(
                    d_model=dim_in,
                    nhead=decoder_num_heads,
                    dim_feedforward=ff_dim,
                    dropout=decoder_dropout,
                    normalize_all=normalize_all,
                )
                for _ in range(decoder_layers)
            ]
        )
        self.fc = nn.Linear(dim_in, output_dim)

        self.reset_parameters()


    def _init_module_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            if module.weight is not None:
                nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            nn.init.xavier_uniform_(module.in_proj_weight)
            if module.in_proj_bias is not None:
                nn.init.zeros_(module.in_proj_bias)
            nn.init.xavier_uniform_(module.out_proj.weight)
            if module.out_proj.bias is not None:
                nn.init.zeros_(module.out_proj.bias)

    def reset_parameters(self) -> None:
        """Explicit random initialization for all D4RT head layers."""
        self.apply(self._init_module_weights)
        with torch.no_grad():
            if self.aux_vis:
                # Always the very last row.
                self.fc.weight[-1, :] = 0.0
                if self.fc.bias is not None:
                    self.fc.bias[-1] = 0.0
            if self.aux_uv:
                # layout [x, y, z, u, v, conf] (+ vis last) -> uv at rows 3-4
                self.fc.weight[3:5, :] = 0.0
                if self.fc.bias is not None:
                    self.fc.bias[3:5] = 0.0




    def _apply_activation(self, x: torch.Tensor, activation: str) -> torch.Tensor:
        if activation == "linear":
            return x
        if activation == "inv_log":
            return torch.sign(x) * torch.expm1(torch.abs(x))
        if activation == "exp":
            return torch.exp(x)
        if activation == "relu":
            return torch.relu(x)
        if activation == "sigmoid":
            return torch.sigmoid(x)
        raise ValueError(f"Unknown activation: {activation}")

    def _apply_conf_activation(self, x: torch.Tensor, conf_activation: str) -> torch.Tensor:
        if conf_activation == "expp1":
            return 1 + torch.exp(x)
        if conf_activation == "expp0":
            return torch.exp(x)
        if conf_activation == "sigmoid":
            return torch.sigmoid(x)
        if conf_activation == "linear":
            return x
        raise ValueError(f"Unknown conf_activation: {conf_activation}")

    def activate_head(self, decoded_query: torch.Tensor):
        """
        decoded_query: (B, num_query, output_dim)
        Per-point stride is always 4 ([x, y, z, conf]). Trailing aux block
        (when present) is appended after the per-point block in the order
        `[u, v, vis]`. `vis` is RAW LOGITS — no activation applied here;
        the loss uses `binary_cross_entropy_with_logits`.

        Layouts:
            no aux:    [x, y, z, conf]            (4)
            aux_uv:    [x, y, z, u, v, conf]      (6)
            aux_vis:   [x, y, z, conf, vis]       (5)
            both:      [x, y, z, u, v, conf, vis] (7)

        Returns:
            pts3d (B, Q, 3), pred_uv (B, Q, 2) or None,
            pred_vis_logits (B, Q) or None, conf (B, Q, 1), None
        """
        last_dim = decoded_query.shape[-1]
        per_point = self._point_stride  # always 4
        aux_uv_extra = self._aux_uv_extra    # 2 if aux_uv else 0
        aux_vis_extra = self._aux_vis_extra  # 1 if aux_vis else 0

        pts3d = self._apply_activation(decoded_query[..., :3], self.activation)
        if self.aux_uv:
            pred_uv = decoded_query[..., 3:5]
            conf = self._apply_conf_activation(decoded_query[..., 5:6], self.conf_activation)
        else:
            pred_uv = None
            conf = self._apply_conf_activation(decoded_query[..., 3:4], self.conf_activation)
        if self.aux_vis:
            pred_vis_logits = decoded_query[..., -1]  # (B, Q)
        else:
            pred_vis_logits = None
        return pts3d, pred_uv, pred_vis_logits, conf, None

    def retrieve_local_rgb(self, images: torch.Tensor, uv: torch.Tensor, frame_indices: torch.Tensor):
        mlp_size = self.local_patch_size
        if self.local_patch_extract_size is not None:
            extract_size = self.local_patch_extract_size
        else:
            W = images.shape[-1]
            scale = W / self.local_patch_ref_width
            extract_size = max(mlp_size, round(mlp_size * scale))
        source_indices = frame_indices.long()
        B, num_query = source_indices.shape
        _, S, C, H, W = images.shape
        device = images.device

        center_x = (uv[:, :, 0] * max(W - 1, 0)).round().long().clamp(0, max(W - 1, 0))
        center_y = (uv[:, :, 1] * max(H - 1, 0)).round().long().clamp(0, max(H - 1, 0))
        half_patch = extract_size // 2
        max_start_x = max(W - extract_size, 0)
        max_start_y = max(H - extract_size, 0)
        patch_start_x = (center_x - half_patch).clamp(0, max_start_x)
        patch_start_y = (center_y - half_patch).clamp(0, max_start_y)

        pad_w = max(extract_size - W, 0)
        pad_h = max(extract_size - H, 0)
        if pad_w > 0 or pad_h > 0:
            images_pad = F.pad(images, (0, pad_w, 0, pad_h))
        else:
            images_pad = images

        y_offsets = torch.arange(extract_size, device=device)
        x_offsets = torch.arange(extract_size, device=device)
        y_abs = patch_start_y.unsqueeze(-1) + y_offsets  # [B, Q, P_ext]
        x_abs = patch_start_x.unsqueeze(-1) + x_offsets  # [B, Q, P_ext]

        b_idx = torch.arange(B, device=device)[:, None, None, None]  # [B,1,1,1]
        s_idx = source_indices[:, :, None, None]                     # [B,Q,1,1]
        y_idx = y_abs[:, :, :, None]                                 # [B,Q,P_ext,1]
        x_idx = x_abs[:, :, None, :]                                 # [B,Q,1,P_ext]

        local_patches = images_pad[b_idx, s_idx, :, y_idx, x_idx]
        # [B, Q, P_ext, P_ext, C] -> [B*Q, C, P_ext, P_ext]
        local_patches = local_patches.permute(0, 1, 4, 2, 3).contiguous()
        local_patches = local_patches.reshape(B * num_query, C, extract_size, extract_size)

        if extract_size != mlp_size:
            local_patches = F.interpolate(local_patches, size=(mlp_size, mlp_size), mode='bilinear', align_corners=False)

        patch_embeddings = self.local_patch_mlp(local_patches.reshape(B * num_query, -1))
        return patch_embeddings.reshape(B, num_query, -1)


    def build_query_embed(
        self,
        images: torch.Tensor,
        query_coords: torch.Tensor,
        frame_indices: torch.Tensor,
        uv: torch.Tensor = None,
        cam_tokens: torch.Tensor = None,
        time_tokens: torch.Tensor = None,
        cached_local_patch_emb: torch.Tensor = None,
        return_local_patch_emb: bool = False,
    ):
        fi_width = frame_indices.shape[-1]
        if fi_width == 4:
            # 2D / 3D mode: [p, s, t, c]
            patch_idx = frame_indices[:, :, 0].long()
            src_idx = frame_indices[:, :, 1].long()
            tgt_idx = frame_indices[:, :, 2].long()
            cam_idx = frame_indices[:, :, 3].long()
        else:
            # 2D_local / 3D_local mode: [s, t, c]
            patch_idx = None
            src_idx = frame_indices[:, :, 0].long()
            tgt_idx = frame_indices[:, :, 1].long()
            cam_idx = frame_indices[:, :, 2].long()

        # if self.use_camtoken:
        if cam_tokens is None:
            raise ValueError("use_camtoken=True requires cam_tokens input in build_query_embed.")
        b_idx = torch.arange(cam_tokens.shape[0], device=cam_tokens.device)[:, None]
        c_src = self.src_mlp(cam_tokens[b_idx, src_idx])
        if time_tokens is not None:
            c_tgt = self.tgt_mlp(time_tokens[b_idx, tgt_idx])
        else:
            c_tgt = self.tgt_mlp(cam_tokens[b_idx, tgt_idx])
        if self.ego_centric:
            c_cam = self.cam_mlp(cam_tokens[b_idx, cam_idx])

        # Encode the 3D query coordinate
        coord_embed = self.xyz_embed(query_coords)
        if self.normalize_all:
            coord_embed = self.norm_xyz(coord_embed)

        # Local RGB patch from (u, v)
        if cached_local_patch_emb is not None:
            local_patch_emb = cached_local_patch_emb
        else:
            uv_for_patch = uv
            patch_frame_idx = patch_idx if patch_idx is not None else src_idx
            local_patch_emb = self.retrieve_local_rgb(images, uv_for_patch, patch_frame_idx)

        if self.normalize_all:
            c_src = self.norm_src(c_src)
            c_tgt = self.norm_tgt(c_tgt)
            local_patch_emb = self.norm_local_patch(local_patch_emb)

        # Sum embeddings and normalize by sqrt(num_components)
        components = [coord_embed, c_src, c_tgt, local_patch_emb]
        if self.ego_centric:
            if self.normalize_all:
                c_cam = self.norm_cam(c_cam)
            components.append(c_cam)

        query = sum(components) / math.sqrt(len(components))
        if return_local_patch_emb:
            return query, local_patch_emb
        return query
    
    def compute_memory_kv_cache(
        self,
        aggregated_tokens_list: List[torch.Tensor],
        patch_start_idx: int,
        use_first_half: bool = True,
    ):
        """Pre-compute per-layer KV projections of memory (scene representation).

        Call once per sequence, then pass the result as ``memory_kv_cache`` to
        :meth:`forward` for every query chunk to avoid redundant norm + K/V
        projection work.

        Returns:
            dict with 'per_layer' (list of (cached_k, cached_v) per decoder block),
            'memory' (the flattened memory tensor), and 'cam_tokens'.
        """
        B, S, N, D = aggregated_tokens_list[-1].shape
        D_half = D // 2
        global_scene_rep = aggregated_tokens_list[-1]
        if use_first_half:
            cam_tokens = aggregated_tokens_list[-1][:, :, 0, :D_half]
            global_scene_rep = global_scene_rep[:, :, patch_start_idx:, :D_half]
        else:
            cam_tokens = aggregated_tokens_list[-1][:, :, 0, D_half:]
            global_scene_rep = global_scene_rep[:, :, patch_start_idx:, D_half:]
        _, T, P, _ = global_scene_rep.shape
        memory = global_scene_rep.reshape(B, T * P, D_half)

        per_layer = []
        for block in self.decoder_blocks:
            per_layer.append(block.compute_kv_cache(memory))
        return {
            'per_layer': per_layer,
            'memory': memory,
            'cam_tokens': cam_tokens,
        }

    def forward(
        self,
        aggregated_tokens_list: List[torch.Tensor],
        images: torch.Tensor,
        patch_start_idx: int,
        query_coords: torch.Tensor,
        frame_indices: torch.Tensor,
        uv: torch.Tensor = None,
        use_first_half = True,
        R: int = None,
        time_tokens: torch.Tensor = None,
        delta: torch.Tensor = None,
        cached_local_patch_emb: torch.Tensor = None,
        return_local_patch_emb: bool = False,
        memory_kv_cache = None,
    ):
        self._forward_count = getattr(self, "_forward_count", 0) + 1

        B, S, N, D = aggregated_tokens_list[-1].shape

        D_half = D // 2

        if memory_kv_cache is not None:
            memory = memory_kv_cache['memory']
            cam_tokens = memory_kv_cache['cam_tokens']
            kv_caches = memory_kv_cache['per_layer']
        else:
            global_scene_rep = aggregated_tokens_list[-1]  # B, T, P+1, D (=frame feat + global feat)

            if use_first_half:
                cam_tokens = aggregated_tokens_list[-1][:, :, 0, :D_half]
                global_scene_rep = global_scene_rep[:, :, patch_start_idx:, :D_half]
            else:
                cam_tokens = aggregated_tokens_list[-1][:, :, 0, D_half:]
                global_scene_rep = global_scene_rep[:, :, patch_start_idx:, D_half:]

            _, T, P, _ = global_scene_rep.shape
            memory = global_scene_rep.reshape(B, T * P, D_half)
            kv_caches = [None] * len(self.decoder_blocks)

        _build_result = self.build_query_embed(
            images, query_coords, frame_indices,
            uv=uv, cam_tokens=cam_tokens,
            time_tokens=time_tokens,
            cached_local_patch_emb=cached_local_patch_emb,
            return_local_patch_emb=return_local_patch_emb,
        )
        if return_local_patch_emb:
            query_embed, local_patch_emb_out = _build_result
        else:
            query_embed = _build_result

        for i, block in enumerate(self.decoder_blocks):
            query_embed = block(tgt=query_embed, memory=memory, kv_cache=kv_caches[i])

        query_embed = self.norm(query_embed)
        query_embed = self.fc(query_embed)
        pts3d, pred_uv, pred_vis_logits, conf, _ = self.activate_head(query_embed)
        if return_local_patch_emb:
            return pts3d, pred_uv, pred_vis_logits, conf, None, local_patch_emb_out
        return pts3d, pred_uv, pred_vis_logits, conf, None
