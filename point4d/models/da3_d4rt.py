# Copyright (c) 2026 Minsik Jeon.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# The module layout follows VGGT
# (https://github.com/facebookresearch/vggt).

import json
import logging
import os
import sys

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin  # used for model hub

from point4d.heads.d4rt_head import D4RTHead

logger = logging.getLogger(__name__)

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]

# DA3 backbone (vendored fork, see point4d/backbone/depth_anything_3)
from point4d.backbone.depth_anything_3.model.dualdpt import DualDPT
from point4d.backbone.depth_anything_3.model.cam_dec import CameraDec
from point4d.backbone.depth_anything_3.model.cam_enc import CameraEnc
from point4d.backbone.depth_anything_3.utils.ray_utils import get_extrinsic_from_camray
from point4d.backbone.depth_anything_3.utils.geometry import affine_inverse
from point4d.backbone.depth_anything_3.utils.alignment import compute_sky_mask, set_sky_regions_to_max_depth
from point4d.backbone.depth_anything_3.model.utils.transform import (
    extri_intri_to_pose_encoding,
    pose_encoding_to_extri_intri,
)




class Aggregator_DA3(nn.Module):
    """
    Wraps the DA3 DinoV2 backbone to produce the same output interface as
    Aggregator_DINOv2: (output_list, patch_start_idx).

    output_list: list of tensors [B, S, 1+num_patches, feat_dim]
        where position 0 is the camera token, positions 1: are patch tokens.
        feat_dim = 2*embed_dim when cat_token=True (local + global concat).
    patch_start_idx: 1 (camera token occupies index 0).
    """

    def __init__(self, config_path, img_size=518, patch_size=14, time_encoding_scheme=None, time_encoding_type="sinusoidal", grad_checkpointing=False):
        super().__init__()
        self.time_encoding_scheme = time_encoding_scheme

        path = os.path.abspath(config_path)
        if os.path.isdir(path):
            path = os.path.join(path, "config.json")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"DA3 config not found: {path}")

        with open(path, "r") as f:
            cfg = json.load(f)

        net_cfg = cfg["config"]["net"]
        name = net_cfg["name"]  # "vitl" or "vitg"
        out_layers = net_cfg["out_layers"]
        alt_start = net_cfg.get("alt_start", -1)
        qknorm_start = net_cfg.get("qknorm_start", -1)
        rope_start = net_cfg.get("rope_start", -1)
        cat_token = net_cfg.get("cat_token", True)

        from point4d.backbone.depth_anything_3.model.dinov2.dinov2 import DinoV2

        self.backbone = DinoV2(
            name=name,
            out_layers=out_layers,
            alt_start=alt_start,
            qknorm_start=qknorm_start,
            rope_start=rope_start,
            cat_token=cat_token,
            time_encoding_scheme=time_encoding_scheme,
            time_encoding_type=time_encoding_type,
            grad_checkpointing=False,
        )

        vit = self.backbone.pretrained
        self.embed_dim = vit.embed_dim
        self.depth = vit.n_blocks
        self.patch_size = patch_size
        self.cat_token = cat_token
        self.out_layers = out_layers
        self.num_output_layers = len(out_layers)

        # ImageNet normalization buffers
        self.register_buffer(
            "_resnet_mean",
            torch.FloatTensor(_RESNET_MEAN).view(1, 1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_resnet_std",
            torch.FloatTensor(_RESNET_STD).view(1, 1, 3, 1, 1),
            persistent=False,
        )

    def forward(self, images, cam_token=None, export_feat_layers=[], ref_view_strategy="saddle_balanced"):
        """
        Args:
            images: [B, S, 3, H, W] in range [0, 1].
            cam_token: Optional camera token for conditioning.
            export_feat_layers: List of layer indices for auxiliary features.
            ref_view_strategy: Strategy for reference view selection.
        Returns:
            (output_list, patch_start_idx, feats, aux_feats)
        """
        B, T, C_in, H, W = images.shape

        # Normalize images
        images = (images - self._resnet_mean) / self._resnet_std

        feats, aux_feats = self.backbone(
            images, cam_token=cam_token,
            export_feat_layers=export_feat_layers,
            ref_view_strategy=ref_view_strategy,
        )

        output_list = []
        for feat_tuple in feats:
            patch_feat = feat_tuple[0]
            cam_tok = feat_tuple[1]
            # Prepend camera token at position 0
            cam_expanded = cam_tok.unsqueeze(2)  # [B, S, 1, feat_dim]
            combined = torch.cat([cam_expanded, patch_feat], dim=2)
            output_list.append(combined)

        patch_start_idx = 1  # time tokens passed separately via feats
        return output_list, patch_start_idx, feats, aux_feats



def _build_aggregator_from_da3_config(config_path, img_size=518, patch_size=14, time_encoding_scheme=None, time_encoding_type="sinusoidal", grad_checkpointing=False):
    """
    Build Aggregator_DA3 from a DA3 config.json (e.g. DA3-LARGE-1.1/config.json).
    """
    return Aggregator_DA3(config_path, img_size=img_size, patch_size=patch_size, time_encoding_scheme=time_encoding_scheme, time_encoding_type=time_encoding_type, grad_checkpointing=grad_checkpointing)


def _is_da3_config(config_path):
    """Check whether a config.json is a DA3 config (vs. DINOv2 HF config)."""
    path = os.path.abspath(config_path)
    if os.path.isdir(path):
        path = os.path.join(path, "config.json")
    if not os.path.isfile(path):
        return False
    with open(path, "r") as f:
        cfg = json.load(f)
    return "model_name" in cfg and cfg["model_name"].startswith("da3")




class D4RT_DA3(nn.Module, PyTorchModelHubMixin):
    PATCH_SIZE = 14

    def __init__(self, backbone_config_path, img_size=518, patch_size=14,
                 enable_camera=True, enable_depth=True,
                 decoder_layers=8,
                 ego_centric=True,
                 use_camtoken=False,
                 time_encoding_scheme=None,
                 time_encoding_type="sinusoidal",
                 use_memory_pos_enc=False,
                 aux_uv=False,
                 aux_vis=False,
                 local_patch_extract_size=None,
                 local_patch_ref_width=252,
                 conf_activation="expp1"):
        super().__init__()

        self.ego_centric = ego_centric
        self.use_camtoken = use_camtoken
        self.use_memory_pos_enc = use_memory_pos_enc
        self.d4rt_mode = "3D_oor"   # the only query layout this release supports
        self.aux_uv = aux_uv
        self.aux_vis = aux_vis

        self._is_da3 = False
        if backbone_config_path is not None and _is_da3_config(backbone_config_path):
            # DA3 backbone (Depth Anything 3)
            self._is_da3 = True
            self.aggregator = _build_aggregator_from_da3_config(
                backbone_config_path,
                img_size=img_size,
                patch_size=patch_size,
                time_encoding_scheme=time_encoding_scheme,
                time_encoding_type=time_encoding_type,
                grad_checkpointing=False,
            )
            embed_dim = self.aggregator.embed_dim
            depth = self.aggregator.depth
            # DA3 outputs exactly num_output_layers feature maps;
            # index them as [0, 1, ..., num_output_layers-1].
            intermediate_layer_idx = list(range(self.aggregator.num_output_layers))
            out_channels = [256, 512, 1024, 1024]
            logger.info(
                f"Using DA3 backbone: embed_dim={embed_dim}, depth={depth}, "
                f"out_layers={self.aggregator.out_layers}"
            )

            # Parse DA3 config for head parameters
            cfg_path = os.path.abspath(backbone_config_path)
            if os.path.isdir(cfg_path):
                cfg_path = os.path.join(cfg_path, "config.json")
            with open(cfg_path, "r") as f:
                da3_full_cfg = json.load(f)
            da3_cfg = da3_full_cfg["config"]
            head_cfg = da3_cfg.get("head", {})
            cam_enc_cfg = da3_cfg.get("cam_enc", {})
            cam_dec_cfg = da3_cfg.get("cam_dec", {})

            # DA3 depth+ray head (DualDPT)
            self.depth_head = DualDPT(
                dim_in=head_cfg.get("dim_in", 2 * embed_dim),
                output_dim=head_cfg.get("output_dim", 2),
                features=head_cfg.get("features", 256),
                out_channels=head_cfg.get("out_channels", [256, 512, 1024, 1024]),
            ) if enable_depth else None

            # DA3 camera decoder (CameraDec)
            self.camera_head = CameraDec(
                dim_in=cam_dec_cfg.get("dim_in", 2 * embed_dim),
            ) if enable_camera else None

            self.ray_head = None   # CameraEnc conditions on GT poses; unused at inference

            self.track_head = None

        else:
            raise ValueError("backbone_config_path must point at a DA3 config.json")

        self.intermediate_layer_idx = intermediate_layer_idx
        self.out_channels = out_channels

        patch_size = getattr(self.aggregator, "patch_size", patch_size)

        # D4RT head (always present).
        # Per-point stride is always 4 ([x, y, z, conf]). When aux_uv is on, an
        # extra trailing 2 channels carry a single (u, v) for the whole query.
        # When aux_vis is on, one more trailing channel carries the visibility
        # logit. Trailing aux block order is `[u, v, vis]` (vis is always last).
        per_point_dim = 4
        aux_uv_extra = 2 if aux_uv else 0
        aux_vis_extra = 1 if aux_vis else 0
        rigid_extra = 0
        d4rt_output_dim = per_point_dim + aux_uv_extra + aux_vis_extra + rigid_extra
        self.d4rt_head = D4RTHead(
            dim_in=embed_dim,
            output_dim=d4rt_output_dim,
            aux_uv=aux_uv,
            aux_vis=aux_vis,
            patch_size=patch_size,
            activation="linear",
            conf_activation=conf_activation,
            decoder_layers=decoder_layers,
            decoder_mlp_ratio=4.0,
            ego_centric=ego_centric,
            use_camtoken=use_camtoken,
            use_memory_pos_enc=use_memory_pos_enc,
            time_encoding_scheme=time_encoding_scheme,
            local_patch_extract_size=local_patch_extract_size,
            local_patch_ref_width=local_patch_ref_width,
        )

    def _parse_d4rt_queries(self, query_points, H, W, S, images):
        """Split a query tensor into 3D coordinates and frame indices.

        A query is [x, y, z, p, s, t, c]: the 3D point, the frame its appearance
        patch is taken from, its source frame, the frame to predict, and the
        camera the prediction is expressed in.
        """
        if query_points.shape[-1] != 7:
            raise ValueError(
                f"expected 7-dim queries [x, y, z, p, s, t, c], got {query_points.shape[-1]}")
        q = query_points.to(images.device)
        xyz = q[..., :3].to(dtype=images.dtype)
        frame_indices = q[..., 3:7].long().clamp_(0, max(S - 1, 0))
        return xyz, frame_indices

    def _process_depth_head(self, feats, H, W):
        """Process features through DualDPT depth+ray head."""
        return self.depth_head(feats, H, W, patch_start_idx=0)


    def _process_camera_estimation(self, feats, H, W, output):
        """Estimate camera pose using CameraDec on camera tokens from last layer."""
        if self.camera_head is not None:
            pose_enc = self.camera_head(feats[-1][1])
            # Remove ray info as it's not needed for pose estimation
            if "ray" in output:
                del output.ray
            if "ray_conf" in output:
                del output.ray_conf

            # Convert pose encoding to extrinsics and intrinsics
            c2w, ixt = pose_encoding_to_extri_intri(pose_enc, (H, W))
            output.extrinsics = affine_inverse(c2w)
            output.intrinsics = ixt
        return output

    def _process_mono_sky_estimation(self, output):
        """Handle sky regions by setting them to maximum depth."""
        if "sky" not in output:
            return output
        non_sky_mask = compute_sky_mask(output.sky, threshold=0.3)
        if non_sky_mask.sum() <= 10:
            return output
        if (~non_sky_mask).sum() <= 10:
            return output

        non_sky_depth = output.depth[non_sky_mask]
        if non_sky_depth.numel() > 100000:
            idx = torch.randint(0, non_sky_depth.numel(), (100000,), device=non_sky_depth.device)
            sampled_depth = non_sky_depth[idx]
        else:
            sampled_depth = non_sky_depth
        non_sky_max = torch.quantile(sampled_depth, 0.99)

        output.depth, _ = set_sky_regions_to_max_depth(
            output.depth, None, non_sky_mask, max_depth=non_sky_max
        )
        return output

    def _extract_auxiliary_features(self, aux_feats, feat_layers, H, W):
        """Extract and reshape auxiliary features from specified layers."""
        aux_features = {}
        if aux_feats is None or feat_layers is None or len(feat_layers) == 0:
            return aux_features
        for feat, feat_layer in zip(aux_feats, feat_layers):
            feat_reshaped = feat.reshape(
                feat.shape[0],
                feat.shape[1],
                H // self.PATCH_SIZE,
                W // self.PATCH_SIZE,
                feat.shape[-1],
            )
            aux_features[f"feat_layer_{feat_layer}"] = feat_reshaped
        return aux_features

    # ------------------------------------------------------------------
    # Split forward: backbone-only + D4RT-only (for eval caching)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def forward_backbone(self, images):
        """Run backbone (aggregator + depth/camera heads) and return a cache dict.

        This allows running the expensive backbone once and then calling
        ``forward_d4rt_only`` multiple times with different query chunks.
        """
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
        B, S, _, H, W = images.shape

        cache = {"images": images, "B": B, "S": S, "H": H, "W": W}
        predictions = {}

        if self._is_da3:
            # a) Aggregator
            aggregated_tokens_list, patch_start_idx, feats, aux_feats = self.aggregator(
                images, cam_token=None, export_feat_layers=[],
                ref_view_strategy="first",
            )

            # b) Depth + camera heads
            with torch.autocast(device_type=images.device.type, enabled=False):
                output = self._process_depth_head(feats, H, W)
                output = self._process_camera_estimation(feats, H, W, output)

            if self.depth_head is not None:
                output = self._process_mono_sky_estimation(output)

            if "depth" in output:
                predictions["depth"] = output.depth
                predictions["depth_conf"] = output.depth_conf
            if "extrinsics" in output:
                predictions["extrinsics"] = output.extrinsics
            if "intrinsics" in output:
                predictions["intrinsics"] = output.intrinsics
            if "extrinsics" in output and "intrinsics" in output:
                pred_extri = output.extrinsics[:, :, :3, :]
                pred_intri = output.intrinsics
                pose_enc = extri_intri_to_pose_encoding(pred_extri, pred_intri, image_size_hw=(H, W))
                predictions["pose_enc"] = pose_enc
                predictions["pose_enc_list"] = [pose_enc]
            if "ray" in output:
                predictions["ray"] = output.ray
            if "ray_conf" in output:
                predictions["ray_conf"] = output.ray_conf

            predictions["aux"] = self._extract_auxiliary_features(aux_feats, [], H, W)

            cache["feats"] = feats  # needed for time_tokens extraction
        else:
            # DINOv2 path
            aggregated_tokens_list, patch_start_idx = self.aggregator(images)

            with torch.cuda.amp.autocast(enabled=False):
                if self.camera_head is not None:
                    pose_enc_list = self.camera_head(aggregated_tokens_list)
                    predictions["pose_enc"] = pose_enc_list[-1]
                    predictions["pose_enc_list"] = pose_enc_list

                if self.depth_head is not None:
                    depth, depth_conf = self.depth_head(
                        aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                    )
                    predictions["depth"] = depth
                    predictions["depth_conf"] = depth_conf

        cache["aggregated_tokens_list"] = aggregated_tokens_list
        cache["patch_start_idx"] = patch_start_idx
        cache["predictions"] = predictions
        return cache

    @torch.no_grad()
    def compute_d4rt_kv_cache(self, backbone_cache):
        """Pre-compute memory KV cache for the D4RT head.

        Call once per sequence after forward_backbone, then pass the result as
        ``memory_kv_cache`` to :meth:`forward_d4rt_only` for every query chunk.
        """
        aggregated_tokens_list = backbone_cache["aggregated_tokens_list"]
        patch_start_idx = backbone_cache["patch_start_idx"]
        use_first_half = not self._is_da3  # DA3 uses second half
        with torch.cuda.amp.autocast(enabled=False):
            return self.d4rt_head.compute_memory_kv_cache(
                aggregated_tokens_list, patch_start_idx,
                use_first_half=use_first_half,
            )

    def forward_d4rt_only(self, backbone_cache, query_points, d4rt_uvs=None,
                          d4rt_R=None,
                          cached_local_patch_emb=None, return_local_patch_emb=False,
                          delta=None, memory_kv_cache=None):
        """Run only the D4RT head using cached backbone outputs.

        Args:
            memory_kv_cache: optional, from :meth:`compute_d4rt_kv_cache`.
                When provided, skips per-layer memory norm + K/V projection.
        """
        images = backbone_cache["images"]
        B, S, H, W = backbone_cache["B"], backbone_cache["S"], backbone_cache["H"], backbone_cache["W"]
        aggregated_tokens_list = backbone_cache["aggregated_tokens_list"]
        patch_start_idx = backbone_cache["patch_start_idx"]
        predictions = dict(backbone_cache["predictions"])  # shallow copy

        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)

        if query_points is None or self.d4rt_head is None:
            return predictions

        query_coords, frame_indices = self._parse_d4rt_queries(query_points, H, W, S, images)

        # Normalize d4rt_uvs
        uv_normalized = None
        if d4rt_uvs is not None:
            uv_normalized = d4rt_uvs.to(device=images.device, dtype=images.dtype)
            uv_normalized = uv_normalized / torch.tensor([max(W - 1, 1), max(H - 1, 1)], device=images.device, dtype=images.dtype)
            uv_normalized = uv_normalized.clamp(0.0, 1.0)

        # Time tokens (DA3 only)
        time_tokens = None
        if self._is_da3 and "feats" in backbone_cache:
            feats = backbone_cache["feats"]
            if self.d4rt_head.time_encoding_scheme == "time_token" and len(feats[-1]) >= 3:
                time_tok_raw = feats[-1][2]
                D_half = time_tok_raw.shape[-1] // 2
                time_tokens = time_tok_raw[:, :, D_half:]

        # Run D4RT head
        head_kwargs = dict(
            images=images,
            patch_start_idx=patch_start_idx,
            query_coords=query_coords,
            frame_indices=frame_indices,
            uv=uv_normalized,
            R=d4rt_R,
            cached_local_patch_emb=cached_local_patch_emb,
            return_local_patch_emb=return_local_patch_emb,
        )
        if delta is not None:
            head_kwargs["delta"] = delta
        if self._is_da3:
            head_kwargs["use_first_half"] = False
            head_kwargs["time_tokens"] = time_tokens
        if memory_kv_cache is not None:
            head_kwargs["memory_kv_cache"] = memory_kv_cache

        with torch.cuda.amp.autocast(enabled=False):
            head_out = self.d4rt_head(
                aggregated_tokens_list, **head_kwargs
            )
            if return_local_patch_emb:
                d4rt_pts3d, d4rt_uv, d4rt_vis_logits, d4rt_pts3d_conf, d4rt_rigid, local_patch_emb_out = head_out
            else:
                d4rt_pts3d, d4rt_uv, d4rt_vis_logits, d4rt_pts3d_conf, d4rt_rigid = head_out

        predictions["d4rt_pred"] = d4rt_pts3d
        predictions["d4rt_pred_conf"] = d4rt_pts3d_conf
        if d4rt_uv is not None:
            predictions["d4rt_pred_uv"] = d4rt_uv
        if d4rt_vis_logits is not None:
            predictions["d4rt_pred_vis_logits"] = d4rt_vis_logits
        if d4rt_rigid is not None:
            predictions["d4rt_rigid_rot_6d"] = d4rt_rigid["rot_6d"]
            predictions["d4rt_rigid_trans"] = d4rt_rigid["trans"]
            predictions["d4rt_rigid_patch_conf"] = d4rt_rigid["patch_conf_logits"]

        predictions["images"] = images
        predictions["ego_centric"] = self.ego_centric
        if return_local_patch_emb:
            predictions["local_patch_emb"] = local_patch_emb_out
        return predictions
