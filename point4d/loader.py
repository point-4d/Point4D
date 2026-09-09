"""Build a Point4D model from a config file and load its weights."""

import os

import torch
import yaml

from point4d.models.da3_d4rt import D4RT_DA3

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG = "configs/point4d_vitg_3door.yaml"
DEFAULT_CHECKPOINT = "checkpoints/point4d_final.pt"


def _resolve(path):
    """Config paths are written relative to the repo root."""
    return path if os.path.isabs(path) else os.path.join(_REPO_ROOT, path)


def build_model(config_path=DEFAULT_CONFIG):
    """Instantiate D4RT_DA3 from a config YAML, without loading weights."""
    with open(_resolve(config_path)) as f:
        cfg = yaml.safe_load(f)["model"]
    cfg["backbone_config_path"] = _resolve(cfg["backbone_config_path"])
    return D4RT_DA3(**cfg)


def load_model(config_path=DEFAULT_CONFIG, checkpoint_path=DEFAULT_CHECKPOINT,
               device="cuda"):
    """Build the model and load a checkpoint onto `device`, ready for inference."""
    model = build_model(config_path)

    ckpt_path = _resolve(checkpoint_path)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    # released weights are a plain state_dict; training checkpoints nest it under "model"
    state_dict = ckpt.get("model", ckpt.get("state_dict", ckpt)) if isinstance(ckpt, dict) else ckpt
    if state_dict and all(k.startswith("module.") for k in state_dict):
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    missing = [k for k in missing if not k.endswith("_resnet_mean") and not k.endswith("_resnet_std")]
    if missing:
        print(f"[point4d] {len(missing)} missing keys, e.g. {missing[:5]}")
    if unexpected:
        print(f"[point4d] {len(unexpected)} unexpected keys, e.g. {unexpected[:5]}")

    model.eval()
    return model.to(device)
