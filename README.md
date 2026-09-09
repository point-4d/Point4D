<div align="center">

# Point4D: Long-range 4D Motion Reconstruction

Minsik Jeon &nbsp;·&nbsp; Jay Karhade &nbsp;·&nbsp; Deva Ramanan &nbsp;·&nbsp; Shubham Tulsiani

Carnegie Mellon University

[![arXiv](https://img.shields.io/badge/arXiv-2609.09145-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2609.09145)
[![Project Page](https://img.shields.io/badge/Project_Page-point--4d.github.io-1565c0)](https://point-4d.github.io)
[![Weights](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Weights-yellow)](https://huggingface.co/minsikj/point4d)

<img src="assets/teaser.gif" width="100%" alt="Point4D overview">

</div>

**Point4D** is a feed-forward 4D reconstruction model that tracks points through
long videos by chaining **3D queries** across overlapping chunks — handed over as
a 3D point, a query survives occlusion and leaving the frame, where a reprojected
2D query does not.

## Quick Start

### Installation

```bash
git clone https://github.com/point-4d/Point4D.git
cd Point4D
conda create -n point4d python=3.10 -y && conda activate point4d
pip install -e .
```


### Model Checkpoints

The trained weights are on the Hugging Face hub at
[minsikj/point4d](https://huggingface.co/minsikj/point4d). `load_model()` looks
in `checkpoints/` by default, so put the file there:

```bash
mkdir -p checkpoints
hf download minsikj/point4d point4d_final.pt --local-dir checkpoints
```


If you would rather not use the CLI:

```python
from huggingface_hub import hf_hub_download
hf_hub_download("minsikj/point4d", "point4d_final.pt", local_dir="checkpoints")
```

### Run Demo Inference

```bash
pip install viser          # only needed for the 3D viewer
python scripts/demo.py     # tracks the bundled clip, then opens the viewer
```

`scripts/demo.py` lays a grid of queries over frame 0, tracks them through
`--max_frames` frames, writes `outputs/tracks.npz` and opens an interactive
[viser](https://viser.studio) viewer. `examples/` holds `bouldering_1.mp4`
(the default), `corridor.mp4` and `dog.mp4` — point the demo at one of those,
at your own video, or skip the viewer:

```bash
# the default clip
python scripts/demo.py --input examples/bouldering_1.mp4 --max_frames 0 \
    --chunk_size 64 --overlap 16

# another bundled clip, tracking only the query in mask
python scripts/demo.py --input examples/corridor.mp4 --max_frames 180 \
    --mask examples/corridor_mask.png --query_stride 8

# can use any video, or frame
python scripts/demo.py --input my_video.mp4 --max_frames 300 \
    --chunk_size 48 --overlap 8

# setting chunk size larger than max frame process every frames through single forward step
python scripts/demo.py --input examples/dog.mp4 --max_frames 60 \
    --chunk_size 60

# a directory of images, and no viewer (headless)
python scripts/demo.py --input frames_dir/ --no_viewer
```

- `--max_frames` — how many frames to read; `0` is the whole video. Default 150.
- `--chunk_size` — how many frames go through the model at once. Default 64.
  Anything longer is chunked, `--max_frames 0` included.
- `--overlap` — how many of those frames each chunk shares with the previous
  one. The shared frames are what the handoff aligns the two chunks on.
  Default 16.
- `--query_stride` — pixel spacing of the query grid laid over frame 0.
  Default 20.
- `--mask` — an image mask on frame 0 (anything non-zero is foreground, resized
  to the model's input resolution). Only the grid queries inside it are
  tracked, so `--query_stride` still sets the density and usually wants to come
  down with it. Without `--mask` the full grid is tracked.
- `--port` — port for the viser viewer. Default 8080.
- `--no_viewer` — write `outputs/tracks.npz` and stop, for headless machines.

## Usage

```python
import numpy as np
from point4d.loader import load_model
from point4d.inference import track_long_video, LongTrackConfig

model = load_model(device="cuda")     # uses configs/ and checkpoints/ by default

images = ...                      # (T, H, W, 3) uint8 RGB
query_uv = np.array([[260, 147]], dtype=np.float32)   # pixels in frame 0

traj, conf = track_long_video(model, images, query_uv,
                              LongTrackConfig(chunk_size=64, overlap=16))
# traj: (T, N, 3) in frame 0 camera coordinates,  conf: (T, N)
```

Pass `return_geometry=True` for a third return value carrying the per-frame
scene geometry the tracker computes anyway — depth clouds, camera poses and
intrinsics, in the same coordinate frame as the trajectories. That is what
`scripts/demo.py` draws.

Query pixels are given in the model's input resolution, which
`LongTrackConfig.image_hw` sets (default 294x518).

## Acknowledgement

Our work builds upon several fantastic open-source projects. We would like to acknowledge and thank the authors of: [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3), [VGGT](https://github.com/facebookresearch/vggt), and [Any4D](https://github.com/Any-4D/Any4D).
We also thank the members of the [Physical Perception Lab](https://shubhtuls.github.io/) at CMU for their valuable discussions.

## Citation

```bibtex
@article{jeon2026point4d,
  title={Point4D: Long-range 4D Motion Reconstruction},
  author={Jeon, Minsik and Karhade, Jay and Ramanan, Deva and Tulsiani, Shubham},
  journal={arXiv preprint arXiv:2609.09145},
  year={2026}
}
```
