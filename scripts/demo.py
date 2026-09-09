"""Point4D demo: track a grid of 3D queries through a video and look at the result.

    python scripts/demo.py                                  # the bundled example clip
    python scripts/demo.py --input my_video.mp4
    python scripts/demo.py --input frames_dir/ --max_frames 200 --no_viewer

Writes the trajectories and the per-frame geometry to `--out`, then opens an
interactive viser viewer (`pip install viser`).
"""

import argparse
import colorsys
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from point4d.inference import LongTrackConfig, track_long_video
from point4d.loader import load_model

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLE = os.path.join(REPO, "examples", "bouldering_1.mp4")


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------

def load_frames(path, hw, max_frames):
    """Read a video file or a directory of images as (T, H, W, 3) uint8 RGB."""
    h, w = hw
    frames = []
    if os.path.isdir(path):
        names = sorted(f for f in os.listdir(path)
                       if f.lower().endswith((".jpg", ".jpeg", ".png")))
        for n in names[:max_frames or None]:
            frames.append(cv2.imread(os.path.join(path, n)))
    else:
        cap = cv2.VideoCapture(path)
        while max_frames <= 0 or len(frames) < max_frames:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
        cap.release()
    if not frames:
        raise SystemExit(f"no frames found in {path}")
    return np.stack([cv2.cvtColor(cv2.resize(f, (w, h)), cv2.COLOR_BGR2RGB)
                     for f in frames])


def query_grid(hw, stride, margin=8):
    """Pixel queries on a regular grid over frame 0, away from the border."""
    h, w = hw
    ys = np.arange(margin, h - margin, stride, dtype=np.float32)
    xs = np.arange(margin, w - margin, stride, dtype=np.float32)
    gx, gy = np.meshgrid(xs, ys)
    return np.stack([gx.ravel(), gy.ravel()], axis=-1)


def load_mask(path, hw):
    """A frame-0 mask as a bool (H, W); anything non-zero counts as foreground."""
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise SystemExit(f"could not read mask {path}")
    return cv2.resize(m, (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST) > 0


def query_colors(query_uv, hw):
    """A rainbow down the image, so a track's colour says where it started."""
    t = query_uv[:, 1] / max(hw[0] - 1, 1)
    return np.array([colorsys.hls_to_rgb(0.667 * v, 0.5, 1.0) for v in t],
                    dtype=np.float32)


# ---------------------------------------------------------------------------
# viser viewer
# ---------------------------------------------------------------------------

def run_viewer(images, traj, geom, colors, port, point_size):
    try:
        import viser
        import viser.transforms as vt
    except ImportError:
        raise SystemExit("the 3D viewer needs viser: pip install 'point4d[viz]'")

    T = len(traj)
    server = viser.ViserServer(port=port)
    server.gui.configure_theme(control_layout="floating", show_logo=False)

    # per-frame depth cloud, coloured by the input image
    gh, gw = geom["hw"]
    sy = np.linspace(0, images.shape[1] - 1, gh).round().astype(int)
    sx = np.linspace(0, images.shape[2] - 1, gw).round().astype(int)
    cloud_nodes = []
    for t in range(T):
        pts = geom["points"][t].reshape(-1, 3)
        rgb = images[t][np.ix_(sy, sx)].reshape(-1, 3)
        ok = np.isfinite(pts).all(1) & (np.linalg.norm(pts, axis=1) > 1e-8)
        n = server.scene.add_point_cloud(f"/cloud/{t}", points=pts[ok],
                                         colors=rgb[ok], point_size=point_size)
        n.visible = t == 0
        cloud_nodes.append(n)

    # the tracked points themselves, plus a trail of line segments
    track_nodes, trail_nodes = [], []
    seg_colors = np.repeat(colors[:, None, :], 2, axis=1)
    for t in range(T):
        n = server.scene.add_point_cloud(f"/track/{t}", points=traj[t], colors=colors,
                                         point_size=point_size * 3)
        n.visible = t == 0
        track_nodes.append(n)
        if t > 0:
            seg = np.stack([traj[t - 1], traj[t]], axis=1)
            s = server.scene.add_line_segments(f"/trail/{t}", seg, seg_colors,
                                               thickness=point_size / 2)
            s.visible = False
            trail_nodes.append(s)

    gui_t = server.gui.add_slider("Frame", min=0, max=T - 1, step=1, initial_value=0)
    # the trail can reach back over the whole clip
    gui_trail = server.gui.add_slider("Trail", min=0, max=T - 1, step=1,
                                      initial_value=min(16, T - 1))
    gui_dsize = server.gui.add_slider("Depth pt size", min=0.001, max=0.05, step=0.001,
                                      initial_value=point_size)
    gui_tsize = server.gui.add_slider("Track pt size", min=0.001, max=0.10, step=0.001,
                                      initial_value=point_size * 3)
    gui_lwidth = server.gui.add_slider("Trail width", min=0.001, max=0.05, step=0.001,
                                       initial_value=point_size / 2)
    gui_cloud = server.gui.add_checkbox("Depth points", initial_value=True)
    gui_track = server.gui.add_checkbox("Trajectories", initial_value=True)
    gui_play = server.gui.add_checkbox("Play", initial_value=True)

    def refresh():
        t = gui_t.value
        for i, n in enumerate(cloud_nodes):
            n.visible = gui_cloud.value and i == t
        for i, n in enumerate(track_nodes):
            n.visible = gui_track.value and i == t
        # trail_nodes[i] spans frames i -> i+1, so a trail of W frames is the
        # segments i in [t-W, t-1]; W = 0 draws none
        lo = t - gui_trail.value
        for i, n in enumerate(trail_nodes):
            n.visible = gui_track.value and lo <= i <= t - 1

    for g in (gui_t, gui_trail, gui_cloud, gui_track):
        g.on_update(lambda _: refresh())

    @gui_dsize.on_update
    def _(_):
        for n in cloud_nodes:
            n.point_size = gui_dsize.value

    @gui_tsize.on_update
    def _(_):
        for n in track_nodes:
            n.point_size = gui_tsize.value

    @gui_lwidth.on_update
    def _(_):
        for n in trail_nodes:
            n.thickness = gui_lwidth.value

    refresh()
    print(f"viser running at http://localhost:{port} — Ctrl-C to stop")
    import time
    while True:
        if gui_play.value:
            gui_t.value = (gui_t.value + 1) % T
        time.sleep(1 / 15)


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=EXAMPLE, help="video file or image directory")
    ap.add_argument("--out", default="outputs", help="where to write the results")
    ap.add_argument("--max_frames", type=int, default=150, help="0 = the whole video")
    ap.add_argument("--query_stride", type=int, default=20,
                    help="pixel spacing of the frame-0 query grid")
    ap.add_argument("--mask", default=None,
                    help="image mask on frame 0; keep only the grid queries inside it")
    ap.add_argument("--chunk_size", type=int, default=64)
    ap.add_argument("--overlap", type=int, default=16)
    ap.add_argument("--config", default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--point_size", type=float, default=0.01)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--no_viewer", action="store_true")
    a = ap.parse_args()

    cfg = LongTrackConfig(chunk_size=a.chunk_size, overlap=a.overlap)
    images = load_frames(a.input, cfg.image_hw, a.max_frames)
    uv = query_grid(cfg.image_hw, a.query_stride)
    if a.mask:
        keep = load_mask(a.mask, cfg.image_hw)[uv[:, 1].astype(int), uv[:, 0].astype(int)]
        uv = uv[keep]
        if len(uv) == 0:
            raise SystemExit("the mask covers no grid query — lower --query_stride")
    print(f"{len(images)} frames, {len(uv)} queries, "
          f"chunk {cfg.chunk_size} / overlap {cfg.overlap}")

    kw = {k: v for k, v in (("config_path", a.config),
                            ("checkpoint_path", a.checkpoint)) if v}
    model = load_model(**kw)

    traj, conf, geom = track_long_video(model, images, uv, cfg, return_geometry=True)

    os.makedirs(a.out, exist_ok=True)
    npz = os.path.join(a.out, "tracks.npz")
    np.savez_compressed(npz, traj=traj, conf=conf, query_uv=uv, **geom)
    print("saved", npz)

    colors = query_colors(uv, cfg.image_hw)
    if not a.no_viewer:
        run_viewer(images, traj, geom, colors, a.port, a.point_size)


if __name__ == "__main__":
    main()
