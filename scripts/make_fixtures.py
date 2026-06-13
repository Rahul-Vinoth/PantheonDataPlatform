"""Generate synthetic, messy fixtures mirroring the workspace layout, so the ingester
runs end-to-end without the real /data. Replace with real data when available.

Creates:
  fixtures/ego_raw/factory_001/worker_0/  good clips, a duplicate, a corrupt file,
                                          a .tar shard, and a corrupt .tar
  fixtures/egodex/part0/task0/            paired mp4+hdf5, one missing hdf5,
                                          one hdf5 missing intrinsics
"""
from __future__ import annotations

import shutil
import tarfile
from pathlib import Path

import av
import h5py
import numpy as np

DEFAULT_ROOT = Path(__file__).resolve().parent.parent / "fixtures"


def encode_mp4(path: Path, n_frames: int, fps: int, w: int, h: int, seed: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)
    container = av.open(str(path), "w")
    stream = container.add_stream("libx264", rate=fps)
    stream.width, stream.height, stream.pix_fmt = w, h, "yuv420p"
    for i in range(n_frames):
        img = (base + i) % 255
        frame = av.VideoFrame.from_ndarray(img.astype(np.uint8), format="rgb24")
        for pkt in stream.encode(frame):
            container.mux(pkt)
    for pkt in stream.encode():
        container.mux(pkt)
    container.close()


def make_hdf5(path: Path, n_frames: int, *, intrinsics=True, joints=None, desc=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    joints = joints or ["camera", "leftHand", "rightHand", "leftIndexFingerTip"]
    with h5py.File(path, "w") as f:
        if intrinsics:
            cam = f.create_group("camera")
            cam.create_dataset("intrinsic", data=np.array(
                [[600., 0, 320], [0, 600., 240], [0, 0, 1.]], dtype=np.float32))
        tg = f.create_group("transforms")
        cg = f.create_group("confidences")
        for j in joints:
            T = np.tile(np.eye(4, dtype=np.float32), (n_frames, 1, 1))
            tg.create_dataset(j, data=T)
            cg.create_dataset(j, data=np.ones(n_frames, dtype=np.float32))
        if desc:
            f.attrs["llm_description"] = "pick up the red block and place it in the bin"
            f.attrs["llm_description2"] = "place the red block into the bin"
            f.attrs["which_llm_description"] = 1


def build(root: Path = DEFAULT_ROOT):
    root = Path(root)
    if root.exists():
        shutil.rmtree(root)

    # ---------------- ego_raw ----------------
    er = root / "ego_raw" / "factory_001" / "worker_0"
    encode_mp4(er / "clip_000.mp4", 30, 30, 64, 48, seed=1)
    encode_mp4(er / "clip_001.mp4", 24, 30, 64, 48, seed=2)
    # exact-content duplicate of clip_000 (different path) -> duplicate_clip signal
    shutil.copy(er / "clip_000.mp4", er / "clip_000_copy.mp4")
    # corrupt / unopenable file -> quarantine (access)
    (er / "clip_bad.mp4").write_bytes(b"\x00\x01not a real mp4\xff" * 64)
    # a .tar shard containing one good clip
    tmp = er / "_tartmp.mp4"
    encode_mp4(tmp, 20, 30, 64, 48, seed=3)
    with tarfile.open(er.parent / "shard_00.tar", "w") as tf:
        tf.add(tmp, arcname="clip_tar_000.mp4")
    tmp.unlink()
    # a corrupt tar -> quarantine (access)
    (er.parent / "shard_bad.tar").write_bytes(b"not a tar" * 32)

    # ---------------- egodex ----------------
    ed = root / "egodex" / "part0" / "task0"
    encode_mp4(ed / "0.mp4", 30, 30, 64, 48, seed=10)
    make_hdf5(ed / "0.hdf5", 30)                       # full: pose + intrinsics + lang
    encode_mp4(ed / "1.mp4", 30, 30, 64, 48, seed=11)  # NO hdf5 -> partial (video-only)
    encode_mp4(ed / "2.mp4", 30, 30, 64, 48, seed=12)
    make_hdf5(ed / "2.hdf5", 30, intrinsics=False)     # missing intrinsics -> partial

    print(f"fixtures written to {root}")
    return root


if __name__ == "__main__":
    build()
