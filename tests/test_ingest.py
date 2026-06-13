"""End-to-end ingester tests over synthetic fixtures (no real /data needed)."""
from __future__ import annotations

import sys
from pathlib import Path

import lance
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from pantheon.ingest import ingest_source
from pantheon.registry import select_adapter
import make_fixtures


@pytest.fixture(scope="module")
def lake(tmp_path_factory):
    data = tmp_path_factory.mktemp("data")
    make_fixtures.build(data)
    lh = tmp_path_factory.mktemp("lakehouse")
    ingest_source(data / "ego_raw", lh)
    ingest_source(data / "egodex", lh)
    return lh, data


def _tbl(lh, name):
    return lance.dataset(str(Path(lh) / f"{name}.lance")).to_table().to_pylist()


def test_adapter_selection(lake):
    _, data = lake
    assert select_adapter(data / "ego_raw").name == "ego_raw"
    assert select_adapter(data / "egodex").name == "egodex"


def test_episodes_materialized(lake):
    lh, _ = lake
    eps = _tbl(lh, "episode")
    by_status = {}
    for e in eps:
        by_status[e["quality_status"]] = by_status.get(e["quality_status"], 0) + 1
    assert by_status["ok"] == 5          # 4 ego_raw + 1 full egodex
    assert by_status["partial"] == 2     # missing hdf5 + missing intrinsics


def test_quarantine_not_dropped(lake):
    lh, _ = lake
    q = _tbl(lh, "quarantine")
    assert len(q) == 2                   # corrupt mp4 + corrupt tar
    assert all(r["stage"] == "access" for r in q)


def test_label_less_is_minimal(lake):
    """Raw clip = one video stream, embodiment=unknown, zero optional rows."""
    lh, _ = lake
    streams = _tbl(lh, "stream")
    raw_eps = {e["episode_id"] for e in _tbl(lh, "episode")
               if e["embodiment_id"] == "emb_unknown"}
    for ep in raw_eps:
        s = [x for x in streams if x["episode_id"] == ep]
        assert len(s) == 1 and s[0]["modality"] == "video"


def test_moving_camera_calibration(lake):
    """The annotated case yields a dynamic (moving-camera) calibration whose
    extrinsics reference the camera pose stream."""
    lh, _ = lake
    cal = _tbl(lh, "calibration")
    dyn = [c for c in cal if c["extrinsics_kind"] == "dynamic"]
    assert dyn, "expected a moving-camera calibration"
    assert dyn[0]["extr_dynamic_stream_id"].endswith("/pose/camera")


def test_pose_streams_and_captions(lake):
    lh, _ = lake
    streams = _tbl(lh, "stream")
    assert sum(s["modality"] == "pose_se3" for s in streams) == 8  # 4 joints × 2 eps
    caps = [a for a in _tbl(lh, "annotation") if a["payload_kind"] == "text"]
    assert len(caps) == 4 and all(c["label_raw"] for c in caps)


def test_descriptor_is_self_describing(lake):
    """Every stream carries a descriptor (the §2 contract)."""
    lh, _ = lake
    for s in _tbl(lh, "stream"):
        assert s["descriptor_json"] and s["payload_uri"]
