"""ego_raw adapter — raw egocentric video; the label-less / scale / messy case.

Layout:  <root>/factory_*/worker_*/*.mp4   (+ some shards only as .tar; dupes; truncated)

Exercises: the minimal viable episode (one video stream, embodiment=unknown, zero
optional rows), and messy-data handling — truncation -> `partial`, unopenable ->
quarantine, content duplicates -> a queryable `duplicate_clip` signal.
"""
from __future__ import annotations

import tarfile
import tempfile
from pathlib import Path
from typing import Iterator

from ..io.hashing import file_checksum, stable_id
from ..io.video import probe_video
from ..registry import register_adapter
from ..schema.enums import (
    Modality, RoleClass, ClockKind, ClockUnit, Timing, QualityStatus,
    EmbodimentKind, UNKNOWN_EMBODIMENT_ID,
)
from ..schema.records import (
    CanonicalRecords, Clock, Descriptor, Embodiment, Episode, QualitySignal, Role,
    Source, Stream,
)
from .base import EpisodeUnit, IngestError, SourceAdapter

SOURCE_ID = "ego_raw_factory"
DETECTOR = "ego_raw-0.1"


@register_adapter("ego_raw")
class EgoRawAdapter(SourceAdapter):

    def __init__(self):
        self._seen: dict[str, str] = {}  # checksum -> first unit_ref (dedup within run)

    def source(self) -> Source:
        return Source(
            source_id=SOURCE_ID, name="ego_raw/factory", data_type="egocentric",
            license="unknown", native_format="mp4",
            default_embodiment_id=UNKNOWN_EMBODIMENT_ID, modality_profile=["video"])

    def embodiments(self) -> list[Embodiment]:
        return [Embodiment(UNKNOWN_EMBODIMENT_ID, "unknown", EmbodimentKind.UNKNOWN)]

    def probe(self, root: Path) -> bool:
        root = Path(root)
        if any(root.glob("factory_*")):
            return True
        # defer to a structured adapter if this is a known catalog layout
        # (LeRobot v3 packs its videos as mp4s but is described by meta/info.json)
        if (root / "meta" / "info.json").exists():
            return False
        # raw video only: defer to a richer adapter if mp4s are paired with side-cars
        mp4s = list(root.rglob("*.mp4"))
        if mp4s and not any(m.with_suffix(".hdf5").exists() for m in mp4s):
            return True
        return any(root.rglob("*.tar")) and not mp4s

    def iter_episodes(self, root: Path) -> Iterator[EpisodeUnit]:
        root = Path(root)
        self._seen.clear()
        for mp4 in sorted(root.rglob("*.mp4")):
            yield EpisodeUnit(unit_ref=str(mp4.relative_to(root)), root=root,
                              paths={"video": mp4})
        # .tar shards: surface each mp4 member as a unit (payload referenced in-tar)
        for tar in sorted(root.rglob("*.tar")):
            try:
                with tarfile.open(tar) as tf:
                    members = [m for m in tf.getmembers()
                               if m.isfile() and m.name.endswith(".mp4")]
            except Exception:
                # corrupt archive itself -> a single quarantine-able unit
                yield EpisodeUnit(unit_ref=str(tar.relative_to(root)), root=root,
                                  paths={"tar": tar}, meta={"tar_corrupt": True})
                continue
            for m in members:
                yield EpisodeUnit(
                    unit_ref=f"{tar.relative_to(root)}!{m.name}", root=root,
                    paths={"tar": tar}, meta={"tar_member": m.name})

    def emit(self, unit: EpisodeUnit, *, episode_id: str,
             ingest_run_id: str) -> CanonicalRecords:
        if unit.meta.get("tar_corrupt"):
            raise IngestError(f"corrupt tar archive: {unit.paths['tar']}", stage="access")

        # ----- asset access (may degrade or quarantine) -----
        video_path, cleanup = self._resolve_video(unit)
        try:
            checksum = file_checksum(video_path, max_bytes=8 << 20)
            try:
                info = probe_video(video_path)
            except Exception as e:
                raise IngestError(str(e), stage="access") from e

            signals: list[QualitySignal] = []
            status = QualityStatus.OK
            if info.truncated:
                status = QualityStatus.PARTIAL
                signals.append(self._sig(episode_id, "truncated_video", True))
            dup_of = self._seen.get(checksum)
            if dup_of is not None:
                signals.append(self._sig(episode_id, "duplicate_clip", {"of": dup_of}))
            else:
                self._seen[checksum] = unit.unit_ref
            # raw video carries no calibration -> queryable flag (Part 1 §3.3)
            signals.append(self._sig(episode_id, "uncalibrated", True))

            # ----- schema mapping -----
            clock_id = f"{episode_id}/clk0"
            payload_uri = (f"tar://{unit.paths['tar']}!{unit.meta['tar_member']}"
                           if "tar" in unit.paths else str(video_path))
            ep = Episode(
                episode_id=episode_id, source_id=SOURCE_ID,
                embodiment_id=UNKNOWN_EMBODIMENT_ID, primary_clock_id=clock_id,
                t_start_ns=0, duration_ns=info.duration_ns, quality_status=status,
                ingest_run_id=ingest_run_id, native_root_uri=payload_uri)
            clock = Clock(clock_id, episode_id, ClockKind.DEVICE_MONOTONIC,
                          ClockUnit.NS, is_primary=True)
            stream = Stream(
                stream_id=f"{episode_id}/video0", episode_id=episode_id,
                modality=Modality.VIDEO, role=Role("head_cam", RoleClass.CAMERA),
                clock_id=clock_id, timing=Timing.UNIFORM, rate_hz=info.fps,
                t_start_ns=0, t_end_ns=info.duration_ns, n_samples=info.n_frames,
                payload_uri=payload_uri, payload_format=info.codec,
                payload_locator=({"member": unit.meta["tar_member"]}
                                 if "tar" in unit.paths else None),
                descriptor=Descriptor(shape=[info.height, info.width, 3], dtype="uint8"))
            for s in signals:
                s.stream_id = stream.stream_id
            return CanonicalRecords(episode=ep, clocks=[clock], streams=[stream],
                                    quality_signals=signals)
        finally:
            cleanup()

    # --------------------------------------------------------------- helpers
    def _resolve_video(self, unit: EpisodeUnit):
        """Return (path, cleanup). For tar members, extract to a temp file to probe;
        the canonical payload still references the in-tar location."""
        if "video" in unit.paths:
            return unit.paths["video"], (lambda: None)
        tar, member = unit.paths["tar"], unit.meta["tar_member"]
        try:
            with tarfile.open(tar) as tf:
                tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
                tmp.write(tf.extractfile(member).read())
                tmp.flush(); tmp.close()
        except Exception as e:
            raise IngestError(f"cannot read tar member {member}: {e}", stage="access")
        return Path(tmp.name), (lambda: Path(tmp.name).unlink(missing_ok=True))

    def _sig(self, episode_id: str, kind: str, value) -> QualitySignal:
        return QualitySignal(
            signal_id=stable_id("qs", episode_id, kind), episode_id=episode_id,
            signal_type=kind, value=value, detector_version=DETECTOR)
