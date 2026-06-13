"""EgoDex adapter — richly annotated egocentric (MP4 + paired HDF5).

Layout:  <root>/part*/task*/{idx}.mp4 + {idx}.hdf5
HDF5:    camera/intrinsic (3x3) · transforms/<joint> (N×4×4) · confidences/<joint>
         attrs: llm_description [, llm_description2, which_llm_description]

Exercises: per-frame SE(3) pose streams, the MOVING-CAMERA case (camera extrinsics =
the transforms/camera pose stream), language captions, and partial degradation
(missing hdf5 -> video-only `partial`; missing intrinsics -> uncalibrated `partial`).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import h5py

from ..io.hashing import stable_id
from ..io.video import probe_video
from ..registry import register_adapter
from ..schema.enums import (
    Modality, RoleClass, ClockKind, ClockUnit, Timing, QualityStatus, AnnotationKind,
    AnnotationScope, ProvKind, EmbodimentKind,
)
from ..schema.records import (
    Annotation, CalibrationProfile, CanonicalRecords, Clock, Descriptor, Embodiment,
    Episode, QualitySignal, Role, Source, Stream,
)
from ..schema.enums import ExtrinsicsKind, CalibScope
from .base import EpisodeUnit, IngestError, SourceAdapter

SOURCE_ID = "egodex"
EMB_HANDS = "emb_human_hands"
DETECTOR = "egodex-0.1"
FRAME = "arkit_origin"


@register_adapter("egodex")
class EgoDexAdapter(SourceAdapter):

    def source(self) -> Source:
        return Source(
            source_id=SOURCE_ID, name="EgoDex", data_type="egocentric_human",
            license="apple-ml-research", native_format="hdf5+mp4",
            default_embodiment_id=EMB_HANDS,
            modality_profile=["video", "pose_se3"])

    def embodiments(self) -> list[Embodiment]:
        return [Embodiment(EMB_HANDS, "human_hands", EmbodimentKind.HUMAN,
                           kinematic_model_ref="hand_model_v1")]

    def probe(self, root: Path) -> bool:
        root = Path(root)
        # paired idx.mp4 + idx.hdf5 somewhere under root
        for mp4 in root.rglob("*.mp4"):
            if mp4.with_suffix(".hdf5").exists():
                return True
        return False

    def iter_episodes(self, root: Path) -> Iterator[EpisodeUnit]:
        root = Path(root)
        for mp4 in sorted(root.rglob("*.mp4")):
            paths = {"video": mp4}
            h5 = mp4.with_suffix(".hdf5")
            if h5.exists():
                paths["hdf5"] = h5
            yield EpisodeUnit(unit_ref=str(mp4.relative_to(root)), root=root, paths=paths)

    def emit(self, unit: EpisodeUnit, *, episode_id: str,
             ingest_run_id: str) -> CanonicalRecords:
        video = unit.paths["video"]
        try:
            info = probe_video(video)
        except Exception as e:
            raise IngestError(str(e), stage="access") from e

        clock_id = f"{episode_id}/clk0"
        signals: list[QualitySignal] = []
        status = QualityStatus.OK
        if info.truncated:
            status = QualityStatus.PARTIAL
            signals.append(self._sig(episode_id, "truncated_video", True))

        video_stream = Stream(
            stream_id=f"{episode_id}/video0", episode_id=episode_id,
            modality=Modality.VIDEO, role=Role("head_cam", RoleClass.CAMERA),
            clock_id=clock_id, timing=Timing.UNIFORM, rate_hz=info.fps,
            t_start_ns=0, t_end_ns=info.duration_ns, n_samples=info.n_frames,
            payload_uri=str(video), payload_format=info.codec,
            descriptor=Descriptor(shape=[info.height, info.width, 3], dtype="uint8"))
        streams = [video_stream]
        calibrations: list[CalibrationProfile] = []
        annotations: list[Annotation] = []

        # ----- side-car: degrade gracefully if hdf5 absent (Part 1 §5.4) -----
        if "hdf5" not in unit.paths:
            status = QualityStatus.PARTIAL
            signals.append(self._sig(episode_id, "hdf5_missing", True))
            signals.append(self._sig(episode_id, "uncalibrated", True))
        else:
            try:
                pose_streams, calibrations, annotations, c_signals = self._map_hdf5(
                    unit.paths["hdf5"], episode_id, clock_id, video_stream, info)
            except Exception as e:
                raise IngestError(f"hdf5 parse: {e}", stage="map") from e
            streams += pose_streams
            signals += c_signals
            if any(s.signal_type == "intrinsics_missing" for s in c_signals):
                status = QualityStatus.PARTIAL

        ep = Episode(
            episode_id=episode_id, source_id=SOURCE_ID, embodiment_id=EMB_HANDS,
            primary_clock_id=clock_id, t_start_ns=0, duration_ns=info.duration_ns,
            quality_status=status, ingest_run_id=ingest_run_id,
            native_root_uri=str(video.parent))
        clock = Clock(clock_id, episode_id, ClockKind.DEVICE_MONOTONIC, ClockUnit.NS,
                      is_primary=True)
        return CanonicalRecords(
            episode=ep, clocks=[clock], streams=streams, calibrations=calibrations,
            annotations=annotations, quality_signals=signals)

    # --------------------------------------------------------------- hdf5 mapping
    def _map_hdf5(self, h5_path, episode_id, clock_id, video_stream, info):
        pose_streams: list[Stream] = []
        calibrations: list[CalibrationProfile] = []
        annotations: list[Annotation] = []
        signals: list[QualitySignal] = []

        with h5py.File(h5_path, "r") as f:
            transforms = f.get("transforms")
            camera_pose_stream_id = None
            if transforms is not None:
                for joint in transforms.keys():
                    ds = transforms[joint]
                    n = int(ds.shape[0]) if ds.ndim >= 1 else 0
                    sid = f"{episode_id}/pose/{joint}"
                    if joint == "camera":
                        camera_pose_stream_id = sid
                    pose_streams.append(Stream(
                        stream_id=sid, episode_id=episode_id,
                        modality=Modality.POSE_SE3,
                        role=Role(joint, RoleClass.POSE),
                        group_id=f"{episode_id}/skeleton",
                        clock_id=clock_id, timing=Timing.UNIFORM, rate_hz=info.fps,
                        t_start_ns=0, t_end_ns=info.duration_ns, n_samples=n,
                        payload_uri=str(h5_path), payload_format="hdf5",
                        payload_locator={"dataset": f"transforms/{joint}",
                                         "confidence": f"confidences/{joint}"},
                        descriptor=Descriptor(
                            shape=[4, 4], dtype="float32", rotation_rep="matrix",
                            frame_id=FRAME, channel_layout=["SE3", "confidence"])))

            # camera calibration: intrinsics + MOVING-CAMERA extrinsics (pose stream)
            cam = f.get("camera")
            if cam is not None and "intrinsic" in cam:
                K = cam["intrinsic"][...].tolist()
                calibrations.append(CalibrationProfile(
                    calib_id=f"{episode_id}/calib0", scope=CalibScope.STREAM,
                    target_ref=video_stream.stream_id,
                    intrinsics={"K": K, "width": info.width, "height": info.height},
                    extrinsics_kind=ExtrinsicsKind.DYNAMIC, frame_id=FRAME,
                    extr_dynamic_stream_id=camera_pose_stream_id))
            else:
                signals.append(self._sig(episode_id, "intrinsics_missing", True))
                signals.append(self._sig(episode_id, "uncalibrated", True))

            # language captions from attrs
            for key in ("llm_description", "llm_description2"):
                if key in f.attrs:
                    text = _attr_str(f.attrs[key])
                    if text:
                        annotations.append(Annotation(
                            annotation_id=stable_id("ann", episode_id, key),
                            episode_id=episode_id, kind=AnnotationKind.CAPTION,
                            scope=AnnotationScope.EPISODE, clock_id=clock_id,
                            payload_kind="text", payload={"text": text},
                            provenance_source=ProvKind.MODEL,
                            provenance_model_version="gpt-4",
                            label_raw=text))
        return pose_streams, calibrations, annotations, signals

    def _sig(self, episode_id: str, kind: str, value) -> QualitySignal:
        return QualitySignal(
            signal_id=stable_id("qs", episode_id, kind), episode_id=episode_id,
            signal_type=kind, value=value, detector_version=DETECTOR)


def _attr_str(v) -> str:
    if isinstance(v, bytes):
        return v.decode("utf-8", "replace")
    return str(v)
