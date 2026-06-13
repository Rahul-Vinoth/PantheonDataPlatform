"""Controlled vocabularies for the canonical schema (see docs/canonical_schema.md §1)."""
from __future__ import annotations

from enum import Enum


class StrEnum(str, Enum):
    """str-valued enum so values serialize directly into Arrow string columns."""

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.value


class Modality(StrEnum):
    VIDEO = "video"
    IMAGE_SEQ = "image_seq"
    POSE_SE3 = "pose_se3"
    JOINT_ANGLES = "joint_angles"
    ACTION = "action"
    FORCE_TORQUE = "force_torque"
    TACTILE = "tactile"
    DEPTH = "depth"
    AUDIO = "audio"
    KEYPOINTS = "keypoints"
    BOXES = "boxes"
    MASKS = "masks"
    SCALAR = "scalar"


class RoleClass(StrEnum):
    CAMERA = "camera"
    PROPRIO = "proprio"
    ACTION = "action"
    POSE = "pose"
    TACTILE = "tactile"
    AUDIO = "audio"
    MISC = "misc"


class ClockKind(StrEnum):
    DEVICE_MONOTONIC = "device_monotonic"
    WALLCLOCK = "wallclock"
    FRAME_COUNTER = "frame_counter"


class ClockUnit(StrEnum):
    NS = "ns"
    S = "s"
    FRAME_INDEX = "frame_index"


class Timing(StrEnum):
    UNIFORM = "uniform"
    EXPLICIT = "explicit"


class CalibScope(StrEnum):
    SOURCE = "source"
    EPISODE = "episode"
    STREAM = "stream"


class ExtrinsicsKind(StrEnum):
    STATIC = "static"
    DYNAMIC = "dynamic"  # moving camera: extrinsics are a pose_se3 stream


class AnnotationKind(StrEnum):
    SEGMENT = "segment"
    EVENT = "event"
    CAPTION = "caption"


class AnnotationScope(StrEnum):
    EPISODE = "episode"
    STREAM = "stream"
    FRAME_RANGE = "frame_range"


class QualityStatus(StrEnum):
    OK = "ok"
    PARTIAL = "partial"
    QUARANTINED = "quarantined"


class ProvKind(StrEnum):
    DATASET = "dataset"
    HUMAN = "human"
    MODEL = "model"


class EmbodimentKind(StrEnum):
    HUMAN = "human"
    ROBOT = "robot"
    UNKNOWN = "unknown"


# Sentinel embodiment id used for the label-less case (never SQL null; see §5.2).
UNKNOWN_EMBODIMENT_ID = "emb_unknown"
