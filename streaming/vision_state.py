from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TrackedFace:
    id: int
    user: str
    nx: float
    ny: float
    nw: float
    nh: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "user": self.user,
            "nx": self.nx,
            "ny": self.ny,
            "nw": self.nw,
            "nh": self.nh,
        }


@dataclass(frozen=True)
class TrackedPerson:
    id: int
    user: str
    nx: float
    ny: float
    nw: float
    nh: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "user": self.user,
            "nx": self.nx,
            "ny": self.ny,
            "nw": self.nw,
            "nh": self.nh,
        }


@dataclass(frozen=True)
class FallRegion:
    nx: float
    ny: float
    nw: float
    nh: float

    def to_dict(self) -> dict[str, float]:
        return {
            "nx": self.nx,
            "ny": self.ny,
            "nw": self.nw,
            "nh": self.nh,
        }


@dataclass(frozen=True)
class VisionState:
    ts: float
    fid: int
    fps: float
    pw: int
    ph: int
    faces: Sequence[TrackedFace] = field(default_factory=tuple)
    persons: Sequence[TrackedPerson] = field(default_factory=tuple)
    gesture: str = ""
    fall: Mapping[str, Any] = field(default_factory=dict)
    fall_region: FallRegion | None = None
    poses: Sequence[Mapping[str, object]] = field(default_factory=tuple)
    ir: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "ts": self.ts,
            "fid": self.fid,
            "fps": self.fps,
            "pw": self.pw,
            "ph": self.ph,
            "faces": [face.to_dict() for face in self.faces],
            "persons": [person.to_dict() for person in self.persons],
            "gesture": self.gesture,
            "fall": dict(self.fall),
            "fall_region": self.fall_region.to_dict() if self.fall_region else None,
            "poses": [dict(pose) for pose in self.poses],
            "ir": self.ir,
        }
