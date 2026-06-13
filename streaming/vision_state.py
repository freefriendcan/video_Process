from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


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
class VisionState:
    ts: float
    fid: int
    fps: float
    pw: int
    ph: int
    faces: Sequence[TrackedFace] = field(default_factory=tuple)
    gesture: str = ""
    fall: Mapping[str, Any] = field(default_factory=dict)
    ir: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "ts": self.ts,
            "fid": self.fid,
            "fps": self.fps,
            "pw": self.pw,
            "ph": self.ph,
            "faces": [face.to_dict() for face in self.faces],
            "gesture": self.gesture,
            "fall": dict(self.fall),
            "ir": self.ir,
        }
