"""Shared Detection dataclass used across all ML modules."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    source: str = "thermal"    # thermal | anomaly | radar | fused
    track_id: Optional[int] = None
    temp_celsius: float = 0.0

    @property
    def cx(self) -> float: return (self.x1 + self.x2) / 2
    @property
    def cy(self) -> float: return (self.y1 + self.y2) / 2
    @property
    def w(self)  -> float: return self.x2 - self.x1
    @property
    def h(self)  -> float: return self.y2 - self.y1
    @property
    def area(self) -> float: return self.w * self.h

    def iou(self, other: "Detection") -> float:
        ix1 = max(self.x1, other.x1); iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2); iy2 = min(self.y2, other.y2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0
