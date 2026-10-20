"""Spatial gating: deciding *where* a detection is, not just whether it exists.

The detector reports boxes in frame pixels. Actuation needs an answer to a
different question: is the fire inside the area this nozzle is allowed to wet?
A detection on the far side of a window, or on a neighbouring desk, is a real
fire that must still not trigger *this* valve.

Everything here works in normalised coordinates so a zone definition keeps its
meaning if the camera resolution changes.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import ZoneConfig

Point = tuple[float, float]
BBox = tuple[float, float, float, float]


def _point_in_polygon(point: Point, polygon: list[Point]) -> bool:
    """Ray-casting test; points exactly on an edge are not guaranteed inside."""
    x, y = point
    inside = False
    count = len(polygon)
    for i in range(count):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % count]
        if (y1 > y) != (y2 > y):
            x_cross = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < x_cross:
                inside = not inside
    return inside


@dataclass(frozen=True)
class Zone:
    """A protected area in normalised frame coordinates."""

    name: str
    polygon: list[Point]
    enabled: bool = True

    @classmethod
    def from_config(cls, config: ZoneConfig) -> "Zone":
        return cls(name=config.name, polygon=list(config.polygon), enabled=config.enabled)

    def contains_point(self, point: Point) -> bool:
        return _point_in_polygon(point, self.polygon)

    def coverage(self, bbox: BBox, frame_size: tuple[int, int]) -> float:
        """Fraction of ``bbox`` area that falls inside this zone.

        Sampled on a grid rather than integrated exactly: a 7x7 sample is
        plenty to separate "this fire is in my zone" from "this fire is
        adjacent to my zone", and it costs microseconds.
        """
        width, height = frame_size
        if width <= 0 or height <= 0:
            return 0.0

        x1, y1, x2, y2 = bbox
        if x2 <= x1 or y2 <= y1:
            return 0.0

        samples = 7
        hits = 0
        for ix in range(samples):
            for iy in range(samples):
                fx = (ix + 0.5) / samples
                fy = (iy + 0.5) / samples
                px = (x1 + (x2 - x1) * fx) / width
                py = (y1 + (y2 - y1) * fy) / height
                if self.contains_point((px, py)):
                    hits += 1
        return hits / (samples * samples)


@dataclass(frozen=True)
class ZoneMatch:
    """Result of testing one detection against the configured zones."""

    matched: bool
    zone_name: str | None
    coverage: float

    def describe(self) -> str:
        if self.matched:
            return f"inside zone '{self.zone_name}' ({self.coverage:.0%} coverage)"
        if self.zone_name is not None:
            return f"outside zone '{self.zone_name}' ({self.coverage:.0%} coverage)"
        return "no zone matched"


class ZoneFilter:
    """Decides whether a bounding box is inside any enabled protected zone.

    When no zones are configured the filter is permissive and reports a
    synthetic whole-frame match, which is what ``require_zone: false`` means.
    """

    def __init__(self, zones: list[Zone] | None = None) -> None:
        self._zones = [z for z in (zones or []) if z.enabled]

    @classmethod
    def from_configs(cls, configs: list[ZoneConfig] | None) -> "ZoneFilter":
        return cls([Zone.from_config(c) for c in (configs or [])])

    @property
    def has_zones(self) -> bool:
        return bool(self._zones)

    @property
    def zone_names(self) -> list[str]:
        return [z.name for z in self._zones]

    def evaluate(self, bbox: BBox, frame_size: tuple[int, int]) -> ZoneMatch:
        """Return the best zone match for ``bbox``.

        The best match is the zone covering the largest share of the box, so a
        fire straddling two zones is attributed to whichever it mostly occupies.
        """
        if not self._zones:
            return ZoneMatch(matched=True, zone_name=None, coverage=1.0)

        best: ZoneMatch | None = None
        for zone in self._zones:
            coverage = zone.coverage(bbox, frame_size)
            candidate = ZoneMatch(
                matched=zone.contains_point(_bbox_centre(bbox, frame_size)),
                zone_name=zone.name,
                coverage=coverage,
            )
            if best is None or candidate.coverage > best.coverage:
                best = candidate

        assert best is not None  # guarded by the has-zones check above
        return best


def _bbox_centre(bbox: BBox, frame_size: tuple[int, int]) -> Point:
    width, height = frame_size
    x1, y1, x2, y2 = bbox
    cx = ((x1 + x2) / 2) / width if width else 0.0
    cy = ((y1 + y2) / 2) / height if height else 0.0
    return (cx, cy)
