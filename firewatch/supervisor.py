"""The actuation supervisor: turning detections into a go/no-go decision.

This module exists because *detection is not a decision*. The model answers
"is there fire in this frame?" It does not answer "should I open a valve?"
Those differ in three ways that matter:

1. **Persistence.** A detection is instantaneous; a fire is not. Requiring the
   evidence to hold across frames and over wall-clock time filters the model's
   transient false positives, which are the dominant failure mode.
2. **Location.** A fire elsewhere in the frame is not a fire this nozzle should
   wet. Gating on a zone is what stops the system spraying at the wrong thing.
3. **Irreversibility.** Water cannot be un-sprayed. Every constraint here is
   therefore a *deny by default*: the valve opens only when every condition is
   satisfied, and closes on any doubt.

The state machine::

    IDLE ──evidence──> CONFIRMING ──held long enough──> SPRAYING
      ^                    │                              │
      │                 evidence                     max duration /
      │                  lost                     evidence lost /
      │                    │                        watchdog stale
      │                    v                              v
      └──────────────── IDLE <──── COOLDOWN <─────────────┘
                                   │
                        budget exhausted
                                   v
                              LOCKED_OUT  (needs a human)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable

from .actuator import Actuator, ActuatorEvent, build_actuator
from .config import ActuationConfig
from .geometry import ZoneFilter
from .types import DetectionClass, FrameResult

logger = logging.getLogger(__name__)


class SupervisorState(str, Enum):
    """Where the supervisor is in its decision cycle."""

    IDLE = "idle"
    CONFIRMING = "confirming"
    SPRAYING = "spraying"
    COOLDOWN = "cooldown"
    LOCKED_OUT = "locked_out"


class DenyReason(str, Enum):
    """Why the supervisor declined to spray. Every deny is recorded."""

    ACTUATION_DISABLED = "actuation_disabled"
    LOCKED_OUT = "locked_out"
    NO_EVIDENCE = "no_evidence"
    BELOW_CONFIDENCE = "below_confidence"
    TOO_SMALL = "too_small"
    OUTSIDE_ZONE = "outside_zone"
    INSUFFICIENT_STREAK = "insufficient_streak"
    ARM_DELAY_PENDING = "arm_delay_pending"
    STALE_HEARTBEAT = "stale_heartbeat"
    IN_COOLDOWN = "in_cooldown"
    ALREADY_SPRAYING = "already_spraying"


@dataclass(frozen=True)
class Evidence:
    """The strongest qualifying detection in a frame, if any."""

    present: bool
    confidence: float = 0.0
    label: DetectionClass | None = None
    zone_name: str | None = None
    bbox: tuple[float, float, float, float] | None = None
    deny_reason: DenyReason = DenyReason.NO_EVIDENCE

    def to_dict(self) -> dict[str, Any]:
        return {
            "present": self.present,
            "confidence": round(self.confidence, 4),
            "label": self.label.value if self.label else None,
            "zone": self.zone_name,
            "bbox": [round(v, 2) for v in self.bbox] if self.bbox else None,
            "deny_reason": self.deny_reason.value,
        }


@dataclass(frozen=True)
class ActuationDecision:
    """One evaluation of the decision logic, suitable for the audit log."""

    state: SupervisorState
    action: str  # "none" | "open" | "close"
    allowed: bool
    reason: str
    detail: str = ""
    confidence: float = 0.0
    frame_index: int = -1
    streak: int = 0
    shots_used: int = 0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        return {
            "event": "actuation_decision",
            "state": self.state.value,
            "action": self.action,
            "allowed": self.allowed,
            "reason": self.reason,
            "detail": self.detail,
            "confidence": round(self.confidence, 4),
            "frame_index": self.frame_index,
            "streak": self.streak,
            "shots_used": self.shots_used,
            "timestamp": self.timestamp.isoformat(),
        }


class ActuationSupervisor:
    """Decides whether to spray, and drives the actuator when it decides yes.

    All timing is read from an injected clock so the state machine is
    deterministic in tests; the real clock is ``time.monotonic``, which is
    immune to wall-clock adjustments mid-run.
    """

    def __init__(
        self,
        config: ActuationConfig,
        actuator: Actuator | None = None,
        clock: Callable[[], float] = time.monotonic,
        zone_filter: ZoneFilter | None = None,
        on_decision: Callable[[ActuationDecision], None] | None = None,
        on_actuator_event: Callable[[ActuatorEvent], None] | None = None,
    ) -> None:
        self._config = config
        self._actuator = actuator if actuator is not None else build_actuator(config)
        self._clock = clock
        self._zones = (
            zone_filter
            if zone_filter is not None
            else ZoneFilter.from_configs(config.zones)
        )
        self._on_decision = on_decision
        self._on_actuator_event = on_actuator_event

        self._state = SupervisorState.IDLE
        self._streak = 0
        self._evidence_since: float | None = None
        self._last_observation: float | None = None
        self._spray_started: float | None = None
        self._cooldown_until: float | None = None
        self._shots_used = 0
        self._last_evidence_lost: float | None = None
        self.decisions: list[ActuationDecision] = []
        self.actuator_events: list[ActuatorEvent] = []

    # -- introspection -----------------------------------------------------

    @property
    def state(self) -> SupervisorState:
        return self._state

    @property
    def shots_used(self) -> int:
        return self._shots_used

    @property
    def is_armed(self) -> bool:
        return self._config.enable_actuation and self._state is not SupervisorState.LOCKED_OUT

    @property
    def zone_names(self) -> list[str]:
        return self._zones.zone_names

    def heartbeat(self) -> None:
        """Record that the detector is alive.

        The pipeline calls this every frame, detections or not. If it stops
        arriving, the supervisor treats the detector as dead and refuses to
        spray - a dead process must not leave a valve open.
        """
        self._last_observation = self._clock()

    def reset(self, reason: str = "operator reset") -> None:
        """Clear a lockout so the system can spray again.

        Deliberately manual: an automatic reset would let the device spend its
        whole shot budget on a persistent false positive.
        """
        if self._state is SupervisorState.LOCKED_OUT:
            logger.warning("Actuation lockout cleared: %s", reason)
        self._state = SupervisorState.IDLE
        self._streak = 0
        self._evidence_since = None
        self._spray_started = None
        self._cooldown_until = None
        self._shots_used = 0
        self._last_evidence_lost = None

    # -- the decision path -------------------------------------------------

    def observe(
        self, result: FrameResult, frame_size: tuple[int, int] | None = None
    ) -> ActuationDecision:
        """Feed one frame's detections through the decision logic.

        Returns the decision taken. The actuator is commanded as a side effect
        when - and only when - the decision allows it.
        """
        self.heartbeat()
        now = self._clock()

        if not self._config.enable_actuation:
            return self._decide(
                action="none",
                allowed=False,
                reason=DenyReason.ACTUATION_DISABLED.value,
                detail="enable_actuation is false; detection only",
                frame_index=result.frame_index,
            )

        if self._state is SupervisorState.LOCKED_OUT:
            return self._decide(
                action="none",
                allowed=False,
                reason=DenyReason.LOCKED_OUT.value,
                detail=(
                    f"shot budget exhausted ({self._shots_used}/"
                    f"{self._config.max_shots}); human reset required"
                ),
                frame_index=result.frame_index,
            )

        # Enforce time-based transitions that do not depend on new evidence.
        # If the timer path changed anything (max spray reached, watchdog
        # expired, cooldown elapsed) that outcome takes precedence over this
        # frame's evidence and must be the decision we surface, otherwise a
        # fault close would be silently swallowed.
        state_before = self._state
        timer_decision = self._enforce_timers(now, frame_index=result.frame_index)
        if timer_decision.action != "none" or self._state is not state_before:
            return timer_decision

        evidence = self._evaluate_evidence(result, frame_size)
        return self._advance(evidence, now, frame_index=result.frame_index)

    def tick(self) -> ActuationDecision | None:
        """Enforce timers without new frame data.

        Called between frames so a long spray still terminates on schedule even
        if the stream stalls. Returns a decision only when it changed
        something, so a quiet stream does not flood the audit log.
        """
        if not self._config.enable_actuation:
            return None
        before = self._state
        decision = self._enforce_timers(self._clock(), frame_index=-1)
        if self._state is not before:
            return decision
        return None

    def _enforce_timers(self, now: float, frame_index: int) -> ActuationDecision:
        """Apply max-spray, watchdog, cooldown and evidence-loss timeouts."""
        if self._state is SupervisorState.SPRAYING:
            assert self._spray_started is not None
            elapsed = now - self._spray_started

            if elapsed >= self._config.max_spray_seconds:
                return self._stop_spraying(
                    reason="max_spray_seconds reached",
                    detail=f"sprayed for {elapsed:.1f}s, hard limit "
                    f"{self._config.max_spray_seconds}s",
                    frame_index=frame_index,
                    fault=False,
                )

            if self._watchdog_expired(now):
                age = now - (self._last_observation or now)
                return self._stop_spraying(
                    reason=DenyReason.STALE_HEARTBEAT.value,
                    detail=f"detector heartbeat {age:.1f}s old; failing closed",
                    frame_index=frame_index,
                    fault=True,
                )

            if (
                self._config.stop_on_evidence_loss
                and self._last_evidence_lost is not None
                and elapsed >= self._config.min_spray_seconds
            ):
                return self._stop_spraying(
                    reason="evidence lost",
                    detail="fire no longer visible; closing valve",
                    frame_index=frame_index,
                    fault=False,
                )

        if self._state is SupervisorState.COOLDOWN:
            assert self._cooldown_until is not None
            if now >= self._cooldown_until:
                if self._shots_used >= self._config.max_shots:
                    self._state = SupervisorState.LOCKED_OUT
                    return self._decide(
                        action="none",
                        allowed=False,
                        reason=DenyReason.LOCKED_OUT.value,
                        detail=(
                            f"shot budget exhausted ({self._shots_used}/"
                            f"{self._config.max_shots}); human reset required"
                        ),
                        frame_index=frame_index,
                    )
                self._state = SupervisorState.IDLE
                self._streak = 0
                self._evidence_since = None
                self._last_evidence_lost = None

        return self._decide(
            action="none",
            allowed=False,
            reason="no_change",
            detail="",
            frame_index=frame_index,
        )

    def _watchdog_expired(self, now: float) -> bool:
        if self._last_observation is None:
            return True
        return (now - self._last_observation) > self._config.watchdog_timeout_seconds

    def _evaluate_evidence(
        self, result: FrameResult, frame_size: tuple[int, int] | None
    ) -> Evidence:
        """Find the strongest detection that qualifies to actuate on."""
        if frame_size is None:
            # Without frame dimensions, zones cannot be tested. Refusing is the
            # safe answer: never actuate on evidence that cannot be located.
            if self._config.require_zone and self._zones.has_zones:
                return Evidence(present=False, deny_reason=DenyReason.OUTSIDE_ZONE)

        candidates = [
            d
            for d in result.detections
            if d.label.value in self._config.trigger_classes
        ]
        if not candidates:
            return Evidence(present=False, deny_reason=DenyReason.NO_EVIDENCE)

        strongest_deny = DenyReason.NO_EVIDENCE
        best: Evidence | None = None

        for detection in candidates:
            if detection.confidence < self._config.min_confidence:
                strongest_deny = DenyReason.BELOW_CONFIDENCE
                continue
            if detection.area < self._config.min_bbox_area_px:
                strongest_deny = DenyReason.TOO_SMALL
                continue

            zone_name = None
            if self._config.require_zone and self._zones.has_zones:
                if frame_size is None:
                    strongest_deny = DenyReason.OUTSIDE_ZONE
                    continue
                match = self._zones.evaluate(detection.bbox, frame_size)
                if not match.matched:
                    strongest_deny = DenyReason.OUTSIDE_ZONE
                    continue
                zone_name = match.zone_name

            candidate = Evidence(
                present=True,
                confidence=detection.confidence,
                label=detection.label,
                zone_name=zone_name,
                bbox=detection.bbox,
            )
            if best is None or candidate.confidence > best.confidence:
                best = candidate

        if best is not None:
            return best
        return Evidence(present=False, deny_reason=strongest_deny)

    def _advance(
        self, evidence: Evidence, now: float, frame_index: int
    ) -> ActuationDecision:
        """Move the state machine given the current evidence."""
        if not evidence.present:
            self._streak = 0
            self._evidence_since = None
            if self._state is SupervisorState.SPRAYING:
                # Note the moment fire vanished; the timer path closes the
                # valve on a later tick so min_spray_seconds is respected.
                if self._last_evidence_lost is None:
                    self._last_evidence_lost = now
                return self._decide(
                    action="none",
                    allowed=False,
                    reason="evidence_lost_while_spraying",
                    detail="no fire visible; valve will close",
                    frame_index=frame_index,
                )
            if self._state is SupervisorState.CONFIRMING:
                self._state = SupervisorState.IDLE
                return self._decide(
                    action="none",
                    allowed=False,
                    reason=DenyReason.NO_EVIDENCE.value,
                    detail="evidence lost while confirming; disarmed",
                    frame_index=frame_index,
                )
            return self._decide(
                action="none",
                allowed=False,
                reason=evidence.deny_reason.value,
                detail="no qualifying detection",
                frame_index=frame_index,
            )

        # Evidence present from here on.
        self._last_evidence_lost = None
        self._streak += 1
        if self._evidence_since is None:
            self._evidence_since = now

        if self._state is SupervisorState.SPRAYING:
            return self._decide(
                action="none",
                allowed=False,
                reason=DenyReason.ALREADY_SPRAYING.value,
                detail="continuing current spray",
                confidence=evidence.confidence,
                frame_index=frame_index,
            )

        if self._state is SupervisorState.COOLDOWN:
            return self._decide(
                action="none",
                allowed=False,
                reason=DenyReason.IN_COOLDOWN.value,
                detail="evidence seen but device is in cooldown",
                confidence=evidence.confidence,
                frame_index=frame_index,
            )

        if self._state is SupervisorState.IDLE:
            self._state = SupervisorState.CONFIRMING
            return self._decide(
                action="none",
                allowed=False,
                reason="confirming",
                detail=f"evidence started; streak 1/{self._config.consecutive_frames}",
                confidence=evidence.confidence,
                frame_index=frame_index,
            )

        # CONFIRMING: check every gate before allowing a spray.
        if self._streak < self._config.consecutive_frames:
            return self._decide(
                action="none",
                allowed=False,
                reason=DenyReason.INSUFFICIENT_STREAK.value,
                detail=f"streak {self._streak}/{self._config.consecutive_frames}",
                confidence=evidence.confidence,
                frame_index=frame_index,
            )

        assert self._evidence_since is not None
        held = now - self._evidence_since
        if held < self._config.arm_delay_seconds:
            return self._decide(
                action="none",
                allowed=False,
                reason=DenyReason.ARM_DELAY_PENDING.value,
                detail=f"held {held:.1f}s of {self._config.arm_delay_seconds}s",
                confidence=evidence.confidence,
                frame_index=frame_index,
            )

        if self._watchdog_expired(now):
            return self._decide(
                action="none",
                allowed=False,
                reason=DenyReason.STALE_HEARTBEAT.value,
                detail="detector heartbeat stale; refusing to spray",
                confidence=evidence.confidence,
                frame_index=frame_index,
            )

        if self._shots_used >= self._config.max_shots:
            self._state = SupervisorState.LOCKED_OUT
            return self._decide(
                action="none",
                allowed=False,
                reason=DenyReason.LOCKED_OUT.value,
                detail=(
                    f"shot budget exhausted ({self._shots_used}/"
                    f"{self._config.max_shots}); human reset required"
                ),
                confidence=evidence.confidence,
                frame_index=frame_index,
            )

        return self._start_spraying(evidence, now, frame_index=frame_index)

    # -- actuator commands -------------------------------------------------

    def _start_spraying(
        self, evidence: Evidence, now: float, frame_index: int
    ) -> ActuationDecision:
        self._state = SupervisorState.SPRAYING
        self._spray_started = now
        self._shots_used += 1
        self._last_evidence_lost = None

        try:
            event = self._actuator.open_valve()
        except Exception as exc:  # noqa: BLE001 - a failed open must not crash the loop
            logger.exception("Actuator failed to open")
            self._state = SupervisorState.LOCKED_OUT
            return self._decide(
                action="open",
                allowed=False,
                reason="actuator_fault",
                detail=f"failed to open valve: {exc}",
                confidence=evidence.confidence,
                frame_index=frame_index,
            )

        self._record_actuator_event(event)
        detail = f"spraying; shot {self._shots_used}/{self._config.max_shots}"
        if event.simulated:
            detail += " (DRY RUN - no water)"
        if evidence.zone_name:
            detail += f"; zone '{evidence.zone_name}'"
        return self._decide(
            action="open",
            allowed=True,
            reason="fire_confirmed",
            detail=detail,
            confidence=evidence.confidence,
            frame_index=frame_index,
        )

    def _stop_spraying(
        self, reason: str, detail: str, frame_index: int, fault: bool
    ) -> ActuationDecision:
        try:
            event = self._actuator.close_valve()
            self._record_actuator_event(event)
        except Exception as exc:  # noqa: BLE001 - always try to close, always report
            logger.exception("Actuator failed to close")
            self._state = SupervisorState.LOCKED_OUT
            self._spray_started = None
            return self._decide(
                action="close",
                allowed=False,
                reason="actuator_fault",
                detail=f"FAILED TO CLOSE VALVE: {exc}",
                frame_index=frame_index,
            )

        self._spray_started = None
        if fault:
            self._state = SupervisorState.LOCKED_OUT
        else:
            self._state = SupervisorState.COOLDOWN
            self._cooldown_until = self._clock() + self._config.cooldown_seconds
            self._streak = 0
            self._evidence_since = None

        return self._decide(
            action="close",
            allowed=True,
            reason=reason,
            detail=detail,
            frame_index=frame_index,
        )

    # -- bookkeeping -------------------------------------------------------

    def _record_actuator_event(self, event: ActuatorEvent) -> None:
        self.actuator_events.append(event)
        if self._on_actuator_event is not None:
            self._on_actuator_event(event)

    def _decide(
        self,
        action: str,
        allowed: bool,
        reason: str,
        detail: str = "",
        confidence: float = 0.0,
        frame_index: int = -1,
    ) -> ActuationDecision:
        decision = ActuationDecision(
            state=self._state,
            action=action,
            allowed=allowed,
            reason=reason,
            detail=detail,
            confidence=confidence,
            frame_index=frame_index,
            streak=self._streak,
            shots_used=self._shots_used,
        )
        if action != "none" or allowed or reason not in {"no_change", "no_evidence"}:
            self.decisions.append(decision)
        if self._on_decision is not None and (action != "none" or allowed):
            self._on_decision(decision)
        return decision
