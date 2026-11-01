"""Tests for the turret hardware boundary."""

from __future__ import annotations

import pytest

from firewatch.config import TurretConfig
from firewatch.turret import (
    DryRunTurret,
    SerialTurret,
    TurretError,
    build_turret,
)


def _config(**overrides) -> TurretConfig:
    base = {
        "enable_turret": True,
        "dry_run": True,
        "min_pan": -60.0,
        "max_pan": 60.0,
        "min_tilt": -10.0,
        "max_tilt": 45.0,
    }
    base.update(overrides)
    return TurretConfig(**base)


class TestTurretConfigValidation:
    def test_disabled_by_default(self) -> None:
        assert TurretConfig().enable_turret is False
        assert TurretConfig().dry_run is True

    def test_rejects_inverted_pan_range(self) -> None:
        with pytest.raises(ValueError, match="min_pan"):
            _config(min_pan=10.0, max_pan=-10.0)

    def test_rejects_inverted_tilt_range(self) -> None:
        with pytest.raises(ValueError, match="min_tilt"):
            _config(min_tilt=45.0, max_tilt=-10.0)

    def test_rejects_home_outside_envelope(self) -> None:
        with pytest.raises(ValueError, match="outside the travel envelope"):
            _config(home_pan=90.0)

    def test_rejects_live_turret_without_a_port(self) -> None:
        # A live turret with no serial port cannot be commanded or stopped.
        with pytest.raises(ValueError, match="serial_port is unset"):
            _config(dry_run=False)

    def test_accepts_live_turret_with_a_port(self) -> None:
        config = _config(dry_run=False, serial_port="/dev/ttyUSB0")
        assert config.dry_run is False

    def test_rejects_non_positive_aim_tolerance(self) -> None:
        with pytest.raises(ValueError):
            _config(aim_tolerance_degrees=0.0)


class TestDryRunTurret:
    def test_starts_unhomed(self) -> None:
        turret = DryRunTurret(_config())
        assert turret.is_homed is False
        assert turret.position is None

    def test_home_sets_the_datum(self) -> None:
        turret = DryRunTurret(_config(home_pan=5.0, home_tilt=3.0))
        event = turret.home()
        assert turret.is_homed is True
        assert turret.position == (5.0, 3.0)
        assert event.simulated is True
        assert event.success is True

    def test_goto_records_position(self) -> None:
        turret = DryRunTurret(_config())
        turret.home()
        event = turret.goto(12.5, -4.0)
        assert turret.position == (12.5, -4.0)
        assert event.action == "goto"

    def test_goto_clamps_to_envelope(self) -> None:
        # The envelope is the software guard against aiming somewhere the
        # mechanism is not allowed to point.
        turret = DryRunTurret(_config())
        turret.home()
        turret.goto(500.0, 500.0)
        assert turret.position == (60.0, 45.0)

    def test_goto_clamps_negative_side(self) -> None:
        turret = DryRunTurret(_config())
        turret.home()
        turret.goto(-500.0, -500.0)
        assert turret.position == (-60.0, -10.0)

    def test_events_are_recorded_in_order(self) -> None:
        turret = DryRunTurret(_config())
        turret.home()
        turret.goto(10.0, 5.0)
        turret.goto(20.0, 10.0)
        assert [e.action for e in turret.events] == ["home", "goto", "goto"]

    def test_event_serialises(self) -> None:
        turret = DryRunTurret(_config())
        payload = turret.home().to_dict()
        assert payload["event"] == "turret"
        assert payload["simulated"] is True

    def test_close_is_safe_when_never_opened(self) -> None:
        DryRunTurret(_config()).close()


class TestBuildTurret:
    def test_returns_none_when_disabled(self) -> None:
        assert build_turret(TurretConfig()) is None

    def test_returns_dry_run_when_configured(self) -> None:
        assert isinstance(build_turret(_config()), DryRunTurret)


class TestSerialTurret:
    def test_requires_a_port(self) -> None:
        with pytest.raises(ValueError, match="serial_port"):
            SerialTurret(_config(serial_port=None))

    def test_parse_position_reads_both_axes(self) -> None:
        assert SerialTurret._parse_position("OK GOTO pan=12.5 tilt=-4.0") == (12.5, -4.0)

    def test_parse_position_rejects_missing_axis(self) -> None:
        with pytest.raises(TurretError, match="missing pan/tilt"):
            SerialTurret._parse_position("OK PONG")

    def test_parse_position_rejects_malformed_number(self) -> None:
        with pytest.raises(TurretError, match="malformed pan"):
            SerialTurret._parse_position("OK GOTO pan=abc tilt=1.0")


class FakeSerial:
    """A scripted stand-in for a serial port.

    Lets the real SerialTurret command/response logic be tested without an
    ESP32 on the other end. It records what was written so the wire protocol
    itself can be asserted.
    """

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.written: list[str] = []
        self.closed = False

    def write(self, data: bytes) -> None:
        self.written.append(data.decode("ascii"))

    def flush(self) -> None:
        return None

    def readline(self) -> bytes:
        if not self._responses:
            return b""
        return (self._responses.pop(0) + "\n").encode("ascii")

    def reset_input_buffer(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True


def _serial_turret(monkeypatch, responses: list[str], **overrides) -> SerialTurret:
    """Build a SerialTurret backed by a FakeSerial."""
    import sys

    import firewatch.turret as turret_module

    fake = FakeSerial(responses)
    monkeypatch.setattr(turret_module, "time", _NoSleep())
    # The lazy ``import serial`` inside SerialTurret.__init__ resolves through
    # sys.modules, so the fake has to be installed there rather than patched
    # onto the turret module.
    monkeypatch.setitem(
        sys.modules, "serial", type("M", (), {"Serial": lambda **kw: fake})
    )

    turret = SerialTurret(_config(dry_run=False, serial_port="/dev/fake", **overrides))
    turret._fake = fake  # type: ignore[attr-defined]
    return turret


class _NoSleep:
    """Stands in for the time module so tests do not wait for settle delays."""

    def sleep(self, _seconds: float) -> None:
        return None


class TestSerialTurretProtocol:
    def test_home_parses_the_datum(self, monkeypatch) -> None:
        turret = _serial_turret(monkeypatch, ["OK HOME pan=1.5 tilt=-2.5"])
        event = turret.home()
        assert turret.is_homed is True
        assert turret.position == (1.5, -2.5)
        assert event.simulated is False
        assert turret._fake.written == ["HOME\n"]

    def test_goto_sends_three_decimals(self, monkeypatch) -> None:
        turret = _serial_turret(
            monkeypatch, ["OK HOME pan=0.0 tilt=0.0", "OK GOTO pan=12.250 tilt=-4.500"]
        )
        turret.home()
        turret.goto(12.25, -4.5)
        assert turret._fake.written[-1] == "GOTO 12.250 -4.500\n"

    def test_goto_before_homing_is_refused(self, monkeypatch) -> None:
        # Without a datum an absolute angle is meaningless, so the move must
        # not be sent at all.
        turret = _serial_turret(monkeypatch, ["OK GOTO pan=0 tilt=0"])
        with pytest.raises(TurretError, match="not homed"):
            turret.goto(10.0, 5.0)
        assert turret._fake.written == []

    def test_err_response_becomes_an_exception(self, monkeypatch) -> None:
        # A rejected command must surface, never be swallowed: a turret that
        # did not move is exactly what a human needs to know before water flows.
        turret = _serial_turret(monkeypatch, ["ERR GOTO outside envelope"])
        with pytest.raises(TurretError, match="rejected"):
            turret.home()

    def test_empty_response_times_out(self, monkeypatch) -> None:
        turret = _serial_turret(monkeypatch, [])
        with pytest.raises(TurretError, match="no response"):
            turret.home()

    def test_close_releases_the_port(self, monkeypatch) -> None:
        turret = _serial_turret(monkeypatch, ["OK HOME pan=0.0 tilt=0.0"])
        turret.home()
        turret.close()
        assert turret._fake.closed is True
