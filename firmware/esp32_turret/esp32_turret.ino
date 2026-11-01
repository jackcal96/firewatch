/*
 * ESP32 turret + pump controller for firewatch.
 *
 * This sketch is the other end of the serial protocol implemented by
 * firewatch/turret.py. It owns three things the host must not:
 *
 *   1. Stepping the pan and tilt axes to absolute angles.
 *   2. Homing against limit switches, which is what makes an "absolute"
 *      angle mean anything on an open-loop stepper.
 *   3. Driving the pump and the normally-closed solenoid relay.
 *
 * It re-enforces the travel envelope in firmware. The host also enforces it,
 * but a serial glitch, a corrupted line, or a bug in the host must not be
 * able to drive the mount past its mechanical limits, so the limit is applied
 * here too rather than trusted.
 *
 * Protocol (ASCII, one command per line, one response line back):
 *
 *   PING              -> OK PONG
 *   HOME              -> OK HOME pan=<p> tilt=<t>
 *   GOTO <pan> <tilt> -> OK GOTO pan=<p> tilt=<t>
 *   POS               -> OK POS pan=<p> tilt=<t>
 *   JOG <axis> <deg>  -> OK JOG pan=<p> tilt=<t>      (manual, for calibration)
 *   WATER ON          -> OK WATER on
 *   WATER OFF         -> OK WATER off
 *   ENVELOPE          -> OK ENVELOPE pan=<min> <max> tilt=<min> <max>
 *   STOP              -> OK STOP                      (kills water, freezes motion)
 *
 * Any rejected command answers ERR <reason>. The host treats that as a fault,
 * which is what stops a spray when the mount cannot do what it was asked.
 *
 * Wiring assumed (edit the pin constants for your board):
 *   Pan  STEP/DIR/ENABLE  -> 26 / 25 / 27
 *   Tilt STEP/DIR/ENABLE  -> 14 / 12 / 13
 *   Limit switches        -> 34 (pan), 35 (tilt)   input-only pins, active LOW
 *   Pump relay            -> 32
 *   Solenoid valve relay  -> 33
 */

#include <Arduino.h>

// ---------------------------------------------------------------- pins

constexpr int PAN_STEP_PIN = 26;
constexpr int PAN_DIR_PIN = 25;
constexpr int PAN_ENABLE_PIN = 27;
constexpr int TILT_STEP_PIN = 14;
constexpr int TILT_DIR_PIN = 12;
constexpr int TILT_ENABLE_PIN = 13;

constexpr int PAN_LIMIT_PIN = 34;
constexpr int TILT_LIMIT_PIN = 35;

constexpr int PUMP_RELAY_PIN = 32;
constexpr int VALVE_RELAY_PIN = 33;

// ------------------------------------------------------- mechanics

// Steps per degree of output shaft rotation. A 200-step motor at 16
// microsteps through a 4:1 reduction is 200 * 16 * 4 / 360 = 35.56.
// Measure yours: command a known move and read the angle back with a level.
constexpr float PAN_STEPS_PER_DEG = 35.56f;
constexpr float TILT_STEPS_PER_DEG = 35.56f;

// Software envelope. Must match the host's TurretConfig, and must be narrower
// than the mechanism's real travel so the switches are a backstop, not a stop.
constexpr float MIN_PAN_DEG = -60.0f;
constexpr float MAX_PAN_DEG = 60.0f;
constexpr float MIN_TILT_DEG = -10.0f;
constexpr float MAX_TILT_DEG = 45.0f;

// Speed and acceleration. Slow is deliberate: the mount carries a nozzle and
// a hose, and a fast stop whips both. Tune up only after checking for
// missed steps by re-homing and comparing the datum.
constexpr float MAX_STEPS_PER_SEC = 800.0f;
constexpr float ACCEL_STEPS_PER_SEC2 = 1600.0f;

// How long the host may go silent before the water is cut. Covers the case
// where the host crashes or the USB cable is pulled mid-spray.
constexpr unsigned long HOST_TIMEOUT_MS = 5000;

// -------------------------------------------------------- state

struct Axis {
  int stepPin;
  int dirPin;
  float stepsPerDeg;
  float minDeg;
  float maxDeg;
  float angleDeg;
  long stepPosition;
  int direction;
};

Axis pan = {PAN_STEP_PIN, PAN_DIR_PIN, PAN_STEPS_PER_DEG,
            MIN_PAN_DEG, MAX_PAN_DEG, 0.0f, 0, 1};
Axis tilt = {TILT_STEP_PIN, TILT_DIR_PIN, TILT_STEPS_PER_DEG,
             MIN_TILT_DEG, MAX_TILT_DEG, 0.0f, 0, 1};

bool homed = false;
bool waterOn = false;
bool stopped = false;
unsigned long lastHostContact = 0;

// --------------------------------------------------- relay control

void allOff() {
  waterOn = false;
  // Active-LOW relay modules are the common case. Both the pump and the valve
  // are switched so that either one failing off stops the water.
  digitalWrite(PUMP_RELAY_PIN, HIGH);
  digitalWrite(VALVE_RELAY_PIN, HIGH);
}

void setWater(bool on) {
  waterOn = on;
  digitalWrite(PUMP_RELAY_PIN, on ? LOW : HIGH);
  digitalWrite(VALVE_RELAY_PIN, on ? LOW : HIGH);
  Serial.printf("OK WATER %s\n", on ? "on" : "off");
}

// ------------------------------------------------------- stepper

void enableDrivers(bool enable) {
  digitalWrite(PAN_ENABLE_PIN, enable ? LOW : HIGH);
  digitalWrite(TILT_ENABLE_PIN, enable ? LOW : HIGH);
}

void stepOnce(Axis &axis) {
  digitalWrite(axis.stepPin, HIGH);
  delayMicroseconds(3);
  digitalWrite(axis.stepPin, LOW);
  delayMicroseconds(3);
}

// Move one axis by a number of steps with a trapezoidal speed ramp, stopping
// immediately if its limit switch trips. Returns the steps actually taken, so
// the caller can tell a completed move from an interrupted one.
long moveSteps(Axis &axis, long steps, int limitPin) {
  if (steps == 0) {
    return 0;
  }

  int dir = steps > 0 ? 1 : -1;
  long remaining = labs(steps);
  long done = 0;
  float speed = 0.0f;

  if (dir != axis.direction) {
    // Respect the driver's setup time before the direction pin changes.
    digitalWrite(axis.dirPin, dir > 0 ? HIGH : LOW);
    delayMicroseconds(10);
    axis.direction = dir;
  }

  unsigned long lastStep = micros();
  while (remaining > 0) {
    if (digitalRead(limitPin) == LOW) {
      break;  // hit the switch: stop, do not grind against it
    }
    if (stopped) {
      break;
    }

    speed = min(speed + ACCEL_STEPS_PER_SEC2 * 0.00005f, MAX_STEPS_PER_SEC);
    unsigned long interval = (unsigned long)(1000000.0f / speed);

    unsigned long now = micros();
    if (now - lastStep >= interval) {
      stepOnce(axis);
      lastStep = now;
      done += dir;
      remaining--;
      axis.stepPosition += dir;
    }
  }
  return done;
}

// -------------------------------------------------------- homing

bool homeAxis(Axis &axis, int limitPin, float datumDeg) {
  // Back off the switch first: homing must approach from the same side every
  // time, or the datum moves by the switch's hysteresis.
  axis.direction = -axis.direction;
  moveSteps(axis, (long)(2.0f * axis.stepsPerDeg), limitPin);
  axis.direction = -axis.direction;
  delay(200);

  // Creep toward the switch slowly until it trips.
  long maxCreep = (long)(70.0f * axis.stepsPerDeg);
  for (long i = 0; i < maxCreep; i++) {
    if (digitalRead(limitPin) == LOW) {
      break;
    }
    stepOnce(axis);
    delayMicroseconds(900);
    axis.stepPosition += (axis.direction > 0 ? 1 : -1);
  }
  if (digitalRead(limitPin) != LOW) {
    return false;  // never found it: refuse to call this homed
  }

  axis.stepPosition = 0;
  axis.angleDeg = datumDeg;
  return true;
}

void doHome() {
  allOff();
  enableDrivers(true);
  stopped = false;
  homed = false;

  if (!homeAxis(pan, PAN_LIMIT_PIN, 0.0f)) {
    Serial.println("ERR HOME pan limit not found");
    return;
  }
  if (!homeAxis(tilt, TILT_LIMIT_PIN, 0.0f)) {
    Serial.println("ERR HOME tilt limit not found");
    return;
  }

  homed = true;
  Serial.printf("OK HOME pan=%.2f tilt=%.2f\n", pan.angleDeg, tilt.angleDeg);
}

// -------------------------------------------------------- goto

void doGoto(float panDeg, float tiltDeg) {
  if (!homed) {
    Serial.println("ERR GOTO not homed");
    return;
  }
  if (stopped) {
    Serial.println("ERR GOTO stopped");
    return;
  }
  if (panDeg < pan.minDeg || panDeg > pan.maxDeg ||
      tiltDeg < tilt.minDeg || tiltDeg > tilt.maxDeg) {
    // Refuse rather than clamp. A clamped move would report success while
    // pointing somewhere the host did not ask for, and the host gates water
    // on that report.
    Serial.printf("ERR GOTO outside envelope pan=%.2f tilt=%.2f\n", panDeg, tiltDeg);
    return;
  }

  moveSteps(pan, (long)((panDeg - pan.angleDeg) * pan.stepsPerDeg), PAN_LIMIT_PIN);
  pan.angleDeg = panDeg;
  moveSteps(tilt, (long)((tiltDeg - tilt.angleDeg) * tilt.stepsPerDeg), TILT_LIMIT_PIN);
  tilt.angleDeg = tiltDeg;

  Serial.printf("OK GOTO pan=%.2f tilt=%.2f\n", pan.angleDeg, tilt.angleDeg);
}

// -------------------------------------------------------- serial

void handleLine(String line) {
  line.trim();
  if (line.length() == 0) {
    return;
  }
  lastHostContact = millis();

  if (line.equalsIgnoreCase("PING")) {
    Serial.println("OK PONG");
  } else if (line.equalsIgnoreCase("HOME")) {
    doHome();
  } else if (line.equalsIgnoreCase("POS")) {
    Serial.printf("OK POS pan=%.2f tilt=%.2f\n", pan.angleDeg, tilt.angleDeg);
  } else if (line.equalsIgnoreCase("ENVELOPE")) {
    Serial.printf("OK ENVELOPE pan=%.1f %.1f tilt=%.1f %.1f\n",
                  pan.minDeg, pan.maxDeg, tilt.minDeg, tilt.maxDeg);
  } else if (line.equalsIgnoreCase("STOP")) {
    stopped = true;
    allOff();
    Serial.println("OK STOP");
  } else if (line.equalsIgnoreCase("WATER ON")) {
    if (!homed) {
      Serial.println("ERR WATER not homed");
    } else {
      setWater(true);
    }
  } else if (line.equalsIgnoreCase("WATER OFF")) {
    setWater(false);
  } else if (line.startsWith("GOTO ")) {
    float p = 0, t = 0;
    if (sscanf(line.c_str() + 5, "%f %f", &p, &t) == 2) {
      doGoto(p, t);
    } else {
      Serial.println("ERR GOTO malformed");
    }
  } else if (line.startsWith("JOG ")) {
    // Manual jog for calibration: JOG pan 5.0
    char axisName[8];
    float delta = 0;
    if (sscanf(line.c_str() + 4, "%7s %f", axisName, &delta) == 2) {
      bool isPan = String(axisName).equalsIgnoreCase("pan");
      Axis &axis = isPan ? pan : tilt;
      int limitPin = isPan ? PAN_LIMIT_PIN : TILT_LIMIT_PIN;
      float target = axis.angleDeg + delta;
      if (target < axis.minDeg || target > axis.maxDeg) {
        Serial.println("ERR JOG outside envelope");
      } else {
        moveSteps(axis, (long)(delta * axis.stepsPerDeg), limitPin);
        axis.angleDeg = target;
        Serial.printf("OK JOG pan=%.2f tilt=%.2f\n", pan.angleDeg, tilt.angleDeg);
      }
    } else {
      Serial.println("ERR JOG malformed");
    }
  } else {
    Serial.println("ERR unknown command");
  }
}

void setup() {
  Serial.begin(115200);
  delay(200);

  pinMode(PAN_STEP_PIN, OUTPUT);
  pinMode(PAN_DIR_PIN, OUTPUT);
  pinMode(PAN_ENABLE_PIN, OUTPUT);
  pinMode(TILT_STEP_PIN, OUTPUT);
  pinMode(TILT_DIR_PIN, OUTPUT);
  pinMode(TILT_ENABLE_PIN, OUTPUT);

  pinMode(PUMP_RELAY_PIN, OUTPUT);
  pinMode(VALVE_RELAY_PIN, OUTPUT);
  allOff();

  pinMode(PAN_LIMIT_PIN, INPUT_PULLUP);
  pinMode(TILT_LIMIT_PIN, INPUT_PULLUP);

  enableDrivers(false);

  lastHostContact = millis();
  // Announce readiness. The host waits a moment after opening the port before
  // its first command, so this line is informational rather than required.
  Serial.println("OK READY firewatch turret");
}

void loop() {
  if (Serial.available() > 0) {
    String line = Serial.readStringUntil('\n');
    handleLine(line);
  }

  // Serial watchdog: if the host goes quiet mid-spray, cut the water. The
  // host has its own watchdog on the detector; this one covers the cable.
  if (waterOn && (millis() - lastHostContact) > HOST_TIMEOUT_MS) {
    allOff();
    Serial.println("ERR WATER watchdog: no host contact, water off");
  }
}
