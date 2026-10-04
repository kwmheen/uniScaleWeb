/** Unity Interaction 스크립트와 같은 선택·스케일·이동 규칙. */

export const METHODS = [
  ["uniDepth", "1 uniDepth"],
  ["uniAngle", "2 uniAngle"],
  ["uniMicro", "3 uniMicro"],
  ["uniSemi", "4 uniSemi"],
  ["biSemi", "5 biSemi"],
  ["biDistance", "6 biDistance"],
];

const WRIST = 0;
const THUMB_MCP = 2;
const THUMB_TIP = 4;
const INDEX_MCP = 5;
const INDEX_TIP = 8;
const MIDDLE_MCP = 9;
const MIDDLE_TIP = 12;
const PINKY_MCP = 17;

const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
const len = (v) => Math.hypot(v[0], v[1], v[2]);
const norm = (v) => {
  const l = len(v);
  return l < 1e-8 ? [0, 0, 0] : [v[0] / l, v[1] / l, v[2] / l];
};
const dist3 = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);

function clampScale(scale, min, max) {
  return Math.min(max, Math.max(min, scale));
}

function speedLevel(absValue, slow, normal) {
  if (absValue <= slow) return "slow";
  if (absValue <= normal) return "normal";
  return "fast";
}

function speedMultiplier(level, slow, normal, fast, cap) {
  const base = level === "slow" ? slow : level === "fast" ? fast : normal;
  return Math.min(base, cap);
}

function normalizeAngle(delta) {
  let angle = delta;
  while (angle > 180) angle -= 360;
  while (angle < -180) angle += 360;
  return angle;
}

class ImmediateClutch {
  constructor() {
    this.direction = 1;
    this.ready = false;
  }
  reset() {
    this.direction = 1;
    this.ready = false;
  }
  step(delta) {
    if (!this.ready) {
      this.direction = delta > 0 ? 1 : -1;
      this.ready = true;
    }
    const crossed = (this.direction > 0 && delta < 0) || (this.direction < 0 && delta > 0);
    if (crossed) this.direction *= -1;
    return this.direction;
  }
}

class DelayedClutch {
  constructor() {
    this.direction = 1;
    this.ready = false;
    this.changing = false;
    this.changeTime = 0;
  }
  reset() {
    this.direction = 1;
    this.ready = false;
    this.changing = false;
    this.changeTime = 0;
  }
  step(delta, now, threshold, stability) {
    if (!this.ready) {
      this.direction = delta > 0 ? 1 : -1;
      this.ready = true;
    }
    const past = this.direction > 0 ? -delta >= threshold : delta >= threshold;
    if (past) {
      if (!this.changing) {
        this.changing = true;
        this.changeTime = now;
      } else if (now - this.changeTime >= stability) {
        this.direction *= -1;
        this.changing = false;
      }
    } else {
      this.changing = false;
    }
    return this.direction;
  }
}

export function palmLineAngle(pixel) {
  const lineX = pixel[MIDDLE_MCP][0] - pixel[WRIST][0];
  const lineY = pixel[MIDDLE_MCP][1] - pixel[WRIST][1];
  if (lineX * lineX + lineY * lineY < 1e-6) return 0;
  return (Math.atan2(lineX, -lineY) * 180) / Math.PI;
}

export function palmFacingScore(world, isRight, xRotation, palmSign) {
  const wrist = world[WRIST];
  const indexMcp = world[INDEX_MCP];
  const pinky = world[PINKY_MCP];
  const normal = norm(
    isRight ? cross(sub(indexMcp, wrist), sub(pinky, wrist)) : cross(sub(pinky, wrist), sub(indexMcp, wrist)),
  );
  const ax = (xRotation * Math.PI) / 180;
  const towardCamera = [0, -Math.sin(ax), -Math.cos(ax)];
  return palmSign * dot(normal, towardCamera);
}

function planeLineAngle(p1, p2, p3, lineStart, lineEnd, plane) {
  const axis = plane === "XZ" ? [0, 2] : plane === "YZ" ? [1, 2] : [0, 1];
  const vec = (from, to) => [to[axis[0]] - from[axis[0]], to[axis[1]] - from[axis[1]]];
  let v1 = vec(p1, p2);
  let v2 = vec(p1, p3);
  let line = vec(lineStart, lineEnd);
  const l1 = Math.hypot(v1[0], v1[1]);
  const l2 = Math.hypot(v2[0], v2[1]);
  const ll = Math.hypot(line[0], line[1]);
  if (l1 < 1e-3 || l2 < 1e-3 || ll < 1e-3) return 0;
  v1 = [v1[0] / l1, v1[1] / l1];
  v2 = [v2[0] / l2, v2[1] / l2];
  line = [line[0] / ll, line[1] / ll];
  let planeDir = [v1[0] + v2[0], v1[1] + v2[1]];
  const planeLen = Math.hypot(planeDir[0], planeDir[1]);
  if (planeLen < 1e-8) return 0;
  planeDir = [planeDir[0] / planeLen, planeDir[1] / planeLen];
  const product = Math.min(1, Math.max(-1, planeDir[0] * line[0] + planeDir[1] * line[1]));
  return (Math.acos(product) * 180) / Math.PI;
}

export class Card {
  constructor(name, nx, ny, nw, nh, color) {
    this.name = name;
    this.nx = nx;
    this.ny = ny;
    this.nw = nw;
    this.nh = nh;
    this.color = color;
    this.scale = 1;
    this.home = { nx, ny, scale: 1 };
  }
  get area() {
    return this.nw * this.nh * this.scale * this.scale;
  }
  contains(x, y, width, height) {
    const cx = this.nx * width;
    const cy = this.ny * height;
    const halfW = (this.nw * width * this.scale) / 2;
    const halfH = (this.nh * height * this.scale) / 2;
    return Math.abs(x - cx) <= halfW && Math.abs(y - cy) <= halfH;
  }
  rect(width, height) {
    const boxW = this.nw * width * this.scale;
    const boxH = this.nh * height * this.scale;
    return [this.nx * width - boxW / 2, this.ny * height - boxH / 2, boxW, boxH];
  }
  reset() {
    this.nx = this.home.nx;
    this.ny = this.home.ny;
    this.scale = this.home.scale;
  }
}

export function createCards() {
  return [new Card("Object", 0.5, 0.46, 0.22, 0.28, "#2a52be")];
}

function gazedObject(ctx) {
  const target = ctx.objects[0] ?? null;
  if (!target) return null;
  if (ctx.aimMode !== "ray") return target;
  if (!ctx.gaze.tracked) return null;
  return target.contains(ctx.gaze.x, ctx.gaze.y, ctx.width, ctx.height) ? target : null;
}

class BiDistance {
  constructor() {
    this.minScale = 0.1;
    this.maxScale = 10;
    this.zoomSensitivity = 1;
    this.pinchStabilityTime = 0.1;
    this.minHandSeparation = 0.02;
    this.free = false;
    this.continuousZoomSpeed = 0.25;
    this.slowThreshold = 0.05;
    this.normalThreshold = 0.1;
    this.maxScaleSpeed = 2;
    this.slowM = 0.25;
    this.normalM = 1;
    this.fastM = 1.25;
    this.selected = null;
    this.initialDistance = 0;
    this.initialScale = 1;
    this.wasLeft = false;
    this.wasRight = false;
    this.pinchStart = 0;
    this.stable = false;
    this.clutch = new ImmediateClutch();
    this.currentDistance = 0;
  }
  setFree(enabled) {
    this.free = enabled;
  }
  reset() {
    this.deselect();
  }
  distance(ctx) {
    if (!ctx.left?.tracked || !ctx.right?.tracked) return 0;
    return Math.max(dist3(ctx.left.camera[INDEX_TIP], ctx.right.camera[INDEX_TIP]), this.minHandSeparation);
  }
  update(ctx) {
    const leftPinch = Boolean(ctx.left?.tracked && ctx.left.indexPinching);
    const rightPinch = Boolean(ctx.right?.tracked && ctx.right.indexPinching);
    const both = leftPinch && rightPinch;
    if (leftPinch !== rightPinch) {
      if (this.selected) this.deselect();
      this.wasLeft = leftPinch;
      this.wasRight = rightPinch;
      return;
    }
    const started = both && (!this.wasLeft || !this.wasRight);
    if (started) {
      this.pinchStart = ctx.time;
      this.stable = false;
      const current = this.distance(ctx);
      if (current > 0) {
        this.initialDistance = current;
        this.clutch.reset();
      }
    }
    if (both && !this.stable && ctx.time - this.pinchStart >= this.pinchStabilityTime) this.stable = true;
    if (!both) {
      this.stable = false;
      this.clutch.reset();
    }
    this.wasLeft = leftPinch;
    this.wasRight = rightPinch;
    const gazed = gazedObject(ctx);
    if (gazed && both && this.stable && !this.selected) {
      this.selected = gazed;
      this.initialDistance = this.distance(ctx);
      this.initialScale = gazed.scale;
      this.clutch.reset();
    }
    if (this.selected && both && this.stable) this.zoom(ctx);
    else if (!both && this.selected) this.deselect();
  }
  zoom(ctx) {
    const current = this.distance(ctx);
    this.currentDistance = current;
    if (this.initialDistance <= 0 || current <= 0 || !this.selected) return;
    if (this.free) {
      const delta = current - this.initialDistance;
      const direction = this.clutch.step(delta);
      const mult = speedMultiplier(
        speedLevel(Math.abs(delta), this.slowThreshold, this.normalThreshold),
        this.slowM,
        this.normalM,
        this.fastM,
        this.maxScaleSpeed,
      );
      const scaleDelta = direction * this.continuousZoomSpeed * mult * ctx.dt;
      this.selected.scale = clampScale(this.selected.scale * (1 + scaleDelta), this.minScale, this.maxScale);
      return;
    }
    const ratio = current / this.initialDistance;
    const zoom = Math.pow(ratio, this.zoomSensitivity);
    this.selected.scale = clampScale(this.initialScale * zoom, this.minScale, this.maxScale);
  }
  deselect() {
    this.selected = null;
    this.initialDistance = 0;
    this.initialScale = 1;
    this.stable = false;
    this.clutch.reset();
  }
  status() {
    return `양손 거리 ${this.currentDistance.toFixed(3)}m  선택 ${this.selected?.name ?? "-"}`;
  }
}

class BiSemi {
  constructor() {
    this.dominantRight = true;
    this.middleThreshold = 0.04;
    this.minScale = 0.1;
    this.maxScale = 10;
    this.zoomSensitivity = 0.75;
    this.free = false;
    this.continuousZoomSpeed = 0.25;
    this.directionThreshold = 0.02;
    this.directionStability = 0.2;
    this.slowThreshold = 0.025;
    this.normalThreshold = 0.05;
    this.maxScaleSpeed = 2;
    this.slowM = 0.25;
    this.normalM = 1.25;
    this.fastM = 1.25;
    this.selected = null;
    this.initialScale = 1;
    this.wasDh = false;
    this.initialDistance = 0;
    this.ready = false;
    this.clutch = new DelayedClutch();
    this.dhDistance = 0;
    this.ndhDistance = 0;
  }
  setFree(enabled) {
    this.free = enabled;
  }
  setDominantRight(value) {
    this.dominantRight = value;
  }
  reset() {
    this.deselect();
    this.wasDh = false;
  }
  dhPinching(hand) {
    if (!hand?.tracked) return false;
    return hand.middlePinching || hand.middleThumb <= this.middleThreshold;
  }
  ndhDistanceOf(ctx) {
    const hand = ctx.nondominant(this.dominantRight);
    return hand?.tracked ? hand.indexThumb : 0;
  }
  update(ctx) {
    const dh = ctx.dominant(this.dominantRight);
    const pinching = this.dhPinching(dh);
    this.dhDistance = dh?.middleThumb ?? 0;
    const gazed = gazedObject(ctx);
    if (gazed && pinching && !this.wasDh && !this.selected) {
      this.selected = gazed;
      this.initialScale = gazed.scale;
      this.initialDistance = this.ndhDistanceOf(ctx);
      this.ready = this.initialDistance > 0;
      this.clutch.reset();
    } else if (!pinching && this.wasDh && this.selected) {
      this.deselect();
    }
    if (this.selected && pinching) this.scale(ctx);
    this.wasDh = pinching;
  }
  scale(ctx) {
    if (!this.ready || !this.selected) return;
    const current = this.ndhDistanceOf(ctx);
    this.ndhDistance = current;
    if (current <= 0) return;
    if (this.free) {
      const delta = current - this.initialDistance;
      const direction = this.clutch.step(delta, ctx.time, this.directionThreshold, this.directionStability);
      const mult = speedMultiplier(
        speedLevel(Math.abs(delta), this.slowThreshold, this.normalThreshold),
        this.slowM,
        this.normalM,
        this.fastM,
        this.maxScaleSpeed,
      );
      const scaleDelta = direction * this.continuousZoomSpeed * mult * ctx.dt;
      this.selected.scale = clampScale(this.selected.scale * (1 + scaleDelta), this.minScale, this.maxScale);
      return;
    }
    const ratio = current / this.initialDistance;
    const zoom = 1 + (ratio - 1) * this.zoomSensitivity;
    this.selected.scale = clampScale(this.initialScale * zoom, this.minScale, this.maxScale);
  }
  deselect() {
    this.selected = null;
    this.initialScale = 1;
    this.initialDistance = 0;
    this.ready = false;
    this.clutch.reset();
  }
  status() {
    return `DH ${this.wasDh ? "세미핀치" : "대기"} ${this.dhDistance.toFixed(3)}m  NDH ${this.ndhDistance.toFixed(3)}m  선택 ${this.selected?.name ?? "-"}`;
  }
}

class DhGazePinch {
  constructor() {
    this.dominantRight = true;
    this.sensitivity = 1.5;
    this.method = "uniDepth";
    this.selected = null;
    this.wasPinching = false;
    this.handAnchor = [0, 0];
    this.objectAnchor = [0.5, 0.5];
    this.yielded = false;
  }
  setDominantRight(value) {
    this.dominantRight = value;
  }
  reset() {
    this.selected = null;
    this.wasPinching = false;
    this.yielded = false;
  }
  pinching(ctx) {
    const hand = ctx.dominant(this.dominantRight);
    return Boolean(hand?.tracked && hand.indexPinching);
  }
  update(ctx) {
    const left = Boolean(ctx.left?.tracked && ctx.left.indexPinching);
    const right = Boolean(ctx.right?.tracked && ctx.right.indexPinching);
    if (this.method === "biDistance" && left && right) {
      this.yielded = true;
      this.wasPinching = this.pinching(ctx);
      return;
    }
    this.yielded = false;
    const pinching = this.pinching(ctx);
    if (pinching && !this.wasPinching) {
      const gazed = gazedObject(ctx);
      const hand = ctx.dominant(this.dominantRight);
      if (gazed && hand) {
        this.selected = gazed;
        this.handAnchor = [...hand.pixel[WRIST]];
        this.objectAnchor = [gazed.nx, gazed.ny];
      }
    } else if (!pinching && this.wasPinching) {
      this.selected = null;
    }
    this.wasPinching = pinching;
    if (!this.selected) return;
    const hand = ctx.dominant(this.dominantRight);
    if (!hand || ctx.width <= 0 || ctx.height <= 0) return;
    const dx = (hand.pixel[WRIST][0] - this.handAnchor[0]) * this.sensitivity;
    const dy = (hand.pixel[WRIST][1] - this.handAnchor[1]) * this.sensitivity;
    this.selected.nx = Math.min(0.92, Math.max(0.08, this.objectAnchor[0] + dx / ctx.width));
    this.selected.ny = Math.min(0.9, Math.max(0.1, this.objectAnchor[1] + dy / ctx.height));
  }
  status() {
    const state = this.yielded ? "일시정지" : this.selected ? "이동" : "대기";
    return `시선 핀치 ${state}  선택 ${this.selected?.name ?? "-"}`;
  }
}

class UniAngle {
  constructor() {
    this.dominantRight = true;
    this.minScale = 0.1;
    this.maxScale = 10;
    this.scaleSensitivity = 1;
    this.rotationSensitivity = 1;
    this.free = false;
    this.continuousZoomSpeed = 0.75;
    this.slowThreshold = 5;
    this.normalThreshold = 10;
    this.maxScaleSpeed = 2;
    this.slowM = 0.25;
    this.normalM = 1;
    this.fastM = 1.25;
    this.selected = null;
    this.initialScale = 1;
    this.wasPinching = false;
    this.initialAngle = 0;
    this.ready = false;
    this.clutch = new ImmediateClutch();
    this.angle = 0;
  }
  setFree(enabled) {
    this.free = enabled;
  }
  setDominantRight(value) {
    this.dominantRight = value;
  }
  reset() {
    this.deselect();
    this.wasPinching = false;
  }
  update(ctx) {
    const hand = ctx.nondominant(this.dominantRight);
    const pinching = Boolean(hand?.tracked && hand.indexPinching);
    this.angle = hand?.tracked ? palmLineAngle(hand.pixel) : 0;
    const gazed = gazedObject(ctx);
    if (gazed && pinching && !this.wasPinching && !this.selected) {
      this.selected = gazed;
      this.initialScale = gazed.scale;
    } else if (!pinching && this.wasPinching && this.selected) {
      this.deselect();
    }
    if (this.selected) {
      if (pinching) {
        if (!this.wasPinching) {
          this.initialAngle = this.angle;
          this.ready = true;
          this.clutch.reset();
        } else this.apply(ctx);
      } else if (this.wasPinching) {
        this.ready = false;
        this.initialAngle = 0;
      }
    }
    if (!pinching && this.wasPinching) this.clutch.reset();
    this.wasPinching = pinching;
  }
  apply(ctx) {
    if (!this.ready || !this.selected) return;
    const delta = normalizeAngle(this.angle - this.initialAngle);
    if (this.free) {
      const direction = this.clutch.step(delta);
      const mult = speedMultiplier(
        speedLevel(Math.abs(delta), this.slowThreshold, this.normalThreshold),
        this.slowM,
        this.normalM,
        this.fastM,
        this.maxScaleSpeed,
      );
      const scaleDelta = direction * this.continuousZoomSpeed * mult * ctx.dt;
      this.selected.scale = clampScale(this.selected.scale * (1 + scaleDelta), this.minScale, this.maxScale);
      return;
    }
    const ratio = 1 + (delta / 90) * this.scaleSensitivity * this.rotationSensitivity;
    this.selected.scale = clampScale(this.initialScale * ratio, this.minScale, this.maxScale);
  }
  deselect() {
    this.selected = null;
    this.initialScale = 1;
    this.ready = false;
    this.initialAngle = 0;
    this.clutch.reset();
  }
  status() {
    return `손바닥 각도 ${this.angle.toFixed(1)}°  선택 ${this.selected?.name ?? "-"}`;
  }
}

class UniDepth {
  constructor() {
    this.dominantRight = true;
    this.minScale = 0.1;
    this.maxScale = 10;
    this.zoomSensitivity = 1;
    this.pushPullSensitivity = 1;
    this.free = false;
    this.continuousZoomSpeed = 0.25;
    this.slowThreshold = 0.025;
    this.normalThreshold = 0.05;
    this.maxScaleSpeed = 2;
    this.slowM = 0.25;
    this.normalM = 1;
    this.fastM = 1.25;
    this.selected = null;
    this.initialScale = 1;
    this.wasPinching = false;
    this.initialZ = 0;
    this.ready = false;
    this.clutch = new ImmediateClutch();
    this.z = 0;
  }
  setFree(enabled) {
    this.free = enabled;
  }
  setDominantRight(value) {
    this.dominantRight = value;
  }
  reset() {
    this.deselect();
    this.wasPinching = false;
  }
  update(ctx) {
    const hand = ctx.nondominant(this.dominantRight);
    const pinching = Boolean(hand?.tracked && hand.indexPinching);
    this.z = hand?.tracked ? hand.camera[INDEX_TIP][2] : 0;
    const gazed = gazedObject(ctx);
    if (gazed && pinching && !this.wasPinching && !this.selected) {
      this.selected = gazed;
      this.initialScale = gazed.scale;
    } else if (!pinching && this.wasPinching && this.selected) {
      this.deselect();
    }
    if (this.selected) {
      if (pinching) {
        if (!this.wasPinching) {
          if (hand?.tracked) {
            this.initialZ = hand.camera[INDEX_TIP][2];
            this.ready = true;
            this.clutch.reset();
          }
        } else this.apply(ctx);
      } else if (this.wasPinching) {
        this.ready = false;
        this.initialZ = 0;
      }
    }
    if (!pinching && this.wasPinching) this.clutch.reset();
    this.wasPinching = pinching;
  }
  apply(ctx) {
    if (!this.ready || !this.selected) return;
    const zDistance = this.z - this.initialZ;
    if (this.free) {
      const direction = this.clutch.step(-zDistance);
      const mult = speedMultiplier(
        speedLevel(Math.abs(zDistance), this.slowThreshold, this.normalThreshold),
        this.slowM,
        this.normalM,
        this.fastM,
        this.maxScaleSpeed,
      );
      const scaleDelta = direction * this.continuousZoomSpeed * mult * ctx.dt;
      this.selected.scale = clampScale(this.selected.scale * (1 + scaleDelta), this.minScale, this.maxScale);
      return;
    }
    const zoom = 1 - zDistance * this.pushPullSensitivity * this.zoomSensitivity;
    this.selected.scale = clampScale(this.initialScale * zoom, this.minScale, this.maxScale);
  }
  deselect() {
    this.selected = null;
    this.initialScale = 1;
    this.ready = false;
    this.initialZ = 0;
    this.clutch.reset();
  }
  status() {
    const delta = this.ready ? this.z - this.initialZ : 0;
    return `깊이 Δ ${delta.toFixed(3)}m  선택 ${this.selected?.name ?? "-"}`;
  }
}

class UniMicro {
  constructor() {
    this.dominantRight = true;
    this.minScale = 0.1;
    this.maxScale = 10;
    this.zoomSensitivity = 2;
    this.free = false;
    this.continuousZoomSpeed = 0.25;
    this.tapOn = 30;
    this.tapOff = 80;
    this.holdTime = 0.5;
    this.changeThreshold = 0.1;
    this.ignoreTime = 1;
    this.selected = null;
    this.initialScale = 1;
    this.wasTabbing = false;
    this.tabState = false;
    this.tabStart = 0;
    this.ready = false;
    this.initialKnuckle = 0;
    this.initialTip = 0;
    this.lastChange = 0;
    this.lastScale = 0;
    this.queue = [];
    this.window = 10;
    this.combined = 180;
    this.zoomFactor = 0;
  }
  setFree(enabled) {
    this.free = enabled;
  }
  setDominantRight(value) {
    this.dominantRight = value;
  }
  reset() {
    this.deselect();
    this.wasTabbing = false;
  }
  toLocal(hand) {
    const origin = hand.world[WRIST];
    const across = sub(hand.world[INDEX_MCP], hand.world[PINKY_MCP]);
    const up = norm(
      hand.handedness === "Right"
        ? cross(sub(hand.world[INDEX_MCP], origin), sub(hand.world[PINKY_MCP], origin))
        : cross(sub(hand.world[PINKY_MCP], origin), sub(hand.world[INDEX_MCP], origin)),
    );
    let xAxis = norm(across);
    const zAxis = up;
    let yAxis = cross(zAxis, xAxis);
    if (len(yAxis) < 1e-8) return (point) => sub(point, origin);
    yAxis = norm(yAxis);
    xAxis = norm(cross(yAxis, zAxis));
    return (point) => {
      const d = sub(point, origin);
      return [dot(d, xAxis), dot(d, yAxis), dot(d, zAxis)];
    };
  }
  combinedAngle(hand) {
    if (!hand?.tracked) return 180;
    const convert = this.toLocal(hand);
    const thumbMcp = convert(hand.world[THUMB_MCP]);
    const indexMcp = convert(hand.world[INDEX_MCP]);
    const indexTip = convert(hand.world[INDEX_TIP]);
    const thumbTip = convert(hand.world[THUMB_TIP]);
    const xz = planeLineAngle(thumbMcp, indexMcp, indexTip, thumbMcp, thumbTip, "XZ");
    const yz = planeLineAngle(thumbMcp, indexMcp, indexTip, thumbMcp, thumbTip, "YZ");
    return Math.hypot(xz, yz);
  }
  tabbing(hand) {
    if (!hand?.tracked) {
      this.tabState = false;
      return false;
    }
    if (!this.tabState) {
      if (this.combined < this.tapOn) this.tabState = true;
    } else if (this.combined >= this.tapOff) {
      this.tabState = false;
    }
    return this.tabState;
  }
  shouldStop(ctx, tabbing) {
    if (!tabbing && this.wasTabbing) return true;
    if (ctx.time - this.tabStart < this.ignoreTime) return false;
    if (!this.selected) return false;
    return ctx.time - this.lastChange >= this.holdTime;
  }
  update(ctx) {
    const hand = ctx.nondominant(this.dominantRight);
    this.combined = this.combinedAngle(hand);
    const tabbing = this.tabbing(hand);
    if (tabbing && !this.wasTabbing) this.tabStart = ctx.time;
    const gazed = gazedObject(ctx);
    if (gazed && tabbing && !this.selected) this.select(gazed, ctx.time);
    else if (this.shouldStop(ctx, tabbing) && this.selected) this.deselect();
    if (this.selected) {
      if (tabbing) {
        if (!this.wasTabbing) this.initialize(hand);
        else {
          this.apply(ctx, hand);
          this.track(ctx);
        }
      } else if (this.shouldStop(ctx, tabbing)) {
        this.clearGesture();
      }
    }
    this.wasTabbing = tabbing;
  }
  select(target, now) {
    this.clearGesture();
    this.selected = target;
    this.initialScale = target.scale;
    this.lastScale = target.scale;
    this.lastChange = now;
  }
  initialize(hand) {
    if (!hand?.tracked) return;
    const horizontal = (point) => [point[0], 0, point[2]];
    const thumb = horizontal(hand.world[THUMB_TIP]);
    const knuckle = horizontal(hand.world[INDEX_MCP]);
    const tip = horizontal(hand.world[INDEX_TIP]);
    this.initialKnuckle = dist3(thumb, knuckle);
    this.initialTip = dist3(thumb, tip);
    this.ready = true;
  }
  rawZoom(hand) {
    if (!hand?.tracked) return 0;
    const horizontal = (point) => [point[0], 0, point[2]];
    const thumb = horizontal(hand.world[THUMB_TIP]);
    const knuckle = horizontal(hand.world[INDEX_MCP]);
    const tip = horizontal(hand.world[INDEX_TIP]);
    const knuckleDelta = dist3(thumb, knuckle) - this.initialKnuckle;
    const tipDelta = dist3(thumb, tip) - this.initialTip;
    const average = (this.initialKnuckle + this.initialTip) * 0.5;
    return average > 0 ? (-tipDelta + knuckleDelta) / average : 0;
  }
  apply(ctx, hand) {
    if (!this.ready || !this.selected) return;
    const raw = this.rawZoom(hand);
    this.queue.push(raw);
    while (this.queue.length > this.window) this.queue.shift();
    this.zoomFactor = this.queue.reduce((sum, value) => sum + value, 0) / this.queue.length;
    if (Math.abs(this.zoomFactor) <= 0.001) return;
    if (this.free) {
      const scaleDelta = this.zoomFactor * this.continuousZoomSpeed * ctx.dt;
      this.selected.scale = clampScale(this.selected.scale * (1 + scaleDelta), this.minScale, this.maxScale);
      return;
    }
    const multiplier = 1 + this.zoomFactor * this.zoomSensitivity;
    this.selected.scale = clampScale(this.initialScale * multiplier, this.minScale, this.maxScale);
  }
  track(ctx) {
    if (!this.selected) return;
    if (Math.abs(this.selected.scale - this.lastScale) > this.changeThreshold) {
      this.lastChange = ctx.time;
      this.lastScale = this.selected.scale;
    }
  }
  clearGesture() {
    this.ready = false;
    this.initialKnuckle = 0;
    this.initialTip = 0;
    this.queue = [];
    this.tabState = false;
    this.zoomFactor = 0;
  }
  deselect() {
    this.selected = null;
    this.initialScale = 1;
    this.tabStart = 0;
    this.lastScale = 0;
    this.lastChange = 0;
    this.clearGesture();
  }
  status() {
    return `마이크로 ${this.tabState ? "탭" : "해제"} ${this.combined.toFixed(1)}°  줌 ${this.zoomFactor.toFixed(3)}  선택 ${this.selected?.name ?? "-"}`;
  }
}

class UniSemi {
  constructor() {
    this.dominantRight = true;
    this.lockThreshold = 0.55;
    this.unlockThreshold = 0.4;
    this.xRotation = 20;
    this.palmSign = -1;
    this.minScale = 0.1;
    this.maxScale = 10;
    this.zoomSensitivity = 0.75;
    this.free = false;
    this.continuousZoomSpeed = 0.25;
    this.directionThreshold = 0.02;
    this.directionStability = 0.2;
    this.slowThreshold = 0.025;
    this.normalThreshold = 0.05;
    this.maxScaleSpeed = 2;
    this.slowM = 0.25;
    this.normalM = 1.25;
    this.fastM = 1.25;
    this.selected = null;
    this.initialScale = 1;
    this.locked = false;
    this.initialDistance = 0;
    this.ready = false;
    this.clutch = new DelayedClutch();
    this.zDot = 0;
    this.currentDistance = 0;
  }
  setFree(enabled) {
    this.free = enabled;
  }
  setDominantRight(value) {
    this.dominantRight = value;
  }
  togglePalm() {
    this.palmSign *= -1;
  }
  reset() {
    this.deselect();
    this.locked = false;
  }
  update(ctx) {
    const hand = ctx.nondominant(this.dominantRight);
    this.zDot = hand?.tracked
      ? palmFacingScore(hand.world, hand.handedness === "Right", this.xRotation, this.palmSign)
      : 0;
    const threshold = this.locked ? this.unlockThreshold : this.lockThreshold;
    this.locked = Boolean(hand?.tracked && this.zDot > threshold);
    const gazed = gazedObject(ctx);
    if (gazed && this.locked && !this.selected) {
      this.selected = gazed;
      this.initialScale = gazed.scale;
      this.initialDistance = hand.indexThumb;
      this.ready = this.initialDistance > 0;
      this.clutch.reset();
    } else if (!this.locked && this.selected) {
      this.deselect();
    }
    if (this.selected && this.locked) this.scale(hand, ctx);
  }
  scale(hand, ctx) {
    if (!this.ready || !this.selected) return;
    const current = hand?.tracked ? hand.indexThumb : 0;
    this.currentDistance = current;
    if (current <= 0) return;
    if (this.free) {
      const delta = current - this.initialDistance;
      const direction = this.clutch.step(delta, ctx.time, this.directionThreshold, this.directionStability);
      const mult = speedMultiplier(
        speedLevel(Math.abs(delta), this.slowThreshold, this.normalThreshold),
        this.slowM,
        this.normalM,
        this.fastM,
        this.maxScaleSpeed,
      );
      const scaleDelta = direction * this.continuousZoomSpeed * mult * ctx.dt;
      this.selected.scale = clampScale(this.selected.scale * (1 + scaleDelta), this.minScale, this.maxScale);
      return;
    }
    const ratio = current / this.initialDistance;
    const zoom = 1 + (ratio - 1) * this.zoomSensitivity;
    this.selected.scale = clampScale(this.initialScale * zoom, this.minScale, this.maxScale);
  }
  deselect() {
    this.selected = null;
    this.initialScale = 1;
    this.initialDistance = 0;
    this.ready = false;
    this.clutch.reset();
  }
  status() {
    return `손바닥 ${this.locked ? "잠금" : "해제"} Z ${this.zDot.toFixed(2)}  거리 ${this.currentDistance.toFixed(3)}m  선택 ${this.selected?.name ?? "-"}`;
  }
}

export class InteractionController {
  constructor() {
    this.method = "uniDepth";
    this.free = false;
    this.dominantRight = true;
    this.biDistance = new BiDistance();
    this.biSemi = new BiSemi();
    this.uniAngle = new UniAngle();
    this.uniDepth = new UniDepth();
    this.uniMicro = new UniMicro();
    this.uniSemi = new UniSemi();
    this.move = new DhGazePinch();
    this.scales = {
      uniDepth: this.uniDepth,
      uniAngle: this.uniAngle,
      uniMicro: this.uniMicro,
      uniSemi: this.uniSemi,
      biSemi: this.biSemi,
      biDistance: this.biDistance,
    };
    this.applySettings();
  }
  get active() {
    return this.scales[this.method];
  }
  setMethod(method) {
    if (method === this.method) return;
    this.active.reset();
    this.method = method;
    this.applySettings();
  }
  toggleFree() {
    this.free = !this.free;
    this.applySettings();
  }
  toggleDominant() {
    this.dominantRight = !this.dominantRight;
    this.applySettings();
  }
  resetAll() {
    Object.values(this.scales).forEach((interaction) => interaction.reset());
    this.move.reset();
  }
  update(ctx) {
    this.move.method = this.method;
    this.move.update(ctx);
    this.active.update(ctx);
  }
  applySettings() {
    Object.values(this.scales).forEach((interaction) => {
      interaction.setFree(this.free);
      interaction.setDominantRight?.(this.dominantRight);
    });
    this.move.setDominantRight(this.dominantRight);
    this.move.method = this.method;
  }
  lines() {
    const clutch = this.free ? "ClutchingFree" : "Clutching";
    const dominant = this.dominantRight ? "오른손" : "왼손";
    return [
      `${this.method}  |  ${clutch}  |  우세손 ${dominant}`,
      this.move.status(),
      this.active.status(),
    ];
  }
}

export function makeContext(time, dt, width, height, left, right, gaze, objects, aimMode) {
  return {
    time,
    dt,
    width,
    height,
    left,
    right,
    gaze,
    objects,
    aimMode,
    dominant: (dominantRight) => (dominantRight ? right : left),
    nondominant: (dominantRight) => (dominantRight ? left : right),
  };
}
