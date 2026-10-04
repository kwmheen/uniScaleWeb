import {
  FilesetResolver,
  HandLandmarker,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/+esm";
import {
  METHODS,
  InteractionController,
  createCards,
  makeContext,
} from "./gestures.js";

const WASM = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/wasm";
const HAND_MODEL = new URL("../models/hand_landmarker.task", import.meta.url).href;

const WRIST = 0;
const THUMB_TIP = 4;
const INDEX_TIP = 8;
const MIDDLE_MCP = 9;
const MIDDLE_TIP = 12;
const CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17],
];
const THEME = "#2a52be";
const DISTRICTS = [
  { x: 0.22, y: 0.32, name: "Harbor" },
  { x: 0.48, y: 0.42, name: "Square" },
  { x: 0.74, y: 0.28, name: "Hill" },
  { x: 0.32, y: 0.7, name: "Park" },
  { x: 0.68, y: 0.74, name: "Station" },
];
const NEARBY = [
  { x: 0.15, y: 0.22, name: "Lighthouse" },
  { x: 0.28, y: 0.38, name: "Pier" },
  { x: 0.2, y: 0.42, name: "Warehouse" },
  { x: 0.43, y: 0.35, name: "Fountain" },
  { x: 0.78, y: 0.2, name: "Tower" },
  { x: 0.27, y: 0.64, name: "Pond" },
  { x: 0.74, y: 0.66, name: "Clock" },
];
const DETAILS = [
  { x: 0.11, y: 0.16, name: "Buoy", color: "#c8392c" },
  { x: 0.185, y: 0.2, name: "Bell" },
  { x: 0.46, y: 0.37, name: "Coin" },
  { x: 0.3, y: 0.66, name: "Bench" },
  { x: 0.8, y: 0.24, name: "Flag" },
];
const WINGS = [
  { x: 0.24, y: 0.28, name: "South" },
  { x: 0.5, y: 0.24, name: "East" },
  { x: 0.3, y: 0.62, name: "West" },
  { x: 0.52, y: 0.78, name: "Core" },
  { x: 0.82, y: 0.72, name: "North" },
];
const PARTS = [
  { x: 0.74, y: 0.58, name: "Bracket" },
  { x: 0.2, y: 0.22, name: "Slot" },
  { x: 0.46, y: 0.18, name: "Rail" },
  { x: 0.26, y: 0.54, name: "Boss" },
  { x: 0.48, y: 0.7, name: "Plate" },
];
const HOLES = [
  { x: 0.7, y: 0.54, name: "6 mm", color: "#c8392c" },
  { x: 0.78, y: 0.62, name: "4 mm" },
  { x: 0.18, y: 0.2, name: "8 mm" },
  { x: 0.44, y: 0.16, name: "3 mm" },
  { x: 0.5, y: 0.74, name: "10 mm" },
];
const GOALS = {
  map: "Near the Harbor there is a Lighthouse. Find the red buoy.",
  cad: "On the North wing there is a Bracket. Find the 6 mm hole.",
};
const APPS = ["base", "map", "cad"];

const video = document.querySelector("#video");
const canvas = document.querySelector("#canvas");
const stage = document.querySelector("#stage");
const startButton = document.querySelector("#start");
const verdictBox = document.querySelector("#verdict");
const checksBox = document.querySelector("#checks");
const modeBox = document.querySelector("#modes");
const ctx2d = canvas.getContext("2d");

const controller = new InteractionController();
const cards = createCards();
const smooth = {
  Left: { image: null, world: null, index: false, middle: false },
  Right: { image: null, world: null, index: false, middle: false },
};

let handLandmarker = null;
let running = false;
let lastTime = 0;
let lastStamp = 0;
let fps = 0;
let appMode = "base";
let loopError = "";
let cameraOn = false;
let lastReport = null;
let shuttingDown = false;
let aliveTimer = 0;

const latch = (active, distance, on, off) => (active ? distance <= off : distance <= on);

function landmarksToArray(landmarks) {
  return landmarks.map((landmark) => [landmark.x, landmark.y, landmark.z ?? 0]);
}

function ema(previous, current, alpha) {
  if (!previous) return current.map((point) => [...point]);
  return current.map((point, index) =>
    point.map((value, axis) => alpha * value + (1 - alpha) * previous[index][axis]),
  );
}

function cameraOf(pixel, world, width, height) {
  const palm = Math.hypot(pixel[MIDDLE_MCP][0] - pixel[WRIST][0], pixel[MIDDLE_MCP][1] - pixel[WRIST][1]);
  const palmPx = Math.max(palm, 8);
  const metersPerPixel = 0.09 / palmPx;
  const depth = 0.55 * (160 / palmPx);
  return world.map((point, index) => [
    (pixel[index][0] - width / 2) * metersPerPixel,
    (height / 2 - pixel[index][1]) * metersPerPixel,
    depth + (point[2] - world[0][2]),
  ]);
}

function readHands(result, width, height) {
  const found = {};
  const groups = result.landmarks ?? [];
  const worlds = result.worldLandmarks ?? [];
  const labels = result.handedness ?? result.handednesses ?? [];
  const seen = new Set();

  groups.forEach((group, index) => {
    if (!worlds[index] || !labels[index]?.[0]) return;
    let label = labels[index][0].categoryName || labels[index][0].displayName || "Left";
    if (label !== "Left" && label !== "Right") label = index === 0 ? "Left" : "Right";
    if (seen.has(label)) return;
    seen.add(label);
    const state = smooth[label];
    const image = ema(state.image, landmarksToArray(group), 0.55);
    const world = ema(state.world, landmarksToArray(worlds[index]), 0.55);
    state.image = image;
    state.world = world;
    const pixel = image.map((point) => [(1 - point[0]) * width, point[1] * height]);
    const indexDistance = Math.hypot(world[INDEX_TIP][0] - world[THUMB_TIP][0], world[INDEX_TIP][1] - world[THUMB_TIP][1], world[INDEX_TIP][2] - world[THUMB_TIP][2]);
    const middleDistance = Math.hypot(world[MIDDLE_TIP][0] - world[THUMB_TIP][0], world[MIDDLE_TIP][1] - world[THUMB_TIP][1], world[MIDDLE_TIP][2] - world[THUMB_TIP][2]);
    state.index = latch(state.index, indexDistance, 0.045, 0.07);
    state.middle = latch(state.middle, middleDistance, 0.05, 0.08);
    found[label] = {
      handedness: label,
      tracked: true,
      world,
      pixel,
      camera: cameraOf(pixel, world, width, height),
      indexThumb: indexDistance,
      middleThumb: middleDistance,
      indexPinching: state.index,
      middlePinching: state.middle,
    };
  });

  for (const label of ["Left", "Right"]) {
    if (seen.has(label)) continue;
    smooth[label].image = null;
    smooth[label].world = null;
    smooth[label].index = false;
    smooth[label].middle = false;
  }
  return [found.Left ?? null, found.Right ?? null];
}

function project(card, mx, my, x, y, w, h) {
  return [
    x + w / 2 + (mx - card.mapX) * w * card.mapZoom,
    y + h / 2 + (my - card.mapY) * h * card.mapZoom,
  ];
}

function inFrame(px, py, x, y, w, h) {
  return px >= x + 10 && px <= x + w - 10 && py >= y + 28 && py <= y + h - 10;
}

function drawMarks(card, places, x, y, w, h, radius, fontSize) {
  ctx2d.font = `${fontSize}px Palatino, Georgia, serif`;
  ctx2d.textBaseline = "middle";
  for (const place of places) {
    const [px, py] = project(card, place.x, place.y, x, y, w, h);
    if (!inFrame(px, py, x, y, w, h)) continue;
    ctx2d.fillStyle = place.color || THEME;
    ctx2d.beginPath();
    ctx2d.arc(px, py, radius, 0, Math.PI * 2);
    ctx2d.fill();
    ctx2d.fillStyle = "#161616";
    ctx2d.fillText(place.name, px + radius + 5, py);
  }
}

function drawMap(card, x, y, w, h) {
  const zoom = card.mapZoom;
  ctx2d.save();
  ctx2d.beginPath();
  ctx2d.rect(x, y, w, h);
  ctx2d.clip();
  ctx2d.translate(x + w / 2, y + h / 2);
  ctx2d.scale(zoom, zoom);
  ctx2d.translate(-card.mapX * w, -card.mapY * h);

  ctx2d.fillStyle = "#d7e6f2";
  ctx2d.fillRect(0, 0, w, h);
  ctx2d.fillStyle = "#efe4cf";
  ctx2d.beginPath();
  ctx2d.moveTo(w * 0.06, h * 0.22);
  ctx2d.lineTo(w * 0.28, h * 0.08);
  ctx2d.lineTo(w * 0.7, h * 0.1);
  ctx2d.lineTo(w * 0.94, h * 0.26);
  ctx2d.lineTo(w * 0.9, h * 0.78);
  ctx2d.lineTo(w * 0.58, h * 0.94);
  ctx2d.lineTo(w * 0.16, h * 0.86);
  ctx2d.closePath();
  ctx2d.fill();

  ctx2d.fillStyle = "#b7c99a";
  ctx2d.fillRect(w * 0.2, h * 0.58, w * 0.2, h * 0.16);
  ctx2d.fillRect(w * 0.62, h * 0.22, w * 0.16, h * 0.12);

  const stroke = (px) => px / zoom;
  ctx2d.strokeStyle = "#7eb0d4";
  ctx2d.lineWidth = stroke(Math.max(6, w * 0.028));
  ctx2d.beginPath();
  ctx2d.moveTo(w * 0.08, h * 0.48);
  ctx2d.quadraticCurveTo(w * 0.4, h * 0.36, w * 0.58, h * 0.52);
  ctx2d.quadraticCurveTo(w * 0.74, h * 0.66, w * 0.96, h * 0.58);
  ctx2d.stroke();

  ctx2d.strokeStyle = "#f7f3ea";
  ctx2d.lineWidth = stroke(Math.max(2, w * 0.012));
  ctx2d.beginPath();
  for (const t of [0.28, 0.46, 0.64, 0.82]) {
    ctx2d.moveTo(w * 0.1, h * t);
    ctx2d.lineTo(w * 0.9, h * t);
    ctx2d.moveTo(w * t, h * 0.14);
    ctx2d.lineTo(w * t, h * 0.88);
  }
  ctx2d.stroke();
  ctx2d.restore();

  ctx2d.save();
  ctx2d.beginPath();
  ctx2d.rect(x, y, w, h);
  ctx2d.clip();
  drawMarks(card, DISTRICTS, x, y, w, h, 5, 18);
  if (zoom >= 3) drawMarks(card, NEARBY, x, y, w, h, 4, 15);
  if (zoom >= 7) drawMarks(card, DETAILS, x, y, w, h, 4, 14);
  ctx2d.restore();

  drawZoomLabel(card, x, y, zoom >= 7 ? "Detail" : zoom >= 3 ? "Nearby" : "Overview");
}

function drawCad(card, x, y, w, h) {
  const zoom = card.mapZoom;
  const stroke = (px) => px / zoom;
  const blocks = [
    [0.08, 0.12, 0.36, 0.3],
    [0.52, 0.1, 0.4, 0.26],
    [0.1, 0.5, 0.32, 0.36],
    [0.48, 0.46, 0.44, 0.42],
  ];
  ctx2d.save();
  ctx2d.beginPath();
  ctx2d.rect(x, y, w, h);
  ctx2d.clip();
  ctx2d.translate(x + w / 2, y + h / 2);
  ctx2d.scale(zoom, zoom);
  ctx2d.translate(-card.mapX * w, -card.mapY * h);

  ctx2d.fillStyle = "#e4edf5";
  ctx2d.fillRect(0, 0, w, h);
  ctx2d.strokeStyle = "#c5d4e6";
  ctx2d.lineWidth = stroke(1);
  ctx2d.beginPath();
  for (const t of [0.2, 0.4, 0.6, 0.8]) {
    ctx2d.moveTo(w * t, h * 0.06);
    ctx2d.lineTo(w * t, h * 0.94);
    ctx2d.moveTo(w * 0.06, h * t);
    ctx2d.lineTo(w * 0.94, h * t);
  }
  ctx2d.stroke();
  ctx2d.strokeStyle = "#16325c";
  ctx2d.lineWidth = stroke(Math.max(1.5, w * 0.004));
  for (const [bx, by, bw, bh] of blocks) {
    ctx2d.strokeRect(w * bx, h * by, w * bw, h * bh);
  }
  ctx2d.strokeRect(w * 0.66, h * 0.5, w * 0.22, h * 0.16);
  ctx2d.lineWidth = stroke(1.5);
  for (const hole of HOLES) {
    ctx2d.beginPath();
    ctx2d.arc(w * hole.x, h * hole.y, Math.max(3, w * 0.012), 0, Math.PI * 2);
    ctx2d.stroke();
  }
  ctx2d.restore();

  ctx2d.save();
  ctx2d.beginPath();
  ctx2d.rect(x, y, w, h);
  ctx2d.clip();
  drawMarks(card, WINGS, x, y, w, h, 5, 18);
  if (zoom >= 3) drawMarks(card, PARTS, x, y, w, h, 4, 15);
  if (zoom >= 7) drawMarks(card, HOLES, x, y, w, h, 4, 14);
  ctx2d.restore();
  drawZoomLabel(card, x, y, zoom >= 7 ? "Callouts" : zoom >= 3 ? "Parts" : "Plan");
}

function drawZoomLabel(card, x, y, layer) {
  ctx2d.fillStyle = "#161616";
  ctx2d.font = "24px Palatino Linotype, Palatino, Georgia, serif";
  ctx2d.textBaseline = "alphabetic";
  ctx2d.fillText(`${card.mapZoom.toFixed(1)}×  ${layer}`, x + 14, y + 32);
}

function drawScene(width, height, left, right, gaze) {
  ctx2d.clearRect(0, 0, width, height);
  if (video.readyState >= 2 && video.videoWidth > 0) {
    ctx2d.save();
    ctx2d.translate(width, 0);
    ctx2d.scale(-1, 1);
    ctx2d.drawImage(video, 0, 0, width, height);
    ctx2d.restore();
  } else {
    ctx2d.fillStyle = "#171a22";
    ctx2d.fillRect(0, 0, width, height);
    ctx2d.fillStyle = "#f4f1ea";
    ctx2d.font = "bold 28px Palatino Linotype, Palatino, Georgia, serif";
    ctx2d.fillText("Camera off", 48, height / 2);
  }
  const scaleSelected = controller.active.selected;
  const moveSelected = controller.move.selected;
  for (const card of cards) {
    const [x, y, w, h] = card.rect(width, height);
    const active = card === scaleSelected || card === moveSelected;
    if (appMode === "map") {
      drawMap(card, x, y, w, h);
    } else if (appMode === "cad") {
      drawCad(card, x, y, w, h);
    } else {
      ctx2d.fillStyle = "rgba(243, 239, 230, 0.82)";
      ctx2d.fillRect(x, y, w, h);
      ctx2d.fillStyle = "#161616";
      ctx2d.font = "24px Palatino Linotype, Palatino, Apple SD Gothic Neo, serif";
      ctx2d.fillText(`${card.name}  ${card.scale.toFixed(2)}`, x + 14, y + 34);
    }
    ctx2d.lineWidth = active ? 3 : 1.5;
    ctx2d.strokeStyle = active ? THEME : "#161616";
    ctx2d.strokeRect(x, y, w, h);
  }

  for (const hand of [left, right]) {
    if (!hand) continue;
    ctx2d.strokeStyle = hand.indexPinching || hand.middlePinching ? THEME : "#f3efe6";
    ctx2d.lineWidth = 3;
    ctx2d.beginPath();
    for (const [start, end] of CONNECTIONS) {
      ctx2d.moveTo(hand.pixel[start][0], hand.pixel[start][1]);
      ctx2d.lineTo(hand.pixel[end][0], hand.pixel[end][1]);
    }
    ctx2d.stroke();
  }
}

function checkItem(label, ok) {
  const item = document.createElement("li");
  const name = document.createElement("span");
  const mark = document.createElement("span");
  name.textContent = label;
  mark.textContent = ok ? "On" : "Off";
  mark.className = ok ? "ok" : "bad";
  item.append(name, mark);
  return item;
}

function publishStatus(report) {
  if (report) lastReport = report;
  report = report ?? lastReport ?? {};
  if (report.stopped || shuttingDown) {
    checksBox.replaceChildren(checkItem("Camera", false), checkItem("Server", false));
    verdictBox.textContent = "Stopped";
    verdictBox.className = "verdict bad";
    return;
  }
  const camera = report.camera ?? cameraOn;
  const hands = report?.hands ?? false;
  const scaling = Boolean(controller.active.selected);
  const moving = Boolean(controller.move.selected);
  const framed = appMode !== "base";
  const checks = [
    checkItem("Camera", camera),
    checkItem("Hands", hands),
    checkItem(framed ? "Zoom" : "Scaling", scaling),
    checkItem(framed ? "Panning" : "Moving", moving),
  ];
  checksBox.replaceChildren(...checks);
  for (const mode of APPS) {
    document.querySelector(`#app-${mode}`)?.classList.toggle("active", appMode === mode);
  }

  let tone = "bad";
  let headline = "Camera off";
  if (!camera) {
    headline = "Camera off";
  } else if (loopError) {
    headline = "Tracking error";
  } else if (scaling) {
    headline = appMode === "base" ? "Scaling" : "Zooming";
    tone = "ok";
  } else if (moving && appMode !== "base") {
    headline = "Panning";
    tone = "ok";
  } else if (moving) {
    headline = "Moving";
    tone = "ok";
  } else if (hands) {
    headline = "Hand tracked";
    tone = "wait";
  } else if (camera) {
    headline = "Show a hand";
    tone = "wait";
  }
  verdictBox.textContent = headline;
  verdictBox.className = `verdict ${tone}`;
  document.querySelector("#clutch").textContent = controller.free ? "ClutchingFree" : "Clutching";
  document.querySelector("#hand").textContent = controller.dominantRight ? "Right hand" : "Left hand";
  for (const button of modeBox.querySelectorAll("button")) {
    button.classList.toggle("active", button.dataset.method === controller.method);
  }
}

async function createLandmarker(factory, modelPath, extra) {
  const base = { modelAssetPath: modelPath };
  try {
    return await factory.createFromOptions(fileset, {
      ...extra,
      baseOptions: { ...base, delegate: "GPU" },
    });
  } catch (error) {
    console.warn(error);
    return factory.createFromOptions(fileset, {
      ...extra,
      baseOptions: { ...base, delegate: "CPU" },
    });
  }
}

let fileset = null;

async function ensureModels() {
  if (handLandmarker) return;
  startButton.textContent = "Loading model";
  fileset = await FilesetResolver.forVisionTasks(WASM);
  handLandmarker = await createLandmarker(HandLandmarker, HAND_MODEL, {
    runningMode: "VIDEO",
    numHands: 2,
    minHandDetectionConfidence: 0.5,
    minHandPresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });
}

function frameStamp() {
  let stamp = performance.now();
  if (stamp <= lastStamp) stamp = lastStamp + 1;
  lastStamp = stamp;
  return stamp;
}

function releaseCamera() {
  running = false;
  cameraOn = false;
  const stream = video.srcObject;
  if (stream) {
    for (const track of stream.getTracks()) track.stop();
  }
  video.srcObject = null;
}

async function shutdownApp() {
  if (shuttingDown) return;
  shuttingDown = true;
  clearInterval(aliveTimer);
  releaseCamera();
  const stopButton = document.querySelector("#stop");
  stopButton.disabled = true;
  stopButton.textContent = "Quitting";
  startButton.hidden = true;
  publishStatus({ camera: false, stopped: true });
  drawScene(canvas.width, canvas.height, null, null, { tracked: false, x: 0, y: 0 });
  try {
    await fetch("/shutdown", { method: "POST", keepalive: true });
  } catch {
    // 서버가 이미 꺼진 경우에도 카메라는 위에서 멈춰 있다.
  }
  stopButton.textContent = "Stopped";
  publishStatus({ camera: false, stopped: true });
}

function startHeartbeat() {
  const beat = async () => {
    if (shuttingDown) return;
    try {
      const response = await fetch("/ping", { cache: "no-store" });
      if (!response.ok) throw new Error("ping");
    } catch {
      if (shuttingDown) return;
      shuttingDown = true;
      clearInterval(aliveTimer);
      releaseCamera();
      const stopButton = document.querySelector("#stop");
      stopButton.disabled = true;
      stopButton.textContent = "Stopped";
      startButton.hidden = true;
      publishStatus({ camera: false, stopped: true });
      drawScene(canvas.width, canvas.height, null, null, { tracked: false, x: 0, y: 0 });
    }
  };
  beat();
  aliveTimer = setInterval(beat, 2000);
}

function loop() {
  if (!running || shuttingDown) return;
  const width = video.videoWidth;
  const height = video.videoHeight;
  if (!width || !height) {
    requestAnimationFrame(loop);
    return;
  }
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
    stage.style.aspectRatio = `${width} / ${height}`;
  }

  const now = performance.now() / 1000;
  const dt = Math.min(Math.max(now - lastTime, 0.0001), 0.05);
  lastTime = now;
  fps = fps ? fps * 0.9 + (1 / dt) * 0.1 : 1 / dt;
  const stamp = frameStamp();

  let left = null;
  let right = null;
  try {
    [left, right] = readHands(handLandmarker.detectForVideo(video, stamp), width, height);
    loopError = "";
  } catch (error) {
    loopError = error.message ?? String(error);
  }

  const gaze = { tracked: true, x: cards[0].nx * width, y: cards[0].ny * height, hit: true, origin: null, dir: null };
  const frame = makeContext(now, dt, width, height, left, right, gaze, cards, "hand");
  controller.update(frame);
  drawScene(width, height, left, right, gaze);
  publishStatus({
    camera: true,
    hands: Boolean(left || right),
    scaling: Boolean(controller.active.selected),
    panning: appMode !== "base" && Boolean(controller.move.selected),
  });
  requestAnimationFrame(loop);
}

async function openCameraStream() {
  const attempts = [
    { audio: false, video: { facingMode: { ideal: "user" }, width: { ideal: 1280 }, height: { ideal: 720 } } },
    { audio: false, video: { width: { ideal: 1280 }, height: { ideal: 720 } } },
    { audio: false, video: true },
  ];
  let lastError = null;
  for (const constraints of attempts) {
    try {
      return await navigator.mediaDevices.getUserMedia(constraints);
    } catch (error) {
      lastError = error;
      if (error?.name === "NotAllowedError" || error?.name === "PermissionDeniedError") throw error;
    }
  }
  throw lastError ?? new Error("camera");
}

async function startCamera() {
  if (running) return;
  startButton.disabled = true;
  try {
    await ensureModels();
  } catch (error) {
    console.error(error);
    startButton.disabled = false;
    startButton.textContent = "Start camera";
    verdictBox.textContent = "Model failed";
    verdictBox.className = "verdict bad";
    return;
  }
  try {
    startButton.textContent = "Connecting";
    const stream = await openCameraStream();
    video.srcObject = stream;
    await video.play();
    cameraOn = true;
    running = true;
    lastTime = performance.now() / 1000;
    startButton.hidden = true;
    loop();
  } catch (error) {
    cameraOn = false;
    startButton.disabled = false;
    startButton.textContent = "Start camera";
    publishStatus({ camera: false });
  }
}

function setApp(mode) {
  appMode = mode;
  for (const card of cards) card.useMode(mode);
  const goal = document.querySelector("#goal");
  goal.hidden = !GOALS[mode];
  if (GOALS[mode]) goal.textContent = GOALS[mode];
  controller.resetAll();
  publishStatus();
  if (!running) drawScene(canvas.width, canvas.height, null, null, { tracked: false, x: 0, y: 0 });
}

function bindControls() {
  for (const [method, label] of METHODS) {
    const button = document.createElement("button");
    button.type = "button";
    button.dataset.method = method;
    button.textContent = label;
    button.addEventListener("click", () => {
      controller.setMethod(method);
      publishStatus();
    });
    modeBox.append(button);
  }
  document.querySelector("#clutch").addEventListener("click", () => {
    controller.toggleFree();
    publishStatus();
  });
  document.querySelector("#hand").addEventListener("click", () => {
    controller.toggleDominant();
    publishStatus();
  });
  document.querySelector("#app-base").addEventListener("click", () => setApp("base"));
  document.querySelector("#app-map").addEventListener("click", () => setApp("map"));
  document.querySelector("#app-cad").addEventListener("click", () => setApp("cad"));
  document.querySelector("#palm").addEventListener("click", () => controller.uniSemi.togglePalm());
  document.querySelector("#reset").addEventListener("click", () => {
    cards.forEach((card) => card.reset());
    controller.resetAll();
  });
  startButton.addEventListener("click", startCamera);
  document.querySelector("#stop").addEventListener("click", shutdownApp);
  window.addEventListener("keydown", (event) => {
    const method = METHODS.find((_, index) => event.key === String(index + 1));
    if (method) controller.setMethod(method[0]);
    else if (event.key === "f" || event.key === "F") controller.toggleFree();
    else if (event.key === "h" || event.key === "H") controller.toggleDominant();
    else if (event.key === "p" || event.key === "P") controller.uniSemi.togglePalm();
    else if (event.key === "r" || event.key === "R") {
      cards.forEach((card) => card.reset());
      controller.resetAll();
    } else if (event.key === "q" || event.key === "Q") {
      setApp(APPS[(APPS.indexOf(appMode) + 1) % APPS.length]);
    }
  });
}

bindControls();
publishStatus({ camera: false });
if (location.protocol !== "file:") startHeartbeat();
drawScene(canvas.width, canvas.height, null, null, { tracked: false, x: 0, y: 0 });
