import {
  FaceLandmarker,
  FilesetResolver,
  HandLandmarker,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/+esm";
import {
  HELP,
  METHODS,
  InteractionController,
  createCards,
  makeContext,
} from "./gestures.js";

const WASM = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/wasm";
const HAND_MODEL = new URL("../models/hand_landmarker.task", import.meta.url).href;
const FACE_MODEL = new URL("../models/face_landmarker.task", import.meta.url).href;

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
const NOSE = 1;
const LEFT_CHEEK = 234;
const RIGHT_CHEEK = 454;

const video = document.querySelector("#video");
const canvas = document.querySelector("#canvas");
const stage = document.querySelector("#stage");
const startButton = document.querySelector("#start");
const verdictBox = document.querySelector("#verdict");
const checksBox = document.querySelector("#checks");
const detailBox = document.querySelector("#detail");
const helpBox = document.querySelector("#help");
const modeBox = document.querySelector("#modes");
const ctx2d = canvas.getContext("2d");

const controller = new InteractionController();
const cards = createCards();
const smooth = {
  Left: { image: null, world: null, index: false, middle: false },
  Right: { image: null, world: null, index: false, middle: false },
};

let handLandmarker = null;
let faceLandmarker = null;
let running = false;
let lastTime = 0;
let lastStamp = 0;
let fps = 0;
let aimMode = "hand";
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

function displayPoint(landmark, width, height) {
  return [(1 - landmark.x) * width, landmark.y * height];
}

function segmentHitsRect(start, end, rect) {
  const [x, y, w, h] = rect;
  if (start[0] >= x && start[0] <= x + w && start[1] >= y && start[1] <= y + h) return start;
  const dx = end[0] - start[0];
  const dy = end[1] - start[1];
  let near = 0;
  let far = 1;
  const p = [-dx, dx, -dy, dy];
  const q = [start[0] - x, x + w - start[0], start[1] - y, y + h - start[1]];
  for (let i = 0; i < 4; i += 1) {
    if (Math.abs(p[i]) < 1e-8) {
      if (q[i] < 0) return null;
    } else {
      const t = q[i] / p[i];
      if (p[i] < 0) near = Math.max(near, t);
      else far = Math.min(far, t);
      if (near > far) return null;
    }
  }
  return [start[0] + dx * near, start[1] + dy * near];
}

function readFaceRay(result, width, height, target) {
  const face = result.faceLandmarks?.[0];
  if (!face || face.length <= RIGHT_CHEEK) {
    return { tracked: false, x: width / 2, y: height / 2, origin: null, dir: null, hit: false, points: face?.length ?? 0 };
  }
  const nose = displayPoint(face[NOSE], width, height);
  const left = displayPoint(face[LEFT_CHEEK], width, height);
  const right = displayPoint(face[RIGHT_CHEEK], width, height);
  const mid = [(left[0] + right[0]) / 2, (left[1] + right[1]) / 2];
  const center = target ? [target.nx * width, target.ny * height] : [width / 2, height / 2];
  const gain = 16;
  const dx = center[0] - nose[0] + (nose[0] - mid[0]) * gain;
  const dy = center[1] - nose[1] + (nose[1] - mid[1]) * gain;
  const mag = Math.hypot(dx, dy) || 1;
  const dir = [dx / mag, dy / mag];
  const reach = Math.hypot(width, height);
  const end = [nose[0] + dir[0] * reach, nose[1] + dir[1] * reach];
  const hitPoint = target ? segmentHitsRect(nose, end, target.rect(width, height)) : null;
  return {
    tracked: Boolean(hitPoint),
    x: hitPoint ? hitPoint[0] : end[0],
    y: hitPoint ? hitPoint[1] : end[1],
    origin: nose,
    dir,
    hit: Boolean(hitPoint),
    points: face.length,
  };
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
    ctx2d.font = "bold 28px Malgun Gothic, sans-serif";
    ctx2d.fillText("카메라가 꺼져 있습니다", 48, height / 2);
  }
  const scaleSelected = controller.active.selected;
  const moveSelected = controller.move.selected;
  for (const card of cards) {
    const [x, y, w, h] = card.rect(width, height);
    const active = card === scaleSelected || card === moveSelected;
    const aimed = aimMode === "ray" ? Boolean(gaze.hit) : active;
    ctx2d.fillStyle = "rgba(243, 239, 230, 0.82)";
    ctx2d.fillRect(x, y, w, h);
    ctx2d.lineWidth = active ? 3 : 1.5;
    ctx2d.strokeStyle = aimed || active ? "#c8392c" : "#161616";
    ctx2d.strokeRect(x, y, w, h);
    ctx2d.fillStyle = "#161616";
    ctx2d.font = "24px Palatino Linotype, Georgia, serif";
    ctx2d.fillText(`${card.name}  ${card.scale.toFixed(2)}`, x + 14, y + 34);
  }

  for (const hand of [left, right]) {
    if (!hand) continue;
    ctx2d.strokeStyle = hand.indexPinching || hand.middlePinching ? "#c8392c" : "#f3efe6";
    ctx2d.lineWidth = 3;
    ctx2d.beginPath();
    for (const [start, end] of CONNECTIONS) {
      ctx2d.moveTo(hand.pixel[start][0], hand.pixel[start][1]);
      ctx2d.lineTo(hand.pixel[end][0], hand.pixel[end][1]);
    }
    ctx2d.stroke();
  }

  if (aimMode === "ray" && gaze.origin && gaze.dir) {
    const reach = Math.hypot(width, height);
    const end = [gaze.origin[0] + gaze.dir[0] * reach, gaze.origin[1] + gaze.dir[1] * reach];
    ctx2d.strokeStyle = gaze.hit ? "#c8392c" : "rgba(243, 239, 230, 0.92)";
    ctx2d.lineWidth = gaze.hit ? 3 : 1.5;
    ctx2d.beginPath();
    ctx2d.moveTo(gaze.origin[0], gaze.origin[1]);
    ctx2d.lineTo(end[0], end[1]);
    ctx2d.stroke();
    ctx2d.fillStyle = "#c8392c";
    ctx2d.fillRect(gaze.origin[0] - 4, gaze.origin[1] - 4, 8, 8);
  }
}

function pinchText(hand, name) {
  if (!hand) return `${name} 없음`;
  return `${name} 검지 ${hand.indexThumb.toFixed(3)}m ${hand.indexPinching ? "핀치" : "열림"} / 중지 ${hand.middleThumb.toFixed(3)}m`;
}

function checkItem(label, ok, detail) {
  const item = document.createElement("li");
  const name = document.createElement("span");
  const mark = document.createElement("span");
  name.textContent = label;
  mark.textContent = ok ? "됨" : "안 됨";
  mark.className = ok ? "ok" : "bad";
  if (detail) name.textContent = `${label} · ${detail}`;
  item.append(name, mark);
  return item;
}

function publishStatus(report) {
  if (report) lastReport = report;
  report = report ?? lastReport ?? {};
  if (report.stopped || shuttingDown) {
    checksBox.replaceChildren(checkItem("카메라 영상", false), checkItem("서버", false));
    verdictBox.textContent = "종료됨";
    verdictBox.className = "verdict bad";
    detailBox.textContent = report.detail || "카메라와 서버가 꺼졌습니다. 이 창을 닫으면 됩니다.";
    return;
  }
  const camera = report.camera ?? cameraOn;
  const face = report?.face ?? false;
  const hands = report?.hands ?? false;
  const gazeOn = report?.gazeOn ?? false;
  const scaling = report?.scaling ?? false;
  const scaleText = report?.scaleText ?? "";
  const aimLabel = aimMode === "hand" ? "대상은 항상 선택" : "레이가 오브젝트에 닿음";
  const checks = [checkItem("카메라 영상", camera)];
  if (aimMode === "ray") checks.push(checkItem("얼굴", face, report?.faceDetail ?? ""));
  checks.push(
    checkItem("손", hands, report?.handDetail ?? ""),
    checkItem(aimLabel, aimMode === "hand" ? camera : Boolean(face && gazeOn)),
    checkItem("크기 조절", scaling, scaleText),
  );
  checksBox.replaceChildren(...checks);
  document.querySelector("#aim-hand")?.classList.toggle("active", aimMode === "hand");
  document.querySelector("#aim-ray")?.classList.toggle("active", aimMode === "ray");
  const aimNote = document.querySelector("#aim-note");
  if (aimNote) {
    aimNote.textContent = aimMode === "hand"
      ? "시선은 맞은 것으로 둡니다. 손 제스처만으로 크기가 바뀝니다."
      : "코끝에서 레이가 나갑니다. 고개를 돌려 오브젝트를 맞춘 뒤 손으로 크기를 바꿉니다.";
  }

  let tone = "bad";
  let headline = "카메라 꺼짐";
  if (!camera) {
    headline = "카메라 꺼짐";
  } else if (loopError) {
    headline = "추적 오류";
  } else if (scaling) {
    headline = "크기 조절 중";
    tone = "ok";
  } else if (aimMode === "hand" && hands) {
    headline = "손 추적 됨";
    tone = "wait";
  } else if (aimMode === "ray" && gazeOn) {
    headline = "레이가 오브젝트에 닿음";
    tone = "wait";
  } else if (aimMode === "ray" && face) {
    headline = "레이가 오브젝트를 비껴 감";
    tone = "wait";
  } else if (camera) {
    headline = aimMode === "hand" ? "손을 보여 주세요" : "얼굴을 보여 주세요";
    tone = "wait";
  }
  verdictBox.textContent = headline;
  verdictBox.className = `verdict ${tone}`;
  detailBox.textContent = [report?.detail ?? "", loopError].filter(Boolean).join(" ");
  helpBox.textContent = HELP[controller.method];
  document.querySelector("#clutch").textContent = controller.free ? "ClutchingFree" : "Clutching";
  document.querySelector("#hand").textContent = controller.dominantRight ? "우세손 오른손" : "우세손 왼손";
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
  if (handLandmarker && faceLandmarker) return;
  startButton.textContent = "모델 불러오는 중";
  fileset = await FilesetResolver.forVisionTasks(WASM);
  handLandmarker = await createLandmarker(HandLandmarker, HAND_MODEL, {
    runningMode: "VIDEO",
    numHands: 2,
    minHandDetectionConfidence: 0.5,
    minHandPresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });
  faceLandmarker = await createLandmarker(FaceLandmarker, FACE_MODEL, {
    runningMode: "VIDEO",
    numFaces: 1,
    outputFaceBlendshapes: false,
    outputFacialTransformationMatrixes: false,
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
  stopButton.textContent = "끄는 중";
  startButton.hidden = true;
  publishStatus({
    camera: false,
    stopped: true,
    detail: "카메라와 서버를 끄는 중입니다.",
  });
  drawScene(canvas.width, canvas.height, null, null, { tracked: false, x: 0, y: 0 });
  try {
    await fetch("/shutdown", { method: "POST", keepalive: true });
  } catch {
    // 서버가 이미 꺼진 경우에도 카메라는 위에서 멈춰 있다.
  }
  stopButton.textContent = "종료됨";
  publishStatus({
    camera: false,
    stopped: true,
    detail: "카메라와 서버가 꺼졌습니다. 이 창을 닫으면 됩니다.",
  });
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
      stopButton.textContent = "종료됨";
      startButton.hidden = true;
      publishStatus({
        camera: false,
        stopped: true,
        detail: "서버가 꺼져 카메라를 멈췄습니다. 이 창을 닫으면 됩니다.",
      });
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
  let ray = { tracked: false, hit: false, points: 0, origin: null, dir: null };
  try {
    [left, right] = readHands(handLandmarker.detectForVideo(video, stamp), width, height);
    ray = readFaceRay(faceLandmarker.detectForVideo(video, stamp), width, height, cards[0]);
    loopError = "";
  } catch (error) {
    loopError = `추적 오류: ${error.message ?? error}`;
  }

  const gaze = aimMode === "hand"
    ? { tracked: true, x: cards[0].nx * width, y: cards[0].ny * height, hit: true, origin: null, dir: null }
    : ray;
  const frame = makeContext(now, dt, width, height, left, right, gaze, cards, aimMode);
  controller.update(frame);
  drawScene(width, height, left, right, gaze);
  const target = cards[0];
  const gazeOn = aimMode === "hand" ? true : Boolean(ray.hit);
  const handDetail = [left ? pinchText(left, "왼손") : "", right ? pinchText(right, "오른손") : ""].filter(Boolean).join(" / ");
  publishStatus({
    camera: true,
    face: ray.points > 0,
    faceDetail: ray.points ? `${ray.points}점` : "",
    hands: Boolean(left || right),
    handDetail,
    gazeOn,
    scaling: Boolean(controller.active.selected),
    scaleText: target ? `${target.scale.toFixed(2)}배` : "",
    detail: `${fps.toFixed(0)} FPS · ${controller.lines().join(" · ")}`,
  });
  requestAnimationFrame(loop);
}

async function startCamera() {
  if (running) return;
  startButton.disabled = true;
  try {
    await ensureModels();
    startButton.textContent = "카메라 연결 중";
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
    });
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
    startButton.textContent = "카메라 시작";
    publishStatus({
      camera: false,
      detail: `카메라를 열지 못했습니다. 브라우저에서 카메라 허용을 눌러 주세요. (${error.message ?? error})`,
    });
  }
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
  document.querySelector("#aim-hand").addEventListener("click", () => {
    aimMode = "hand";
    controller.resetAll();
    publishStatus();
  });
  document.querySelector("#aim-ray").addEventListener("click", () => {
    aimMode = "ray";
    controller.resetAll();
    publishStatus();
  });
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
      aimMode = aimMode === "hand" ? "ray" : "hand";
      controller.resetAll();
      publishStatus();
    }
  });
}

bindControls();
publishStatus({
  camera: false,
  detail: location.protocol === "file:"
    ? "파일을 직접 열면 카메라가 막힙니다. run.bat 으로 여세요."
    : "카메라 시작을 누르면 영상이 이 화면에 그려집니다. 종료를 누르면 카메라와 서버가 같이 꺼집니다.",
});
if (location.protocol !== "file:") startHeartbeat();
drawScene(canvas.width, canvas.height, null, null, { tracked: false, x: 0, y: 0 });
