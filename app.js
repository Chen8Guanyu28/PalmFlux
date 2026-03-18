import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.182.0/build/three.module.js';
import { FilesetResolver, HandLandmarker } from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3';

const MODEL_ASSET = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';
const WASM_ROOT = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm';

const ui = {
  status: document.getElementById('status'),
  tracking: document.getElementById('metric-tracking'),
  openness: document.getElementById('metric-openness'),
  radius: document.getElementById('metric-radius'),
  mode: document.getElementById('metric-mode'),
  btnCamera: document.getElementById('btn-camera'),
  btnPreview: document.getElementById('btn-preview'),
  previewShell: document.getElementById('preview-shell'),
  video: document.getElementById('video'),
  overlay: document.getElementById('overlay'),
};

const overlayCtx = ui.overlay.getContext('2d');

const state = {
  handLandmarker: null,
  cameraReady: false,
  previewVisible: false,
  showDebug: false,
  lastVideoTime: -1,
  lastFrameMs: performance.now(),
  lastSeenHandMs: 0,

  // smoothed continuous control
  smoothedLandmarks: null,
  opennessRaw: 0.55,
  opennessSmooth: 0.55,
  radiusCurrent: 2.2,
  radiusTarget: 2.2,
};

const config = {
  particleCount: 12000,
  morphSpring: 7.2,
  particleDamping: 3.25,

  // Radius range for contraction / expansion
  minRadius: 1.25,
  maxRadius: 3.15,

  // Smoothing
  landmarkAlpha: 0.28,     // lower = steadier, higher = faster
  opennessAlpha: 0.12,     // stronger smoothing on signal
  radiusFollowK: 4.2,      // radius target -> current
  opennessDeadband: 0.015, // suppress tiny flicker

  // Tracker thresholds
  minDetection: 0.62,
  minPresence: 0.62,
  minTracking: 0.62,
};

const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [0, 9], [9, 10], [10, 11], [11, 12],
  [0, 13], [13, 14], [14, 15], [15, 16],
  [0, 17], [17, 18], [18, 19], [19, 20],
  [5, 9], [9, 13], [13, 17],
];

function dist2(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

function remap01(value, inMin, inMax) {
  return clamp((value - inMin) / (inMax - inMin), 0, 1);
}

function smoothScalar(current, target, alpha) {
  return current + (target - current) * alpha;
}

function smoothLandmarks(prev, next, alpha) {
  if (!prev) return next.map((p) => ({ x: p.x, y: p.y, z: p.z }));
  const out = [];
  for (let i = 0; i < next.length; i += 1) {
    out.push({
      x: lerp(prev[i].x, next[i].x, alpha),
      y: lerp(prev[i].y, next[i].y, alpha),
      z: lerp(prev[i].z, next[i].z, alpha),
    });
  }
  return out;
}

function palmCenter(landmarks) {
  const ids = [0, 5, 9, 13, 17];
  let x = 0;
  let y = 0;
  for (const id of ids) {
    x += landmarks[id].x;
    y += landmarks[id].y;
  }
  return { x: x / ids.length, y: y / ids.length };
}

// Robust continuous control signal:
// average fingertip distance to palm center, normalized by palm width.
function computeHandOpenness(landmarks) {
  const center = palmCenter(landmarks);
  const palmWidth = dist2(landmarks[5], landmarks[17]) + 1e-6;
  const tips = [4, 8, 12, 16, 20];

  let sum = 0;
  for (const id of tips) {
    sum += dist2(landmarks[id], center) / palmWidth;
  }
  const mean = sum / tips.length;

  // Typical usable range on webcam is about 0.85 ~ 1.9
  return remap01(mean, 0.88, 1.92);
}

function clearOverlay() {
  overlayCtx.clearRect(0, 0, ui.overlay.width, ui.overlay.height);
}

function drawDebug(landmarks) {
  if (!state.previewVisible || !state.showDebug || !landmarks) {
    clearOverlay();
    return;
  }
  clearOverlay();
  const width = ui.overlay.width;
  const height = ui.overlay.height;

  overlayCtx.save();
  overlayCtx.strokeStyle = 'rgba(255,255,255,0.72)';
  overlayCtx.lineWidth = 2;
  for (const [a, b] of HAND_CONNECTIONS) {
    const pa = landmarks[a];
    const pb = landmarks[b];
    overlayCtx.beginPath();
    overlayCtx.moveTo((1 - pa.x) * width, pa.y * height);
    overlayCtx.lineTo((1 - pb.x) * width, pb.y * height);
    overlayCtx.stroke();
  }
  overlayCtx.fillStyle = 'rgba(92,168,255,0.95)';
  for (const p of landmarks) {
    overlayCtx.beginPath();
    overlayCtx.arc((1 - p.x) * width, p.y * height, 3, 0, Math.PI * 2);
    overlayCtx.fill();
  }
  overlayCtx.restore();
}

async function initHandTracking() {
  ui.status.textContent = 'Loading hand tracker…';
  const vision = await FilesetResolver.forVisionTasks(WASM_ROOT);
  try {
    state.handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: MODEL_ASSET,
        delegate: 'GPU',
      },
      runningMode: 'VIDEO',
      numHands: 1,
      minHandDetectionConfidence: config.minDetection,
      minHandPresenceConfidence: config.minPresence,
      minTrackingConfidence: config.minTracking,
    });
  } catch (error) {
    console.warn('GPU delegate unavailable, switching to CPU.', error);
    state.handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: MODEL_ASSET,
        delegate: 'CPU',
      },
      runningMode: 'VIDEO',
      numHands: 1,
      minHandDetectionConfidence: config.minDetection,
      minHandPresenceConfidence: config.minPresence,
      minTrackingConfidence: config.minTracking,
    });
    ui.mode.textContent = 'CPU tracking';
  }
  ui.status.textContent = 'Tracker ready';
}

async function enableCamera() {
  if (state.cameraReady) return;
  const stream = await navigator.mediaDevices.getUserMedia({
    video: {
      facingMode: 'user',
      width: { ideal: 1280 },
      height: { ideal: 720 },
      frameRate: { ideal: 60, max: 60 },
    },
    audio: false,
  });
  ui.video.srcObject = stream;
  await ui.video.play();
  state.cameraReady = true;
  ui.overlay.width = ui.video.videoWidth || 640;
  ui.overlay.height = ui.video.videoHeight || 480;
  ui.status.textContent = 'Camera live';
}

ui.btnCamera.addEventListener('click', async () => {
  try {
    await enableCamera();
  } catch (error) {
    console.error(error);
    ui.status.textContent = 'Camera failed';
  }
});

ui.btnPreview.addEventListener('click', () => {
  state.previewVisible = !state.previewVisible;
  ui.previewShell.hidden = !state.previewVisible;
  if (!state.previewVisible) clearOverlay();
});

window.addEventListener('keydown', (event) => {
  if (event.key.toLowerCase() === 'd') {
    state.showDebug = !state.showDebug;
    if (!state.showDebug) clearOverlay();
  }
});

// ---- Three.js sphere particles ----
const stage = document.getElementById('stage');
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
stage.appendChild(renderer.domElement);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(55, window.innerWidth / window.innerHeight, 0.1, 100);
camera.position.set(0, 0, 8.8);

const particleGroup = new THREE.Group();
scene.add(particleGroup);

const positions = new Float32Array(config.particleCount * 3);
const velocities = new Float32Array(config.particleCount * 3);
const unitSphere = new Float32Array(config.particleCount * 3);

for (let i = 0; i < config.particleCount; i += 1) {
  const u = Math.random();
  const v = Math.random();
  const theta = 2 * Math.PI * u;
  const phi = Math.acos(2 * v - 1);
  const x = Math.sin(phi) * Math.cos(theta);
  const y = Math.sin(phi) * Math.sin(theta);
  const z = Math.cos(phi);
  const k = i * 3;
  unitSphere[k] = x;
  unitSphere[k + 1] = y;
  unitSphere[k + 2] = z;

  positions[k] = x * state.radiusCurrent;
  positions[k + 1] = y * state.radiusCurrent;
  positions[k + 2] = z * state.radiusCurrent;
}

const geometry = new THREE.BufferGeometry();
geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
const material = new THREE.PointsMaterial({
  size: 0.03,
  transparent: true,
  opacity: 0.94,
  depthWrite: false,
});
const points = new THREE.Points(geometry, material);
particleGroup.add(points);

window.addEventListener('resize', () => {
  renderer.setSize(window.innerWidth, window.innerHeight);
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
});

function updateRadiusFromHand(dt) {
  const alpha = 1 - Math.exp(-config.radiusFollowK * dt);
  state.radiusCurrent = lerp(state.radiusCurrent, state.radiusTarget, alpha);
}

function stepParticles(dt) {
  for (let i = 0; i < config.particleCount; i += 1) {
    const k = i * 3;

    let px = positions[k];
    let py = positions[k + 1];
    let pz = positions[k + 2];
    let vx = velocities[k];
    let vy = velocities[k + 1];
    let vz = velocities[k + 2];

    const tx = unitSphere[k] * state.radiusCurrent;
    const ty = unitSphere[k + 1] * state.radiusCurrent;
    const tz = unitSphere[k + 2] * state.radiusCurrent;

    vx += (tx - px) * config.morphSpring * dt;
    vy += (ty - py) * config.morphSpring * dt;
    vz += (tz - pz) * config.morphSpring * dt;

    const damp = Math.exp(-config.particleDamping * dt);
    vx *= damp;
    vy *= damp;
    vz *= damp;

    px += vx;
    py += vy;
    pz += vz;

    positions[k] = px;
    positions[k + 1] = py;
    positions[k + 2] = pz;
    velocities[k] = vx;
    velocities[k + 1] = vy;
    velocities[k + 2] = vz;
  }

  geometry.attributes.position.needsUpdate = true;
  particleGroup.rotation.y += dt * 0.08;
}

function updateFromTracking(now) {
  let landmarks = null;

  if (
    state.cameraReady &&
    state.handLandmarker &&
    ui.video.readyState >= 2 &&
    ui.video.currentTime !== state.lastVideoTime
  ) {
    state.lastVideoTime = ui.video.currentTime;
    const result = state.handLandmarker.detectForVideo(ui.video, now);
    landmarks = result?.landmarks?.[0] || null;
  }

  if (!landmarks) {
    // Do not drop immediately. Keep last stable radius.
    ui.tracking.textContent = 'searching';
    ui.openness.textContent = state.opennessSmooth.toFixed(2);
    drawDebug(state.smoothedLandmarks);
    return;
  }

  state.lastSeenHandMs = now;
  state.smoothedLandmarks = smoothLandmarks(
    state.smoothedLandmarks,
    landmarks,
    config.landmarkAlpha
  );

  const openness = computeHandOpenness(state.smoothedLandmarks);

  // deadband against tiny noise
  const delta = openness - state.opennessSmooth;
  const filteredTarget = Math.abs(delta) < config.opennessDeadband
    ? state.opennessSmooth
    : openness;

  state.opennessRaw = openness;
  state.opennessSmooth = smoothScalar(
    state.opennessSmooth,
    filteredTarget,
    config.opennessAlpha
  );

  state.radiusTarget = lerp(
    config.minRadius,
    config.maxRadius,
    state.opennessSmooth
  );

  ui.tracking.textContent = 'locked';
  ui.openness.textContent = `${state.opennessSmooth.toFixed(2)} (raw ${state.opennessRaw.toFixed(2)})`;
  ui.radius.textContent = `${state.radiusCurrent.toFixed(2)} → ${state.radiusTarget.toFixed(2)}`;
  drawDebug(state.smoothedLandmarks);
}

function animate(now) {
  const dt = Math.min(0.05, Math.max(0.001, (now - state.lastFrameMs) / 1000));
  state.lastFrameMs = now;

  updateFromTracking(now);
  updateRadiusFromHand(dt);
  stepParticles(dt);

  renderer.render(scene, camera);
  requestAnimationFrame(animate);
}

initHandTracking()
  .then(() => {
    ui.radius.textContent = `${state.radiusCurrent.toFixed(2)} → ${state.radiusTarget.toFixed(2)}`;
    requestAnimationFrame(animate);
  })
  .catch((error) => {
    console.error(error);
    ui.status.textContent = 'Tracker failed';
  });
