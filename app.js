import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.182.0/build/three.module.js';
import { FilesetResolver, HandLandmarker } from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3';

const MODEL_ASSET = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';
const WASM_ROOT = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm';

const ui = {
  status: document.getElementById('status'),
  gesture: document.getElementById('metric-gesture'),
  pinch: document.getElementById('metric-pinch'),
  pinchStrength: document.getElementById('metric-pinch-strength'),
  shape: document.getElementById('metric-shape'),
  mode: document.getElementById('metric-mode'),
  btnCamera: document.getElementById('btn-camera'),
  btnPreview: document.getElementById('btn-preview'),
  btnShape: document.getElementById('btn-shape'),
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
  lastGesture: 'none',
  lastFistTrigger: 0,
  grabbing: false,
  pinchActive: false,
  pinchStrength: 0,
  burst: 0,
  swirl: 0,
  attractStrength: 0,
  morphSpring: 7.5,
  follow: {
    pos: new THREE.Vector3(),
    target: new THREE.Vector3(),
  },
  grabOffset: new THREE.Vector3(),
  maxFollowSpeed: 5.8,
  shapeIndex: 0,
};

const detector = {
  minDetection: 0.55,
  minPresence: 0.55,
  minTracking: 0.55,
};

const config = {
  particles: 12000,
  normalFollowK: 6.0,
  pinchFollowK: 15.0,
  particleDamping: 3.2,
  burstStrength: 2.2,
  swirlStrength: 9.0,
};

const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [0, 9], [9, 10], [10, 11], [11, 12],
  [0, 13], [13, 14], [14, 15], [15, 16],
  [0, 17], [17, 18], [18, 19], [19, 20],
  [5, 9], [9, 13], [13, 17],
];

const stage = document.getElementById('stage');
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
stage.appendChild(renderer.domElement);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(55, window.innerWidth / window.innerHeight, 0.1, 100);
camera.position.set(0, 0, 10);

const particleGroup = new THREE.Group();
scene.add(particleGroup);

const PARTICLE_COUNT = config.particles;
const positions = new Float32Array(PARTICLE_COUNT * 3);
const velocities = new Float32Array(PARTICLE_COUNT * 3);
const targets = new Float32Array(PARTICLE_COUNT * 3);
const burstVectors = new Float32Array(PARTICLE_COUNT * 3);

for (let i = 0; i < PARTICLE_COUNT; i += 1) {
  const k = i * 3;
  positions[k] = (Math.random() - 0.5) * 5;
  positions[k + 1] = (Math.random() - 0.5) * 4;
  positions[k + 2] = (Math.random() - 0.5) * 4;

  const vx = Math.random() * 2 - 1;
  const vy = Math.random() * 2 - 1;
  const vz = Math.random() * 2 - 1;
  const mag = Math.hypot(vx, vy, vz) || 1;
  burstVectors[k] = vx / mag;
  burstVectors[k + 1] = vy / mag;
  burstVectors[k + 2] = vz / mag;
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

function fillSphere(out, radius = 2.8) {
  for (let i = 0; i < PARTICLE_COUNT; i += 1) {
    const u = Math.random();
    const v = Math.random();
    const theta = 2 * Math.PI * u;
    const phi = Math.acos(2 * v - 1);
    const k = i * 3;
    out[k] = radius * Math.sin(phi) * Math.cos(theta);
    out[k + 1] = radius * Math.sin(phi) * Math.sin(theta);
    out[k + 2] = radius * Math.cos(phi);
  }
}

function fillTorus(out, major = 2.25, minor = 0.8) {
  for (let i = 0; i < PARTICLE_COUNT; i += 1) {
    const u = Math.random() * Math.PI * 2;
    const v = Math.random() * Math.PI * 2;
    const k = i * 3;
    out[k] = (major + minor * Math.cos(v)) * Math.cos(u);
    out[k + 1] = (major + minor * Math.cos(v)) * Math.sin(u);
    out[k + 2] = minor * Math.sin(v);
  }
}

function fillHelix(out, radius = 2.0, height = 5.2, turns = 7) {
  const total = turns * Math.PI * 2;
  for (let i = 0; i < PARTICLE_COUNT; i += 1) {
    const t = (i / Math.max(1, PARTICLE_COUNT - 1)) * total;
    const k = i * 3;
    out[k] = radius * Math.cos(t);
    out[k + 1] = radius * Math.sin(t);
    out[k + 2] = (t / total - 0.5) * height;
  }
}

function fillPlane(out, width = 6, height = 4) {
  for (let i = 0; i < PARTICLE_COUNT; i += 1) {
    const k = i * 3;
    out[k] = (Math.random() - 0.5) * width;
    out[k + 1] = (Math.random() - 0.5) * height;
    out[k + 2] = (Math.random() - 0.5) * 0.25;
  }
}

const shapeBuffers = {
  SPHERE: new Float32Array(PARTICLE_COUNT * 3),
  TORUS: new Float32Array(PARTICLE_COUNT * 3),
  HELIX: new Float32Array(PARTICLE_COUNT * 3),
  PLANE: new Float32Array(PARTICLE_COUNT * 3),
};

fillSphere(shapeBuffers.SPHERE);
fillTorus(shapeBuffers.TORUS);
fillHelix(shapeBuffers.HELIX);
fillPlane(shapeBuffers.PLANE);

const shapeNames = Object.keys(shapeBuffers);
function setShapeByIndex(index) {
  state.shapeIndex = (index + shapeNames.length) % shapeNames.length;
  const shapeName = shapeNames[state.shapeIndex];
  targets.set(shapeBuffers[shapeName]);
  ui.shape.textContent = shapeName;
}
function nextShape() {
  setShapeByIndex(state.shapeIndex + 1);
}
setShapeByIndex(0);
ui.btnShape.addEventListener('click', nextShape);

function dist2(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function dist3(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);
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

function screenToWorld(nx, ny, zPlane = 0) {
  const x = (1 - nx) * 2 - 1;
  const y = -(ny * 2 - 1);
  const projected = new THREE.Vector3(x, y, 0.5).unproject(camera);
  const direction = projected.sub(camera.position).normalize();
  const t = (zPlane - camera.position.z) / (direction.z || 1e-6);
  return camera.position.clone().add(direction.multiplyScalar(t));
}

function fingerExtended(landmarks, tip, pip, mcp) {
  const wrist = landmarks[0];
  const tipToWrist = dist3(landmarks[tip], wrist);
  const pipToWrist = dist3(landmarks[pip], wrist);
  const mcpToWrist = dist3(landmarks[mcp], wrist);
  return tipToWrist > pipToWrist && pipToWrist > mcpToWrist * 0.92;
}

function classifyGesture(landmarks, pinch) {
  const index = fingerExtended(landmarks, 8, 6, 5);
  const middle = fingerExtended(landmarks, 12, 10, 9);
  const ring = fingerExtended(landmarks, 16, 14, 13);
  const pinky = fingerExtended(landmarks, 20, 18, 17);
  const thumbSpread = dist2(landmarks[4], landmarks[5]);
  const palmWidth = dist2(landmarks[5], landmarks[17]) + 1e-6;
  const thumbExtended = thumbSpread / palmWidth > 0.45;

  if (pinch.active) return 'pinch';
  if (index && middle && !ring && !pinky) return 'victory';
  if (index && !middle && !ring && !pinky) return 'point';
  if (index && middle && ring && pinky && thumbExtended) return 'open_palm';
  if (!index && !middle && !ring && !pinky) return 'fist';
  return 'hand';
}

function updatePinch(landmarks) {
  const palmWidth = dist2(landmarks[5], landmarks[17]) + 1e-6;
  const normalized = dist2(landmarks[4], landmarks[8]) / palmWidth;
  const pinchOn = 0.34;
  const pinchOff = 0.43;

  if (!state.pinchActive && normalized < pinchOn) state.pinchActive = true;
  if (state.pinchActive && normalized > pinchOff) state.pinchActive = false;

  let strength = 0;
  if (state.pinchActive) {
    const t = (normalized - pinchOn) / (pinchOff - pinchOn);
    strength = 1 - Math.min(1, Math.max(0, t));
  }

  state.pinchStrength = strength;
  return {
    active: state.pinchActive,
    strength,
    normalized,
  };
}

function handleGestureActions(gesture, now) {
  if (gesture === 'fist' && state.lastGesture !== 'fist' && now - state.lastFistTrigger > 650) {
    nextShape();
    state.lastFistTrigger = now;
  }
  if (gesture === 'open_palm' && state.lastGesture !== 'open_palm') {
    state.burst = 1;
  }
  if (gesture === 'victory') {
    state.swirl = Math.min(1, state.swirl + 0.12);
  }
  if (gesture === 'point') {
    state.attractStrength = 1;
  } else {
    state.attractStrength *= 0.92;
  }
  state.lastGesture = gesture;
}

function updateFollowTarget(handWorld, pinching) {
  if (pinching && !state.grabbing) {
    state.grabbing = true;
    state.grabOffset.copy(particleGroup.position).sub(handWorld);
  } else if (!pinching && state.grabbing) {
    state.grabbing = false;
  }

  if (state.grabbing) {
    state.follow.target.copy(handWorld).add(state.grabOffset);
  } else {
    state.follow.target.copy(handWorld);
  }
}

function followStep(dt, pinching) {
  const k = pinching ? config.pinchFollowK : config.normalFollowK;
  const alpha = 1 - Math.exp(-k * dt);
  const next = state.follow.pos.clone().lerp(state.follow.target, alpha);
  const delta = next.sub(state.follow.pos);
  const maxStep = state.maxFollowSpeed * dt;
  if (delta.length() > maxStep) delta.setLength(maxStep);
  state.follow.pos.add(delta);
  particleGroup.position.copy(state.follow.pos);
}

function particleStep(dt) {
  state.burst *= Math.pow(0.025, dt);
  state.swirl *= Math.pow(0.08, dt);

  const burstForce = config.burstStrength * state.burst;
  const swirlForce = config.swirlStrength * state.swirl;
  const attractForce = 14 * state.attractStrength;
  const attractX = state.follow.target.x;
  const attractY = state.follow.target.y;
  const attractZ = state.follow.target.z;

  for (let i = 0; i < PARTICLE_COUNT; i += 1) {
    const k = i * 3;

    let px = positions[k];
    let py = positions[k + 1];
    let pz = positions[k + 2];
    let vx = velocities[k];
    let vy = velocities[k + 1];
    let vz = velocities[k + 2];

    const tx = targets[k];
    const ty = targets[k + 1];
    const tz = targets[k + 2];
    vx += (tx - px) * state.morphSpring * dt;
    vy += (ty - py) * state.morphSpring * dt;
    vz += (tz - pz) * state.morphSpring * dt;

    if (swirlForce > 1e-4) {
      vx += (-py) * swirlForce * dt * 0.08;
      vy += px * swirlForce * dt * 0.08;
    }

    if (burstForce > 1e-4) {
      vx += burstVectors[k] * burstForce * dt;
      vy += burstVectors[k + 1] * burstForce * dt;
      vz += burstVectors[k + 2] * burstForce * dt;
    }

    if (attractForce > 1e-4) {
      const dx = attractX - px;
      const dy = attractY - py;
      const dz = attractZ - pz;
      const d = Math.hypot(dx, dy, dz) + 0.25;
      const s = (attractForce / d) * dt;
      vx += dx * s;
      vy += dy * s;
      vz += dz * s;
    }

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
      minHandDetectionConfidence: detector.minDetection,
      minHandPresenceConfidence: detector.minPresence,
      minTrackingConfidence: detector.minTracking,
    });
  } catch (error) {
    console.warn('GPU delegate unavailable, falling back to CPU.', error);
    state.handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: MODEL_ASSET,
        delegate: 'CPU',
      },
      runningMode: 'VIDEO',
      numHands: 1,
      minHandDetectionConfidence: detector.minDetection,
      minHandPresenceConfidence: detector.minPresence,
      minTrackingConfidence: detector.minTracking,
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

window.addEventListener('resize', () => {
  renderer.setSize(window.innerWidth, window.innerHeight);
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
});

function updateFromTracking(now) {
  let landmarks = null;
  if (
    state.cameraReady
    && state.handLandmarker
    && ui.video.readyState >= 2
    && ui.video.currentTime !== state.lastVideoTime
  ) {
    state.lastVideoTime = ui.video.currentTime;
    const result = state.handLandmarker.detectForVideo(ui.video, now);
    landmarks = result?.landmarks?.[0] || null;
  }

  if (!landmarks) {
    ui.gesture.textContent = 'none';
    ui.pinch.textContent = 'false';
    ui.pinchStrength.textContent = '0.00';
    clearOverlay();
    state.attractStrength *= 0.92;
    return;
  }

  const pinch = updatePinch(landmarks);
  const gesture = classifyGesture(landmarks, pinch);
  handleGestureActions(gesture, now);

  const controlPoint = pinch.active
    ? { x: (landmarks[4].x + landmarks[8].x) / 2, y: (landmarks[4].y + landmarks[8].y) / 2 }
    : palmCenter(landmarks);

  const handWorld = screenToWorld(controlPoint.x, controlPoint.y, 0);
  updateFollowTarget(handWorld, pinch.active);
  drawDebug(landmarks);

  ui.gesture.textContent = gesture;
  ui.pinch.textContent = `${pinch.active} (${pinch.normalized.toFixed(2)})`;
  ui.pinchStrength.textContent = pinch.strength.toFixed(2);
}

function animate(now) {
  const dt = Math.min(0.05, Math.max(0.001, (now - state.lastFrameMs) / 1000));
  state.lastFrameMs = now;

  updateFromTracking(now);
  followStep(dt, state.pinchActive);
  particleStep(dt);

  if (!state.grabbing) {
    particleGroup.rotation.y += dt * 0.12;
  }

  renderer.render(scene, camera);
  requestAnimationFrame(animate);
}

initHandTracking()
  .then(() => {
    requestAnimationFrame(animate);
  })
  .catch((error) => {
    console.error(error);
    ui.status.textContent = 'Tracker failed';
  });
