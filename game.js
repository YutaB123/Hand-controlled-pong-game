// Hand Tennis — 2D Pong controlled by hand position via MediaPipe Hands.
// Left hand drives the left paddle, right hand drives the right paddle.

import { HandLandmarker, FilesetResolver }
  from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

// ---------- DOM ----------
const video       = document.getElementById('video');
const canvas      = document.getElementById('game');
const ctx         = canvas.getContext('2d');
const statusEl    = document.getElementById('status');
const startBtn    = document.getElementById('startBtn');
const leftScoreEl = document.getElementById('leftScore');
const rightScoreEl= document.getElementById('rightScore');

// ---------- Game state ----------
const state = {
  running: false,
  width: 0, height: 0,
  paddleW: 14,
  paddleH: 110,
  // y-positions of paddles (center y, in canvas coords)
  leftY: 0, rightY: 0,
  // smoothed target positions from hand input
  leftTargetY: null, rightTargetY: null,
  ball: { x: 0, y: 0, vx: 0, vy: 0, r: 9 },
  leftScore: 0, rightScore: 0,
  lastTime: performance.now(),
  baseSpeed: 380,           // px/s
  speed: 380,
  serveToRight: true,
};

// ---------- Canvas sizing ----------
function resize() {
  const stage = canvas.parentElement;
  const dpr = window.devicePixelRatio || 1;
  state.width  = stage.clientWidth;
  state.height = stage.clientHeight;
  canvas.width  = state.width  * dpr;
  canvas.height = state.height * dpr;
  canvas.style.width  = state.width  + 'px';
  canvas.style.height = state.height + 'px';
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  // initialise paddle Ys if first run
  if (state.leftTargetY  === null) state.leftTargetY  = state.height / 2;
  if (state.rightTargetY === null) state.rightTargetY = state.height / 2;
  state.leftY  = state.leftTargetY;
  state.rightY = state.rightTargetY;
}
window.addEventListener('resize', resize);

// ---------- Camera ----------
let cameraStarted = false;
async function startCamera() {
  if (cameraStarted) return;
  if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
    const why = location.protocol === 'file:'
      ? 'Camera APIs are blocked on file:// — run this from a local server (e.g. http://localhost:8000) or HTTPS.'
      : 'This page is not in a secure context. Camera APIs require HTTPS or localhost.';
    throw new Error(why);
  }
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: 960, height: 540, facingMode: 'user' },
    audio: false,
  });
  video.srcObject = stream;
  await new Promise(res => { video.onloadedmetadata = () => { video.play(); res(); }; });
  cameraStarted = true;
}

function describeCameraError(err) {
  const name = err?.name || '';
  if (name === 'NotAllowedError' || name === 'SecurityError') {
    return 'Camera permission denied. Click the camera icon in the address bar, allow access, then click Start again.';
  }
  if (name === 'NotFoundError' || name === 'OverconstrainedError') {
    return 'No camera found that matches the requested settings.';
  }
  if (name === 'NotReadableError') {
    return 'Camera is in use by another application. Close other apps using it and click Start again.';
  }
  return err?.message || String(err);
}

// ---------- Hand model ----------
let handLandmarker = null;
async function loadHandModel() {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );
  handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numHands: 2,
    minHandDetectionConfidence: 0.5,
    minHandPresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });
}

// ---------- Hand → paddle mapping ----------
// We don't trust MediaPipe's handedness label (it depends on whether the input
// was mirrored). Instead route by X position in the *mirrored* display frame:
// whichever hand is on the left half of the screen controls the left paddle,
// whichever is on the right half controls the right paddle.
function updatePaddlesFromHands(landmarksList) {
  if (!landmarksList || landmarksList.length === 0) return;

  // Use landmark index 9 — middle-finger MCP, a stable centroid-ish point.
  // Coords are normalised (0..1) in the raw (non-mirrored) camera frame.
  const points = landmarksList.map(lm => ({
    xRaw: lm[9].x,
    y:    lm[9].y,
  }));

  // Mirror X to match the on-screen mirrored video.
  const hands = points.map(p => ({ xMirrored: 1 - p.xRaw, y: p.y }));

  // Split by which side of the screen the hand is on.
  let leftHand = null, rightHand = null;
  for (const h of hands) {
    if (h.xMirrored < 0.5) {
      if (!leftHand || h.xMirrored < leftHand.xMirrored) leftHand = h;
    } else {
      if (!rightHand || h.xMirrored > rightHand.xMirrored) rightHand = h;
    }
  }

  if (leftHand)  state.leftTargetY  = clamp(leftHand.y  * state.height, state.paddleH / 2, state.height - state.paddleH / 2);
  if (rightHand) state.rightTargetY = clamp(rightHand.y * state.height, state.paddleH / 2, state.height - state.paddleH / 2);
}

// ---------- Game logic ----------
function resetBall(toRight) {
  state.ball.x = state.width / 2;
  state.ball.y = state.height / 2;
  state.speed = state.baseSpeed;
  const angle = (Math.random() * 0.6 - 0.3); // -0.3..0.3 rad
  const dir = toRight ? 1 : -1;
  state.ball.vx = Math.cos(angle) * state.speed * dir;
  state.ball.vy = Math.sin(angle) * state.speed;
}

function step(dt) {
  // Smooth paddle Ys toward targets.
  const smooth = 0.25;
  state.leftY  += (state.leftTargetY  - state.leftY)  * smooth;
  state.rightY += (state.rightTargetY - state.rightY) * smooth;

  if (!state.running) return;

  const b = state.ball;
  b.x += b.vx * dt;
  b.y += b.vy * dt;

  // Top/bottom walls
  if (b.y - b.r < 0)              { b.y = b.r;                  b.vy = Math.abs(b.vy); }
  if (b.y + b.r > state.height)   { b.y = state.height - b.r;   b.vy = -Math.abs(b.vy); }

  // Left paddle collision
  const leftPaddleX = 24;
  if (b.x - b.r < leftPaddleX + state.paddleW &&
      b.x - b.r > leftPaddleX - 8 &&
      Math.abs(b.y - state.leftY) < state.paddleH / 2 + b.r &&
      b.vx < 0) {
    bounceOffPaddle(state.leftY, +1);
    b.x = leftPaddleX + state.paddleW + b.r;
  }

  // Right paddle collision
  const rightPaddleX = state.width - 24 - state.paddleW;
  if (b.x + b.r > rightPaddleX &&
      b.x + b.r < rightPaddleX + state.paddleW + 8 &&
      Math.abs(b.y - state.rightY) < state.paddleH / 2 + b.r &&
      b.vx > 0) {
    bounceOffPaddle(state.rightY, -1);
    b.x = rightPaddleX - b.r;
  }

  // Scoring
  if (b.x < -40) {
    state.rightScore++; rightScoreEl.textContent = state.rightScore;
    state.serveToRight = true; resetBall(false);
  } else if (b.x > state.width + 40) {
    state.leftScore++; leftScoreEl.textContent = state.leftScore;
    state.serveToRight = false; resetBall(true);
  }
}

function bounceOffPaddle(paddleY, dirX) {
  const b = state.ball;
  const offset = (b.y - paddleY) / (state.paddleH / 2); // -1..1
  const maxAngle = Math.PI / 3;                          // 60°
  const angle = offset * maxAngle;
  state.speed = Math.min(state.speed * 1.05, 900);
  b.vx = Math.cos(angle) * state.speed * dirX;
  b.vy = Math.sin(angle) * state.speed;
}

// ---------- Rendering ----------
function draw() {
  ctx.clearRect(0, 0, state.width, state.height);

  // Court center dashed line
  ctx.save();
  ctx.strokeStyle = 'rgba(255,255,255,0.18)';
  ctx.lineWidth = 2;
  ctx.setLineDash([10, 14]);
  ctx.beginPath();
  ctx.moveTo(state.width / 2, 0);
  ctx.lineTo(state.width / 2, state.height);
  ctx.stroke();
  ctx.restore();

  // Paddles
  drawPaddle(24, state.leftY,  '#3ddc97');
  drawPaddle(state.width - 24 - state.paddleW, state.rightY, '#ff6b6b');

  // Ball
  ctx.fillStyle = '#fff';
  ctx.shadowColor = 'rgba(255,255,255,0.6)';
  ctx.shadowBlur = 18;
  ctx.beginPath();
  ctx.arc(state.ball.x, state.ball.y, state.ball.r, 0, Math.PI * 2);
  ctx.fill();
  ctx.shadowBlur = 0;
}

function drawPaddle(x, y, color) {
  ctx.fillStyle = color;
  ctx.shadowColor = color;
  ctx.shadowBlur = 20;
  roundRect(ctx, x, y - state.paddleH / 2, state.paddleW, state.paddleH, 6);
  ctx.fill();
  ctx.shadowBlur = 0;
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

// ---------- Loop ----------
let lastVideoTime = -1;
function loop() {
  const now = performance.now();
  const dt = Math.min((now - state.lastTime) / 1000, 0.05);
  state.lastTime = now;

  if (handLandmarker && video.readyState >= 2 && video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    const result = handLandmarker.detectForVideo(video, now);
    updatePaddlesFromHands(result.landmarks);
  }

  step(dt);
  draw();
  requestAnimationFrame(loop);
}

// ---------- Utils ----------
function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

// ---------- Boot ----------
async function boot() {
  resize();
  if (location.protocol === 'file:') {
    statusEl.textContent = 'Open this via a local server (e.g. http://localhost:8000) — file:// blocks the camera.';
  }
  try {
    statusEl.textContent = 'Loading hand model…';
    await loadHandModel();
    statusEl.textContent = 'Click Start to allow camera and play.';
    startBtn.disabled = false;
    resetBall(true);
    requestAnimationFrame(loop);
  } catch (err) {
    console.error(err);
    statusEl.textContent = 'Error loading hand model: ' + (err?.message || err);
  }
}

startBtn.addEventListener('click', async () => {
  if (!cameraStarted) {
    startBtn.disabled = true;
    statusEl.textContent = 'Requesting camera…';
    try {
      await startCamera();
    } catch (err) {
      console.error(err);
      statusEl.textContent = describeCameraError(err);
      startBtn.disabled = false;
      return;
    }
    startBtn.disabled = false;
  }

  if (!state.running) {
    state.running = true;
    state.lastTime = performance.now();
    resetBall(state.serveToRight);
    statusEl.textContent = 'Move both hands in view — left controls left paddle.';
    startBtn.textContent = 'Pause';
  } else {
    state.running = false;
    startBtn.textContent = 'Start Game';
    statusEl.textContent = 'Paused.';
  }
});

startBtn.disabled = true;
boot();
