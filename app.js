// === Real-time Tongue Contour Tracker v2 ===
// Dual-mode: Classic CV + DLC AI

// ─── State ──────────────────────────────────
let stream = null;
let animationId = null;
let frameCount = 0;
let lastFpsTime = performance.now();
let currentFps = 0;
let obsVirtualCamDeviceId = null;  // OBS Virtual Camera device ID
let currentDeviceId = null;        // Currently active device ID
let streamHealthErrors = 0;        // Consecutive stream health failures

// Mode: 'cv' or 'dlc'
let currentMode = 'cv';
let dlcConnected = false;
let dlcLastRequest = 0;
let lastDLCKeypoints = null;

// Processing params
const params = {
  // CV params
  roiY: 30, roiH: 80, blur: 5, threshold: 100,
  edgeLow: 50, edgeHigh: 150, lineWidth: 3,
  showPoints: true, showROI: true, invert: false,
  // DLC params
  dlcInterval: 2,
  showPalateRefs: true,
  showDLCPoints: true,
};

// DOM refs
const sourceVideo = document.getElementById('source-video');
const outputCanvas = document.getElementById('output-canvas');
const ctx = outputCanvas.getContext('2d');
const noSignal = document.getElementById('no-signal');

// ─── Mode Switching ─────────────────────────
function setMode(mode) {
  if (currentMode === mode) return;
  currentMode = mode;

  document.getElementById('btn-mode-cv').classList.toggle('active', mode === 'cv');
  document.getElementById('btn-mode-dlc').classList.toggle('active', mode === 'dlc');

  // Toggle settings panels
  document.getElementById('cv-settings').style.display = mode === 'cv' ? '' : 'none';
  document.getElementById('dlc-settings').style.display = mode === 'dlc' ? '' : 'none';
  document.getElementById('dlc-status').style.display = mode === 'dlc' ? 'flex' : 'none';
  document.getElementById('section-divider').style.display = mode === 'dlc' ? '' : 'none';

  // Toggle legend items
  document.getElementById('legend-roi').style.display = mode === 'cv' ? '' : 'none';
  document.getElementById('legend-palate').style.display = mode === 'dlc' ? '' : 'none';
  document.getElementById('legend-dlc').style.display = mode === 'dlc' ? '' : 'none';

  if (mode === 'dlc') {
    checkDLCServer();
  }

  // Clear canvas
  lastDLCKeypoints = null;
}

// ─── DLC Server Communication ───────────────
function getDLCServerURL() {
  return document.getElementById('dlc-server-url').value.replace(/\/$/, '');
}

async function checkDLCServer() {
  const url = getDLCServerURL();
  const dot = document.getElementById('dlc-status-dot');
  const text = document.getElementById('dlc-status-text');

  dot.className = 'status-dot connecting';
  text.textContent = 'DLC Server: 连接中...';

  try {
    const resp = await fetch(url + '/health', { method: 'GET' });
    const data = await resp.json();
    if (data.status === 'ok') {
      dlcConnected = data.model_loaded || false;
      dot.className = dlcConnected ? 'status-dot connected' : 'status-dot offline';
      text.textContent = dlcConnected
        ? 'DLC Server: 已连接 (模型已加载)'
        : 'DLC Server: 已连接 (模型未加载)';
    } else {
      dlcConnected = false;
      dot.className = 'status-dot offline';
      text.textContent = 'DLC Server: 异常 - ' + JSON.stringify(data);
    }
  } catch (e) {
    dlcConnected = false;
    dot.className = 'status-dot offline';
    text.textContent = 'DLC Server: 未连接 (' + e.message + ')';
  }
}

async function sendFrameToDLC(frameBlob) {
  const url = getDLCServerURL();

  try {
    const t0 = performance.now();
    const resp = await fetch(url + '/infer', {
      method: 'POST',
      body: frameBlob,
      headers: { 'Content-Type': 'image/jpeg' },
    });

    if (!resp.ok) {
      const errData = await resp.json().catch(() => ({}));
      throw new Error(errData.error || 'HTTP ' + resp.status);
    }

    const data = await resp.json();
    const dt = performance.now() - t0;

    // Update status
    document.getElementById('dlc-inference-time').textContent =
      Math.round(dt) + 'ms';

    return data;
  } catch (e) {
    console.warn('DLC inference failed:', e.message);
    document.getElementById('dlc-status-text').textContent =
      'DLC Server: 推理失败 - ' + e.message;
    dlcConnected = false;
    document.getElementById('dlc-status-dot').className = 'status-dot offline';
    return null;
  }
}

// ─── DLC Keypoint Drawing ───────────────────
function drawDLCKeypoints(kps, canvasW, canvasH) {
  if (!kps || !kps.keypoints) return;

  const allKP = kps.keypoints;
  const scaleX = canvasW / (kps.input_size ? kps.input_size[0] : 320);
  const scaleY = canvasH / (kps.input_size ? kps.input_size[1] : 240);

  // Tongue contour (indices 1-10)
  const tonguePoints = allKP.slice(1, 11).filter(
    kp => kp.confidence > 0.1
  );

  if (tonguePoints.length >= 2) {
    // Draw contour line
    ctx.strokeStyle = '#00ff88';
    ctx.lineWidth = params.lineWidth;
    ctx.shadowColor = '#00ff88';
    ctx.shadowBlur = 6;
    ctx.beginPath();
    ctx.moveTo(tonguePoints[0].x * scaleX, tonguePoints[0].y * scaleY);
    for (let i = 1; i < tonguePoints.length; i++) {
      ctx.lineTo(tonguePoints[i].x * scaleX, tonguePoints[i].y * scaleY);
    }
    ctx.stroke();
    ctx.shadowBlur = 0;

    // Draw tongue contour points
    if (params.showPoints) {
      ctx.fillStyle = '#ff6b6b';
      for (const kp of tonguePoints) {
        ctx.beginPath();
        ctx.arc(kp.x * scaleX, kp.y * scaleY, params.lineWidth + 1, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }

  // All DLC keypoints (including vallecula and palate refs)
  if (params.showDLCPoints) {
    for (const kp of allKP) {
      if (kp.confidence < 0.3) continue;

      const x = kp.x * scaleX;
      const y = kp.y * scaleY;

      // Palate reference points (indices 11-18: jinxing...taiyang)
      const isPalate = ['jinxing','muxing','shuixing','huoxing','tuxing','diqiu','yueliang','taiyang'].includes(kp.name);
      const isVallecula = kp.name === 'vallecula';
      const isTongue = kp.name.startsWith('tongue');

      if (isPalate && params.showPalateRefs) {
        ctx.fillStyle = '#ffd700';
        ctx.strokeStyle = '#ffd700';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.arc(x, y, params.lineWidth + 2, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
        // Label
        ctx.fillStyle = '#ffd700';
        ctx.font = '9px Inter, sans-serif';
        ctx.fillText(kp.name, x + 8, y - 4);
      } else if (isTongue) {
        // Already drawn above
      } else if (isVallecula) {
        ctx.fillStyle = '#7b68ee';
        ctx.beginPath();
        ctx.arc(x, y, params.lineWidth + 2, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = '#7b68ee';
        ctx.font = '9px Inter, sans-serif';
        ctx.fillText('vallecula', x + 8, y - 4);
      }
    }
  }
}

// ─── Camera Setup ────────────────────────────
async function enumerateCameras() {
  try {
    // Request permission first (needed for device labels)
    try {
      const tempStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      tempStream.getTracks().forEach(t => t.stop());
    } catch (_) { /* permission may already be granted */ }

    const devices = await navigator.mediaDevices.enumerateDevices();
    const cameras = devices.filter(d => d.kind === 'videoinput');

    // Detect OBS Virtual Camera
    obsVirtualCamDeviceId = null;
    for (const cam of cameras) {
      const label = (cam.label || '').toLowerCase();
      if (label.includes('obs') || label.includes('virtual')) {
        obsVirtualCamDeviceId = cam.deviceId;
        break;
      }
    }

    const select = document.getElementById('camera-select');
    const prevValue = select.value; // preserve current selection if still valid

    select.innerHTML = '<option value="">选择摄像头...</option>';

    // Show OBS Virtual Camera first with special label
    if (obsVirtualCamDeviceId) {
      const obsCam = cameras.find(c => c.deviceId === obsVirtualCamDeviceId);
      if (obsCam) {
        select.innerHTML += `<option value="${obsCam.deviceId}" class="obs-cam-option">
          🎬 ${obsCam.label || 'OBS Virtual Camera'} ← 推荐</option>`;
      }
    }

    cameras.forEach((cam, i) => {
      if (cam.deviceId === obsVirtualCamDeviceId) return; // already added
      select.innerHTML += `<option value="${cam.deviceId}">${cam.label || 'Camera ' + (i+1)}</option>`;
    });

    // Restore previous selection if still valid, otherwise prefer OBS Virtual Cam
    const validIds = cameras.map(c => c.deviceId);
    if (prevValue && validIds.includes(prevValue)) {
      select.value = prevValue;
    } else if (obsVirtualCamDeviceId) {
      select.value = obsVirtualCamDeviceId;
    } else if (cameras.length > 0 && !select.value) {
      select.value = cameras[0].deviceId;
    }

    // Update OBS hint banner
    updateOBSHint(select.value);
  } catch (e) {
    console.warn('Camera enumeration failed:', e);
  }
}

// ─── OBS Compatibility Helpers ───────────────
function updateOBSHint(deviceId) {
  const hint = document.getElementById('obs-hint');
  if (!hint) return;

  if (deviceId === obsVirtualCamDeviceId) {
    // User selected OBS Virtual Camera — ideal setup
    hint.className = 'obs-hint obs-hint-ok';
    hint.innerHTML = '✅ <b>最佳配置</b>：通过OBS虚拟摄像头采集，OBS和Tracker互不干扰。';
  } else if (obsVirtualCamDeviceId && deviceId && deviceId !== obsVirtualCamDeviceId) {
    // OBS VC available but user chose different device — warn about contention
    hint.className = 'obs-hint obs-hint-warn';
    hint.innerHTML = '⚠️ <b>注意</b>：你选择了采集卡直连。如果OBS也使用同一设备，画面会互相抢占。<br>建议切换到 🎬 OBS虚拟摄像头。';
  } else if (!obsVirtualCamDeviceId) {
    // No OBS VC detected
    hint.className = 'obs-hint obs-hint-info';
    hint.innerHTML = '💡 <b>提示</b>：未检测到OBS虚拟摄像头。请在OBS中点击「启动虚拟摄像头」以避免设备争用。';
  }
}

function isUsingOBSVirtualCam() {
  return currentDeviceId === obsVirtualCamDeviceId && obsVirtualCamDeviceId !== null;
}

async function startCapture() {
  const deviceId = document.getElementById('camera-select').value;
  const constraints = {
    video: deviceId ? { deviceId: { exact: deviceId } } : true,
    audio: false
  };

  try {
    // First, stop any existing stream before acquiring new one
    if (stream) {
      stream.getTracks().forEach(t => t.stop());
      stream = null;
    }

    stream = await navigator.mediaDevices.getUserMedia(constraints);
    currentDeviceId = deviceId;
    streamHealthErrors = 0;
    sourceVideo.srcObject = stream;
    sourceVideo.style.display = 'block';
    noSignal.style.display = 'none';

    await sourceVideo.play();

    outputCanvas.width = sourceVideo.videoWidth || 640;
    outputCanvas.height = sourceVideo.videoHeight || 480;

    document.getElementById('btn-start').disabled = true;
    document.getElementById('btn-stop').disabled = false;

    frameCount = 0;
    lastFpsTime = performance.now();
    dlcLastRequest = 0;
    lastDLCKeypoints = null;

    if (currentMode === 'dlc') checkDLCServer();

    processFrame();
    updateOBSHint(deviceId);
  } catch (e) {
    const msg = e.message || String(e);
    if (msg.includes('NotReadableError') || msg.includes('device')) {
      alert('⚠️ 摄像头访问失败: 设备可能被其他程序(如OBS)占用。\n\n' +
            '解决方法:\n' +
            '1. 在OBS中点击「启动虚拟摄像头」\n' +
            '2. 在本页面摄像头列表中选择 🎬 OBS虚拟摄像头\n\n' +
            '错误详情: ' + msg);
    } else {
      alert('摄像头访问失败: ' + msg);
    }
    updateOBSHint(deviceId);
  }
}

function stopCapture() {
  if (animationId) cancelAnimationFrame(animationId);
  if (stream) {
    stream.getTracks().forEach(t => t.stop());
    stream = null;
  }
  sourceVideo.style.display = 'none';
  sourceVideo.srcObject = null;
  noSignal.style.display = 'flex';
  ctx.clearRect(0, 0, outputCanvas.width, outputCanvas.height);

  document.getElementById('btn-start').disabled = false;
  document.getElementById('btn-stop').disabled = true;
  document.getElementById('fps-display').textContent = '0 FPS';

  lastDLCKeypoints = null;
}

async function switchCamera() {
  const newDeviceId = document.getElementById('camera-select').value;
  if (!newDeviceId || newDeviceId === currentDeviceId) return;

  // If no active stream, just update hint — user will click "开始采集" manually
  if (!stream) {
    updateOBSHint(newDeviceId);
    return;
  }

  // Use track replacement instead of full stop/start to avoid
  // releasing the device back to the system (which OBS may grab)
  try {
    const newStream = await navigator.mediaDevices.getUserMedia({
      video: { deviceId: { exact: newDeviceId } },
      audio: false
    });

    const newTrack = newStream.getVideoTracks()[0];
    if (!newTrack) throw new Error('No video track in new stream');

    // Stop old tracks
    stream.getTracks().forEach(t => t.stop());

    // Build new stream with the fresh track
    stream = new MediaStream([newTrack]);
    currentDeviceId = newDeviceId;
    streamHealthErrors = 0;
    sourceVideo.srcObject = stream;
    await sourceVideo.play();

    outputCanvas.width = sourceVideo.videoWidth || 640;
    outputCanvas.height = sourceVideo.videoHeight || 480;

    updateOBSHint(newDeviceId);
  } catch (e) {
    console.error('Camera switch failed:', e);
    // Fall back to full restart
    await startCapture();
  }
}

// ─── Main Processing Loop ───────────────────
async function processFrame() {
  if (!stream) return;

  // Stream health check: verify video track is still live
  const videoTrack = stream.getVideoTracks()[0];
  if (!videoTrack || videoTrack.readyState === 'ended') {
    streamHealthErrors++;
    if (streamHealthErrors <= 3) {
      console.warn('Stream health degraded, attempt recovery', streamHealthErrors);
      // Try to re-acquire the same device
      try {
        const newStream = await navigator.mediaDevices.getUserMedia({
          video: currentDeviceId ? { deviceId: { exact: currentDeviceId } } : true,
          audio: false
        });
        stream.getTracks().forEach(t => t.stop());
        stream = newStream;
        sourceVideo.srcObject = stream;
        await sourceVideo.play();
        streamHealthErrors = 0;
        console.log('Stream recovered');
      } catch (e) {
        console.error('Stream recovery failed:', e);
        if (streamHealthErrors >= 3) {
          stopCapture();
          alert('⚠️ 摄像头信号丢失。请检查OBS虚拟摄像头是否仍在运行，然后重新点击「开始采集」。');
        }
      }
    }
    if (streamHealthErrors >= 3) return;
  } else {
    streamHealthErrors = 0; // healthy
  }

  const w = outputCanvas.width;
  const h = outputCanvas.height;

  // Draw video frame
  ctx.drawImage(sourceVideo, 0, 0, w, h);

  if (currentMode === 'dlc') {
    await processFrameDLC(w, h);
  } else {
    processFrameCV(w, h);
  }

  // FPS
  frameCount++;
  const now = performance.now();
  if (now - lastFpsTime >= 1000) {
    currentFps = Math.round(frameCount / ((now - lastFpsTime) / 1000));
    frameCount = 0;
    lastFpsTime = now;
    document.getElementById('fps-display').textContent = currentFps + ' FPS';
  }

  animationId = requestAnimationFrame(processFrame);
}

// ─── DLC Mode Processing ────────────────────
async function processFrameDLC(w, h) {
  if (!dlcConnected) {
    // Show offline indicator
    ctx.fillStyle = 'rgba(255,107,107,0.15)';
    ctx.fillRect(0, 0, w, h);
    ctx.fillStyle = '#ff6b6b';
    ctx.font = '14px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('DLC Server 未连接 — 检查Windows端服务器', w/2, h/2);
    ctx.textAlign = 'start';
    return;
  }

  const now = performance.now();
  const interval = params.dlcInterval;

  // Send frame every N frames
  if (now - dlcLastRequest > interval * 33) { // ~30fps base, interval * frame time
    dlcLastRequest = now;

    // Capture frame as JPEG blob
    const blob = await new Promise(resolve => {
      outputCanvas.toBlob(resolve, 'image/jpeg', 0.8);
    });

    if (blob) {
      const result = await sendFrameToDLC(blob);
      if (result) {
        lastDLCKeypoints = result;
        dlcConnected = true;
      }
    }
  }

  // Draw last known keypoints
  if (lastDLCKeypoints) {
    drawDLCKeypoints(lastDLCKeypoints, w, h);
  }
}

// ─── CV Image Processing Pipeline ────────────
function processFrameCV(w, h) {
  const frameData = ctx.getImageData(0, 0, w, h);

  // Grayscale
  const gray = rgbToGray(frameData);

  // ROI mask
  const roiY1 = Math.floor(h * params.roiY / 100);
  const roiY2 = Math.floor(h * params.roiH / 100);

  // Box blur
  const blurred = boxBlur(gray, w, h, Math.floor(params.blur / 2));

  // Invert
  if (params.invert) {
    for (let i = 0; i < blurred.length; i++) blurred[i] = 255 - blurred[i];
  }

  // Binary threshold
  const binary = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const idx = y * w + x;
      if (y >= roiY1 && y <= roiY2) {
        binary[idx] = blurred[idx] > params.threshold ? 255 : 0;
      } else {
        binary[idx] = 0;
      }
    }
  }

  // Edge detection
  const edges = simpleEdgeDetect(binary, w, h, roiY1, roiY2, params.edgeLow, params.edgeHigh);

  // Find contour
  const contour = findLongestContour(edges, w, h, roiY1, roiY2);

  // Display
  ctx.putImageData(frameData, 0, 0);

  // ROI rect
  if (params.showROI) {
    ctx.strokeStyle = 'rgba(255,255,255,0.3)';
    ctx.lineWidth = 1;
    ctx.setLineDash([5, 5]);
    ctx.strokeRect(0, roiY1, w, roiY2 - roiY1);
    ctx.setLineDash([]);
  }

  // Contour
  if (contour.length > 0) {
    ctx.strokeStyle = '#00ff88';
    ctx.lineWidth = params.lineWidth;
    ctx.shadowColor = '#00ff88';
    ctx.shadowBlur = 4;
    ctx.beginPath();
    ctx.moveTo(contour[0][0], contour[0][1]);
    for (let i = 1; i < contour.length; i++) {
      ctx.lineTo(contour[i][0], contour[i][1]);
    }
    ctx.stroke();
    ctx.shadowBlur = 0;

    if (params.showPoints) {
      ctx.fillStyle = '#ff6b6b';
      for (let i = 0; i < contour.length; i += Math.max(1, Math.floor(contour.length / 30))) {
        ctx.beginPath();
        ctx.arc(contour[i][0], contour[i][1], params.lineWidth, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }
}

// ─── Image Processing Functions ──────────────
function rgbToGray(imageData) {
  const gray = new Uint8Array(imageData.width * imageData.height);
  const data = imageData.data;
  for (let i = 0; i < gray.length; i++) {
    const offset = i * 4;
    gray[i] = Math.round(data[offset] * 0.299 + data[offset + 1] * 0.587 + data[offset + 2] * 0.114);
  }
  return gray;
}

function boxBlur(src, w, h, radius) {
  if (radius <= 0) return new Uint8Array(src);
  const dst = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    let sum = 0;
    const rowStart = y * w;
    for (let x = 0; x < w; x++) {
      sum += src[rowStart + x];
      if (x >= radius * 2 + 1) sum -= src[rowStart + x - radius * 2 - 1];
      if (x >= radius) dst[rowStart + x - radius] = Math.round(sum / (Math.min(x + 1, radius * 2 + 1, w - x + radius)));
    }
  }
  const result = new Uint8Array(w * h);
  const temp = dst;
  for (let x = 0; x < w; x++) {
    let sum = 0;
    for (let y = 0; y < h; y++) {
      sum += temp[y * w + x];
      if (y >= radius * 2 + 1) sum -= temp[(y - radius * 2 - 1) * w + x];
      if (y >= radius) result[(y - radius) * w + x] = Math.round(sum / (Math.min(y + 1, radius * 2 + 1, h - y + radius)));
    }
  }
  return result;
}

function simpleEdgeDetect(binary, w, h, roiY1, roiY2, lowThresh, highThresh) {
  const edges = new Uint8Array(w * h);
  for (let y = roiY1 + 1; y < roiY2 - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const idx = y * w + x;
      const dy = binary[(y+1)*w+x] - binary[(y-1)*w+x];
      const dx = binary[y*w+x+1] - binary[y*w+x-1];
      const mag = Math.abs(dy) + Math.abs(dx);
      if (mag > lowThresh) {
        edges[idx] = mag > highThresh ? 255 : 128;
      }
    }
  }
  return edges;
}

function findLongestContour(edges, w, h, roiY1, roiY2) {
  const points = [];
  for (let x = 0; x < w; x++) {
    let bestY = -1;
    let bestVal = 0;
    for (let y = roiY1; y < roiY2; y++) {
      const val = edges[y * w + x];
      if (val > bestVal) {
        bestVal = val;
        bestY = y;
      }
    }
    if (bestY >= 0) {
      points.push([x, bestY]);
    }
  }
  return smoothContour(points, 5);
}

function smoothContour(points, windowSize) {
  if (points.length < windowSize) return points;
  const smoothed = [];
  const half = Math.floor(windowSize / 2);
  for (let i = 0; i < points.length; i++) {
    let sumY = 0;
    let count = 0;
    for (let j = Math.max(0, i - half); j <= Math.min(points.length - 1, i + half); j++) {
      sumY += points[j][1];
      count++;
    }
    smoothed.push([points[i][0], Math.round(sumY / count)]);
  }
  return smoothed;
}

// ─── Parameter Updates ───────────────────────
function updateParam(name, value) {
  params[name] = parseInt(value);
  const valMap = {
    roiY: 'roi-y-val', roiH: 'roi-h-val', blur: 'blur-val',
    threshold: 'threshold-val', edgeLow: 'edge-low-val',
    edgeHigh: 'edge-high-val', lineWidth: 'line-width-val',
    dlcInterval: 'dlc-interval-val',
  };
  if (valMap[name]) {
    const suffix = ['roiY','roiH'].includes(name) ? '%' : '';
    document.getElementById(valMap[name]).textContent = value + suffix;
  }
}

function updateCheckbox(name, checked) {
  params[name] = checked;
}

function resetParams() {
  const defaults = {
    roiY: 30, roiH: 80, blur: 5, threshold: 100,
    edgeLow: 50, edgeHigh: 150, lineWidth: 3,
    showPoints: true, showROI: true, invert: false,
    dlcInterval: 2, showPalateRefs: true, showDLCPoints: true,
  };
  Object.assign(params, defaults);

  document.getElementById('roi-y').value = 30; document.getElementById('roi-y-val').textContent = '30%';
  document.getElementById('roi-h').value = 80; document.getElementById('roi-h-val').textContent = '80%';
  document.getElementById('blur').value = 5; document.getElementById('blur-val').textContent = '5';
  document.getElementById('threshold').value = 100; document.getElementById('threshold-val').textContent = '100';
  document.getElementById('edge-low').value = 50; document.getElementById('edge-low-val').textContent = '50';
  document.getElementById('edge-high').value = 150; document.getElementById('edge-high-val').textContent = '150';
  document.getElementById('line-width').value = 3; document.getElementById('line-width-val').textContent = '3';
  document.getElementById('show-points').checked = true;
  document.getElementById('show-roi').checked = true;
  document.getElementById('invert').checked = false;
  document.getElementById('dlc-interval').value = 2; document.getElementById('dlc-interval-val').textContent = '2';
  document.getElementById('show-palate-refs').checked = true;
  document.getElementById('show-dlc-points').checked = true;
}

// ─── Screenshot ──────────────────────────────
function takeScreenshot() {
  const dataUrl = outputCanvas.toDataURL('image/png');
  const gallery = document.getElementById('gallery');
  const grid = document.getElementById('gallery-grid');

  gallery.style.display = 'block';
  const img = document.createElement('img');
  img.src = dataUrl;
  img.onclick = () => {
    const win = window.open();
    win.document.write(`<img src="${dataUrl}" style="max-width:100%;">`);
  };
  grid.appendChild(img);

  const a = document.createElement('a');
  a.href = dataUrl;
  a.download = `tongue-contour-${new Date().toISOString().replace(/[:.]/g, '-')}.png`;
  a.click();
}

function clearGallery() {
  document.getElementById('gallery-grid').innerHTML = '';
  document.getElementById('gallery').style.display = 'none';
}

// ─── Init ────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  enumerateCameras();
  outputCanvas.width = 640;
  outputCanvas.height = 480;
  setMode('cv'); // Default to CV mode

  // Re-detect cameras when devices change (e.g. OBS Virtual Camera started/stopped)
  navigator.mediaDevices.addEventListener('devicechange', () => {
    console.log('Device change detected, re-enumerating cameras...');
    enumerateCameras();
  });
});
