// === Real-time Tongue Contour Tracker ===

// State
let stream = null;
let animationId = null;
let frameCount = 0;
let lastFpsTime = performance.now();
let currentFps = 0;

// Processing params
const params = {
  roiY: 30,      // ROI start Y (percentage)
  roiH: 80,      // ROI end Y (percentage)
  blur: 5,       // Gaussian blur kernel size (must be odd)
  threshold: 100, // Binary threshold
  edgeLow: 50,   // Canny low threshold
  edgeHigh: 150, // Canny high threshold
  lineWidth: 3,  // Contour line width
  showPoints: true,
  showROI: true,
  invert: false,
};

// DOM refs
const sourceVideo = document.getElementById('source-video');
const outputCanvas = document.getElementById('output-canvas');
const ctx = outputCanvas.getContext('2d');
const noSignal = document.getElementById('no-signal');

// ====== Camera Setup ======
async function enumerateCameras() {
  try {
    // Need to get permission first
    const devices = await navigator.mediaDevices.enumerateDevices();
    const cameras = devices.filter(d => d.kind === 'videoinput');
    const select = document.getElementById('camera-select');
    select.innerHTML = '<option value="">选择摄像头...</option>';
    cameras.forEach((cam, i) => {
      select.innerHTML += `<option value="${cam.deviceId}">${cam.label || 'Camera ' + (i+1)}</option>`;
    });
    if (cameras.length > 0 && !select.value) {
      select.value = cameras[0].deviceId;
    }
  } catch (e) {
    console.warn('Camera enumeration failed:', e);
  }
}

async function startCapture() {
  const deviceId = document.getElementById('camera-select').value;
  const constraints = {
    video: deviceId ? { deviceId: { exact: deviceId } } : true,
    audio: false
  };
  
  try {
    stream = await navigator.mediaDevices.getUserMedia(constraints);
    sourceVideo.srcObject = stream;
    sourceVideo.style.display = 'block';
    noSignal.style.display = 'none';
    
    await sourceVideo.play();
    
    // Set canvas size to match video
    outputCanvas.width = sourceVideo.videoWidth || 640;
    outputCanvas.height = sourceVideo.videoHeight || 480;
    
    document.getElementById('btn-start').disabled = true;
    document.getElementById('btn-stop').disabled = false;
    
    // Start processing loop
    frameCount = 0;
    lastFpsTime = performance.now();
    processFrame();
    
    // Enumerate cameras after permission
    enumerateCameras();
  } catch (e) {
    alert('摄像头访问失败: ' + e.message);
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
}

async function switchCamera() {
  if (stream) {
    stopCapture();
    await startCapture();
  }
}

// ====== Image Processing Pipeline ======
function processFrame() {
  if (!stream) return;
  
  const w = outputCanvas.width;
  const h = outputCanvas.height;
  
  // Step 1: Draw video frame to canvas
  ctx.drawImage(sourceVideo, 0, 0, w, h);
  const frameData = ctx.getImageData(0, 0, w, h);
  
  // Step 2: Convert to grayscale
  const gray = rgbToGray(frameData);
  
  // Step 3: Apply ROI mask
  const roiY1 = Math.floor(h * params.roiY / 100);
  const roiY2 = Math.floor(h * params.roiH / 100);
  
  // Step 4: Gaussian blur (simplified box blur for performance)
  const blurred = boxBlur(gray, w, h, Math.floor(params.blur / 2));
  
  // Step 5: Invert if needed (for dark ultrasound images)
  if (params.invert) {
    for (let i = 0; i < blurred.length; i++) blurred[i] = 255 - blurred[i];
  }
  
  // Step 6: Binary threshold within ROI
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
  
  // Step 7: Edge detection (simplified Canny-like)
  const edges = simpleEdgeDetect(binary, w, h, roiY1, roiY2, params.edgeLow, params.edgeHigh);
  
  // Step 8: Find longest contour (tongue surface)
  const contour = findLongestContour(edges, w, h, roiY1, roiY2);
  
  // Step 9: Display
  // Put original frame back
  ctx.putImageData(frameData, 0, 0);
  
  // Draw ROI rectangle
  if (params.showROI) {
    ctx.strokeStyle = 'rgba(255,255,255,0.3)';
    ctx.lineWidth = 1;
    ctx.setLineDash([5, 5]);
    ctx.strokeRect(0, roiY1, w, roiY2 - roiY1);
    ctx.setLineDash([]);
  }
  
  // Draw contour
  if (contour.length > 0) {
    // Draw line
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
    
    // Draw points
    if (params.showPoints) {
      ctx.fillStyle = '#ff6b6b';
      for (let i = 0; i < contour.length; i += Math.max(1, Math.floor(contour.length / 30))) {
        ctx.beginPath();
        ctx.arc(contour[i][0], contour[i][1], params.lineWidth, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }
  
  // FPS calculation
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

// ====== Image Processing Functions ======
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
  // Horizontal pass
  for (let y = 0; y < h; y++) {
    let sum = 0;
    const rowStart = y * w;
    for (let x = 0; x < w; x++) {
      sum += src[rowStart + x];
      if (x >= radius * 2 + 1) sum -= src[rowStart + x - radius * 2 - 1];
      if (x >= radius) dst[rowStart + x - radius] = Math.round(sum / (Math.min(x + 1, radius * 2 + 1, w - x + radius)));
    }
  }
  // Vertical pass (using dst as temp, write back to src behavior)
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
  // Sobel-like gradient in Y direction (since tongue contour is mostly horizontal)
  for (let y = roiY1 + 1; y < roiY2 - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const idx = y * w + x;
      const dy = binary[(y+1)*w+x] - binary[(y-1)*w+x];
      const dx = binary[y*w+x+1] - binary[y*w+x-1];
      const mag = Math.abs(dy) + Math.abs(dx);
      if (mag > lowThresh) {
        edges[idx] = mag > highThresh ? 255 : 128; // Strong vs weak
      }
    }
  }
  return edges;
}

function findLongestContour(edges, w, h, roiY1, roiY2) {
  // For tongue ultrasound, the contour is a bright line across the image
  // Find the brightest pixel in each column within ROI
  const points = [];
  
  for (let x = 0; x < w; x++) {
    let bestY = -1;
    let bestVal = 0;
    
    // Search for strongest edge in this column
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
  
  // Smooth the contour with moving average
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

// ====== Parameter Updates ======
function updateParam(name, value) {
  params[name] = parseInt(value);
  if (name === 'roiY') document.getElementById('roi-y-val').textContent = value + '%';
  if (name === 'roiH') document.getElementById('roi-h-val').textContent = value + '%';
  if (name === 'blur') document.getElementById('blur-val').textContent = value;
  if (name === 'threshold') document.getElementById('threshold-val').textContent = value;
  if (name === 'edgeLow') document.getElementById('edge-low-val').textContent = value;
  if (name === 'edgeHigh') document.getElementById('edge-high-val').textContent = value;
  if (name === 'lineWidth') document.getElementById('line-width-val').textContent = value;
}

function updateCheckbox(name, checked) {
  params[name] = checked;
}

function resetParams() {
  const defaults = { roiY: 30, roiH: 80, blur: 5, threshold: 100, edgeLow: 50, edgeHigh: 150, lineWidth: 3, showPoints: true, showROI: true, invert: false };
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
}

// ====== Screenshot ======
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
  
  // Also download
  const a = document.createElement('a');
  a.href = dataUrl;
  a.download = `tongue-contour-${new Date().toISOString().replace(/[:.]/g, '-')}.png`;
  a.click();
}

function clearGallery() {
  document.getElementById('gallery-grid').innerHTML = '';
  document.getElementById('gallery').style.display = 'none';
}

// ====== Init ======
document.addEventListener('DOMContentLoaded', () => {
  enumerateCameras();
  
  // Initial canvas size
  outputCanvas.width = 640;
  outputCanvas.height = 480;
});
