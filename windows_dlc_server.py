"""
DLC Inference Server — runs on Windows (miniconda3 + DLC).
Accepts frames via HTTP, runs DLC inference, returns keypoints.

Usage (on Windows):
    python windows_dlc_server.py

Then access from browser/WSL at http://<windows-ip>:8766
"""

import json
import base64
import io
import sys
import os
import time
import logging
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

import numpy as np
from PIL import Image

# ─── Config ─────────────────────────────────
CONFIG_PATH = r"C:\DLC\palate-sunjing-2024-08-27\config.yaml"
PORT = 8766
INPUT_W, INPUT_H = 320, 240

BODY_PARTS = [
    "vallecula",
    "tongueRoot1", "tongueRoot2", "tongueBody1", "tongueBody2",
    "tongueDorsum1", "tongueDorsum2", "tongueBlade1", "tongueBlade2",
    "tongueTip1", "tongueTip2",
    "jinxing", "muxing", "shuixing", "huoxing", "tuxing",
    "diqiu", "yueliang", "taiyang",
]
TONGUE_INDICES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
NUM_JOINTS = 19
LOCREF_STDEV = 7.2801

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger('dlc-server')

# ─── Model (loaded once) ────────────────────
sess = None
inputs_tensor = None
outputs_tensor = None
dlc_cfg = None


def load_model():
    """Load DLC model using the DLC prediction pipeline."""
    global sess, inputs_tensor, outputs_tensor, dlc_cfg

    log.info(f"Loading DLC config: {CONFIG_PATH}")
    from deeplabcut.pose_estimation_tensorflow.config import load_config
    dlc_cfg = load_config(CONFIG_PATH)

    log.info("Setting up pose prediction...")
    from deeplabcut.pose_estimation_tensorflow.core import predict
    sess, inputs_tensor, outputs_tensor = predict.setup_pose_prediction(dlc_cfg)

    log.info(f"Model loaded! Input: {inputs_tensor}, Output: {outputs_tensor}")

    # Warmup inference
    log.info("Running warmup inference...")
    dummy = np.random.randint(0, 255, (INPUT_H, INPUT_W, 3), dtype=np.uint8)
    t0 = time.time()
    _ = run_inference(dummy)
    dt = time.time() - t0
    log.info(f"Warmup done in {dt:.1f}s")

    # Benchmark
    times = []
    for _ in range(5):
        t1 = time.time()
        _ = run_inference(dummy)
        times.append(time.time() - t1)
    avg = sum(times[1:]) / (len(times) - 1)
    log.info(f"Steady-state: {avg*1000:.0f}ms/frame (~{1/avg:.0f} FPS)")


def preprocess_frame(pil_image):
    """Preprocess PIL Image → model input array."""
    img = pil_image.resize((INPUT_W, INPUT_H), Image.LANCZOS)
    img = img.convert('RGB')
    arr = np.array(img, dtype=np.float32)
    return arr


def run_inference(frame_array):
    """Run DLC inference on a single frame. Returns keypoints list."""
    global sess, inputs_tensor, outputs_tensor

    if sess is None:
        raise RuntimeError("Model not loaded")

    # DLC expects NHWC float32 input (1, H, W, 3) or batch
    inp = frame_array.reshape(1, INPUT_H, INPUT_W, 3)

    # Run inference
    # outputs_tensor from DLC is typically a list of tensors or a single tensor
    if isinstance(outputs_tensor, list):
        fetches = outputs_tensor
    else:
        fetches = [outputs_tensor]

    results = sess.run(fetches, feed_dict={inputs_tensor: inp})

    # Parse results
    # DLC returns pose in format: scmap, locref, ...
    # The main result is typically the first tensor: (1, H', W', N_joints*3)
    # or separate tensors for scmap + locref
    result = results[0]

    # result shape: (1, H_out, W_out, C) where C = NUM_JOINTS * 3 (scmap + locref_x + locref_y)
    # OR just (1, H_out, W_out, NUM_JOINTS) if no locref
    h_out, w_out = result.shape[1], result.shape[2]
    n_channels = result.shape[3]

    keypoints = []
    num_joints_effective = min(NUM_JOINTS, n_channels)

    for j in range(num_joints_effective):
        smap = result[0, :, :, j]
        max_idx = np.argmax(smap)
        max_y, max_x = divmod(int(max_idx), w_out)
        confidence = float(smap[max_y, max_x])

        # Locref if available
        if n_channels >= NUM_JOINTS * 3:
            dx = result[0, max_y, max_x, j + NUM_JOINTS]
            dy = result[0, max_y, max_x, j + NUM_JOINTS * 2]
        elif n_channels >= NUM_JOINTS * 2:
            dx = result[0, max_y, max_x, j + NUM_JOINTS]
            dy = 0
        else:
            dx = dy = 0

        refined_x = (max_x + dx * LOCREF_STDEV) * (INPUT_W / w_out)
        refined_y = (max_y + dy * LOCREF_STDEV) * (INPUT_H / h_out)

        keypoints.append({
            "name": BODY_PARTS[j],
            "x": round(float(refined_x), 2),
            "y": round(float(refined_y), 2),
            "confidence": round(confidence, 4),
        })

    return keypoints


# ─── HTTP Server ────────────────────────────
class DLCRequestHandler(BaseHTTPRequestHandler):
    """Handle HTTP requests for DLC inference."""

    def do_OPTIONS(self):
        """CORS preflight."""
        self.send_response(200)
        self._cors_headers()
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            self.send_response(200)
            self._cors_headers()
            self.end_headers()
            resp = json.dumps({"status": "ok", "model_loaded": sess is not None})
            self.wfile.write(resp.encode())
        elif parsed.path == "/":
            self.send_response(200)
            self._cors_headers()
            self.end_headers()
            html = """<html><body>
            <h1>DLC Inference Server</h1>
            <p>POST /infer with JSON {image: "base64..."} or multipart file upload</p>
            <p><a href="/health">/health</a></p>
            </body></html>"""
            self.wfile.write(html.encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        parsed = urlparse(self.path)

        if parsed.path == "/infer":
            try:
                content_length = int(self.headers.get('Content-Length', 0))
                content_type = self.headers.get('Content-Type', '')

                # JSON mode: {"image": "base64encoded..."}
                if 'application/json' in content_type:
                    body = self.rfile.read(content_length)
                    data = json.loads(body)
                    img_b64 = data.get('image', '')
                    img_bytes = base64.b64decode(img_b64)

                # Multipart form upload
                elif 'multipart/form-data' in content_type:
                    # Simple boundary parsing
                    boundary = content_type.split('boundary=')[1].encode()
                    body = self.rfile.read(content_length)
                    parts = body.split(b'--' + boundary)
                    img_bytes = None
                    for part in parts:
                        if b'Content-Type: image' in part or b'filename=' in part:
                            header_end = part.find(b'\r\n\r\n')
                            img_bytes = part[header_end + 4:]
                            # Remove trailing \r\n and boundary
                            img_bytes = img_bytes.rstrip(b'\r\n--')
                            break
                    if img_bytes is None:
                        raise ValueError("No image found in multipart upload")

                else:
                    # Raw binary JPEG
                    img_bytes = self.rfile.read(content_length)

                # Decode and process
                pil_img = Image.open(io.BytesIO(img_bytes))
                frame_arr = preprocess_frame(pil_img)
                keypoints = run_inference(frame_arr)

                response = {
                    "keypoints": keypoints,
                    "tongue_contour": [keypoints[i] for i in TONGUE_INDICES],
                    "input_size": [INPUT_W, INPUT_H],
                    "inference_time_ms": 0,  # placeholder
                }

                self.send_response(200)
                self._cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps(response).encode())

            except Exception as e:
                log.error(f"Inference error: {e}")
                self.send_response(500)
                self._cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def _cors_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')

    def log_message(self, format, *args):
        """Suppress default HTTP logging (use our own)."""
        pass


# ─── Main ───────────────────────────────────
def main():
    log.info("=" * 50)
    log.info("DLC Inference Server for Tongue Tracker")
    log.info("=" * 50)

    # Load model
    try:
        load_model()
    except Exception as e:
        log.warning(f"Model load failed, server will start without inference: {e}")

    # Start server
    server = HTTPServer(('0.0.0.0', PORT), DLCRequestHandler)
    log.info(f"Server running on http://0.0.0.0:{PORT}")
    log.info(f"Health check: http://localhost:{PORT}/health")
    log.info(f"POST frames to: http://localhost:{PORT}/infer")
    log.info("Press Ctrl+C to stop")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Shutting down...")
        server.shutdown()


if __name__ == '__main__':
    main()
