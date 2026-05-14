"""
DLC Inference Server — Tongue Contour Tracker Backend
=====================================================
Loads a trained DeepLabCut model and serves real-time keypoint
inference over HTTP. Runs on Windows (requires DLC environment).

Usage:
    python windows_dlc_server.py
    python windows_dlc_server.py --model "E:\DLC\palate-sunjing-2024-08-27"
    python windows_dlc_server.py --port 8766 --input-size 320 240

The tongue-tracker web app connects to http://localhost:8766/infer
"""

import argparse
import base64
import io
import json
import logging
import os
import sys
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

import numpy as np
from PIL import Image

# ─── Defaults ────────────────────────────────
DEFAULT_MODEL_PATHS = [
    r"E:\DLC\palate-sunjing-2024-08-27",
    r"C:\DLC\palate-sunjing-2024-08-27",
    r"D:\DLC\palate-sunjing-2024-08-27",
]

BODY_PARTS = [
    "vallecula",
    "tongueRoot1", "tongueRoot2", "tongueBody1", "tongueBody2",
    "tongueDorsum1", "tongueDorsum2", "tongueBlade1", "tongueBlade2",
    "tongueTip1", "tongueTip2",
    "jinxing", "muxing", "shuixing", "huoxing", "tuxing",
    "diqiu", "yueliang", "taiyang",
]
TONGUE_INDICES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # tongueRoot1..tongueTip2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("dlc-server")

# ─── Model State ─────────────────────────────
sess = None
inputs_tensor = None
outputs_tensor = None
dlc_cfg = None
INPUT_W, INPUT_H = 320, 240
LOCREF_STDEV = 7.2801
STRIDE = 8


def find_model_path():
    """Auto-detect model path from CLI args, env var, or defaults."""
    parser = argparse.ArgumentParser(description="DLC Inference Server")
    parser.add_argument(
        "--model", "-m",
        help="Path to DLC project directory (e.g. E:\\DLC\\palate-sunjing-2024-08-27)",
    )
    parser.add_argument("--port", "-p", type=int, default=8766)
    parser.add_argument(
        "--input-size", "-s", nargs=2, type=int, default=[320, 240],
        metavar=("W", "H"),
    )
    args = parser.parse_args()

    # Priority: CLI > env var > defaults
    if args.model:
        path = args.model
    elif "DLC_MODEL_PATH" in os.environ:
        path = os.environ["DLC_MODEL_PATH"]
    else:
        path = None
        for p in DEFAULT_MODEL_PATHS:
            if os.path.isdir(p):
                path = p
                break

    if not path or not os.path.isdir(path):
        log.error("Model path not found! Tried: %s", DEFAULT_MODEL_PATHS)
        log.error("Specify with: python windows_dlc_server.py --model <path>")
        sys.exit(1)

    return path, args.port, args.input_size[0], args.input_size[1]


def fix_config_project_path(config_path, project_path):
    """DLC config.yaml has a hardcoded project_path. Fix it to match actual location."""
    import yaml
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        old_path = cfg.get("project_path", "")
        if old_path != project_path:
            log.info("Updating config project_path: %s → %s", old_path, project_path)
            cfg["project_path"] = project_path
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
    except Exception as e:
        log.warning("Could not fix config project_path: %s", e)


def load_model(model_path):
    """Load DLC model using the official prediction pipeline."""
    global sess, inputs_tensor, outputs_tensor, dlc_cfg, STRIDE, LOCREF_STDEV

    config_path = os.path.join(model_path, "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.yaml not found at {config_path}")

    # Fix project_path in config so DLC can find its files
    fix_config_project_path(config_path, model_path)

    log.info("Loading DLC config: %s", config_path)
    from deeplabcut.pose_estimation_tensorflow.config import load_config
    dlc_cfg = load_config(config_path)

    # Extract key params
    STRIDE = dlc_cfg.get("stride", 8)
    LOCREF_STDEV = dlc_cfg.get("locref_stdev", 7.2801)
    log.info("Model params: stride=%d, locref_stdev=%.4f", STRIDE, LOCREF_STDEV)

    log.info("Setting up pose prediction...")
    from deeplabcut.pose_estimation_tensorflow.core import predict
    sess, inputs_tensor, outputs_tensor = predict.setup_pose_prediction(dlc_cfg)

    log.info("Model loaded. Running warmup + benchmark...")

    # Warmup
    dummy = np.random.randint(0, 255, (INPUT_H, INPUT_W, 3), dtype=np.uint8)
    t0 = time.time()
    pose = predict.getpose(dummy, dlc_cfg, sess, inputs_tensor, outputs_tensor)
    warmup_time = time.time() - t0
    log.info("Warmup: %.2fs (pose shape: %s)", warmup_time, pose.shape)

    # Benchmark (5 runs, discard first)
    times = []
    for _ in range(6):
        t1 = time.time()
        _ = predict.getpose(dummy, dlc_cfg, sess, inputs_tensor, outputs_tensor)
        times.append(time.time() - t1)
    avg = sum(times[1:]) / len(times[1:])
    log.info("Steady-state: %.0fms/frame (~%d FPS)", avg * 1000, int(1 / avg))

    return predict


def run_inference(frame_array):
    """Run inference on a preprocessed frame. Returns (keypoints_list, inference_time_ms)."""
    from deeplabcut.pose_estimation_tensorflow.core import predict

    t0 = time.time()
    pose = predict.getpose(frame_array, dlc_cfg, sess, inputs_tensor, outputs_tensor)
    dt_ms = (time.time() - t0) * 1000

    # pose shape: (num_joints, 3) = [x, y, confidence]
    keypoints = []
    for i, name in enumerate(BODY_PARTS):
        if i < len(pose):
            keypoints.append({
                "name": name,
                "x": round(float(pose[i, 0]), 2),
                "y": round(float(pose[i, 1]), 2),
                "confidence": round(float(pose[i, 2]), 4),
            })
        else:
            keypoints.append({
                "name": name,
                "x": 0, "y": 0, "confidence": 0,
            })

    return keypoints, dt_ms


# ─── HTTP Server ─────────────────────────────
class DLCRequestHandler(BaseHTTPRequestHandler):
    """HTTP handler for DLC inference requests."""

    def _cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

    def _json_response(self, data, status=200):
        self.send_response(status)
        self._cors_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode())

    def do_OPTIONS(self):
        self.send_response(200)
        self._cors_headers()
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == "/health":
            self._json_response({
                "status": "ok",
                "model_loaded": sess is not None,
                "input_size": [INPUT_W, INPUT_H],
                "num_keypoints": len(BODY_PARTS),
                "tongue_indices": TONGUE_INDICES,
            })

        elif parsed.path == "/" or parsed.path == "":
            self.send_response(200)
            self._cors_headers()
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            html = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>DLC Server</title>
<style>body{font-family:monospace;background:#111;color:#0f0;padding:2em;}
a{color:#0ff}</style></head><body>
<h1>👅 DLC Inference Server</h1>
<p>Status: {status}</p>
<p>Model: {model_loaded}</p>
<p>Input: {w}x{h}</p>
<p>Keypoints: {n}</p>
<hr>
<h3>API</h3>
<ul>
<li><code>GET /health</code> — server status</li>
<li><code>POST /infer</code> — send JPEG/PNG frame, get keypoints JSON</li>
</ul>
</body></html>""".format(
                status="🟢 Running" if sess else "🔴 No model",
                model_loaded="✅ Loaded" if sess else "❌ Not loaded",
                w=INPUT_W, h=INPUT_H, n=len(BODY_PARTS),
            )
            self.wfile.write(html.encode())

        else:
            self._json_response({"error": "Not found"}, 404)

    def do_POST(self):
        parsed = urlparse(self.path)

        if parsed.path != "/infer":
            self._json_response({"error": "Not found"}, 404)
            return

        if sess is None:
            self._json_response({"error": "Model not loaded"}, 503)
            return

        try:
            content_length = int(self.headers.get("Content-Length", 0))
            content_type = self.headers.get("Content-Type", "")

            # Decode image from request body
            if "application/json" in content_type:
                body = self.rfile.read(content_length)
                data = json.loads(body)
                img_b64 = data.get("image", "")
                img_bytes = base64.b64decode(img_b64)

            elif "multipart/form-data" in content_type:
                boundary = content_type.split("boundary=")[1].encode()
                body = self.rfile.read(content_length)
                parts = body.split(b"--" + boundary)
                img_bytes = None
                for part in parts:
                    if b"Content-Type: image" in part or b"filename=" in part:
                        header_end = part.find(b"\r\n\r\n")
                        if header_end >= 0:
                            img_bytes = part[header_end + 4:]
                            img_bytes = img_bytes.rstrip(b"\r\n--")
                            break
                if img_bytes is None:
                    raise ValueError("No image found in multipart upload")

            else:
                # Raw binary (JPEG/PNG)
                img_bytes = self.rfile.read(content_length)

            if not img_bytes or len(img_bytes) < 100:
                raise ValueError("Empty or invalid image data")

            # Preprocess: resize to model input size
            pil_img = Image.open(io.BytesIO(img_bytes))
            pil_img = pil_img.resize((INPUT_W, INPUT_H), Image.LANCZOS)
            pil_img = pil_img.convert("RGB")
            frame_arr = np.array(pil_img, dtype=np.uint8)

            # Run inference
            keypoints, dt_ms = run_inference(frame_arr)

            # Build tongue contour (filtered by confidence)
            tongue_contour = [keypoints[i] for i in TONGUE_INDICES]
            tongue_valid = [kp for kp in tongue_contour if kp["confidence"] > 0.1]

            response = {
                "keypoints": keypoints,
                "tongue_contour": tongue_contour,
                "tongue_valid": tongue_valid,
                "input_size": [INPUT_W, INPUT_H],
                "inference_time_ms": round(dt_ms, 1),
                "num_keypoints": len(keypoints),
            }

            self._json_response(response)

        except Exception as e:
            log.error("Inference error: %s", e, exc_info=True)
            self._json_response({"error": str(e)}, 500)

    def log_message(self, format, *args):
        """Suppress default HTTP request logging."""
        pass


# ─── Main ────────────────────────────────────
def main():
    model_path, port, input_w, input_h = find_model_path()

    global INPUT_W, INPUT_H
    INPUT_W, INPUT_H = input_w, input_h

    log.info("=" * 55)
    log.info("  DLC Inference Server — Tongue Contour Tracker")
    log.info("=" * 55)
    log.info("  Model path : %s", model_path)
    log.info("  Input size : %dx%d", INPUT_W, INPUT_H)
    log.info("  Keypoints  : %d (tongue: indices %s)", len(BODY_PARTS), TONGUE_INDICES)
    log.info("  Port       : %d", port)
    log.info("=" * 55)

    # Load model
    try:
        load_model(model_path)
    except Exception as e:
        log.error("Failed to load model: %s", e, exc_info=True)
        log.warning("Server will start WITHOUT model. Inference will return 503.")

    # Start HTTP server
    server = HTTPServer(("0.0.0.0", port), DLCRequestHandler)
    log.info("Server running at http://localhost:%d", port)
    log.info("Health check: http://localhost:%d/health", port)
    log.info("POST frames:  http://localhost:%d/infer", port)
    log.info("Press Ctrl+C to stop.")
    log.info("")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Shutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
