#!/usr/bin/env python3
import argparse
import copy
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]

import sys

sys.path.insert(0, str(REPO_ROOT))

from Soccer.config_paths import init_param_read_path, init_param_write_path  # noqa: E402
from Soccer.Vision.camera import Camera  # noqa: E402
from Soccer.Vision.reload import Image  # noqa: E402


DEFAULT_DETECTION_KEYS = {
    0: None,
    1: "current",
    2: "orange ball",
    3: "blue posts",
    4: "yellow posts",
    5: "white posts",
}


class ThresholdTunerState:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.lock = threading.Lock()
        self.thresholds = self._load_thresholds()
        self.device = self._first_device()
        self.blob_detection = 1
        self.running = True
        self.camera = None
        self.last_frame = None
        self.last_error = None
        self.frame_id = 0

    def _load_thresholds(self):
        path = init_param_read_path(REPO_ROOT, "Real/Real_Thresholds.json")
        with open(path, "r") as f:
            return json.load(f)

    def _first_device(self):
        for key, value in self.thresholds.items():
            if isinstance(value, dict) and "th" in value:
                return key
        return "orange ball"

    def snapshot_state(self):
        with self.lock:
            return {
                "thresholds": copy.deepcopy(self.thresholds),
                "device": self.device,
                "devices": [k for k, v in self.thresholds.items() if isinstance(v, dict) and "th" in v],
                "blob_detection": self.blob_detection,
                "frame_id": self.frame_id,
                "last_error": self.last_error,
            }

    def update(self, payload):
        with self.lock:
            if "device" in payload and payload["device"] in self.thresholds:
                self.device = payload["device"]
            if "blob_detection" in payload:
                self.blob_detection = int(payload["blob_detection"])
            if "thresholds" in payload:
                for key, value in payload["thresholds"].items():
                    if key in self.thresholds:
                        self.thresholds[key] = value
            if "exposure" in payload:
                self.thresholds["exposure"] = int(payload["exposure"])
                if self.camera:
                    self.camera.set_controls({"ExposureTime": int(payload["exposure"])})
            if "gain" in payload:
                self.thresholds["gain"] = float(payload["gain"])
                if self.camera:
                    self.camera.set_controls({"AnalogueGain": float(payload["gain"])})
            if payload.get("save"):
                self.save_locked()
        return self.snapshot_state()

    def save_locked(self):
        path = init_param_write_path(REPO_ROOT, "Real/Real_Thresholds.json")
        with open(path, "w") as f:
            json.dump(self.thresholds, f)
        print(f"Saved thresholds to {path}", flush=True)

    def _draw_blobs(self, image):
        with self.lock:
            thresholds = copy.deepcopy(self.thresholds)
            device = self.device
            blob_detection = self.blob_detection
        key = DEFAULT_DETECTION_KEYS.get(blob_detection)
        if key == "current":
            key = device
        if not key or key not in thresholds:
            return
        threshold = thresholds[key]
        try:
            for blob in image.find_blobs(
                [threshold["th"]],
                pixels_threshold=threshold["pixel"],
                area_threshold=threshold["area"],
                merge=True,
            ):
                image.draw_rectangle(blob.rect(), color=(0, 0, 255))
        except Exception as exc:
            with self.lock:
                self.last_error = f"blob detection failed: {exc}"

    def camera_loop(self):
        self.camera = Camera()
        with self.lock:
            exposure = self.thresholds.get("exposure")
            gain = self.thresholds.get("gain")
            self.camera.camera_lores = (self.width, self.height)
        self.camera.start(exposure=exposure, gain=gain)
        print(f"Camera started: {self.width}x{self.height}", flush=True)
        try:
            while self.running:
                frame, _ = self.camera.snapshot()
                color = Image(frame)
                self._draw_blobs(color)
                with self.lock:
                    th = self.thresholds[self.device]["th"]
                binary = Image(frame, copy=False).binary(th, to_three_channels=True)
                combined = cv2.hconcat([color.img, binary])
                with self.lock:
                    self.last_frame = combined
                    self.frame_id += 1
                    self.last_error = None
        except Exception as exc:
            with self.lock:
                self.last_error = str(exc)
            raise
        finally:
            if self.camera:
                self.camera.stop()

    def encode_jpeg(self, quality):
        with self.lock:
            frame = None if self.last_frame is None else self.last_frame.copy()
        if frame is None:
            return None
        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        if not ok:
            return None
        return encoded.tobytes()


class Handler(BaseHTTPRequestHandler):
    server_version = "RokiThresholdTuner/1.0"

    def log_message(self, fmt, *args):
        return

    @property
    def state(self):
        return self.server.state

    def _send_json(self, data, status=200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/state":
            self._send_json(self.state.snapshot_state())
            return
        if parsed.path == "/frame.jpg":
            query = parse_qs(parsed.query)
            quality = int(query.get("quality", ["80"])[0])
            body = self.state.encode_jpeg(quality)
            if body is None:
                self.send_response(503)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self._send_json({"error": "not found"}, status=404)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/control":
            self._send_json({"error": "not found"}, status=404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        try:
            payload = json.loads(body.decode("utf-8"))
            self._send_json(self.state.update(payload))
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=400)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=650)
    args = parser.parse_args()

    state = ThresholdTunerState(args.width, args.height)
    camera_thread = threading.Thread(target=state.camera_loop, daemon=True)
    camera_thread.start()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.state = state
    print(f"Threshold tuner robot server listening on {args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    finally:
        state.running = False
        server.server_close()


if __name__ == "__main__":
    main()
