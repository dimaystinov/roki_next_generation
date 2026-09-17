# Video output and future robot control

## Current implementation

`Soccer/Vision/display.py` offers two display backends:

- `null`: no display, the default on the robot;
- `imshow`: local OpenCV windows, useful for development.

Select one with `DISPLAY_BACKEND` or `ROKI_DISPLAY_BACKEND`.
An unknown/obsolete backend name falls back to `null`.
There is no embedded RTSP server or GstRtspServer dependency.

Camera capture uses direct libcamera. See `LIBCAMERA_UNICAM_SYNC.md` for BGR
output, buffer ownership and the image-provided Unicam metadata extension.

`tools/manual_msgpack_server.py` already contains manual control and a separate
RTP/JPEG-over-UDP video path addressed to a client. It uses GStreamer core,
`appsrc`, `videoconvert`, `v4l2jpegenc`, `rtpjpegpay` and `udpsink`.
Removing the old display server does not remove that direct UDP path.

## Deferred work agreed with the image developer

The MsgPack protocol will later expand to robot startup/control, match
telemetry, debugging, calibration, localisation viewing and manual control.
The protocol is not redesigned in this change. The current server is a
reference implementation, not the final contract.

Competition scope: autonomous football and manually controlled football.
Other discipline code remains in the repository but is outside the current
localisation review.

Neural inference uses `Soccer/Vision/neural.py` to supervise a private child
process running the same current Python executable. Only the child imports
OpenVINO. `neuro_client.py` retains compatibility aliases. The old Python
worker/service is removed; the private child requires no service setup. It requires the image-provided native MYRIAD blob
plugin (the integrated OpenVINO 2026.5 fork, or the original standalone
2026.0 build) and a BGR U8 NHWC YOLOv5 blob with preprocessing
embedded. Set `ROKI_NCS2_BLOB`, `ROKI_NCS2_PLUGIN` and `NCS2_FIRMWARE_DIR`.
Missing model/device/plugin disables the neural detector with a logged error;
existing strategy readiness checks allow colour-based ball detection.
USB exceptions, child crashes and response timeouts disable neural inference;
football search/speed methods fall back within the failing call using a fresh
camera snapshot. Defaults: `ROKI_NCS2_TIMEOUT=1` seconds per response and
`ROKI_NCS2_STARTUP_TIMEOUT=15` seconds for startup, plus at most 0.2 seconds
waiting for child cleanup (not a hard real-time OS guarantee). No automatic
retries occur during a match; call `glob.neural_vision_enable()` explicitly to
retry after repairing the device. `neural.error` records the failure.

A reusable 16 MiB mmap holds one BGR frame (one copy, no extra colour
conversion); a private socketpair carries small control messages and centres.
There is one request in flight, no frame backlog and no stale response cache.
Linux uses anonymous RAM-backed memfd; only the development-host fallback uses
TemporaryFile. The Linux child is killed when its parent dies. Closing the
detector kills the child without waiting for native OpenVINO destructors.
See `docs/NCS2_RESILIENCE_RU.md` for failure tests and target verification.
This is not a CPU OpenVINO fallback. Physical NCS2 inference remains unverified.

The historical football candidate `orange_ball_on_green_only` and old
YOLO preprocessing were recovered from commit `ad372a887a6b62f51c2cba7a7156363dc642c882`
outside the application repository. The label interpretation follows the old
code (ball=0, basket=1); the model's current suitability must be tested on field
images. Neural preprocessing uses exact letterbox padding, and postprocessing
uses xywh for OpenCV NMS and selects the strongest valid detection.

## Boundaries to retain during future integration

- One camera owner: video/debug clients should consume its frames, rather than
  opening a competing capture pipeline during autonomous operation.
- Keep robot control responsive when clients disconnect or video consumers
  cannot keep up; avoid unbounded frame queues.
- Localisation telemetry should identify the pose source, measurement age and
  frame ID, rather than presenting every stored coordinate as a fresh estimate.
- Do not claim NCS2 support from an OpenVINO version string alone; prove device
  enumeration, model compilation and sustained inference on the target image.

Built-in MYRIAD registration is reused. An explicit `ROKI_NCS2_PLUGIN`
selects one library through a private Core XML configuration, avoiding duplicate
registration or an unintended dispatch group in the newer OpenVINO fork.
