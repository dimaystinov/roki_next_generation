# Direct libcamera and Unicam frame IDs

The robot uses the Python `libcamera` API directly. Picamera2 is no longer
required. The target is Raspberry Pi CM4/VC4 with
`libcamera 0.7.2+rpt20260817` with the `rpi.UnicamSequence` extension supplied
by the robot image. The library patch is included in `buildroot/0001-rpi-unicam-sequence.patch`;
the image developer integrates it into the Buildroot package patch queue.

## Image and frame number

`Soccer/Vision/camera.py` configures one `VideoRecording` output:

- size: 800 × 650 by default (existing `camera_lores` setting);
- libcamera pixel format: `RGB888`, which maps to V4L2 `BGR24` and supplies
  B,G,R bytes in memory, ready for OpenCV;
- sensor timing: 16700 µs by default, with exposure/gain controls as before;
- no application RAW stream, no software demosaic or colour conversion.

The underlying camera pipeline still uses its internal Bayer input as usual.
The image patch copies only that input buffer's hardware sequence into the
request metadata before sending the frame to the ISP:

```python
sequence = request.metadata[libcamera.controls.rpi.UnicamSequence]
```

`UnicamSequence` is an **image-provided extension**, not a field in
unmodified `0.7.2+rpt20260817`. The stock tag exports `FrameMetadata.sequence`,
but an ISP output buffer has an ISP sequence, not an authoritative Unicam ID.
`Request.sequence` and `rpi.ControlListSequence` are also not sensor frame IDs.

The adapter never derives IDs from timestamps or FPS and never renumbers them.
Zero is valid, skipped frames leave gaps, and the underlying 32-bit counter
can wrap. Startup frames may already have been discarded before the first
usable frame arrives, so the first returned ID is not necessarily zero.
Missing patch/bindings/metadata causes an explicit error with no fallback.

## Buffers and consumers

A capture thread keeps requests circulating and retains only the latest
completed request. It recycles older frames without mapping or copying their
pixels for the consumer. Main buffers are mapped once at startup.

`snapshot()` returns `(bgr_array, unicam_sequence)` from one completed request.
It honours row stride, synchronises DMA-BUF CPU access and makes one copy of
the requested BGR image before requeueing its buffer. This is a lifetime copy,
not a colour conversion: existing vision code retains and edits frames after
`snapshot()` returns. Returning a view and immediately requeueing would let the
camera overwrite data still being processed.

`set_controls({...})` sends named controls on the next recycled request.
`capture_metadata()` returns metadata from a fresh snapshot for calibration.
`stop()` joins capture, stops libcamera, drains cancelled requests, unmaps
buffers and releases the device. One `Camera` may own the process-wide
CameraManager ready queue at a time.

## IMU and restarts

`Vision_RPI` resets STM strobe history before starting the camera. Its snapshot
passes the exact Unicam ID to `GetIMUFrame`, even when vision is not running
in a background thread. Calls with no image ID still use `GetIMULatest`.
An unavailable indexed sample must not silently become a latest sample.

`Glob.camera_reset()` stops the old vision consumer and camera before creating
new vision and resetting STM history. It retains the existing STM channel,
so motion and localisation do not keep stale channel references.

The electrical alignment of STM STROBE counting with Unicam's origin still
requires a CM4 test. No arbitrary +1/+2 offset is introduced in software.

## Build and validation

The image developer applies the library patch in Buildroot and rebuilds
libcamera, the RPi IPA and Python bindings together. The application requires
`libcamera.controls.rpi.UnicamSequence` and that field in completed VC4 request
metadata. See `buildroot/README.md` and `buildroot/LIBCAMERA_PATCH_RU.md` for the
included patch and image integration steps.

On a development machine:

```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests
```

On the updated CM4 image, with other camera owners stopped, from repo root:

```sh
python3 tools/test/check_libcamera.py --frames 100 --restarts 2
```

This camera-only check does not initialise Roki or move servos. It logs IDs,
ID gaps, image shape and sensor timestamps, deliberately pauses its consumer,
and restarts capture. Timestamps are diagnostic only. Also verify BGR colours
with a known coloured target, and then independently compare camera IDs with
STM history across capture pauses and camera restarts.

Network streaming/protocol changes are a separate task. Existing manual and
calibration tools now use the same camera adapter; no new streaming command
is introduced here.
