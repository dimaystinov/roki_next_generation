"""Supervised YOLOv5 detector using the current Python and modern NCS2 plugin.

Only the child process loads OpenVINO. Native hangs/crashes disable detection
without taking the robot down. The blob embeds BGR U8 NHWC preprocessing.
"""
import io
import logging
import os
from pathlib import Path
import threading
import numpy as np

log = logging.getLogger(__name__)


def letterbox(frame, width, height):
    import cv2
    h, w = frame.shape[:2]
    scale = min(width / w, height / h)
    rw, rh = round(w * scale), round(h * scale)
    left, top = (width-rw)//2, (height-rh)//2
    image = frame if (w, h) == (rw, rh) else cv2.resize(frame, (rw, rh), interpolation=cv2.INTER_LINEAR)
    if (rw, rh) != (width, height):
        image = cv2.copyMakeBorder(image, top, height-rh-top, left, width-rw-left,
                                   cv2.BORDER_CONSTANT, value=(114,114,114))
    return np.ascontiguousarray(image), (rw/w, rh/h, left, top)


def detection_center(predictions, label, transform, frame_shape):
    """Decoded YOLOv5 xywh/objectness/class scores from the historical model."""
    import cv2
    predictions = np.asarray(predictions, dtype=np.float32)
    if predictions.ndim != 2 or predictions.shape[1] < 6:
        raise ValueError('Expected YOLOv5 output [anchors, 5 + classes]')
    if label >= predictions.shape[1]-5:
        return 0, 0
    class_ids = predictions[:,5:].argmax(axis=1)
    mask = ((class_ids == label) & (predictions[:,4] >= .4) &
            (predictions[:,5+label] > .25) & np.isfinite(predictions).all(axis=1) &
            (predictions[:,2] > 0) & (predictions[:,3] > 0))
    rows = predictions[mask]
    if not len(rows):
        return 0, 0
    boxes = rows[:,:4].copy()
    boxes[:,:2] -= boxes[:,2:]/2  # NMSBoxes expects xywh, not xyxy.
    indices = np.asarray(cv2.dnn.NMSBoxes(boxes.tolist(), rows[:,4].tolist(), .4, .6)).reshape(-1)
    sx, sy, left, top = transform
    height, width = frame_shape[:2]
    for i in sorted(indices, key=lambda i: -float(rows[i,4])):
        x, y = (float(rows[i,0])-left)/sx, (float(rows[i,1])-top)/sy
        if 0 <= x < width and 0 <= y < height:
            return int(x), int(y)
    return 0, 0


def create_ncs2_core(plugin_path=None):
    """Use built-in MYRIAD, or select an explicit library without duplicate registration."""
    import openvino as ov
    if plugin_path:
        import tempfile
        import xml.etree.ElementTree as ET
        root = ET.Element('ie')
        plugins = ET.SubElement(root, 'plugins')
        path = Path(plugin_path)
        location = str(path.resolve()) if path.parent != Path('.') else str(path)
        ET.SubElement(plugins, 'plugin', name='MYRIAD', location=location)
        # XML registration precedes the compile-time registry in OpenVINO.
        # This selects one explicit library instead of appending a dispatch candidate.
        with tempfile.TemporaryDirectory(prefix='roki-ncs2-core-') as directory:
            config = Path(directory) / 'plugins.xml'
            ET.ElementTree(root).write(config, encoding='utf-8')
            return ov.Core(str(config))
    core = ov.Core()
    if 'MYRIAD' not in core.get_versions('MYRIAD'):
        core.register_plugin('libopenvino_ncs2_plugin.so', 'MYRIAD')
    return core


class _NativeDetector:
    def __init__(self, role='other', display=None, *, model_path=None, plugin_path=None):
        self.role, self.display = role, display
        self.enabled, self.error = False, None
        self._lock = threading.Lock()
        self._core = self._compiled = self._request = None
        path = Path(model_path or os.environ.get('ROKI_NCS2_BLOB', '/usr/share/roki/ball.blob'))
        try:
            if not path.is_file():
                raise FileNotFoundError(f'NCS2 blob not found: {path}; set ROKI_NCS2_BLOB')
            import openvino as ov
            plugin = plugin_path or os.environ.get('ROKI_NCS2_PLUGIN')
            self._core = create_ncs2_core(plugin)
            properties = {}
            if os.environ.get('NCS2_FIRMWARE_DIR'):
                properties['NCS2_FIRMWARE_DIR'] = os.environ['NCS2_FIRMWARE_DIR']
            self._compiled = self._core.import_model(io.BytesIO(path.read_bytes()), 'MYRIAD', properties)
            if len(self._compiled.inputs) != 1 or len(self._compiled.outputs) != 1:
                raise ValueError('Ball detector requires one input and one decoded YOLOv5 output')
            port = self._compiled.input()
            shape = list(port.shape)
            if port.element_type != ov.Type.u8 or len(shape) != 4 or shape[0] != 1 or shape[3] != 3:
                raise ValueError(f'Expected BGR U8 NHWC blob input, got {port.element_type} {shape}')
            self.height, self.width = shape[1:3]
            out = self._compiled.output()
            shape = list(out.shape)
            if (out.element_type not in (ov.Type.f16, ov.Type.f32) or len(shape) != 3 or
                    shape[0] != 1 or not 6 <= shape[2] <= 128):
                raise ValueError(f'Expected decoded YOLOv5 float [1, anchors, classes+5], got {out.element_type} {shape}')
            self._request = self._compiled.create_infer_request()
            self.enabled = True
        except Exception as exc:
            self.error = str(exc)
            self._request = self._compiled = self._core = None
            log.error('NCS2 detector unavailable: %s', exc)

    def is_ready(self):
        return self.enabled

    def object_detect_single(self, frame, obj):
        if obj not in ('ball','basket'):
            raise ValueError(f'Unsupported object: {obj}')
        with self._lock:
            if not self.enabled:
                return 0, 0
            try:
                if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3 or min(frame.shape[:2]) < 1:
                    raise ValueError('Camera frame must be nonempty BGR uint8 HWC')
                image, transform = letterbox(frame, self.width, self.height)
                self._request.infer({0: image[np.newaxis]})
                predictions = self._request.get_output_tensor().data[0]
                return detection_center(predictions, 0 if obj == 'ball' else 1, transform, frame.shape)
            except Exception as exc:
                self.error, self.enabled = str(exc), False
                log.error('NCS2 inference stopped: %s', exc)
                return 0, 0

    def ball_detect_single(self, frame):
        return self.object_detect_single(frame, 'ball')

    def basket_detect_single(self, frame):
        return self.object_detect_single(frame, 'basket')

    def close(self):
        with self._lock:
            self.enabled = False
            self._request = self._compiled = self._core = None

    def stop(self):
        self.close()


class Neural:
    """One child per detector, one frame in flight, permanently disabled on fault.

    Recreate explicitly after repairing the device; no retry loop during a match.
    Timeouts bound IPC waiting, not OS scheduling or process creation latency.
    """
    def __init__(self, role='other', display=None, *, model_path=None,
                 plugin_path=None, inference_timeout=None, startup_timeout=None):
        import atexit
        import mmap
        import socket
        import subprocess
        import sys
        import tempfile
        from .neural_ipc import BUFFER_BYTES, receive
        self.role, self.display = role, display
        self.enabled, self.error = False, None
        self._lock = threading.Lock()
        self._process = self._socket = self._buffer = None
        self._sequence = 0
        child_socket = None
        try:
            self._timeout = float(inference_timeout if inference_timeout is not None
                                  else os.environ.get('ROKI_NCS2_TIMEOUT', '1'))
            startup = float(startup_timeout if startup_timeout is not None
                            else os.environ.get('ROKI_NCS2_STARTUP_TIMEOUT', '15'))
            if not all(np.isfinite(t) and t > 0 for t in (self._timeout, startup)):
                raise ValueError('NCS2 timeouts must be finite and positive')
            path = Path(model_path or os.environ.get('ROKI_NCS2_BLOB', '/usr/share/roki/ball.blob'))
            if not path.is_file():
                raise FileNotFoundError(f'NCS2 blob not found: {path}; set ROKI_NCS2_BLOB')
            self._socket, child_socket = socket.socketpair()
            # Linux memfd is RAM-backed, anonymous and needs no resource tracker.
            # TemporaryFile is the POSIX development-host fallback.
            storage = (os.fdopen(os.memfd_create('roki-ncs2'), 'w+b')
                       if hasattr(os, 'memfd_create') else tempfile.TemporaryFile())
            with storage:
                storage.truncate(BUFFER_BYTES)
                self._buffer = mmap.mmap(storage.fileno(), BUFFER_BYTES)
                env = os.environ.copy()
                root = str(Path(__file__).resolve().parents[2])
                env['PYTHONPATH'] = root + os.pathsep + env.get('PYTHONPATH', '')
                self._process = subprocess.Popen(
                    [sys.executable, '-m', 'Soccer.Vision.neural_worker',
                     str(child_socket.fileno()), str(storage.fileno()), str(os.getpid()),
                     str(path.resolve()), str(plugin_path or env.get('ROKI_NCS2_PLUGIN', ''))],
                    pass_fds=(child_socket.fileno(), storage.fileno()), env=env,
                    stdin=subprocess.DEVNULL)
            child_socket.close()
            child_socket = None
            reply = receive(self._socket, startup)
            if reply.get('status') != 'ready':
                raise RuntimeError(reply.get('error', 'NCS2 startup failed'))
            self.enabled = True
            atexit.register(self.close)
        except Exception as exc:
            self._fail(exc)
        finally:
            if child_socket is not None:
                child_socket.close()

    def _dispose(self):
        import subprocess
        # Never wait for native graph/device destructors in the robot process.
        if self._process is not None:
            try:
                if self._process.poll() is None:
                    self._process.kill()
                self._process.wait(timeout=.2)
            except (OSError, subprocess.TimeoutExpired):
                pass  # An uninterruptible kernel wait must not block the robot.
        for name in ('_socket', '_buffer'):
            resource = getattr(self, name)
            if resource is not None:
                resource.close()
                setattr(self, name, None)

    def _fail(self, exc):
        self.enabled, self.error = False, str(exc) or type(exc).__name__
        self._dispose()
        log.error('NCS2 disabled; using colour vision: %s', self.error)

    def is_ready(self):
        if self.enabled and self._process.poll() is not None:
            # A concurrent request owns cleanup; do not close its shared buffer.
            if self._lock.acquire(blocking=False):
                try:
                    self._fail(RuntimeError(f'NCS2 process exited: {self._process.returncode}'))
                finally:
                    self._lock.release()
            return False
        return self.enabled

    def object_detect_single(self, frame, obj):
        from .neural_ipc import BUFFER_BYTES, receive, send
        if obj not in ('ball', 'basket'):
            raise ValueError(f'Unsupported object: {obj}')
        # No backlog of stale frames behind a stalled inference.
        if not self._lock.acquire(blocking=False):
            return 0, 0
        try:
            if not self.enabled:
                return 0, 0
            if (not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or
                    frame.ndim != 3 or frame.shape[2] != 3 or min(frame.shape[:2]) < 1 or
                    frame.nbytes > BUFFER_BYTES):
                raise ValueError('Camera frame must be BGR uint8 HWC, at most 16 MiB')
            view = np.ndarray(frame.shape, dtype=np.uint8, buffer=self._buffer)
            try:
                np.copyto(view, frame)
            finally:
                del view
            self._sequence += 1
            send(self._socket, {'id': self._sequence, 'shape': frame.shape, 'object': obj}, self._timeout)
            reply = receive(self._socket, self._timeout)
            if reply.get('id') != self._sequence or reply.get('status') != 'ok':
                raise RuntimeError(reply.get('error', 'Invalid NCS2 reply'))
            return tuple(reply['center'])
        except Exception as exc:
            self._fail(exc)
            return 0, 0
        finally:
            self._lock.release()

    def ball_detect_single(self, frame):
        return self.object_detect_single(frame, 'ball')

    def basket_detect_single(self, frame):
        return self.object_detect_single(frame, 'basket')

    def close(self):
        import atexit
        with self._lock:
            self.enabled = False
            self._dispose()
            atexit.unregister(self.close)

    def stop(self):
        self.close()
