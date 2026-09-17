"""Direct libcamera capture for CM4, with the roki UnicamSequence metadata patch.

The ISP supplies BGR bytes; no RAW output stream or CPU colour conversion is
needed. A snapshot owns its pixels so recycling DMA buffers cannot corrupt it.
"""

import fcntl
import selectors
import struct
import threading
import time

import numpy as np


# Linux DMA-BUF read access (cache coherency on ARM).
_DMA_BUF_IOCTL_SYNC = 0x40086200
_DMA_BUF_READ_START = struct.pack("Q", 1)
_DMA_BUF_READ_END = struct.pack("Q", 5)


class Camera:
    # CameraManager's ready queue is process-wide; this wrapper owns it alone.
    _manager_owner = threading.Lock()

    def __init__(self, camera_num=0, timeout=3.0):
        self.camera_num = camera_num
        self.timeout = timeout
        self.camera_lores = (800, 650)  # Existing callers use this for output size.
        self.last_frame_number = None
        self.last_metadata = {}
        self._condition = threading.Condition()
        self._lifecycle = threading.RLock()
        self._stop_event = threading.Event()
        self._thread = None
        self._manager = self._camera = self._config = self._allocator = None
        self._stream = None
        self._requests = []
        self._mapped = {}
        self._latest = None
        self._error = None
        self._pending_controls = {}
        self._started = self._acquired = self._owns_manager = False

    def start(self, exposure=None, gain=None, frame_duration_us=16700, neural=False,
              sensor_size=None, sensor_bit_depth=None):
        """Start one ISP stream. Requires the matching libcamera image patch."""
        with self._lifecycle:
            if self._owns_manager:
                raise RuntimeError("Camera is already started; stop it before restarting")
            import libcamera as libcam
            from libcamera.utils import MappedFrameBuffer

            self._libcam = libcam
            try:
                self._sequence_control = libcam.controls.rpi.UnicamSequence
            except AttributeError as exc:
                raise RuntimeError(
                    "The robot image must provide libcamera with the Buildroot-maintained "
                    "UnicamSequence extension and rebuilt Python bindings "
                    "(rpi.UnicamSequence is missing)") from exc
            if not self._manager_owner.acquire(blocking=False):
                raise RuntimeError("Another Camera already owns libcamera in this process")
            self._owns_manager = True
            self._stop_event.clear()
            self._error = None
            self.last_frame_number = None
            self.last_metadata = {}
            self._pending_controls = {}
            try:
                self._manager = libcam.CameraManager.singleton()
                self._camera = self._manager.cameras[self.camera_num]
                self._camera.acquire()
                self._acquired = True
                self._config = self._camera.generate_configuration([libcam.StreamRole.VideoRecording])
                if self._config is None or len(self._config) != 1:
                    raise RuntimeError("Camera must support one ISP output stream")
                main = self._config.at(0)
                # libcamera RGB888 maps to V4L2 BGR24: B,G,R bytes in memory.
                main.pixel_format = libcam.PixelFormat("RGB888")
                main.size = libcam.Size(*self.camera_lores)
                main.buffer_count = 4
                if sensor_size is not None:
                    sensor = libcam.SensorConfiguration()
                    sensor.output_size = libcam.Size(*sensor_size)
                    sensor.bit_depth = int(sensor_bit_depth)
                    self._config.sensor_config = sensor
                if self._config.validate() == libcam.CameraConfiguration.Status.Invalid:
                    raise RuntimeError("Invalid camera configuration")
                if (str(main.pixel_format) != "RGB888" or
                        (main.size.width, main.size.height) != self.camera_lores):
                    raise RuntimeError("Camera cannot supply the requested BGR size directly")
                self._camera.configure(self._config)
                self._stream = main.stream
                self._stride = main.stride
                self._allocator = libcam.FrameBufferAllocator(self._camera)
                self._allocator.allocate(self._stream)
                buffers = self._allocator.buffers(self._stream)
                if not buffers:
                    raise RuntimeError("No camera buffers allocated")
                for i, buffer in enumerate(buffers):
                    if len(buffer.planes) != 1:
                        raise RuntimeError("Packed BGR output must have one plane")
                    self._mapped[i] = MappedFrameBuffer(buffer).mmap()
                    request = self._camera.create_request(i)
                    request.add_buffer(self._stream, buffer)
                    self._requests.append(request)
                controls = {"AwbEnable": bool(neural)}
                if frame_duration_us is not None:
                    controls["FrameDurationLimits"] = (frame_duration_us, frame_duration_us)
                if exposure is not None:
                    controls["ExposureTime"] = int(exposure)
                if gain is not None:
                    controls["AnalogueGain"] = float(gain)
                if exposure is not None and gain is not None:
                    controls["AeEnable"] = False
                self._camera.start(self._control_ids(controls))
                self._started = True
                for request in self._requests:
                    self._camera.queue_request(request)
                self._thread = threading.Thread(target=self._capture_loop,
                                                name="libcamera-capture", daemon=True)
                self._thread.start()
                return True
            except BaseException:
                self.stop()
                raise

    def _control_ids(self, values):
        return {getattr(self._libcam.controls, name): value for name, value in values.items()}

    def set_controls(self, values):
        """Apply named libcamera controls to the next recycled request."""
        with self._condition:
            if not self._started or self._stop_event.is_set():
                raise RuntimeError("Camera is not running")
            self._pending_controls.update(self._control_ids(values))

    def _requeue(self, request):
        # Called under _condition, including when applying pending controls.
        request.reuse()
        for control, value in self._pending_controls.items():
            request.set_control(control, value)
        self._pending_controls.clear()
        self._camera.queue_request(request)

    def _capture_loop(self):
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(self._manager.event_fd, selectors.EVENT_READ)
                while not self._stop_event.is_set():
                    if not selector.select(0.2):
                        continue
                    ready = self._manager.get_ready_requests()
                    with self._condition:
                        for request in ready:
                            if self._stop_event.is_set():
                                break
                            if request.status != self._libcam.Request.Status.Complete:
                                raise RuntimeError("libcamera cancelled a capture request")
                            if request.buffers[self._stream].metadata.status != self._libcam.FrameMetadata.Status.Success:
                                # Includes startup/unconverged frames. Never renumber them.
                                self._requeue(request)
                                continue
                            sequence = request.metadata.get(self._sequence_control)
                            if not isinstance(sequence, int) or not 0 <= sequence <= 0xffffffff:
                                raise RuntimeError("Completed frame has no rpi.UnicamSequence; check the VC4 image patch")
                            if self._latest is not None:
                                self._requeue(self._latest)
                            self._latest = request
                        self._condition.notify_all()
        except Exception as exc:
            with self._condition:
                self._error = exc
                self._condition.notify_all()

    def _copy_bgr(self, request):
        buffer = request.buffers[self._stream]
        fd = buffer.planes[0].fd
        width, height = self.camera_lores
        if buffer.metadata.planes[0].bytes_used < (height - 1) * self._stride + width * 3:
            raise RuntimeError("Incomplete BGR buffer")
        fcntl.ioctl(fd, _DMA_BUF_IOCTL_SYNC, _DMA_BUF_READ_START)
        try:
            # Account for ISP row padding. One memcpy, no colour conversion.
            image = np.ndarray((height, width, 3), dtype=np.uint8,
                               buffer=self._mapped[request.cookie].planes[0],
                               strides=(self._stride, 3, 1))
            return image.copy()
        finally:
            fcntl.ioctl(fd, _DMA_BUF_IOCTL_SYNC, _DMA_BUF_READ_END)

    def snapshot(self):
        deadline = time.monotonic() + self.timeout
        with self._condition:
            while self._latest is None and self._error is None:
                if not self._started or self._stop_event.is_set():
                    raise RuntimeError("Camera is not running")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("No completed libcamera frame")
                self._condition.wait(remaining)
            if self._error is not None:
                raise RuntimeError("libcamera capture failed") from self._error
            if self._stop_event.is_set():
                raise RuntimeError("Camera is stopping")
            request, self._latest = self._latest, None
            try:
                # Never substitute Request.sequence, ISP sequence, timestamps or FPS.
                frame_number = int(request.metadata[self._sequence_control])
                image = self._copy_bgr(request)
                self.last_frame_number = frame_number
                self.last_metadata = {control.name: value for control, value in request.metadata.items()}
                return image, frame_number
            finally:
                self._requeue(request)

    def capture_metadata(self):
        """Capture fresh metadata for calibration tools, keyed by control name."""
        with self._condition:
            self.snapshot()
            return dict(self.last_metadata)

    def stop(self):
        """Stop capture, drain requests, unmap buffers and release the camera."""
        with self._lifecycle:
            if not self._owns_manager:
                return
            self._stop_event.set()
            with self._condition:
                self._condition.notify_all()
            if self._thread is not None:
                self._thread.join()
                self._thread = None
            try:
                if self._started:
                    self._camera.stop()
                    self._manager.get_ready_requests()  # Release Python queue references.
            finally:
                self._started = False
                self._latest = None
                self._requests.clear()
                try:
                    for mapped in self._mapped.values():
                        mapped.munmap()
                finally:
                    self._mapped.clear()
                    self._allocator = None
                    try:
                        if self._acquired:
                            self._camera.release()
                    finally:
                        self._acquired = False
                        self._camera = self._manager = self._config = None
                        self._stream = None
                        self._owns_manager = False
                        self._manager_owner.release()
