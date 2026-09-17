"""Hardware-independent regression checks for the actual camera adapter."""
import os
import sys
import threading
import time
from types import ModuleType, SimpleNamespace as NS
from unittest.mock import patch

import numpy as np
import pytest

from Soccer.Vision.camera import Camera


class Control:
    def __init__(self, name):
        self.name = name


class Request:
    def __init__(self, cookie):
        self.cookie = cookie
        self.buffers = {}
        self.metadata = {}
        self.status = 1
        self.sequence = 999  # Deliberately unrelated to Unicam.
        self.controls = {}

    def add_buffer(self, stream, buffer):
        self.buffers[stream] = buffer

    def reuse(self):
        self.metadata = {}
        self.controls = {}

    def set_control(self, control, value):
        self.controls[control] = value


class Hardware:
    def __init__(self):
        self.pending = []
        self.ready = []
        self.lock = threading.Lock()
        self.event_fd, self.write_fd = os.pipe()
        os.set_blocking(self.event_fd, False)
        self.acquired = False
        self.stopped = False
        self.maps = []
        self.controls = {}
        self.roles = None
        self.fail_queue = False
        self.main = NS(stream=object(), size=NS(width=2, height=2),
                       stride=8, pixel_format='RGB888', buffer_count=4)
        self.buffers = [NS(planes=[NS(fd=i + 100)], data=bytearray(16),
                           metadata=NS(sequence=777, status=0, planes=[NS(bytes_used=16)]))
                        for i in range(4)]
        self.module = ModuleType('libcamera')
        self.module.CameraManager = NS(singleton=lambda: self)
        self.cameras = [self]
        self.module.StreamRole = NS(VideoRecording='video')
        self.module.PixelFormat = str
        self.module.Size = lambda w, h: NS(width=w, height=h)
        self.module.CameraConfiguration = NS(Status=NS(Invalid=2))
        self.module.Request = NS(Status=NS(Complete=1))
        self.module.FrameMetadata = NS(Status=NS(Success=0))
        self.module.controls = NS(**{name: Control(name) for name in
                                   ('AwbEnable', 'AeEnable', 'FrameDurationLimits',
                                    'ExposureTime', 'AnalogueGain')})
        self.sequence_control = Control('UnicamSequence')
        self.module.controls.rpi = NS(UnicamSequence=self.sequence_control)
        self.module.FrameBufferAllocator = lambda camera: NS(
            allocate=lambda stream: len(self.buffers), buffers=lambda stream: self.buffers)
        self.utils = ModuleType('libcamera.utils')
        self.utils.MappedFrameBuffer = self.mapping

    def mapping(self, buffer):
        mapped = NS(planes=[buffer.data], closed=False)
        mapped.mmap = lambda: mapped
        mapped.munmap = lambda: setattr(mapped, 'closed', True)
        self.maps.append(mapped)
        return mapped

    def acquire(self):
        self.acquired = True

    def release(self):
        self.acquired = False

    def generate_configuration(self, roles):
        self.roles = roles
        main = self.main
        class Config:
            def __len__(self): return 1
            def at(self, i): return main
            def validate(self): return 0
        return Config()

    def configure(self, config): pass
    def create_request(self, cookie): return Request(cookie)

    def start(self, controls):
        self.controls = controls
        self.stopped = False

    def stop(self):
        self.stopped = True
        self.pending.clear()

    def queue_request(self, request):
        if self.fail_queue:
            raise RuntimeError('injected queue failure')
        # DMA may overwrite it as soon as it is requeued.
        next(iter(request.buffers.values())).data[:] = b'\x00' * 16
        with self.lock:
            self.pending.append(request)

    def get_ready_requests(self):
        with self.lock:
            try: os.read(self.event_fd, 1000)
            except BlockingIOError: pass
            ready, self.ready = self.ready, []
            return ready

    def deliver(self, sequence, status=0, include_sequence=True):
        with self.lock:
            request = self.pending.pop(0)
            request.status = 1
            buffer = next(iter(request.buffers.values()))
            buffer.metadata.status = status
            buffer.data[:] = bytes([1, 2, 3, 4, 5, 6, 99, 99,
                                    7, 8, 9, 10, 11, 12, 99, 99])
            request.metadata = {self.sequence_control: sequence} if include_sequence else {}
            self.ready.append(request)
            os.write(self.write_fd, b'x')
        return request


@pytest.fixture
def capture():
    hardware = Hardware()
    camera = Camera(timeout=0.5)
    camera.camera_lores = (2, 2)
    with patch.dict(sys.modules, {'libcamera': hardware.module, 'libcamera.utils': hardware.utils}), \
            patch('Soccer.Vision.camera.fcntl.ioctl') as ioctl:
        try:
            yield camera, hardware, ioctl
        finally:
            camera.stop()
            os.close(hardware.event_fd)
            os.close(hardware.write_fd)


def test_bgr_copy_preserves_padding_order_and_hardware_zero(capture):
    camera, hardware, ioctl = capture
    camera.start(exposure=500, gain=8.0)
    hardware.deliver(0)
    image, number = camera.snapshot()
    assert number == 0  # Not request=999, ISP=777 or a fabricated start offset.
    np.testing.assert_array_equal(image, np.arange(1, 13, dtype=np.uint8).reshape(2, 2, 3))
    assert image.flags.c_contiguous
    assert hardware.roles == ['video']  # No RAW stream.
    assert len(ioctl.call_args_list) == 2
    assert camera.last_metadata['UnicamSequence'] == 0


def test_hardware_gaps_are_not_renumbered(capture):
    camera, hardware, _ = capture
    camera.start()
    for number in [17, 25, 0xffffffff, 0]:
        hardware.deliver(number)
        assert camera.snapshot()[1] == number


def test_late_consumer_gets_latest_request_without_copying_older_frames(capture):
    camera, hardware, _ = capture
    camera.start()
    with patch.object(camera, '_copy_bgr', wraps=camera._copy_bgr) as copy:
        hardware.deliver(2)
        hardware.deliver(9)
        deadline = time.monotonic() + 1
        with camera._condition:
            while camera._latest is None or camera._latest.metadata.get(hardware.sequence_control) != 9:
                assert time.monotonic() < deadline
                camera._condition.wait(0.01)
        assert copy.call_count == 0
        assert camera.snapshot()[1] == 9
        assert copy.call_count == 1


def test_startup_frame_is_recycled_without_resetting_sequence(capture):
    camera, hardware, _ = capture
    camera.start()
    hardware.deliver(0, status=3)
    hardware.deliver(4)
    assert camera.snapshot()[1] == 4


def test_missing_patch_fails_before_camera_acquisition(capture):
    camera, hardware, _ = capture
    del hardware.module.controls.rpi.UnicamSequence
    with pytest.raises(RuntimeError, match='rebuilt Python bindings'):
        camera.start()
    assert not hardware.acquired


def test_missing_frame_metadata_does_not_fall_back_to_isp(capture):
    camera, hardware, _ = capture
    camera.start()
    hardware.deliver(0, include_sequence=False)
    with pytest.raises(RuntimeError, match='capture failed') as error:
        camera.snapshot()
    assert 'UnicamSequence' in str(error.value.__cause__)


def test_failed_pixel_copy_still_recycles_request(capture):
    camera, hardware, _ = capture
    camera.start()
    hardware.deliver(5)
    with patch.object(camera, '_copy_bgr', side_effect=ValueError('bad buffer')):
        with pytest.raises(ValueError, match='bad buffer'):
            camera.snapshot()
    assert len(hardware.pending) == 4


def test_controls_are_sent_on_recycled_request(capture):
    camera, hardware, _ = capture
    camera.start()
    camera.set_controls({'ExposureTime': 1234})
    request = hardware.deliver(10)
    camera.snapshot()
    assert request.controls[hardware.module.controls.ExposureTime] == 1234


def test_stop_and_restart_discard_previous_generation(capture):
    camera, hardware, _ = capture
    camera.start()
    hardware.deliver(80)
    assert camera.snapshot()[1] == 80
    old_maps = list(hardware.maps)
    camera.stop()
    assert hardware.stopped and not hardware.acquired
    assert all(mapped.closed for mapped in old_maps)
    camera.start()
    hardware.deliver(0)
    assert camera.snapshot()[1] == 0


def test_timeout_is_bounded(capture):
    camera, hardware, _ = capture
    camera.timeout = 0.01
    camera.start()
    with pytest.raises(TimeoutError):
        camera.snapshot()


def test_failed_start_releases_camera_and_mappings(capture):
    camera, hardware, _ = capture
    hardware.fail_queue = True
    with pytest.raises(RuntimeError, match='injected queue failure'):
        camera.start()
    assert not hardware.acquired and hardware.stopped
    assert all(mapped.closed for mapped in hardware.maps)
    assert not camera._owns_manager
