"""Run the real supervisor/IPC with a fake hardware API in a real child Python."""
import os
import time
import numpy as np
import pytest
from Soccer.Vision.neural import Neural


@pytest.fixture
def fake_runtime(tmp_path, monkeypatch):
    # USB/native failures cannot be reproduced with hardware on the CI host.
    (tmp_path / 'openvino.py').write_text('''
import os, time
import numpy as np
from types import SimpleNamespace as NS
Type = NS(u8='u8', f16='f16', f32='f32')
class Core:
    def __init__(self, config=None): pass
    def get_versions(self, name): return {}
    def register_plugin(self, *args): pass
    def import_model(self, stream, *args):
        mode = stream.read().decode()
        if mode == 'boot_hang': time.sleep(60)
        if mode == 'boot_crash': os._exit(91)
        if mode == 'missing': raise RuntimeError('USB device unavailable')
        return Compiled(mode)
class Compiled:
    inputs = [NS(element_type='u8', shape=[1,8,8,3])]
    outputs = [NS(element_type='f32', shape=[1,1,6])]
    def __init__(self, mode): self.mode = mode
    def input(self): return self.inputs[0]
    def output(self): return self.outputs[0]
    def create_infer_request(self): return Request(self.mode)
class Request:
    def __init__(self, mode): self.mode = mode
    def infer(self, inputs):
        if self.mode == 'hang': time.sleep(60)
        if self.mode == 'crash': os._exit(92)
        if self.mode == 'error': raise RuntimeError('USB disconnected')
        self.x = int(inputs[0][0,0,0,0])
    def get_output_tensor(self):
        return NS(data=np.array([[[self.x,4,2,2,.9,.9]]], dtype=np.float32))
''')
    monkeypatch.setenv('PYTHONPATH', str(tmp_path) + os.pathsep + os.environ.get('PYTHONPATH', ''))
    def create(mode, **kwargs):
        path = tmp_path / (mode + '.blob')
        path.write_text(mode)
        return Neural(model_path=path, inference_timeout=.3, startup_timeout=5, **kwargs)
    return create


def test_success_returns_current_frame_and_close_reaps_child(fake_runtime):
    detector = fake_runtime('ok')
    try:
        assert detector.is_ready(), detector.error
        child = detector._process
        for x in (2, 6, 3):
            assert detector.ball_detect_single(np.full((8,8,3), x, np.uint8)) == (x,4)
    finally:
        detector.close()
    assert child.poll() is not None
    detector.close()


@pytest.mark.parametrize('mode', ['error', 'hang', 'crash'])
def test_inference_fault_is_bounded_and_does_not_retry(fake_runtime, mode):
    detector = fake_runtime(mode)
    try:
        assert detector.is_ready(), detector.error
        child = detector._process
        start = time.monotonic()
        assert detector.ball_detect_single(np.zeros((8,8,3), np.uint8)) == (0,0)
        assert time.monotonic() - start < 1.5
        assert not detector.is_ready() and detector.error
        assert child.poll() is not None
        start = time.monotonic()
        for _ in range(100):
            assert detector.ball_detect_single(np.zeros((8,8,3), np.uint8)) == (0,0)
        assert time.monotonic() - start < .2
    finally:
        detector.close()


@pytest.mark.parametrize('mode', ['boot_hang', 'boot_crash', 'missing'])
def test_startup_fault_does_not_escape_constructor(fake_runtime, mode):
    start = time.monotonic()
    detector = fake_runtime(mode)
    try:
        assert time.monotonic() - start < 6
        assert not detector.is_ready() and detector.error
        assert detector.ball_detect_single(np.zeros((8,8,3), np.uint8)) == (0,0)
    finally:
        detector.close()


def test_idle_child_death_detected_before_strategy_selects_neural(fake_runtime):
    detector = fake_runtime('ok')
    try:
        assert detector.is_ready(), detector.error
        detector._process.kill()
        detector._process.wait(timeout=2)
        assert not detector.is_ready()
        assert detector.error
    finally:
        detector.close()


@pytest.mark.parametrize('method', ['seek_Ball_In_Frame_N', 'detect_Ball_Speed_N'])
def test_vision_switches_to_colour_in_the_failing_call(fake_runtime, method):
    from types import SimpleNamespace as NS
    from Soccer.Vision.class_Vision_General import Vision_General
    detector = fake_runtime('error')
    vision = Vision_General.__new__(Vision_General)
    vision.glob = NS(neural=detector, event_type='FIRA')
    vision.snapshot = lambda: (True, np.zeros((8,8,3), np.uint8), 0,0,0,0)
    vision.seek_Ball_In_Frame = lambda with_Localization: (True,.12,1.5,'colour blob')
    vision.detect_Ball_Speed = lambda: (True,.12,1.5,[.1,.2])
    try:
        if method == 'seek_Ball_In_Frame_N':
            assert vision.seek_Ball_In_Frame_N(False) == (True,.12,1.5)
        else:
            assert vision.detect_Ball_Speed_N() == (True,.12,1.5,[.1,.2])
    finally:
        detector.close()


def test_explicit_reenable_closes_previous_child(fake_runtime, monkeypatch):
    from Soccer.Localisation.class_Glob import Glob
    import Soccer.Vision.neural as module
    old, new = fake_runtime('ok'), fake_runtime('ok')
    glob = Glob.__new__(Glob)
    glob.role, glob.display, glob.neural = 'forward', None, old
    child = old._process
    monkeypatch.setattr(module, 'Neural', lambda *args: new)
    try:
        glob.neural_vision_enable()
        assert child.poll() is not None
        assert glob.neural.is_ready()
    finally:
        old.close()
        new.close()
