import numpy as np
import sys
from types import SimpleNamespace as NS
from unittest.mock import Mock
from Soccer.Vision.neural import Neural, _NativeDetector, detection_center, letterbox


def test_letterbox_preserves_bgr_and_padding():
    frame = np.zeros((650,800,3),dtype=np.uint8)
    frame[:] = [7,30,201]
    image, transform = letterbox(frame,640,640)
    assert image.shape == (640,640,3)
    assert transform == (.8,.8,0,60)
    np.testing.assert_array_equal(image[320,320], [7,30,201])
    np.testing.assert_array_equal(image[0,0], [114,114,114])


def test_center_maps_back_to_camera_and_chooses_best_ball():
    predictions = np.array([[320,320,20,20,.9,.9,.1], [100,100,20,20,.5,.8,.2]])
    assert detection_center(predictions,0,(.8,.8,0,60),(650,800,3)) == (400,325)


def test_detections_in_padding_are_rejected():
    predictions = np.array([[320,20,10,10,.9,.9]])
    assert detection_center(predictions,0,(.8,.8,0,60),(650,800,3)) == (0,0)


def test_missing_class_and_nonfinite_predictions_are_rejected():
    predictions = np.array([[320,320,20,20,.9,np.nan]])
    assert detection_center(predictions,0,(.8,.8,0,60),(650,800,3)) == (0,0)
    assert detection_center(predictions,1,(.8,.8,0,60),(650,800,3)) == (0,0)


def test_missing_model_disables_detector_with_explicit_reason(tmp_path):
    neural = Neural(model_path=tmp_path/'missing.blob')
    assert not neural.is_ready()
    assert 'ROKI_NCS2_BLOB' in neural.error
    assert neural.ball_detect_single(np.zeros((10,10,3),np.uint8)) == (0,0)
    neural.close()


def test_odd_frame_dimensions_produce_exact_model_size():
    image, (sx,sy,left,top) = letterbox(np.zeros((333,777,3),np.uint8),640,640)
    assert image.shape == (640,640,3)
    assert abs(sx*777-640) < 1e-6
    assert int(round(sy*333))+2*top in (640,639)


def test_native_detector_reuses_request_and_stops_after_backend_error(tmp_path, monkeypatch):
    path = tmp_path/'ball.blob'
    path.write_bytes(b'test-only-model')
    input_port = NS(element_type='u8',shape=[1,640,640,3])
    output_port = NS(element_type='f16',shape=[1,1,7])
    request = NS(infer=Mock(), get_output_tensor=lambda: NS(
        data=np.array([[[320,320,20,20,.9,.9,.1]]],dtype=np.float16)))
    compiled = NS(inputs=[input_port],outputs=[output_port],input=lambda: input_port,
                  output=lambda: output_port,create_infer_request=Mock(return_value=request))
    core = NS(register_plugin=Mock(),import_model=Mock(return_value=compiled))
    monkeypatch.setitem(sys.modules,'openvino',NS(Core=lambda: core,Type=NS(u8='u8',f16='f16',f32='f32')))
    neural = _NativeDetector(model_path=path,plugin_path='/test/plugin.so')
    assert neural.is_ready()
    frame = np.zeros((650,800,3),np.uint8)
    frame[:] = [7,30,201]
    for _ in range(2):
        assert neural.ball_detect_single(frame) == (400,325)
    compiled.create_infer_request.assert_called_once()
    batch = request.infer.call_args.args[0][0]
    assert batch.shape == (1,640,640,3) and batch.dtype == np.uint8
    np.testing.assert_array_equal(batch[0,320,320],[7,30,201])
    assert core.import_model.call_args.args[1] == 'MYRIAD'
    request.infer.side_effect = RuntimeError('USB disconnected')
    assert neural.ball_detect_single(frame) == (0,0)
    assert not neural.is_ready() and neural.error == 'USB disconnected'
    calls = request.infer.call_count
    assert neural.ball_detect_single(frame) == (0,0)
    assert request.infer.call_count == calls
    neural.close()
    assert neural._request is None
