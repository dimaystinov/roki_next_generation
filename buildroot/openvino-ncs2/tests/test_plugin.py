"""Integration against real OpenVINO/Python and the explicitly fake USB library.

Pass the test plugin path. This does not test hardware inference accuracy.
"""
import io
import struct
import sys

import numpy as np
import openvino as ov


def blob():
    b = bytearray(320)
    def put(pos, value):
        struct.pack_into('<I', b, pos, value)
    for pos, value in {52:9709, 56:320, 60:6, 68:1, 72:1, 80:8, 84:8,
                       116:132, 120:192, 128:252}.items():
        put(pos, value)
    for start, name in [(132, b'input'), (192, b'output')]:
        put(start+8,16)
        b[start+12:start+12+len(name)] = name
        for offset, value in {28:1, 32:0x21, 36:2, 40:3, 44:0, 48:3, 52:8}.items():
            put(start+offset, value)
    for pos, value in {252:3,256:2,260:1,264:4}.items():
        put(pos,value)
    return bytes(b)


def expect_error(call, text):
    try:
        call()
    except RuntimeError as exc:
        assert text in str(exc), str(exc)
    else:
        raise AssertionError('Expected error: ' + text)


def main():
    core = ov.Core()
    core.register_plugin(sys.argv[1], 'MYRIAD')
    assert core.get_property('MYRIAD','AVAILABLE_DEVICES') == ['TEST-NOT-A-USB-DEVICE']
    compiled = core.import_model(io.BytesIO(blob()), 'MYRIAD', {})
    assert list(compiled.input().shape) == [2,3]
    assert compiled.input().element_type == ov.Type.u8
    assert compiled.input().any_name == 'input'
    assert compiled.output().any_name == 'output'
    assert compiled.get_property('EXECUTION_DEVICES') == ['MYRIAD']
    request = compiled.create_infer_request()
    frame = np.arange(6,dtype=np.uint8).reshape(2,3)
    np.testing.assert_array_equal(request.infer({0:frame})[0],frame)
    # Physical padding round trip, alternating data and requests to expose stale buffers.
    second = compiled.create_infer_request()
    for value in range(20):
        supplied = frame+value
        req = request if value%2 else second
        req.start_async({0:supplied}); req.wait()
        np.testing.assert_array_equal(req.get_output_tensor().data,supplied)
    queue = ov.AsyncInferQueue(compiled, 4)
    seen = {}
    def done(req, number):
        seen[number] = req.get_output_tensor().data.copy()
    queue.set_callback(done)
    for value in range(20): queue.start_async({0:frame+value}, value)
    queue.wait_all()
    for value in range(20): np.testing.assert_array_equal(seen[value],frame+value)
    exported = compiled.export_model()
    reloaded = core.import_model(exported,'MYRIAD',{})
    np.testing.assert_array_equal(reloaded({0:frame})[0],frame)
    # The graph is opaque: unsupported compilation must not silently fall back to CPU.
    param = ov.opset13.parameter([2,3], np.uint8)
    model = ov.Model([ov.opset13.relu(param)], [param])
    expect_error(lambda: core.compile_model(model,'MYRIAD'), '.blob only')
    expect_error(lambda: core.import_model(io.BytesIO(b'bad'),'MYRIAD',{}), 'invalid blob length')
    expect_error(lambda: core.set_property('MYRIAD',{'UNKNOWN':1}), 'unsupported property')
    timeout = np.full((2,3),255,dtype=np.uint8)
    expect_error(lambda: request.infer({0:timeout}), 'mvnc status -6')
    expect_error(lambda: request.infer({0:frame}), 'reload model')
    expect_error(lambda: second.infer({0:frame}), 'reload model')
    print('PASS: OpenVINO',ov.get_version(),'Python',sys.version.split()[0],
          'import/export, padded tensors, sync/async, request isolation, error recovery; FAKE USB ONLY')


if __name__ == '__main__':
    main()
