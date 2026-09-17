#!/usr/bin/env python3
"""Probe a modern OpenVINO MYRIAD plugin in this Python process, without robot hardware APIs.

This is a diagnostic, not an implementation of NCS2 support. No CPU fallback.
Use --model to test compilation/import; --inputs accepts an NPZ keyed input_0,
input_1, etc. with already preprocessed tensors matching the model exactly.
"""
import argparse
import io
import json
import platform
from pathlib import Path
import sys
import time


def probe(args, report):
    report['stage'] = 'import_openvino'
    import openvino as ov

    report['openvino'] = ov.get_version()
    core = ov.Core()
    if args.plugin:
        report['stage'] = 'register_plugin'
        core.register_plugin(str(args.plugin.resolve()), 'MYRIAD')
    report['stage'] = 'load_myriad_plugin'
    versions = core.get_versions('MYRIAD')
    report['plugins'] = {name: {'description': version.description, 'build': version.build_number}
                         for name, version in versions.items()}
    report['stage'] = 'enumerate_myriad_devices'
    report['devices'] = core.get_property('MYRIAD', 'AVAILABLE_DEVICES')
    if not report['devices']:
        raise RuntimeError('MYRIAD plugin loaded, but no NCS2 is available')
    if args.model is None:
        report.update(status='device_detected', inference_verified=False)
        return

    import numpy as np

    report['stage'] = 'compile_or_import_model'
    if args.model.suffix.lower() == '.blob':
        compiled = core.import_model(io.BytesIO(args.model.read_bytes()), 'MYRIAD', {})
    else:
        compiled = core.compile_model(core.read_model(str(args.model)), 'MYRIAD')
    report['inputs'] = [{'shape': str(port.partial_shape), 'type': str(port.element_type)}
                        for port in compiled.inputs]
    report['outputs'] = [{'shape': str(port.partial_shape), 'type': str(port.element_type)}
                         for port in compiled.outputs]
    report['stage'] = 'prepare_inputs'
    tensors = {}
    if args.inputs:
        with np.load(args.inputs, allow_pickle=False) as data:
            for i in range(len(compiled.inputs)):
                tensors[i] = np.ascontiguousarray(data[f'input_{i}'])
        report['input_source'] = 'supplied_npz'
    else:
        for i, port in enumerate(compiled.inputs):
            if port.partial_shape.is_dynamic:
                raise ValueError('Dynamic input requires explicit --inputs')
            tensors[i] = np.zeros(tuple(port.shape), dtype=port.element_type.to_dtype())
        report['input_source'] = 'synthetic_zeros'
    for i, port in enumerate(compiled.inputs):
        tensor = tensors[i]
        if tensor.dtype != np.dtype(port.element_type.to_dtype()):
            raise ValueError(f'input_{i} dtype {tensor.dtype} does not match {port.element_type}')
        if not port.partial_shape.compatible(ov.PartialShape(list(tensor.shape))):
            raise ValueError(f'input_{i} shape {tensor.shape} does not match {port.partial_shape}')
    report['stage'] = 'infer_myriad'
    request = compiled.create_infer_request()
    durations = []
    for _ in range(args.iterations):
        start = time.perf_counter()
        request.infer(tensors)
        durations.append((time.perf_counter() - start) * 1000)
        for i in range(len(compiled.outputs)):
            if not np.isfinite(request.get_output_tensor(i).data).all():
                raise RuntimeError(f'Output {i} contains NaN or infinity')
    report.update(status='inference_completed', inference_verified=True,
                  recognition_accuracy_verified=False, iterations=args.iterations,
                  first_inference_ms=durations[0], mean_inference_ms=sum(durations) / len(durations))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plugin', type=Path, help='Modern-runtime-compatible MYRIAD .so')
    parser.add_argument('--model', type=Path, help='IR .xml/ONNX to compile, or MYRIAD .blob to import')
    parser.add_argument('--inputs', type=Path, help='NPZ with preprocessed input_0, input_1, ...')
    parser.add_argument('--iterations', type=int, default=10)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error('--iterations must be positive')
    if args.inputs and args.model is None:
        parser.error('--inputs requires --model')
    report = {'python': sys.version, 'machine': platform.machine(), 'platform': platform.platform(),
              'requested_device': 'MYRIAD', 'status': 'error', 'inference_verified': False}
    try:
        probe(args, report)
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
        return 1
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
