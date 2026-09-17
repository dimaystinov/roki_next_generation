#!/usr/bin/env python3
"""Camera-only smoke check on CM4. Does not open Roki or command servos."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from Soccer.Vision.camera import Camera


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames', type=int, default=100)
    parser.add_argument('--restarts', type=int, default=2)
    parser.add_argument('--frame-duration-us', type=int, default=16700)
    args = parser.parse_args()
    if args.frames < 1 or args.restarts < 1 or args.frame_duration_us < 1:
        parser.error('frames, restarts and frame-duration-us must be positive')
    camera = Camera()
    for generation in range(args.restarts):
        previous = None
        try:
            camera.start(frame_duration_us=args.frame_duration_us, neural=True)
            started = time.monotonic()
            for i in range(args.frames):
                image, sequence = camera.snapshot()
                if image.shape != (650, 800, 3) or str(image.dtype) != 'uint8':
                    raise RuntimeError(f'Unexpected image: {image.shape}, {image.dtype}')
                delta = None if previous is None else (sequence - previous) & 0xffffffff
                if delta == 0:
                    raise RuntimeError('Duplicate Unicam sequence')
                print(json.dumps({'generation': generation, 'sequence': sequence,
                                  'delta': delta, 'shape': image.shape,
                                  'sensor_timestamp': camera.last_metadata.get('SensorTimestamp')}))
                previous = sequence
                if i == args.frames // 2:
                    # Force a slow consumer; IDs must keep hardware gaps.
                    time.sleep(0.25)
            print(json.dumps({'generation': generation, 'frames': args.frames,
                              'elapsed_s': time.monotonic() - started}))
        finally:
            camera.stop()


if __name__ == '__main__':
    main()
