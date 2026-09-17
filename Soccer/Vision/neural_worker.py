"""Private child entry point. Uses the same interpreter as the robot."""
import os
import sys


def main():
    import socket
    import threading
    from .neural_ipc import BUFFER_BYTES, receive, send
    sock_fd, memory_fd, parent_pid = map(int, sys.argv[1:4])
    # Watch parent lifetime independently of inference.
    # Linux additionally delivers SIGKILL even when native code holds the GIL.
    if sys.platform.startswith('linux'):
        import ctypes
        import signal
        if ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), 'Cannot set NCS2 parent-death signal')
    if os.getppid() != parent_pid:
        return
    def watch_parent():
        import time
        while os.getppid() == parent_pid:
            time.sleep(.5)
        os._exit(0)
    threading.Thread(target=watch_parent, daemon=True).start()
    sock = socket.socket(fileno=sock_fd)
    try:
        import mmap
        import numpy as np
        import cv2  # Include OpenCV loading in the startup deadline, not the first frame.
        from .neural import _NativeDetector
        memory = mmap.mmap(memory_fd, BUFFER_BYTES, access=mmap.ACCESS_READ)
        os.close(memory_fd)
        detector = _NativeDetector(model_path=sys.argv[4], plugin_path=sys.argv[5])
        if not detector.is_ready():
            send(sock, {'status': 'error', 'error': detector.error[:2000]}, 1)
            return
        send(sock, {'status': 'ready'}, 1)
        while True:
            request = receive(sock, None)
            shape = request['shape']
            if (len(shape) != 3 or shape[2] != 3 or
                    any(type(n) is not int or n <= 0 for n in shape) or
                    shape[0] * shape[1] * 3 > BUFFER_BYTES):
                raise ValueError('Invalid shared frame dimensions')
            frame = np.ndarray(shape, dtype=np.uint8, buffer=memory)
            center = detector.object_detect_single(frame, request['object'])
            if not detector.is_ready():
                send(sock, {'id': request['id'], 'status': 'error', 'error': detector.error[:2000]}, 1)
                return
            send(sock, {'id': request['id'], 'status': 'ok', 'center': center}, 1)
    except EOFError:
        return
    except Exception as exc:
        try:
            send(sock, {'status': 'error', 'error': str(exc)[:2000]}, 1)
        except Exception:
            pass


if __name__ == '__main__':
    try:
        main()
    finally:
        # Native USB cleanup may hang too. The OS owns final reclamation here.
        os._exit(0)
