"""Private bounded control messages; pixels live in a single shared mmap."""
import json
import time

BUFFER_BYTES = 16 * 1024 * 1024
MAX_MESSAGE = 4096


def send(sock, message, timeout):
    data = json.dumps(message).encode() + b'\n'
    if len(data) > MAX_MESSAGE:
        raise ValueError('NCS2 control message too large')
    sock.settimeout(timeout)
    sock.sendall(data)


def receive(sock, timeout):
    deadline = None if timeout is None else time.monotonic() + timeout
    data = bytearray()
    while len(data) < MAX_MESSAGE:
        remaining = None if deadline is None else deadline - time.monotonic()
        if remaining is not None and remaining <= 0:
            raise TimeoutError('NCS2 response deadline exceeded')
        sock.settimeout(remaining)
        part = sock.recv(MAX_MESSAGE - len(data))
        if not part:
            raise EOFError('NCS2 process closed its connection')
        data.extend(part)
        if data.endswith(b'\n'):
            reply = json.loads(data)
            if not isinstance(reply, dict):
                raise ValueError('Invalid NCS2 control message')
            return reply
    raise ValueError('NCS2 control message too large')
