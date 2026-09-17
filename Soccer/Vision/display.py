import os


class NullDisplay:
    backend_name = "null"

    def show(self, name, frame):
        return False

    def close(self):
        pass


class OpenCvDisplay:
    backend_name = "imshow"

    def __init__(self, wait_key_ms=1):
        self.wait_key_ms = wait_key_ms

    def show(self, name, frame):
        import cv2

        if frame is None:
            return False
        cv2.imshow(str(name), frame)
        cv2.waitKey(self.wait_key_ms)
        return True

    def close(self):
        import cv2

        cv2.destroyAllWindows()


def create_display(params, simulation):
    backend = os.environ.get("ROKI_DISPLAY_BACKEND", params.get("DISPLAY_BACKEND", ""))
    backend = str(backend).strip().lower()
    if not backend:
        backend = "null" if simulation == 5 else "imshow"
    if backend == "imshow":
        return OpenCvDisplay(wait_key_ms=int(params.get("DISPLAY_WAIT_KEY_MS", 1)))
    return NullDisplay()
