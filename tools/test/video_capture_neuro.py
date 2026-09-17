import Roki
from Soccer.Vision.camera import Camera
from time import time
import numpy as np
import cv2

camera = Camera()
camera.start(neural=True)

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('/home/pi/Desktop/output2.avi', fourcc, 20.0, (640,  640))


mb = Roki.Motherboard()
Roki.MbDefaultConfig(mb)
rcb = Roki.Rcb4(mb)

sd1 = Roki.Rcb4.ServoData()
sd1.Id = 12
sd1.Sio = 2
sd1.Data = -2000 + 7500

sd2 = Roki.Rcb4.ServoData()
sd2.Id = 4
sd2.Sio = 1
sd2.Data = -3350 + 7500

sd3 = Roki.Rcb4.ServoData()
sd3.Id = 1
sd3.Sio = 1
sd3.Data = 250 + 7500

sd4 = Roki.Rcb4.ServoData()
sd4.Id = 4
sd4.Sio = 2
sd4.Data = 3350 + 7500

sd5 = Roki.Rcb4.ServoData()
sd5.Id = 1
sd5.Sio = 2
sd5.Data = -250 + 7500


#rcb.setServoPosAsync([sd1, sd2, sd3, sd4, sd5], 80, 79)
#rcb.setServoPosAsync([sd1], 80, 79)

def letterbox(img, size=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    w, h = size

    # Scale ratio (new / old)
    r = min(h / shape[0], w / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = w - new_unpad[0], h - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (w, h)
        ratio = w / shape[1], h / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border

    top2, bottom2, left2, right2 = 0, 0, 0, 0
    if img.shape[0] != h:
        top2 = (h - img.shape[0])//2
        bottom2 = top2
        img = cv2.copyMakeBorder(img, top2, bottom2, left2, right2, cv2.BORDER_CONSTANT, value=color)  # add border
    elif img.shape[1] != w:
        left2 = (w - img.shape[1])//2
        right2 = left2
        img = cv2.copyMakeBorder(img, top2, bottom2, left2, right2, cv2.BORDER_CONSTANT, value=color)  # add border
    return img
counter = 0
start = time()
while True:
    frame, frame_number = camera.snapshot()
    frame = letterbox(frame)
    out.write(frame)
    cv2.imshow("Output", frame)
    key = cv2.waitKey(1)
    counter +=1
    if key == ord('q'):
        break
time_elapsed = time() - start
camera.stop()
out.release()
cv2.destroyAllWindows()
print('fps = ', counter/ time_elapsed)


