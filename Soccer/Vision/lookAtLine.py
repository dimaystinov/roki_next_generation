import cv2
import time
from Soccer.Vision.led_blink import Led

def detect_aruco_markers(frame, led):
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_100)    #  словарь аруко маркеров
    parameters = cv2.aruco.DetectorParameters_create()    # Параметры детектора

    corners, ids, rejected_img_points = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=parameters)   # Обнаружение маркеров
    
    #print(ids)
    
    if ids is not None:
        # границы обнаруженных маркеров
        #frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        # перебор всех обнаруженных маркеры
        for i in range(len(ids)):
            #  координаты углов маркера
            if ids[i][0] == 88:
                led.blink.set()
                corner = corners[i][0]
                top_left = tuple(corner[0].astype(int))
                top_right = tuple(corner[1].astype(int))
                size = top_right[0] - top_left[0]
                side_shift =  400 - int((top_right[0] + top_left[0])/2)
                #print('size = ', size, 'side_shift = ', side_shift )
    else:
        size = side_shift = 0
    return frame , size, side_shift

def track_from_vision(vision, size, side_shift, stopFlag):
    led = getattr(vision, "led", None) or Led()
    count = 0
    start_time = time.perf_counter()
    while not stopFlag.value:
        frame, frame_number = vision.camera.snapshot()
        if frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        image, size.value, side_shift.value = detect_aruco_markers(gray, led)
        vision.display_camera_image(image, "Line")
        count += 1
        if count == 100:
            count = 0
            time_elapsed = time.perf_counter() - start_time
            print('Rate : ', int(100 / time_elapsed), ' FPS')
            start_time = time.perf_counter()
        if size.value > 180:
            stopFlag.value = True
            break

def track_order_from_vision(vision, turn_shift, stopFlag):
    size = 0
    side_shift = 0
    led = getattr(vision, "led", None) or Led()
    count = 0
    start_time = time.perf_counter()
    while not stopFlag.value:
        frame, frame_number = vision.camera.snapshot()
        if frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        image, size, side_shift = detect_aruco_markers(gray, led)
        vision.display_camera_image(image, "Line")
        if size > 90:
            turn_shift.value = 4
        elif side_shift > 0:
            turn_shift.value = 2
        elif side_shift < 0:
            turn_shift.value = 3
        else:
            turn_shift.value = 1
        count += 1
        if count == 100:
            count = 0
            time_elapsed = time.perf_counter() - start_time
            print('Rate : ', int(100 / time_elapsed), ' FPS')
            start_time = time.perf_counter()

def standalone_camera_loop():
    from Soccer.Vision.camera import Camera
    from libcamera import controls

    led = Led()
    camera = Camera()
    camera.start()
    camera.set_controls({"AeExposureMode": controls.AeExposureModeEnum.Short})
    count = 0
    start_time = time.perf_counter()
    try:
        while True:
            image, frame_number = camera.snapshot()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image, size, side_shift = detect_aruco_markers(image, led)
            cv2.imshow("Line", image)
            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                break
            count += 1
            if count == 100:
                count = 0
                time_elapsed = time.perf_counter() - start_time
                print('Rate : ', int(100 / time_elapsed), ' FPS')
                start_time = time.perf_counter()
            if size > 90:
                print('Reverse')
            elif side_shift > 0:
                print('Go Left')
            elif side_shift < 0:
                print('Go Right')
            else:
                print('Go Straight')
    finally:
        camera.stop()
        cv2.destroyAllWindows()

if __name__== "__main__":
    standalone_camera_loop()
