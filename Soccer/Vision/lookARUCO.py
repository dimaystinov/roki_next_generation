import cv2
import time
import numpy as np
from Soccer.Vision.led_blink import Led
import math
import json

undistortPointMap = np.load("Soccer/Vision/undistortPointMap_x_y.npy")
P_matrix = np.load("Soccer/Vision/Camera_calibration_P.npy")
undistort_cx, undistort_cy = P_matrix[0,2], P_matrix[1,2]
focal_length_horizontal = P_matrix[0,0]
with open("Soccer/Vision/calibration_matrix.json", "r") as f:
    data = json.load(f)
mtx = np.asarray(data['camera_matrix'])
dist = np.asarray(data['dist_coeff'])
markerSizeInCM = 16

def detect_aruco_markers(frame, led, ID):
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_100)    #  словарь аруко маркеров
    parameters = cv2.aruco.DetectorParameters_create()    # Параметры детектора

    corners_raw, ids_raw, rejected_img_points = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=parameters)   # Обнаружение маркеров
    corners = []
    ids = None
    if ids_raw is not None:
        for i in range(len(ids_raw)):
            if ids_raw[i][0] == ID:
                ids = np.array([ID] ,dtype= np.int32)
                corners.append(corners_raw[i])
    if ids is not None:
        rvec , tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners, markerSizeInCM, mtx, dist)
        try:
            distance = tvec[0][0][2]
        except Exception: distance = 0
        #print(ids)
        # границы обнаруженных маркеров
        frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        # перебор всех обнаруженных маркеры
        #  координаты углов маркера
        led.blink.set()
        #os.system("espeak -ven-m1 -a100 'Yes'")
        corner = corners[0][0]
        top_left = tuple(corner[0].astype(int))
        top_right = tuple(corner[1].astype(int))
        bottom_left = tuple(corner[2].astype(int))
        
        tr_x, tr_y = undistortPointMap[top_right[0] * 2, top_right[1] * 2]
        tl_x, tl_y = undistortPointMap[top_left[0] * 2, top_left[1] * 2]
        size = tr_x - tl_x
        cx = int((top_right[0] + top_left[0])/2)
        cy = int((top_left[1] + bottom_left[1])/2)
        side_shift =  400 - int((top_right[0] + top_left[0])/2)
        u, v = undistortPointMap[cx * 2, cy * 2]
        aruco_angle_horizontal = math.atan((undistort_cx -u)/ focal_length_horizontal)
        #print('size = ', size, 'side_shift = ', side_shift )

    else:
        size = side_shift = aruco_angle_horizontal = distance = 0
    return frame , size, side_shift, aruco_angle_horizontal, distance

def track_from_vision(vision, size, side_shift, aruco_angle_horizontal, distance, stopFlag, ID):
    led = getattr(vision, "led", None) or Led()
    count = 0
    start_time = time.perf_counter()
    while not stopFlag.value:
        frame, frame_number = vision.camera.snapshot()
        if frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        image, size.value, side_shift.value, aruco_angle_horizontal.value, distance.value = detect_aruco_markers(gray, led, ID)
        vision.display_camera_image(image, "ARUCO")
        count += 1
        if count == 100:
            count = 0
            time_elapsed = time.perf_counter() - start_time
            print('Rate : ', int(100 / time_elapsed), ' FPS')
            start_time = time.perf_counter()
        if size.value > 180:
            print('exit')
            stopFlag.value = True
            break

def standalone_camera_loop(ID=88):
    from Soccer.Vision.camera import Camera
    from libcamera import controls

    led = Led()
    camera = Camera()
    camera.start(exposure=500, gain=8.0)
    count = 0
    start_time = time.perf_counter()
    try:
        while True:
            image, frame_number = camera.snapshot()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image, size, side_shift, aruco_angle_horizontal, distance = detect_aruco_markers(image, led, ID)
            cv2.imshow("ARUCO", image)
            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                break
            count += 1
            if count == 100:
                count = 0
                time_elapsed = time.perf_counter() - start_time
                print('Rate : ', int(100 / time_elapsed), ' FPS')
                start_time = time.perf_counter()
            print("aruco_angle_horizontal: ", aruco_angle_horizontal)
            print('distance :', distance)
            print('size :', size)
            if size > 150:
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

def evaluate_distance(corners):
    markerSizeInCM = 16
    rvec , tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners, markerSizeInCM, mtx, dist)
    return tvec[2]

if __name__== "__main__":
    standalone_camera_loop()
