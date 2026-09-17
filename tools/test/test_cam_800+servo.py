import cv2
import time, math, json, struct
#import reload
from scipy.spatial.transform import Rotation as R
import numpy as np
import Roki
#from camera import Camera
#from imu_in_head import Imu_in_head
#from class_stm_channel import STM_channel
import threading
import sys

sys.path.append('/home/pi/Desktop/Roki_2_Soccer')

import Soccer.Vision.reload as reload
from Soccer.Vision.class_Vision_RPI import Vision_RPI as Vision
from Soccer.Localisation.class_Glob import Glob
from Soccer.Vision.reload import Image
from Soccer.Motion.class_Motion_real import Motion_real

from Soccer.Motion.class_stm_channel import STM_channel

with open("/home/pi/Desktop/Init_params/Real/Real_Thresholds.json", "r") as f:
    threshold_Dict = json.loads(f.read())
    
undistortPointMap = np.load("/home/pi/Desktop/Roki_2_Soccer/Soccer/Vision/undistortPointMap_x_y.npy")
P_matrix = np.load("/home/pi/Desktop/Roki_2_Soccer/Soccer/Vision/Camera_calibration_P.npy")
focal_length_horizontal = P_matrix[0,0] #243.54 /2
focal_length_vertical = P_matrix[1,1] #219.9 / 2
ball_diameter = 80
camera_elevation = 490
    
def image_point_to_relative_coord_on_floor(column, row, camera_pitch, camera_roll, camera_elevation, for_ball = False):
    # camera roll = - imu roll
    if for_ball: camera_elevation -=  ball_diameter / 2
    u, v = undistortPointMap[column *2, row*2]
    #u = int(u/2)
    #v = int(v/2)
    width, height ,_ = undistortPointMap.shape
    #print('width, height :', width, height)
    cx = width / 2
    cy = height / 2
    point_angle_horizontal = (cx -u)/ focal_length_horizontal        # argument of math.atan
    point_angle_vertical = (cy -v)/ focal_length_vertical            # argument of math.atan
    point_vector = [1, point_angle_horizontal, point_angle_vertical]    # arguments of math.tan
    pitch_rotation = R.from_euler('y', [camera_pitch], degrees=False)
    roll_rotation = R.from_euler('x', [camera_roll], degrees=False)
    point_vector = pitch_rotation.apply(point_vector)
    point_vector = roll_rotation.apply(point_vector)
    if point_vector[0,2] >= 0: return None
    relative_x_on_floor = point_vector[0,0] * camera_elevation / (- point_vector[0,2])
    relative_y_on_floor = point_vector[0,1] * camera_elevation / (- point_vector[0,2])
    return relative_x_on_floor, relative_y_on_floor, v

frame_duration_us = 16700
# camera = Camera()
# stm_channel = STM_channel()
# stm_channel.mb.SetIMUStrobeOffset(3)   # the best value of strobe offset is 9 for 30fps and 5 for 60fps
# stm_channel.mb.ConfigureStrobeFilter(frame_duration_us//1000, 4)
# stm_channel.mb.ResetStrobeContainers()
# camera.start(frame_duration_us=frame_duration_us)
#time.sleep(1)

glob = Glob(5, '/home/pi/Desktop/Roki_2_Soccer/')
vision = Vision(glob)
motion = Motion_real(glob, vision)

sd = Roki.Rcb4.ServoData()
sd.Id = 12
sd.Sio = 2
# for i in range(2000):
#     sd.Data = 7500 + int(500 * math.sin(0.05 * i))
#     stm_channel.rcb.setServoPosAsync([sd], int(2))
glob.stm_channel.rcb.motionPlay(25)
a = 0
filtered = 0
distances = []
distances_filtered = []
vs = []
pitches = []
orders = []
encoders = []
amplitude = 0.4
start_time = time.perf_counter()
cycles = 100
for i in range(cycles):
    angle = amplitude * i
    angle = angle % (2 * math.pi)
    sd.Data = 5500 + int(500 * math.sin(angle))
    glob.stm_channel.rcb.setServoPosAsync([sd], int(5), 4)
    result, sensor_bytes = glob.stm_channel.rcb.moveRamToComCmdSynchronize(0x00a2, 2)
    sensor = struct.unpack('<h', bytes(sensor_bytes))

    camera_result, cv2_image, pitch, roll, yaw, pan = vision.snapshot()
    color_image = reload.Image(cv2_image)
    ball_column, ball_row = 0, 0
    for blob in color_image.find_blobs([threshold_Dict['orange ball']['th']], #20, 20):
                                       pixels_threshold = threshold_Dict['orange ball']['pixel'],
                                       area_threshold = threshold_Dict['orange ball']['area']):
        color_image.draw_rectangle(blob.rect())
        if blob.cy() > ball_row:
            ball_row = blob.cy()
            ball_column = blob.cx()
    cv2.imshow("Camera", color_image.img)
    cv2.waitKey(10)

    start0 = time.perf_counter()
    if ball_column != 0 and ball_row !=0:
        relative_x_on_floor, relative_y_on_floor, v = image_point_to_relative_coord_on_floor(ball_column, ball_row,
                                                    pitch, roll, camera_elevation, for_ball = True)
        #print('relative_x_on_floor = ', relative_x_on_floor, 'camera_pitch = ', camera_pitch, 'ball_row = ', ball_row, 'v = ', v )
        #print('relative_x_on_floor = ', relative_x_on_floor, 'relative_y_on_floor = ', relative_y_on_floor, 'imu_pitch = ', imu_pitch, 'imu_roll =', imu_roll)
        #print('time_calac = ', time.perf_counter() - start0)
        filtered = filtered * 0.80 + relative_x_on_floor * 0.20
        distances_filtered.append(filtered)
        distances.append(relative_x_on_floor)
        vs.append(-(v -780) * 7)
        pitches.append(pitch * 1000)
        orders.append(sd.Data - 7500)
        encoders.append(sensor)
#     im = cv2.resize(color_image.img, [800, 650])
#     cv2.imshow("Camera", im)
#     cv2.waitKey(10)
time_elapsed = time.perf_counter() - start_time
print('time elapsed :', time_elapsed)
print('Rate : ', int(cycles/ time_elapsed), ' FPS')
glob.stm_channel.mb.ResetBodyQueue()
cv2.destroyAllWindows()
vision.camera.stop()
