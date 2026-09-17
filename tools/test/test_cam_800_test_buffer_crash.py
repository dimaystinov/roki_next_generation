import cv2
import time, math, json
#import reload
from scipy.spatial.transform import Rotation as R
import numpy as np

import sys

sys.path.append('/home/pi/Desktop/Roki_2_Soccer')

import Soccer.Vision.reload as reload
from Soccer.Vision.class_Vision_RPI import Vision_RPI as Vision
from Soccer.Localisation.class_Glob import Glob
from Soccer.Vision.reload import Image
from Soccer.Motion.class_Motion_real import Motion_real

#from camera import Camera
#from imu_in_head import Imu_in_head
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
#camera = Camera()

# stm_channel = STM_channel()
# stm_channel.mb.SetIMUStrobeOffset(5)   # the best value of strobe offset is 9 for 30fps and 5 for 60fps
# stm_channel.mb.ConfigureStrobeFilter(frame_duration_us//1000, 4)
# stm_channel.mb.ResetStrobeContainers()
# camera.start(frame_duration_us=frame_duration_us)

glob = Glob(5, '/home/pi/Desktop/Roki_2_Soccer/')
vision = Vision(glob)
motion = Motion_real(glob, vision)

a = 0
start_time = time.perf_counter()
cycles = 100
while a < cycles:
#     # get image and its number
#     im, frame_number, total_number_of_frames, frame_timestamp, frame_duration= camera.snapshot()
#     # get quaternion and Euler angles
#     #imu_pitch, imu_roll, imu_yaw = imu_in_head.pitch_roll_yaw(degrees=False)
#     #print( 'frame_number = ', frame_number)
#     imu_pitch, imu_roll, imu_yaw = stm_channel.pitch_roll_yaw_from_imu_in_head(frame_number = None, degrees=False)
#     #print('\r', 'frame_number = ', frame_number, 'yaw = ', imu_yaw, 'pitch = ', imu_pitch, 'roll = ', imu_roll, end='')
#     camera_pitch = imu_pitch - math.pi/2 - 0.016  # calibration of horizon
#     camera_roll = - imu_roll
    
    camera_result, cv2_image, pitch, roll, yaw, pan = vision.snapshot()
    
    color_image = reload.Image(cv2_image)
    ball_column, ball_row = 0, 0
    blobs = color_image.find_blobs([threshold_Dict['orange ball']['th']],# 20, 20):
                                       pixels_threshold = threshold_Dict['orange ball']['pixel'],
                                       area_threshold = threshold_Dict['orange ball']['area'])
    len_blobs = len(blobs)
    height = 650 # camera vertical resolution
    order = []
    for i in range(len_blobs):
        order.append([height - blobs[i].cy(), i,0,0])
    sorted_order = sorted(order)
    new_order_len = min(10, len_blobs)
    new_order = sorted_order[:new_order_len]
    for i in range(new_order_len):
        ball_column = blobs[new_order[i][1]].cx()
        ball_row = blobs[new_order[i][1]].cy()
        relative_x_on_floor, relative_y_on_floor, v = image_point_to_relative_coord_on_floor(ball_column, ball_row,
                                                    pitch, roll, camera_elevation, for_ball = True)
        new_order[i][0] = int(math.sqrt(relative_x_on_floor *relative_x_on_floor + relative_y_on_floor * relative_y_on_floor))
        new_order[i][2] = int(relative_x_on_floor)
        new_order[i][3] = int(relative_y_on_floor)
    sorted_new_order = sorted(new_order)
    if len_blobs != 0:
        blob = blobs[sorted_new_order[0][1]]
        color_image.draw_rectangle(blob.rect())

        print('a:', a, 'blobs: ', len_blobs, 'relative_x_on_floor = ', sorted_new_order[0][2], 'relative_y_on_floor = ', sorted_new_order[0][3],
               'imu_pitch = ', round(pitch, 2), 'imu_roll =', round(roll, 2))
            #print('time_calac = ', time.perf_counter() - start0)
    a += 1
    im = cv2.resize(color_image.img, [800, 650])
    cv2.imshow("Camera", im)
    cv2.waitKey(10)
time_elapsed = time.perf_counter() - start_time
print('time elapsed :', time_elapsed)
print('Rate : ', int(cycles/ time_elapsed), ' FPS')
cv2.destroyAllWindows()
vision.camera.stop()
