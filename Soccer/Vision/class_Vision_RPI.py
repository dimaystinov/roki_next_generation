import json
import time, math, json, struct
#import reload
import numpy as np
from Soccer.Vision.camera import Camera
from Soccer.Vision.class_Vision_General import Vision_General
from Soccer.Vision.led_blink import Led
from Soccer.config_paths import init_param_read_path
import os


CAMERA_FRAME_DURATION_US = 16700

class Vision_RPI(Vision_General):
    def __init__(self, glob):
        super().__init__(glob)
        with open(init_param_read_path(self.glob.current_work_directory, "Real/Real_Thresholds.json"), "r") as f:
            self.TH = json.loads(f.read())
        self.undistortPointMap = np.load(self.glob.current_work_directory + "Soccer/Vision/undistortPointMap_x_y.npy")
        P_matrix = np.load(self.glob.current_work_directory + "Soccer/Vision/Camera_calibration_P.npy")
        self.undistort_cx, self.undistort_cy = P_matrix[0,2], P_matrix[1,2]
        self.focal_length_horizontal = P_matrix[0,0]
        self.focal_length_vertical = P_matrix[1,1]
        self.camera = self.glob.camera
        ok = self.glob.stm_channel.mb.SetIMUStrobeOffset(0)   # the best value of strobe offset is 9 for 30fps and 5 for 60fps
        ok = self.glob.stm_channel.mb.ConfigureStrobeFilter(CAMERA_FRAME_DURATION_US//1000, 4)
        ok = self.glob.stm_channel.mb.ResetStrobeContainers()
        self.camera.start(exposure = self.TH['exposure'], gain = self.TH['gain'], frame_duration_us = CAMERA_FRAME_DURATION_US, neural = self.glob.neural_vision)
        self.led = Led()
        self.camera_sleep = 0.1
        

    def snapshot(self):
        self.image, frame_number = self.camera.snapshot()
        #time.sleep(0.1)
        return_value, imu_pitch, imu_roll, imu_yaw = self.glob.stm_channel.pitch_roll_yaw_from_imu_in_head(frame_number = frame_number , degrees=False)
        if return_value:
            self.pan = - self.glob.motion.neck_pan * self.glob.motion.TIK2RAD
            self.pitch, self.roll, self.yaw = imu_pitch, imu_roll, imu_yaw
            return True, self.image, imu_pitch, imu_roll, imu_yaw, self.pan
        else: return False, None, None, None, None, None

    def display_camera_image(self, image, window = 'Vision Sensor'):
        self.glob.display.show(window, image)

    def visible_reaction_ball(self):
        self.led.blink.set()
        #os.system("espeak -ven-m1 -a100 'Ball'")
        print('I see ball')

    def undistort_points(self, column, row):
        u, v = self.undistortPointMap[column *2, row*2]
        return u, v

if __name__=="__main__":
    v = Vision_RPI(1)
    print(v.TH['orange ball'])
