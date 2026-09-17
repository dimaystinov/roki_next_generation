import sys
import os
import math
import json
import time
import Soccer.utility
import threading
import random
#from Soccer.Motion.class_stm_channel import STM_channel
#from Soccer.Vision.class_Vision_RPI import Vision_RPI
from multiprocessing import Value
from Robots.roki2met import roki2met
import datetime
import numpy as np
from ctypes import c_bool
from Soccer.config_paths import init_param_read_path, init_param_write_path
#from Soccer.Motion.Soccer_monitor import launcher
#from Soccer.Localisation.class_Glob import monitor
import datetime


def coord2yaw(x, y):
    if x == 0:
        if y > 0 : yaw = math.pi/2
        else: yaw = -math.pi/2
    else: yaw = math.atan(y/x)
    if x < 0:
        if yaw > 0: yaw -= math.pi
        else: yaw += math.pi
    return yaw

class GoalKeeper:
    def __init__(self, motion, local, glob):
        self.motion = motion
        self.local = local
        self.glob = glob
        self.direction_To_Guest = 0

    def turn_Face_To_Guest(self):
        if self.local.coord_odometry[0] < 0:
            self.motion.turn_To_Course(0)
            self.direction_To_Guest = 0
            return
        elif self.local.coord_odometry[0] > 0.8 and abs(self.local.coord_odometry[1]) > 0.6:
            self.direction_To_Guest = math.atan(-self.local.coord_odometry[1]/(1.8-self.local.coord_odometry[0]))
            self.motion.turn_To_Course(self.direction_To_Guest)
        elif self.local.coord_odometry[0] < 1.5 and abs(self.local.coord_odometry[1]) < 0.25:
            if (1.8-self.local.ball_odometry[0]) == 0: self.direction_To_Guest = 0
            else: self.direction_To_Guest = math.atan((0.4* (round(random.random(),0)*2 - 1)-
                                                       self.local.ball_odometry[1])/(1.8-self.local.ball_odometry[0]))
            self.motion.turn_To_Course(self.direction_To_Guest)
            return
        else:
            self.direction_To_Guest = math.atan(-self.local.coord_odometry[1]/(2.8-self.local.coord_odometry[0]))
            self.motion.turn_To_Course(self.direction_To_Guest)

    def goto_Center(self):                      #Function for reterning to center position
        print('Function for reterning to center position')
        player_X_m = self.local.coord_odometry[0]
        player_Y_m = self.local.coord_odometry[1]
        duty_position_x = - self.glob.landmarks['FIELD_LENGTH']/2 + 0.4
        distance_to_target = math.sqrt((duty_position_x -player_X_m)**2 + (0 - player_Y_m)**2 )
        if distance_to_target > 0.5 :
            target_in_front_of_duty_position = [duty_position_x + 0.15, 0]
            if distance_to_target > 1: stop_Over = True
            else: stop_Over = False
            self.motion.far_distance_plan_approach(target_in_front_of_duty_position, self.direction_To_Guest, stop_Over = stop_Over)
        else:
            if (duty_position_x -player_X_m)==0:
                alpha = math.copysign(math.pi/2, (0 - player_Y_m) )
            else:
                if (duty_position_x - player_X_m)> 0: alpha = math.atan((0 - player_Y_m)/(duty_position_x -player_X_m))
                else: alpha = math.atan((0 - player_Y_m)/(duty_position_x - player_X_m)) + math.pi
            napravl = alpha - self.motion.imu_body_yaw()
            dist_mm = distance_to_target * 1000
            self.motion.near_distance_omni_motion(dist_mm, napravl)
        self.turn_Face_To_Guest()

    def ball_Speed_Dangerous(self):
        pass
    def fall_to_Defence(self):
        print('fall to defence')
    def get_Up_from_defence(self):
        print('up from defence')
    def scenario_A1(self, dist, napravl):#The robot knock out the ball to the side of the enemy
        print('The robot knock out the ball to the side of the enemy')
        for i in range(10):
            if dist > 0.5 :
                if dist > 1: stop_Over = True
                else: stop_Over = False
                self.motion.far_distance_plan_approach(self.local.ball_odometry, self.direction_To_Guest, stop_Over = stop_Over)
            self.turn_Face_To_Guest()
            success_Code = self.motion.near_distance_ball_approach_and_kick(self.direction_To_Guest)
            if success_Code == False and self.motion.falling_Flag != 0: return
            if success_Code == False : break
            #success_Code, napravl, dist, speed = self.motion.seek_Ball_In_Pose(fast_Reaction_On = False)
            dist = self.glob.ball_distance
            if dist > 1 : break
        target_course1 = self.local.coord_odometry[2] +math.pi
        self.motion.turn_To_Course(target_course1)
        self.goto_Center()

    def scenario_A2(self, dist, napravl):#The robot knock out the ball to the side of the enemy
        print('The robot knock out the ball to the side of the enemy')
        self.scenario_A1( dist, napravl)

    def scenario_A3(self, dist, napravl):#The robot knock out the ball to the side of the enemy
        print('The robot knock out the ball to the side of the enemy')
        self.scenario_A1( dist, napravl)

    def scenario_A4(self, dist, napravl):#The robot knock out the ball to the side of the enemy
        print('The robot knock out the ball to the side of the enemy')
        self.scenario_A1( dist, napravl)

    def scenario_B1(self):#the robot moves to the left and stands on the same axis as the ball and the opponents' goal
        print('the robot moves to the left 4 steps')
        if self.local.ball_odometry[1] > self.local.coord_odometry[1]:
            if self.local.ball_odometry[1] > 0.4: 
                if self.local.coord_odometry[1] < 0.4:
                    self.motion.near_distance_omni_motion( 1000*(0.4 - self.local.coord_odometry[1]), math.pi/2)
            else:
                self.motion.near_distance_omni_motion( 1000*(self.local.ball_odometry[1] - self.local.coord_odometry[1]), math.pi/2)
        self.turn_Face_To_Guest()

    def scenario_B2(self):#the robot moves to the left and stands on the same axis as the ball and the opponents' goal
        print('the robot moves to the left 4 steps')
        self.scenario_B1()

    def scenario_B3(self):#the robot moves to the right and stands on the same axis as the ball and the opponents' goal
        print('the robot moves to the right 4 steps')
        #self.motion.first_Leg_Is_Right_Leg = True
        #self.motion.near_distance_omni_motion( 110, -math.pi/2)
        if self.local.ball_odometry[1] < self.local.coord_odometry[1]:
            if self.local.ball_odometry[1] < -0.4: 
                if self.local.coord_odometry[1] > -0.4:
                    self.motion.near_distance_omni_motion( 1000*(0.4 + self.local.coord_odometry[1]), -math.pi/2)
            else:
                self.motion.near_distance_omni_motion( 1000*(-self.local.ball_odometry[1] + self.local.coord_odometry[1]), -math.pi/2)
        self.turn_Face_To_Guest()

    def scenario_B4(self):#the robot moves to the right and stands on the same axis as the ball and the opponents' goal
        print('the robot moves to the right 4 steps')
        self.scenario_B3()

class Forward:
    def __init__(self, motion, local, glob):
        self.motion = motion
        self.local = local
        self.glob = glob
        self.direction_To_Guest = 0

    def dir_To_Guest(self):
        if self.local.ball_odometry[0] < 0:
            self.direction_To_Guest = 0
        elif self.local.ball_odometry[0] > 0.8 and abs(self.local.ball_odometry[1]) > 0.6:
            self.direction_To_Guest = math.atan(-self.local.ball_odometry[1]/(1.8-self.local.ball_odometry[0]))
        elif self.local.ball_odometry[0] < 1.5 and abs(self.local.ball_odometry[1]) < 0.25:
            if (1.8-self.local.ball_odometry[0]) == 0: self.direction_To_Guest = 0
            else:
                if abs(self.local.ball_odometry[1]) > 0.2:
                    self.direction_To_Guest = math.atan((math.copysign(0.2, self.local.ball_odometry[1])-
                                                       self.local.ball_odometry[1])/(1.8-self.local.ball_odometry[0]))
                else:
                    self.direction_To_Guest = math.atan((0.2* (round(random.random(),0)*2 - 1)-
                                                       self.local.ball_odometry[1])/(1.8-self.local.ball_odometry[0]))
        else:
            self.direction_To_Guest = math.atan(-self.local.coord_odometry[1]/(2.8-self.local.coord_odometry[0]))
        return self.direction_To_Guest

    def turn_Face_To_Guest(self):
        self.dir_To_Guest()
        self.motion.turn_To_Course(self.direction_To_Guest)

class Forward_Vector_Matrix:
    def __init__(self, motion, local, glob):
        self.motion = motion
        self.local = local
        self.glob = glob
        self.direction_To_Guest = 0
        self.kick_Power = 1

    def dir_To_Guest(self):
        if abs(self.glob.ball_coord[0])  >  self.glob.landmarks["FIELD_LENGTH"] / 2:
            ball_x = math.copysign(self.glob.landmarks["FIELD_LENGTH"] / 2, self.glob.ball_coord[0])
        else: ball_x = self.glob.ball_coord[0]
        if abs(self.glob.ball_coord[1])  >  self.glob.landmarks["FIELD_WIDTH"] / 2:
            ball_y = math.copysign(self.glob.landmarks["FIELD_WIDTH"] / 2, self.glob.ball_coord[1])
        else: ball_y = self.glob.ball_coord[1]
        col = math.floor((ball_x + self.glob.landmarks["FIELD_LENGTH"] / 2) / (self.glob.landmarks["FIELD_LENGTH"] / self.glob.COLUMNS))
        row = math.floor((- ball_y + self.glob.landmarks["FIELD_WIDTH"] / 2) / (self.glob.landmarks["FIELD_WIDTH"] / self.glob.ROWS))
        if col >= self.glob.COLUMNS : col = self.glob.COLUMNS - 1
        if row >= self.glob.ROWS : row = self.glob.ROWS -1
        self.direction_To_Guest = self.glob.strategy_data[(col * self.glob.ROWS + row) * 2 + 1] / 40
        self.kick_Power = self.glob.strategy_data[(col * self.glob.ROWS + row) * 2]
        self.glob.direction_to_guest = self.direction_To_Guest
        self.glob.kick_power_planned = self.kick_Power
        self.glob.strategy_cell = [row, col]
        #print('direction_To_Guest = ', math.degrees(self.direction_To_Guest))
        return row, col

    def turn_Face_To_Guest(self):
        self.dir_To_Guest()
        self.motion.turn_To_Course(self.direction_To_Guest)

class Player():
    def __init__(self, role, second_pressed_button, glob, motion, local):
        self.role = role   #'goalkeeper', 'penalty_Goalkeeper', 'forward', 'penalty_Shooter'
        self.second_pressed_button = second_pressed_button
        self.glob = glob
        self.motion = motion
        self.local = local
        self.g = None
        self.f = None
        if self.glob.SIMULATION == 5:
            from Soccer.Motion.class_stm_channel import STM_channel
            from Soccer.Vision.class_Vision_RPI import Vision_RPI
            self.STM_channel = STM_channel
            self.Vision_RPI = Vision_RPI

    def play_game(self):
        if self.role == 'goalkeeper': self.goalkeeper_main_cycle()
        if self.role == 'penalty_Goalkeeper': self.penalty_Goalkeeper_main_cycle()
        if self.role == 'FIRA_penalty_Goalkeeper': self.FIRA_penalty_Goalkeeper_main_cycle()
        if self.role == 'side_to_side': self.side_to_side_main_cycle()
        if self.role == 'forward': self.forward_main_cycle(self.second_pressed_button)
        if self.role == 'forward_v2': self.forward_v2_main_cycle()
        if self.role == 'marathon':  self.marathon_main_cycle()
        if self.role == 'penalty_Shooter': self.penalty_Shooter_main_cycle()
        if self.role == 'FIRA_penalty_Shooter': self.FIRA_penalty_Shooter_main_cycle()
        if self.role == 'run_test': self.run_test_main_cycle(self.second_pressed_button)
        if self.role == 'jump_test': self.jump_test(self.second_pressed_button)
        if self.role == 'rotation_test': self.rotation_test_main_cycle()
        if self.role == 'sidestep_test': self.sidestep_test_main_cycle()
        if self.role == 'obstacle_runner': self.obstacle_runner_main_cycle()
        #if self.role == 'kick_test': self.test_walk_main_cycle()
        if self.role == 'ball_moving': self.ball_moving_main_cycle()
        if self.role == 'dance': self.dance_main_cycle()
        if self.role == 'basketball': self.basketball_main_cycle(self.second_pressed_button)
        if self.role == 'weight_lifting': self.weight_lifting(self.second_pressed_button)
        if self.role == 'corner_kick_1': self.corner_kick_1_main_cycle()
        if self.role == 'corner_kick_2': self.corner_kick_2_main_cycle()
        if self.role == 'triple_jump': self.triple_jump_main_cycle(self.second_pressed_button)
        if self.role == 'sprint': self.sprint(self.second_pressed_button)
        if self.role == 'kick_test': self.kick_test(self.second_pressed_button)
        #print('self.glob.SIMULATION:', self.glob.SIMULATION)
        if [0,1,3].count(self.glob.SIMULATION) == 1:
            self.motion.sim_Stop()
            if self.glob.SIMULATION != 0:
                self.motion.print_Diagnostics()
                pass
            self.motion.sim_Disable()
        #sys.exit(1)

    def jump_test(self, second_pressed_button):
        #var = roki2met.roki2met.jump_mode
        #intercom = self.glob.stm_channel.zubr       # used for communication between head and zubr-controller with memIGet/memISet commands
        #self.glob.rcb.motionPlay(23)
        self.motion.head_Return(0, -2000)
        if second_pressed_button == 'jump_forward':
            #intercom.memISet(var, 1001)
            #self.glob.rcb.motionPlay(7)
            #while(intercom.memIGet(var) != 0): time.sleep(0.1)
            for _ in range(10): 
                self.motion.play_Soft_Motion_Slot(motion_list = self.motion.jump_motion_forward)
                time.sleep(0.5)
                self.motion.jump_turn(0)
                time.sleep(0.5)
        if second_pressed_button == 'jump_backward':
            #intercom.memISet(var, 1002)
            #self.glob.rcb.motionPlay(7)
            #while(intercom.memIGet(var) != 0): time.sleep(0.1)
            for _ in range(10): 
                self.motion.play_Soft_Motion_Slot(motion_list = self.motion.jump_motion_backward)
                time.sleep(0.5)
                self.motion.jump_turn(0)
                time.sleep(0.5)
        if second_pressed_button == 'jump_left':
            #intercom.memISet(var, 1003)
            #self.glob.rcb.motionPlay(7)
            #while(intercom.memIGet(var) != 0): time.sleep(0.1)
            for _ in range(10): 
                self.motion.play_Soft_Motion_Slot(motion_list = self.motion.jump_motion_left)
                time.sleep(0.5)
                self.motion.jump_turn(0)
                time.sleep(0.5)
        if second_pressed_button == 'jump_right':
            #intercom.memISet(var, 1004)
            #self.glob.rcb.motionPlay(7)
            #while(intercom.memIGet(var) != 0): time.sleep(0.1)
            for _ in range(10): 
                self.motion.play_Soft_Motion_Slot(motion_list = self.motion.jump_motion_right)
                time.sleep(0.5)
                self.motion.jump_turn(0)
                time.sleep(0.5)
        if second_pressed_button == 'jump_on_spot':
            for _ in range(10): 
                self.motion.one_jump_forward( fraction = 0)
                time.sleep(0.5)
                self.motion.jump_turn(0)
                time.sleep(0.5)
        pass

    def rotation_test_main_cycle(self):
        motion = [
            [ 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [ 2, -700, 0, 0, 0, 0, 1000, 0, 0, 0, 0, 0, 700, 0, 0, 0, 0, 1000, 0, 0, 0, 0],
            [ 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [ 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            ]
        self.motion.head_Return(0, self.motion.neck_play_pose)
        time.sleep(1)
        self.motion.refresh_Orientation()
        initial_yaw = self.motion.body_euler_angle['yaw']
        jump_value = 1000           # jumping to Clock-Wise
        motion[1][6] = motion[1][17] = jump_value
        for _ in range(3):
            self.motion.play_Soft_Motion_Slot(  motion_list = motion)
            time.sleep(1)
        time.sleep(2)
        self.motion.refresh_Orientation()
        angle_cw = self.motion.body_euler_angle['yaw'] - initial_yaw
        if angle_cw > 0 : angle_cw -= 2 * math.pi
        self.motion.params['CALIBRATED_CW_YAW'] = round(angle_cw / 3, 3)
        initial_yaw_ccw = self.motion.body_euler_angle['yaw']
        jump_value = -1000          # jumping to CCW
        motion[1][6] = motion[1][17] = jump_value
        for _ in range(3):
            self.motion.play_Soft_Motion_Slot(  motion_list = motion)
            time.sleep(1)
        time.sleep(2)
        self.motion.refresh_Orientation()
        angle_ccw = self.motion.body_euler_angle['yaw'] - initial_yaw_ccw
        if angle_ccw < 0 : angle_cw += 2 * math.pi
        self.motion.params['CALIBRATED_CCW_YAW'] = round(angle_ccw / 3, 3)
        number_Of_Cycles = 10
        stepLength = 0
        sideLength = 0
        self.motion.params['ROTATION_YIELD_RIGHT'] = 0.23
        self.motion.params['ROTATION_YIELD_LEFT'] = 0.23
        self.motion.refresh_Orientation()
        first_yaw_measurement = self.motion.imu_body_yaw()
        rotation = -0.23
        self.motion.first_Leg_Is_Right_Leg = True
        invert = 1
        self.motion.walk_Initial_Pose()
        for cycle in range(number_Of_Cycles):
            self.motion.walk_Cycle(stepLength,sideLength, invert * rotation, cycle, number_Of_Cycles)
        self.motion.walk_Final_Pose()
        self.motion.refresh_Orientation()
        second_yaw_measurement = self.motion.imu_body_yaw()
        time.sleep(2)
        rotation = 0.23
        self.motion.first_Leg_Is_Right_Leg = False
        invert = -1
        self.motion.walk_Initial_Pose()
        for cycle in range(number_Of_Cycles):
            self.motion.walk_Cycle(stepLength,sideLength, invert * rotation, cycle, number_Of_Cycles)
        self.motion.walk_Final_Pose()
        self.motion.refresh_Orientation()
        third_yaw_measurement = self.motion.imu_body_yaw()
        if self.motion.glob.SIMULATION == 5:
            filename = init_param_read_path(self.glob.current_work_directory, "Real/Real_params.json")
            write_filename = init_param_write_path(self.glob.current_work_directory, "Real/Real_params.json")
        else:
            filename = init_param_read_path(self.glob.current_work_directory, "Sim/Sim_params.json")
            write_filename = init_param_write_path(self.glob.current_work_directory, "Sim/Sim_params.json")
        with open(filename, "r") as f:
            params = json.loads(f.read())
            rotation_yield_right = abs(second_yaw_measurement - first_yaw_measurement) / 10
            rotation_yield_left = abs(third_yaw_measurement - second_yaw_measurement) / 10
            #rotation_yield = (rotation_yield_right + rotation_yield_left)/2
            params['ROTATION_YIELD_RIGHT'] = round(rotation_yield_right, 3)
            params['ROTATION_YIELD_LEFT'] = round(rotation_yield_left, 3)
        jsonstring = '{\n"BODY_TILT_AT_WALK": ' + str(params["BODY_TILT_AT_WALK"]) \
                    + ',\n"BODY_TILT_AT_WALK_BACKWARDS": ' + str(params["BODY_TILT_AT_WALK_BACKWARDS"]) \
                    + ',\n"SOLE_LANDING_SKEW": ' + str(params["SOLE_LANDING_SKEW"]) \
                    + ',\n"BODY_TILT_AT_KICK": ' + str(params["BODY_TILT_AT_KICK"]) \
                    + ',\n"ROTATION_YIELD_RIGHT": ' + str(params["ROTATION_YIELD_RIGHT"]) \
                    + ',\n"ROTATION_YIELD_LEFT": ' + str(params["ROTATION_YIELD_LEFT"]) \
                    + ',\n"CALIBRATED_CW_YAW": ' + str(self.motion.params['CALIBRATED_CW_YAW']) \
                    + ',\n"CALIBRATED_CCW_YAW": ' + str(self.motion.params['CALIBRATED_CCW_YAW']) \
                    + ',\n"JUMP_FORWARD_TEST": ' + str(self.motion.params['JUMP_FORWARD_TEST']) \
                    + ',\n"JUMP_BACKWARD_TEST": ' + str(self.motion.params['JUMP_BACKWARD_TEST']) \
                    + ',\n"JUMP_LEFT_TEST": ' + str(self.motion.params['JUMP_LEFT_TEST']) \
                    + ',\n"JUMP_RIGHT_TEST": ' + str(self.motion.params['JUMP_RIGHT_TEST']) \
                    + ',\n"MOTION_SHIFT_TEST_X": ' + str(params["MOTION_SHIFT_TEST_X"]) \
                    + ',\n"MOTION_SHIFT_TEST_Y": ' + str(params["MOTION_SHIFT_TEST_Y"]) \
                    + ',\n"SIDE_STEP_RIGHT_TEST_RESULT": ' + str(params["SIDE_STEP_RIGHT_TEST_RESULT"]) \
                    + ',\n"SIDE_STEP_LEFT_TEST_RESULT": ' + str(params["SIDE_STEP_LEFT_TEST_RESULT"]) \
                    + ',\n"RUN_TEST_10_STEPS": ' + str(params["RUN_TEST_10_STEPS"]) \
                    + ',\n"RUN_TEST_20_STEPS": ' + str(params["RUN_TEST_20_STEPS"]) \
                    + ',\n"KICK_ADJUSTMENT_DISTANCE_1": ' + str(params["KICK_ADJUSTMENT_DISTANCE_1"]) \
                    + ',\n"KICK_ADJUSTMENT_DISTANCE_2": ' + str(params["KICK_ADJUSTMENT_DISTANCE_2"]) \
                    + ',\n"KICK_OFFSET_OF_BALL": ' + str(params["KICK_OFFSET_OF_BALL"]) \
                    + ',\n"IMU_DRIFT_IN_DEGREES_DURING_6_MIN_MEASUREMENT": ' + str(params["IMU_DRIFT_IN_DEGREES_DURING_6_MIN_MEASUREMENT"]) \
                    + '\n}'
        with open(write_filename, "w") as f:
            f.write(jsonstring)
            #json.dump(params_new, f)
        self.motion.turn_To_Course(math.pi/3*2)
        self.motion.turn_To_Course(0)

    def run_test_main_cycle(self, pressed_button, stepLength = 64):
        if pressed_button == 'head_tilt_calibration':
            #execfile("Head_Tilt_Calibration.py")
            if self.motion.head_Tilt_Calibration():
                self.motion.rcb.motionPlay(25)          # zummer
            else: 
                self.motion.rcb.motionPlay(25)          # zummer
                time.sleep(0.2)
                self.motion.rcb.motionPlay(25)          # zummer
            return
        self.motion.head_Return(0, self.motion.neck_play_pose)
        if pressed_button == 'side_step_right' or pressed_button =='side_step_left' :
            self.sidestep_test_main_cycle(pressed_button)
            return
        if pressed_button == 'rotation_right' or pressed_button =='rotation_left' :
            self.rotation_test_main_cycle()
            return

        stepLength = 64
        self.motion.gaitHeight = 180
        if pressed_button == 'spot_run': stepLength = 0
        if pressed_button == 'run_backwards': 
            stepLength = -50
            self.motion.stepHeight = 40
        number_Of_Cycles = 20
        self.motion.amplitude = 32
        if pressed_button == 'short_run': number_Of_Cycles = 10
        sideLength = 0
        #self.motion.first_Leg_Is_Right_Leg = False
        if self.motion.first_Leg_Is_Right_Leg: invert = -1
        else: invert = 1
        if self.glob.hardcode_walking: self.motion.hardcode_walk_Init()
        else: self.motion.walk_Initial_Pose()
        number_Of_Cycles += 1
        for cycle in range(number_Of_Cycles):
            stepLength1 = stepLength
            if cycle ==0 : stepLength1 = stepLength/3
            if cycle ==1 : stepLength1 = stepLength/3 * 2
            self.motion.refresh_Orientation()
            rotation = 0 + invert * self.motion.imu_body_yaw() * 1.1
            #if rotation > 0: rotation *= 1.5
            rotation = self.motion.normalize_rotation(rotation)
            #rotation = 0
            if self.glob.hardcode_walking:
                last = (cycle == (number_Of_Cycles - 1))
                self.motion.hardcode_walk_Cycle(stepLength1, sideLength, rotation, 0, last=last)
            else: self.motion.walk_Cycle(stepLength1,sideLength, rotation,cycle, number_Of_Cycles)
        self.motion.walk_Final_Pose()

    def sidestep_test_main_cycle(self, pressed_button):
        number_Of_Cycles = 20
        stepLength = 0 #64
        sideLength = 20
        if pressed_button == 'side_step_left':
            self.motion.first_Leg_Is_Right_Leg = False
        if self.motion.first_Leg_Is_Right_Leg: invert = -1
        else: invert = 1
        #if pressed_button == 'side_step_left': sideLength = -20
        #else: sideLength = 20
        #invert = 1
        self.motion.walk_Initial_Pose()
        for cycle in range(number_Of_Cycles):
            stepLength1 = stepLength
            #if cycle ==0 : stepLength1 = stepLength/3
            #if cycle ==1 : stepLength1 = stepLength/3 * 2
            self.motion.refresh_Orientation()
            rotation = 0 + invert * self.motion.imu_body_yaw() * 1.1
            rotation = self.motion.normalize_rotation(rotation)
            #rotation = 0
            self.motion.walk_Cycle(stepLength1,sideLength, rotation,cycle, number_Of_Cycles)
        self.motion.walk_Final_Pose()

    def norm_yaw(self, yaw):
        yaw %= 2 * math.pi
        if yaw > math.pi:  yaw -= 2* math.pi
        if yaw < -math.pi: yaw += 2* math.pi
        return yaw

    def forward_main_cycle(self, pressed_button):
        if self.glob.camera_streaming: self.glob.vision.camera_thread.start()
        self.glob.monitor_is_on = True
        self.f = Forward_Vector_Matrix(self.motion, self.local, self.glob)
        if self.glob.SIMULATION == 5:
            self.motion.rcb.motionPlay(25)          # zummer
            labels = [[], [], [], ['start', 'start_later'], []]
            pressed_button = self.motion.push_Button(labels)
        second_player_timer = time.time()
        if pressed_button == 'start':
            self.motion.kick_off_ride()
        while (True):
            if self.glob.SIMULATION == 5:
                if (time.perf_counter() - self.motion.start_point_for_imu_drift) > 360:
                    self.motion.jump_turn(0)
                    os.system("espeak -ven-m1 -a200 'Six minutes of game is over. I look to zero direction'")
                    break
            else:
                if self.motion.trigger_counter > 18000:
                    break
            if self.motion.falling_Flag != 0:
                if self.motion.falling_Flag == 3: break
                self.motion.falling_Flag = 0
                #self.local.coordinate_fall_reset()
                self.motion.head_Return(0, self.motion.neck_play_pose)
            if self.glob.camera_down_Flag == True: self.glob.camera_reset()
                    #print('Camera resetting')
                    #self.glob.camera_down_Flag = False
                    #self.glob.vision.event.set()
                    #new_stm_channel  = self.STM_channel(self.glob)
                    #self.glob.stm_channel = new_stm_channel
                    #self.glob.rcb = self.glob.stm_channel.rcb
                    #new_vision = self.Vision_RPI(self.glob)
                    #self.glob.vision = new_vision
                    #self.motion.vision = self.glob.vision
                    #self.local.vision = self.glob.vision
                    ##self.glob.vision.camera_thread.start()
            #success_Code, napravl, dist, speed = self.motion.seek_Ball_In_Pose(fast_Reaction_On = True, with_Localization = False,
            #                                                                  very_Fast = True, first_look_point=first_look_point)
            #time.sleep(1) # this is to look around for ball 
            if not self.glob.camera_streaming:
                if self.glob.robot_see_ball <= 10:   # must be 0
                    self.motion.head_Return(0, self.motion.neck_play_pose)
                    success_Code, napravl, dist, speed = self.motion.seek_Ball_In_Pose(fast_Reaction_On = True, with_Localization = True,
                                                                                      very_Fast = False)
                    self.motion.head_Return(0, self.motion.neck_play_pose)
            #self.glob.pf_coord = self.local.coord_odometry
            self.glob.local.localisation_Complete()
            time_elapsed = time.time() - second_player_timer
            if self.glob.SIMULATION == 5: frozen_time = 10 
            else: frozen_time = 10
            if pressed_button == 'start_later' and time_elapsed < frozen_time : 
                time.sleep(0.02)
                continue
            self.f.dir_To_Guest()
            print('direction_To_Guest = ', round(math.degrees(self.f.direction_To_Guest)), 'degrees')
            print('coord =', round(self.local.coord_odometry[0],2), round(self.local.coord_odometry[1],2), 'ball =', round(self.local.ball_odometry[0],2), round(self.local.ball_odometry[1],2))
            if self.glob.robot_see_ball < -6:
                print('Seek ball')
                self.motion.jump_turn(self.local.coord_odometry[2]+ 2 * math.pi / 3)
                continue
            player_from_ball_yaw = coord2yaw(self.local.coord_odometry[0] - self.local.ball_odometry[0],
                                                          self.local.coord_odometry[1] - self.local.ball_odometry[1]) - self.f.direction_To_Guest
            player_from_ball_yaw = self.norm_yaw(player_from_ball_yaw)
            player_in_front_of_ball = -math.pi/2 < player_from_ball_yaw < math.pi/2
            player_in_fast_kick_position = (player_from_ball_yaw > 2.0 or player_from_ball_yaw < -2.0) and self.glob.ball_distance < 0.6
            if self.glob.ball_distance > 0.35  and not player_in_fast_kick_position:
                direction_To_Ball = math.atan2((self.local.ball_odometry[1] - self.local.coord_odometry[1]), (self.local.ball_odometry[0] - self.local.coord_odometry[0]))
                print('napravl :', self.glob.ball_course)
                print('direction_To_Ball', direction_To_Ball)
                #self.motion.far_distance_plan_approach(self.local.ball_odometry, self.f.direction_To_Guest, stop_Over = stop_Over)
                #self.motion.far_distance_straight_approach(self.local.ball_odometry, direction_To_Ball, stop_Over = False)
                self.motion.far_distance_straight_approach_streaming()
                continue
            if player_in_front_of_ball or not player_in_fast_kick_position:
                self.go_Around_Ball(self.glob.ball_distance, self.glob.ball_course)
                continue
            if player_in_fast_kick_position:
                self.motion.jump_turn(self.f.direction_To_Guest)
                if self.f.kick_Power == 1: self.motion.kick_power = 100
                if self.f.kick_Power == 2: self.motion.kick_power = 60
                if self.f.kick_Power == 3: self.motion.kick_power = 20
                success_Code = self.motion.near_distance_ball_approach_and_kick_streaming(self.f.direction_To_Guest)

    def goalkeeper_main_cycle(self):
        self.glob.monitor_is_on = True
        def ball_position_is_dangerous(row, col):
            danger = False
            danger = (col <= (round(self.glob.COLUMNS / 3) - 1))
            if ((row <= (round(self.glob.ROWS / 3) - 1) or row >= round(self.glob.ROWS * 2 / 3)) and col == 0) or (col == 1 and (row == 0 or row == (self.glob.ROWS -1))):
               danger = False
            return danger
        #second_player_timer = time.time()
        self.f = Forward_Vector_Matrix(self.motion, self.local, self.glob)
        self.motion.near_distance_omni_motion(300, 0)                    # get out from goal
        fast_Reaction_On = True
        while (True):
            if self.motion.falling_Flag != 0:
                if self.motion.falling_Flag == 3: break
                self.motion.falling_Flag = 0
                #self.local.coordinate_fall_reset()
                self.motion.head_Return(0, self.motion.neck_play_pose)
            if self.glob.camera_down_Flag == True: self.glob.camera_reset()
            #if self.local.ball_odometry[0] <= 0.15:
            #    success_Code, napravl, dist, speed =  self.motion.watch_Ball_In_Pose()
            #else:
            #    success_Code, napravl, dist, speed = self.motion.seek_Ball_In_Pose(fast_Reaction_On = fast_Reaction_On)
            #time.sleep(1) # this is to look around for ball
            #if self.glob.robot_see_ball <= 0:
            self.motion.head_Return(0, self.motion.neck_play_pose)
            success_Code, napravl, dist, speed = self.motion.seek_Ball_In_Pose(fast_Reaction_On = True, with_Localization = False,
                                                                                very_Fast = False)
            self.motion.head_Return(0, self.motion.neck_play_pose)
            napravl, dist, speed = self.glob.ball_course, self.glob.ball_distance, self.glob.ball_speed
            #if abs(speed[0]) > 0.02 and dist < 1 :                         # if dangerous tangential speed
            #    fast_Reaction_On = True
            #    if speed[0] > 0:
            #        if self.local.coord_odometry[1] < 0.35:
            #            self.motion.play_Soft_Motion_Slot(name ='PenaltyDefenceL')
            #    else:
            #        if self.local.coord_odometry[1] > -0.35:
            #            self.motion.play_Soft_Motion_Slot(name ='PenaltyDefenceR')
            #    self.motion.pause_in_ms(3000)
            #    self.motion.falling_Test()
            #    continue
            #if speed[1] < - 0.01 and dist < 1.5 :                          # if dangerous front speed
            #    fast_Reaction_On = True
            #    self.motion.play_Soft_Motion_Slot(name = 'PanaltyDefenceReady_Fast')
            #    self.motion.play_Soft_Motion_Slot(name = 'PenaltyDefenceF')
            #    self.motion.pause_in_ms(3000)
            #    self.motion.play_Soft_Motion_Slot(name = 'Get_Up_From_Defence')
            #    continue
            #if (time.time() - second_player_timer) < 10 : continue
            row, col = self.f.dir_To_Guest()
            #print('direction_To_Guest = ', math.degrees(self.f.direction_To_Guest), 'degrees')
            #print('goalkeeper coord =', self.glob.pf_coord, 'ball =', self.glob.ball_coord, 'row =', row, 'col =', col, 'ball_position_is_dangerous =', ball_position_is_dangerous(row,col))
            if dist == 0 and self.glob.robot_see_ball <= 0:
                if self.local.coord_odometry[0] > -1.3:
                    print('goalkeeper turn_To_Course(pi*2/3)')
                    self.motion.jump_turn(self.local.coord_odometry[2]+ 2 * math.pi / 3)
                continue
            if ball_position_is_dangerous(row, col):
                fast_Reaction_On = True
                player_from_ball_yaw = coord2yaw(self.local.coord_odometry[0] - self.local.ball_odometry[0], self.local.coord_odometry[1] - self.local.ball_odometry[1]) - self.f.direction_To_Guest
                player_from_ball_yaw = self.norm_yaw(player_from_ball_yaw)
                player_in_front_of_ball = -math.pi/2 < player_from_ball_yaw < math.pi/2
                player_in_fast_kick_position = (player_from_ball_yaw > 2 or player_from_ball_yaw < -2) and dist < 0.6
                if dist > 0.35 and not player_in_fast_kick_position:
                    if dist > 3: stop_Over = True
                    else: stop_Over = False
                    print('goalkeeper far_distance_plan_approach')
                    direction_To_Ball = math.atan2((self.local.ball_odometry[1] - self.local.coord_odometry[1]), (self.local.ball_odometry[0] - self.local.coord_odometry[0]))
                    #self.motion.far_distance_straight_approach(self.local.ball_odometry, direction_To_Ball, stop_Over = stop_Over)
                    self.motion.far_distance_straight_approach_streaming()
                    #self.f.turn_Face_To_Guest()
                    continue
                if player_in_front_of_ball or not player_in_fast_kick_position:
                    self.go_Around_Ball(dist, napravl)
                    continue
                print('goalkeeper turn_To_Course(direction_To_Guest)')
                self.motion.jump_turn(self.f.direction_To_Guest)
                print('goalkeeper near_distance_ball_approach_and_kick')
                self.motion.kick_power = 100
                success_Code = self.motion.near_distance_ball_approach_and_kick_streaming(self.f.direction_To_Guest)

            else:
                fast_Reaction_On = False
                duty_x_position =  min((-self.glob.landmarks['FIELD_LENGTH']/2 + 0.4),(self.local.ball_odometry[0]-self.glob.landmarks['FIELD_LENGTH']/2)/2)
                duty_y_position = self.local.ball_odometry[1] * (duty_x_position + self.glob.landmarks['FIELD_LENGTH']/2) / (self.local.ball_odometry[0] + self.glob.landmarks['FIELD_LENGTH']/2)
                duty_distance = math.sqrt((duty_x_position - self.local.coord_odometry[0])**2 + (duty_y_position - self.local.coord_odometry[1])**2)
                #print('duty_x_position =', duty_x_position, 'duty_y_position =', duty_y_position)
                if duty_distance < 0.2 : continue
                elif duty_distance <  3: #   0.6 :
                    print('goalkeeper turn_To_Course(0)')
                    self.motion.jump_turn(0)
                    duty_direction = coord2yaw(duty_x_position - self.local.coord_odometry[0], duty_y_position - self.local.coord_odometry[1])
                    print('goalkeeper near_distance_omni_motion')
                    self.motion.near_distance_omni_motion(duty_distance * 1000, duty_direction)
                    print('goalkeeper turn_To_Course(0)')
                    self.motion.jump_turn(0)
                else:
                    direction_To_Duty = math.atan2((duty_y_position - self.local.coord_odometry[1]), (duty_x_position - self.local.coord_odometry[0]))
                    self.motion.far_distance_straight_approach([duty_x_position , duty_y_position], direction_To_Duty, gap = 0, stop_Over = False)
                    self.motion.jump_turn(0)

    def FIRA_penalty_Shooter_main_cycle(self):
        if self.glob.camera_streaming: self.glob.vision.camera_thread.start()
        self.f = Forward_Vector_Matrix(self.motion, self.local, self.glob)
        self.motion.head_Return(0, -1500)
        time.sleep(1)
        self.motion.jump_turn(self.motion.params['PENALTY_JUMP_TURN_ANGLE'], jumps_limit = 10) # 0.5
        self.motion.kick_power = self.motion.params['PENALTY_FIRST_KICK_POWER_20-100']
        #self.motion.kick(True, small = True)
        success_Code, napravl, dist, speed = self.motion.seek_Ball_In_Pose(fast_Reaction_On = True, with_Localization = False)
        kick_offset = 0
        kick_by_right = 1
        if success_Code:
            if napravl < 0 : kick_by_right = 1
            else: kick_by_right = 0
            kick_offset = int(abs(math.sin(napravl) * dist * 1000)) - 62
        if self.glob.SIMULATION == 5:
            self.motion.hard_kick(kick_by_right = kick_by_right, kick_offset = kick_offset)
        else:
            self.motion.kick( first_Leg_Is_Right_Leg=kick_by_right, kick_offset = kick_offset)
        time.sleep(2)
        self.motion.jump_turn(0.75, jumps_limit = 10)
        #self.motion.walk_Final_Pose_After_Kick()
        for _ in range(5):
            self.motion.falling_Test()
            time.sleep(0.5)
        #self.motion.jump_turn(self.f.direction_To_Guest)
        self.motion.near_distance_omni_motion(500, 0) #math.pi/2)
        self.motion.jump_turn(0, jumps_limit = 10)
        self.motion.near_distance_omni_motion(300, 0)
        #return
        while (True):
            if self.motion.falling_Flag != 0:
                if self.motion.falling_Flag == 3: break
                self.motion.falling_Flag = 0
                #self.local.coordinate_fall_reset()
                self.motion.head_Return(0, self.motion.neck_play_pose)
            if self.glob.camera_down_Flag == True: self.glob.camera_reset()
            if self.glob.robot_see_ball <= 10:   # must be 0
                self.motion.head_Return(0, self.motion.neck_play_pose)
                success_Code, napravl, dist, speed = self.motion.seek_Ball_In_Pose(fast_Reaction_On = True, with_Localization = False,
                                                                                  very_Fast = False)
            self.glob.local.localisation_Complete()
            self.f.dir_To_Guest()
            print('direction_To_Guest = ', round(math.degrees(self.f.direction_To_Guest)), 'degrees')
            print('coord =', round(self.local.coord_odometry[0],2), round(self.local.coord_odometry[1],2), 'ball =', round(self.local.ball_odometry[0],2), round(self.local.ball_odometry[1],2))
            if self.glob.robot_see_ball < 0:
                print('Seek ball')
                self.motion.jump_turn(self.local.coord_odometry[2]+ 2 * math.pi / 3)
                continue
            player_from_ball_yaw = coord2yaw(self.local.coord_odometry[0] - self.local.ball_odometry[0],
                                                          self.local.coord_odometry[1] - self.local.ball_odometry[1]) - self.f.direction_To_Guest
            player_from_ball_yaw = self.norm_yaw(player_from_ball_yaw)
            player_in_front_of_ball = -math.pi/2 < player_from_ball_yaw < math.pi/2
            player_in_fast_kick_position = (player_from_ball_yaw > 2.0 or player_from_ball_yaw < -2.0) and self.glob.ball_distance < 0.6
            if self.glob.ball_distance > 0.35  and not player_in_fast_kick_position:
                direction_To_Ball = math.atan2((self.local.ball_odometry[1] - self.local.coord_odometry[1]), (self.local.ball_odometry[0] - self.local.coord_odometry[0]))
                print('napravl :', self.glob.ball_course)
                print('direction_To_Ball', direction_To_Ball)
                self.motion.far_distance_straight_approach_streaming()
                continue
            if player_in_front_of_ball or not player_in_fast_kick_position:
                self.go_Around_Ball(self.glob.ball_distance, self.glob.ball_course)
                continue
            if player_in_fast_kick_position:
                self.motion.jump_turn(self.f.direction_To_Guest)
                if self.f.kick_Power == 1: self.motion.kick_power = 100
                if self.f.kick_Power == 2: self.motion.kick_power = 60
                if self.f.kick_Power == 3: self.motion.kick_power = 20
                success_Code = self.motion.near_distance_ball_approach_and_kick_streaming_FIRA(self.f.direction_To_Guest)

    def FIRA_penalty_Shooter_main_cycle_(self):
        if self.glob.camera_streaming: self.glob.vision.camera_thread.start()
        self.f = Forward_Vector_Matrix(self.motion, self.local, self.glob)
        self.motion.head_Return(0, -1500)
        time.sleep(1)
        #self.motion.jump_turn(self.motion.params['PENALTY_JUMP_TURN_ANGLE'], jumps_limit = 10) # 0.5
        #self.motion.kick_power = self.motion.params['PENALTY_FIRST_KICK_POWER_20-100']
        #self.motion.kick(True, small = True)
        self.motion.jump_turn(0.6, jumps_limit = 10)
        #self.motion.walk_Final_Pose_After_Kick()
        for _ in range(5):
            self.motion.falling_Test()
            time.sleep(0.5)
        #self.motion.jump_turn(self.f.direction_To_Guest)
        self.motion.stepHeight = 10
        self.motion.gaitHeight = 180
        self.walk_straight_vision(number_Of_Cycles = 50, stepLength = 15, sideLength = 0, respect_body_tilt = False, direction = 0.6)
        self.motion.stepHeight = 32
        self.motion.gaitHeight = 180
        self.f.dir_To_Guest()
        self.motion.one_jump_backward()
        success_Code = self.motion.near_distance_ball_approach_and_kick_streaming(self.f.direction_To_Guest)
        return
        #self.motion.near_distance_omni_motion(500, 0) #math.pi/2)
        #self.motion.jump_turn(0, jumps_limit = 10)
        #self.motion.near_distance_omni_motion(300, 0)
        #return
        #for _ in range(50): self.motion.one_jump_forward(fraction = 1, hands_on = True)
        while (True):
            if self.motion.falling_Flag != 0:
                if self.motion.falling_Flag == 3: break
                self.motion.falling_Flag = 0
                #self.local.coordinate_fall_reset()
                self.motion.head_Return(0, self.motion.neck_play_pose)
            if self.glob.camera_down_Flag == True: self.glob.camera_reset()
            if self.glob.robot_see_ball <= 10:   # must be 0
                self.motion.head_Return(0, self.motion.neck_play_pose)
                success_Code, napravl, dist, speed = self.motion.seek_Ball_In_Pose(fast_Reaction_On = True, with_Localization = True,
                                                                                  very_Fast = False)
            self.glob.local.localisation_Complete()
            self.f.dir_To_Guest()
            print('direction_To_Guest = ', round(math.degrees(self.f.direction_To_Guest)), 'degrees')
            print('coord =', round(self.local.coord_odometry[0],2), round(self.local.coord_odometry[1],2), 'ball =', round(self.local.ball_odometry[0],2), round(self.local.ball_odometry[1],2))
            if self.glob.robot_see_ball < 0:
                print('Seek ball')
                self.motion.jump_turn(self.local.coord_odometry[2]+ 2 * math.pi / 3)
                continue
            player_from_ball_yaw = coord2yaw(self.local.coord_odometry[0] - self.local.ball_odometry[0],
                                                          self.local.coord_odometry[1] - self.local.ball_odometry[1]) - self.f.direction_To_Guest
            player_from_ball_yaw = self.norm_yaw(player_from_ball_yaw)
            player_in_front_of_ball = -math.pi/2 < player_from_ball_yaw < math.pi/2
            player_in_fast_kick_position = (player_from_ball_yaw > 2.0 or player_from_ball_yaw < -2.0) and self.glob.ball_distance < 0.6
            if self.glob.ball_distance > 0.35  and not player_in_fast_kick_position:
                direction_To_Ball = math.atan2((self.local.ball_odometry[1] - self.local.coord_odometry[1]), (self.local.ball_odometry[0] - self.local.coord_odometry[0]))
                print('napravl :', self.glob.ball_course)
                print('direction_To_Ball', direction_To_Ball)
                self.motion.far_distance_straight_approach_streaming()
                continue
            if player_in_front_of_ball or not player_in_fast_kick_position:
                self.go_Around_Ball(self.glob.ball_distance, self.glob.ball_course)
                continue
            if player_in_fast_kick_position:
                self.motion.jump_turn(self.f.direction_To_Guest)
                if self.f.kick_Power == 1: self.motion.kick_power = 100
                if self.f.kick_Power == 2: self.motion.kick_power = 60
                if self.f.kick_Power == 3: self.motion.kick_power = 20
                success_Code = self.motion.near_distance_ball_approach_and_kick_streaming(self.f.direction_To_Guest)

    def penalty_Shooter_main_cycle(self):
        #self.f = Forward(self.motion, self.local, self.glob)
        #first_shoot = True
        #first_look_point = [0.9, 0]
        #self.glob.vision.camera_thread.start()
        #self.motion.control_Head_motion_thread.start()
        #self.motion.kick_off_ride()
        #while (True):
        #    if self.motion.falling_Flag != 0:
        #        if self.motion.falling_Flag == 3: break
        #        self.motion.falling_Flag = 0
        #        self.local.coordinate_fall_reset()
        #    #success_Code, napravl, dist, speed = self.motion.seek_Ball_In_Pose(fast_Reaction_On = True, with_Localization = False,
        #    #                                                                  very_Fast = False, first_look_point=first_look_point )
        #    if self.glob.robot_see_ball > 0: self.local.ball_position_calculation()
        #    first_look_point = self.local.ball_odometry
        #    self.f.dir_To_Guest()
        #    print('ball_coord = ', self.local.ball_odometry)
        #    print('direction_To_Guest = ', math.degrees(self.f.direction_To_Guest), 'degrees')
        #    if self.glob.robot_see_ball <= 0:
        #        self.motion.turn_To_Course(self.local.coord_odometry[2]+ 2 * math.pi / 3)
        #        continue
        #    player_from_ball_yaw = coord2yaw(self.local.coord_odometry[0] - self.local.ball_odometry[0],
        #                                                  self.local.coord_odometry[1] - self.local.ball_odometry[1]) - self.f.direction_To_Guest
        #    player_from_ball_yaw = self.norm_yaw(player_from_ball_yaw)
        #    player_in_front_of_ball = -math.pi/2 < player_from_ball_yaw < math.pi/2
        #    player_in_fast_kick_position = (player_from_ball_yaw > 2.5 or player_from_ball_yaw < -2.5) and self.glob.ball_distance < 0.6
        #    if self.glob.ball_distance > 0.35  and not player_in_fast_kick_position:
        #        if self.glob.ball_distance> 3: stop_Over = True
        #        else: stop_Over = False
        #        direction_To_Ball = math.atan2((self.local.ball_odometry[1] - self.local.coord_odometry[1]), (self.local.ball_odometry[0] - self.local.coord_odometry[0]))
        #        self.motion.far_distance_straight_approach(self.local.ball_odometry, direction_To_Ball, stop_Over = stop_Over)
        #        continue
        #    if player_in_front_of_ball or not player_in_fast_kick_position:
        #        self.go_Around_Ball(self.glob.ball_distance, self.glob.ball_course)
        #        continue
        #    self.motion.turn_To_Course(self.f.direction_To_Guest)
        #    #if first_shoot:
        #    success_Code = self.motion.near_distance_ball_approach_and_kick_streaming(self.f.direction_To_Guest)
        #    first_shoot = False
        self.glob.monitor_is_on = True
        self.f = Forward_Vector_Matrix(self.motion, self.local, self.glob)
        while (True):
            if self.motion.falling_Flag != 0:
                if self.motion.falling_Flag == 3: break
                self.motion.falling_Flag = 0
                #self.local.coordinate_fall_reset()
                self.motion.head_Return(0, self.motion.neck_play_pose)
            if self.glob.camera_down_Flag == True: self.glob.camera_reset()
                    #print('Camera resetting')
                    #self.glob.camera_down_Flag = False
                    #self.glob.vision.event.set()
                    #new_stm_channel  = self.STM_channel(self.glob)
                    #self.glob.stm_channel = new_stm_channel
                    #self.glob.rcb = self.glob.stm_channel.rcb
                    #new_vision = self.Vision_RPI(self.glob)
                    #self.glob.vision = new_vision
                    #self.motion.vision = self.glob.vision
                    #self.local.vision = self.glob.vision
                    ##self.glob.vision.camera_thread.start()
            #success_Code, napravl, dist, speed = self.motion.seek_Ball_In_Pose(fast_Reaction_On = True, with_Localization = False,
            #                                                                  very_Fast = True, first_look_point=first_look_point)
            #time.sleep(1) # this is to look around for ball 
            if self.glob.robot_see_ball <= 0:
                self.motion.head_Return(0, self.motion.neck_play_pose)
                success_Code, napravl, dist, speed = self.motion.seek_Ball_In_Pose(fast_Reaction_On = True, with_Localization = False,
                                                                                  very_Fast = False)
                self.motion.head_Return(0, self.motion.neck_play_pose)
            #self.glob.pf_coord = self.local.coord_odometry
            self.glob.local.localisation_Complete()
            self.f.dir_To_Guest()
            print('direction_To_Guest = ', round(math.degrees(self.f.direction_To_Guest)), 'degrees')
            print('coord =', round(self.local.coord_odometry[0],2), round(self.local.coord_odometry[1],2), 'ball =', round(self.local.ball_odometry[0],2), round(self.local.ball_odometry[1],2))
            if self.glob.robot_see_ball < -6:
                print('Seek ball')
                self.motion.jump_turn(self.local.coord_odometry[2]+ 2 * math.pi / 3)
                continue
            player_from_ball_yaw = coord2yaw(self.local.coord_odometry[0] - self.local.ball_odometry[0],
                                                          self.local.coord_odometry[1] - self.local.ball_odometry[1]) - self.f.direction_To_Guest
            player_from_ball_yaw = self.norm_yaw(player_from_ball_yaw)
            player_in_front_of_ball = -math.pi/2 < player_from_ball_yaw < math.pi/2
            player_in_fast_kick_position = (player_from_ball_yaw > 2.0 or player_from_ball_yaw < -2.0) and self.glob.ball_distance < 0.6
            if self.glob.ball_distance > 0.35  and not player_in_fast_kick_position:
                direction_To_Ball = math.atan2((self.local.ball_odometry[1] - self.local.coord_odometry[1]), (self.local.ball_odometry[0] - self.local.coord_odometry[0]))
                print('napravl :', self.glob.ball_course)
                print('direction_To_Ball', direction_To_Ball)
                #self.motion.far_distance_plan_approach(self.local.ball_odometry, self.f.direction_To_Guest, stop_Over = stop_Over)
                #self.motion.far_distance_straight_approach(self.local.ball_odometry, direction_To_Ball, stop_Over = False)
                self.motion.far_distance_straight_approach_streaming()
                continue
            if player_in_front_of_ball or not player_in_fast_kick_position:
                self.go_Around_Ball(self.glob.ball_distance, self.glob.ball_course)
                continue
            if player_in_fast_kick_position:
                self.motion.jump_turn(self.f.direction_To_Guest)
                if self.f.kick_Power == 1: self.motion.kick_power = 100
                if self.f.kick_Power == 2: self.motion.kick_power = 60
                if self.f.kick_Power == 3: self.motion.kick_power = 20
                success_Code = self.motion.near_distance_ball_approach_and_kick_streaming(self.f.direction_To_Guest)


    def FIRA_penalty_Goalkeeper_main_cycle(self):
        self.motion.with_Vision = False
        self.glob.camera_streaming = False
        self.motion.head_Return(0, -1500)
        self.g = GoalKeeper(self.motion, self.local, self.glob)
        self.glob.obstacleAvoidanceIsOn = False
        first_Get_Up = True
        while (True):
            if self.motion.falling_Flag != 0:
                if self.motion.falling_Flag == 3: break
                self.motion.falling_Flag = 0
                self.local.coordinate_fall_reset()
            self.motion.jump_turn(0)
            success_Code, napravl, dist, speed = self.motion.seek_Ball_In_Pose(fast_Reaction_On = True, with_Localization = False,
                                                                              very_Fast = True )
            if dist != 0:
                #displacement = self.glob.ball_coord[1] - self.glob.pf_coord[1]
                displacement = dist * math.sin(napravl)
                if abs(displacement) > 0.1:
                    self.motion.near_distance_omni_motion(abs(displacement) * 1000, math.copysign(math.pi/2, displacement))


    def penalty_Goalkeeper_main_cycle(self):
        self.g = GoalKeeper(self.motion, self.local, self.glob)
        self.glob.obstacleAvoidanceIsOn = False
        first_Get_Up = True
        while (True):
            dist = -1.0
            if self.motion.falling_Flag != 0:
                if self.motion.falling_Flag == 3: break
                self.motion.falling_Flag = 0
                self.local.coordinate_fall_reset()
                self.g.turn_Face_To_Guest()
                if first_Get_Up:
                    first_Get_Up = False
                    self.g.goto_Center()
            while(dist < 0):
                a, napravl, dist, speed = self.motion.watch_Ball_In_Pose(penalty_Goalkeeper = True)
                napravl = self.glob.ball_course
                dist = self.glob.ball_distance
                speed = self.glob.ball_speed
                #print('speed = ', speed, 'dist  =', dist , 'napravl =', napravl)
                if abs(speed[0]) > 0.002 and dist < 1 :                         # if dangerous tangential speed
                    if speed[0] > 0:
                        self.motion.play_Motion_Slot(name ='PenaltyDefenceL')
                    else:
                        self.motion.play_Motion_Slot(name ='PenaltyDefenceR')
                    continue
                if speed[1] < - 0.01 and dist < 1.5 :                          # if dangerous front speed
                    self.motion.play_Motion_Slot(name = 'PanaltyDefenceReady_Fast')
                    self.motion.play_Motion_Slot(name = 'PenaltyDefenceF')
                    self.motion.pause_in_ms(5000)
                    self.motion.play_Motion_Slot(name = 'Get_Up_From_Defence')

                if (dist == 0 and napravl == 0) or dist > 2.5:
                    continue
                old_neck_pan, old_neck_tilt = self.motion.head_Up()
                if (dist <= 0.5         and 0 <= napravl <= math.pi/4):         self.g.scenario_A1( dist, napravl)
                if (dist <= 0.5         and math.pi/4 < napravl <= math.pi/2):  self.g.scenario_A2( dist, napravl)
                if (dist <= 0.5         and 0 >= napravl >= -math.pi/4):        self.g.scenario_A3( dist, napravl)
                if (dist <= 0.7         and -math.pi/4 > napravl >= -math.pi/2): self.g.scenario_A4( dist, napravl)
                if ((0.5 < dist < self.glob.landmarks['FIELD_LENGTH']/2) and (math.pi/18 <= napravl <= math.pi/4)): self.g.scenario_B1()
                if ((0.5 < dist < self.glob.landmarks['FIELD_LENGTH']/2) and (math.pi/4 < napravl <= math.pi/2)): self.g.scenario_B2()
                if ((0.5 < dist < self.glob.landmarks['FIELD_LENGTH']/2) and (-math.pi/18 >= napravl >= -math.pi/4)): self.g.scenario_B3()
                if ((0.7 < dist < self.glob.landmarks['FIELD_LENGTH']/2) and (-math.pi/4 > napravl >= -math.pi/2)): self.g.scenario_B4()
                self.motion.head_Return(old_neck_pan, old_neck_tilt)

    def go_Around_Ball(self, dist, napravl):
        print('go_Around_Ball')
        turning_radius = 0.20 # meters
        approach_distance = self.glob.params["KICK_ADJUSTMENT_DISTANCE_1"]
        #first_look_point= self.glob.ball_coord
        #success_Code, napravl, dist, speed = self.motion.seek_Ball_In_Pose(fast_Reaction_On = True, with_Localization = False,
        #                                                                  very_Fast = True, first_look_point=first_look_point)
        #if success_Code != True: return
        if dist > 0.5: return
        correction_x = dist * math.cos(napravl)
        correction_y = dist * math.sin(napravl)
        if self.glob.use_particle_filter:
            current_yaw = self.glob.pf_coord[2]
        else:
            current_yaw = self.local.coord_odometry[2]
        alpha = self.f.direction_To_Guest - current_yaw
        alpha = self.motion.norm_yaw(alpha)
        initial_body_yaw = current_yaw
        correction_napravl = math.atan2(correction_y, (correction_x - turning_radius))
        correction_dist = math.sqrt(correction_y**2 + (correction_x - turning_radius)**2)
        #old_neck_pan, old_neck_tilt = self.motion.head_Up()
        #self.motion.first_Leg_Is_Right_Leg = True

        #self.motion.walk_Initial_Pose()
        if napravl * alpha > 0:
            #self.motion.turn_To_Course(self.glob.pf_coord[2] + napravl, one_Off_Motion = False)
            self.motion.jump_turn(self.norm_yaw(current_yaw + napravl))
            #self.motion.walk_Restart()
            self.motion.first_Leg_Is_Right_Leg = True
            self.motion.walk_Initial_Pose()
            #self.motion.near_distance_omni_motion((dist - turning_radius) * 1000 , 0, one_Off_Motion = False)
            self.motion.near_distance_omni_motion(dist * 1000 - approach_distance , 0, one_Off_Motion = False)
            alpha = self.motion.norm_yaw(alpha - napravl)
            initial_body_yaw += napravl
        else:
            self.motion.jump_turn(self.norm_yaw(current_yaw + napravl))
            displacement = correction_dist*1000 * math.sin(correction_napravl)
            if displacement > 0:  self.motion.first_Leg_Is_Right_Leg = False
            else: self.motion.first_Leg_Is_Right_Leg = True
            self.motion.walk_Initial_Pose()
            #self.motion.near_distance_omni_motion(correction_dist*1000, correction_napravl, one_Off_Motion = False)
            self.motion.near_distance_omni_motion(dist * 1000 - approach_distance, 0, one_Off_Motion = False)
        if alpha >= 0:
            if self.motion.first_Leg_Is_Right_Leg == False:
                change_legs = True
                self.motion.walk_Cycle(0, 0, 0, 1, 3, half = True)
            else:
                change_legs = False
                #self.motion.walk_Restart()
            self.motion.first_Leg_Is_Right_Leg = True
            side_step_yield = self.motion.side_step_right_yield
            invert = 1
        else:
            if self.motion.first_Leg_Is_Right_Leg == True:
                change_legs = True
                self.motion.walk_Cycle(0, 0, 0, 1, 3, half = True)
            else:
                change_legs = False
                #self.motion.walk_Restart()
            self.motion.first_Leg_Is_Right_Leg = False
            side_step_yield = self.motion.side_step_left_yield * 0.75
            invert = -1
        #print('6self.motion.first_Leg_Is_Right_Leg:', self.motion.first_Leg_Is_Right_Leg)
        yaw_increment_at_side_step =  math.copysign(2 * math.asin(side_step_yield / 2 / (turning_radius * 1000)), alpha)
        number_Of_Cycles = int(round(abs(alpha / yaw_increment_at_side_step)))+1
        sideLength = 20 * 0.75
        for cycle in range(1, number_Of_Cycles):
            stepLength = 0
            if self.glob.robot_see_ball == 5:
                stepLength = self.glob.ball_distance * 1000 * math.cos(self.glob.ball_course) - approach_distance
                if abs(stepLength) > 30: stepLength *= 30/abs(stepLength)
                print("stepLength: ", stepLength)
            self.motion.refresh_Orientation()
            rotation = initial_body_yaw + cycle * yaw_increment_at_side_step - self.motion.imu_body_yaw() * 1
            rotation = self.motion.normalize_rotation(rotation)
            #rotation = 0
            self.motion.walk_Cycle(stepLength, sideLength, invert*rotation, cycle, number_Of_Cycles)
        self.motion.walk_Final_Pose()
        #self.motion.first_Leg_Is_Right_Leg = True

    def basketball_main_cycle(self, pressed_button):
        
        intercom = self.glob.stm_channel.zubr       # used for communication between head and zubr-controller with memIGet/memISet commands

        #throw = [
        #        [ 100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 450, 0, 4700, 2667, 0, 0, 0, 0, 0, 0 ],
        #        [ 100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 450, 0, 4700, 2667, 0, 0, 0, 0, 0, 0 ],
        #        [ 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2000, 0, 4700, 0, 0, 0, 0, 0, 0, 0 ],
        #        [ 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2000, 0, 4700, 0, 0, 0, 0, 0, 0, 0 ],
        #        [ 50, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]
        #        ]
        #throw[2][18] -= int(self.motion.params['BASKETBALL_DISTANCE'])
        #throw[3][18] -= int(self.motion.params['BASKETBALL_DISTANCE'])
        #pickUp = [
        #        [ 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ],
        #        [ 100, 0, -900, 0, -3300, 0, 0, 0, 2667, 0, -3500, 0, 0, 900, 0, 3300, 0, 0, 0, -2667, 0, 3500, 0, 0, 0, 0, 0, 0 ],
        #        [ 50, 0, -900, 0, -3300, 0, 0, 600, 2300, 385, -3500, 0, 0, 900, 0, 3300, 0, 0, -600, -2300, -385, 3500, 0, 0, 0, 0, 0, 0 ],
        #        [ 50, 0, 0, 0, 0, 0, 0, 600, 2667, 385, -2667, 0, 0, 0, 0, 0, 0, 0, -600, -2667, -385, 2667, 0, 0, 0, 0, 0, 0 ],
        #        [ 50, 0, 0, 0, 0, 0, 0, 600, 3000, 385, -3000, 0, 0, 0, 0, 0, 0, 0, -700, -2000, -385, 2500, 0, 0 ],
        #        [ 50, 0, 0, 0, 0, 0, 0, 700, 3425, 385, -3300, 0, 0, 0, 0, 0, 0, 0, -900, -1500, -385, 2200, 0, 0, 0, 0, 0, 0 ],
        #        [ 50, 0, 0, 0, 0, 0, 0, 910, 2667, 600, -2600, 0, 0, 0, 0, 0, 0, 0, -2100, -600, -680, 1000, 0, 0, 0, 0, 0, 0 ],
        #        [ 50, 0, 0, 0, 0, 0, 0, 0, 2667, 600, -2600, 0, 0, 0, 0, 0, 0, 0, -2100, 0, -680, 1000, 0, 0 ],
        #        [ 50, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1100, 0, -680, 2000, 0, 0, 0, 0, 0, 0 ],
        #        [ 50, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -433, 0, -680, 2667, 0, 0, 0, 0, 0, 0 ],
        #        [ 100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -433, 0, 2500, 2667, 0, 0, 0, 0, 0, 0 ],
        #        [ 100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -350, 0, 4700, 2667, 0, 0, 0, 0, 0, 0 ]
        #        ]
        #for i in range(2, 8,1):
        #    pickUp[i][9] += int(self.motion.params['BASKETBALL_CLAMPING'])
        #    pickUp[i][20] -= int(self.motion.params['BASKETBALL_CLAMPING'])
        #for i in range(4):
        #    throw[i][19] += int(self.motion.params['BASKETBALL_DIRECTION'])

        if pressed_button == 'approach_test' or pressed_button == 'start_with_approach' :
            position_x, position_y = 0, 0
            self.motion.head_Return(0, -2000)
            for _ in range(500):
                result, course, distance = self.glob.vision.seek_Ball_In_Frame_N(with_Localization = False)
                x = distance * math.cos(course) * 1000 - 15
                y = distance * math.sin(course) * 1000
                if abs(y) > 10:
                    if y > 0:
                        fraction = min(1, abs(y) / self.glob.jump_left_yield)
                        self.motion.one_jump_left(fraction)
                        position_y += self.glob.jump_left_yield * fraction
                    if y < 0:
                        fraction = min(1, abs(y) / self.glob.jump_right_yield)
                        self.motion.one_jump_right(fraction)
                        position_y -= self.glob.jump_right_yield * fraction
                    time.sleep(0.5)
                    self.motion.jump_turn(0)
                    time.sleep(0.5)
                if x > 10:
                    fraction = min(1, abs(x) / self.glob.jump_forward_yield)
                    self.motion.one_jump_forward(fraction)
                    position_x += self.glob.jump_forward_yield * fraction
                    time.sleep(0.5)
                    self.motion.jump_turn(0)
                    time.sleep(0.5)
                self.motion.refresh_Orientation()
                if abs(y) < 10 and x < 10: 
                    break
            # Basketball_PickUp start
            var = roki2met.roki2met.Basketball_PickUp_v2_S1
            self.glob.rcb.motionPlay(13)                                # Basketball_PickUp
            while True:
                ok, frameCount = intercom.memIGet(var.frameCount)
                if ok: print('frameCount :', frameCount)
                else: print(intercom.GetError())
                if frameCount == 1: break
                time.sleep(0.25)
            intercom.memISet(var.clamping, int(self.motion.params['BASKETBALL_CLAMPING']))         # clamping gap for ball gripping -50 best value
            intercom.memISet(var.steps, 0)    # side shift steps to provide 80mm shifting to right. 17 is the best value
            intercom.memISet(var.pitStop, 1)                                                       # ignition
            time.sleep(13)
            # Basketball_PickUp end
            target_pos = [0, -80]
            for _ in range(500):
                shift_x = target_pos[0] - position_x
                shift_y = target_pos[1] - position_y
                if abs(shift_x) > 5:
                    if (shift_x) < 0:
                        fraction = min(1, abs(shift_x) / self.glob.jump_backward_yield)
                        self.motion.one_jump_backward(fraction, hands_on = False)
                        position_x -= self.glob.jump_backward_yield * fraction
                    else:
                        fraction = min(1, abs(shift_x) / self.glob.jump_forward_yield)
                        self.motion.one_jump_forward(fraction, hands_on = False)
                        position_x += self.glob.jump_forward_yield * fraction
                    time.sleep(0.5)
                    self.motion.jump_turn(0, hands_on = False)
                    time.sleep(0.5)
                if abs(shift_y) > 5:
                    if (shift_y) < 0:
                        fraction = min(1, abs(shift_y) / self.glob.jump_right_yield)
                        self.motion.one_jump_right(fraction, hands_on = False)
                        position_y -= self.glob.jump_right_yield * fraction
                    else:
                        fraction = min(1, abs(shift_y * 1000) / self.glob.jump_left_yield)
                        self.motion.one_jump_left(fraction, hands_on = False)
                        position_y += self.glob.jump_left_yield * fraction
                    time.sleep(0.5)
                    self.motion.jump_turn(0, hands_on = False)
                    time.sleep(0.5)
                if abs(shift_y) < 5 and abs(shift_x) < 5: break
            # Basketball_PickUp start
            self.motion.head_Return(0, 0)
            var = roki2met.roki2met.Basketball_PickUp_v2_S2
            self.glob.rcb.motionPlay(14)                                # Basketball_PickUp
            while True:
                ok, frameCount = intercom.memIGet(var.frameCount)
                if ok: print('frameCount :', frameCount)
                else: print(intercom.GetError())
                if frameCount == 1: break
                time.sleep(0.25)
            intercom.memISet(var.clamping, int(self.motion.params['BASKETBALL_CLAMPING']))         # clamping gap for ball gripping -50 best value
            intercom.memISet(var.steps, 0)    # side shift steps to provide 80mm shifting to right. 17 is the best value
            intercom.memISet(var.pitStop, 1)                                                       # ignition
            time.sleep(25)
            # Basketball_PickUp end
            #return

        if pressed_button == 'approach_test_low_level' :
            jump_mode = roki2met.roki2met.jump_motions_local.jump_mode_local
            jump_motions = roki2met.roki2met.jump_motions_local
            self.motion.head_Return(0, -2000)
            for _ in range(500):
                result, course, distance = self.glob.vision.seek_Ball_In_Frame_N(with_Localization = False)
                x = distance * math.cos(course) * 1000
                y = distance * math.sin(course) * 1000
                if abs(y) > 10:
                    self.glob.rcb.motionPlay(7)
                    while True:
                        ok, frameCount = intercom.memIGet(jump_motions.frameCount)
                        if ok: print('frameCount :', frameCount)
                        else: print(intercom.GetError())
                        if frameCount == 1: break
                        time.sleep(0.25)
                    if y > 0:
                        intercom.memISet(jump_mode, 103) # jump left
                        self.local.coord_shift[1] = self.glob.jump_left_yield / 1000
                    if y < 0:
                        intercom.memISet(jump_mode, 104) # jump right
                        self.local.coord_shift[1] = - self.glob.jump_right_yield / 1000
                    intercom.memISet(jump_motions.pitStop, 1)
                    time.sleep(0.5)
                    self.motion.jump_turn(0)
                if x > 10:
                    self.glob.rcb.motionPlay(7)
                    while True:
                        ok, frameCount = intercom.memIGet(jump_motions.frameCount)
                        if ok: print('frameCount :', frameCount)
                        else: print(intercom.GetError())
                        if frameCount == 1: break
                        time.sleep(0.25)
                    intercom.memISet(jump_mode, 101)   # jump forward
                    self.local.coord_shift[0] = self.glob.jump_forward_yield / 1000
                    intercom.memISet(jump_motions.pitStop, 1)
                    time.sleep(0.5)
                    self.motion.jump_turn(0)
                self.motion.refresh_Orientation()
                self.local.coordinate_record(odometry = True, shift = True)
                self.local.refresh_odometry()
                if abs(y) < 10 and x < 0: 
                    break
            self.motion.head_Return(0, 0)
            # Basketball_PickUp start
            var = roki2met.roki2met.Basketball_PickUp_v2_S1
            self.glob.rcb.motionPlay(13)                                # Basketball_PickUp
            while True:
                ok, frameCount = intercom.memIGet(var.frameCount)
                if ok: print('frameCount :', frameCount)
                else: print(intercom.GetError())
                if frameCount == 1: break
                time.sleep(0.25)
            intercom.memISet(var.clamping, int(self.motion.params['BASKETBALL_CLAMPING']))         # clamping gap for ball gripping -50 best value
            intercom.memISet(var.steps, 0)    # side shift steps to provide 80mm shifting to right. 17 is the best value
            intercom.memISet(var.pitStop, 1)                                                       # ignition
            time.sleep(35)
            # Basketball_PickUp end
            target_pos = [0, -0.08, 0]
            for _ in range(500):
                print('coord_odometry : ', self.local.coord_odometry)
                shift_x = target_pos[0] - self.local.coord_odometry[0]
                shift_y = target_pos[1] - self.local.coord_odometry[1]
                if abs(shift_x) > 0.005:
                    self.glob.rcb.motionPlay(7)
                    while True:
                        ok, frameCount = intercom.memIGet(jump_motions.frameCount)
                        if ok: print('frameCount :', frameCount)
                        else: print(intercom.GetError())
                        if frameCount == 1: break
                        time.sleep(0.25)
                    if (shift_x) < 0:
                        if abs(shift_x) > self.glob.jump_backward_yield / 1000:
                            intercom.memISet(jump_mode, 102)   # jump backward
                            self.local.coord_shift[0] = - self.glob.jump_backward_yield / 1000
                        else:
                            attenuation = int(abs(shift_x) /  self.glob.jump_backward_yield * 1000 * 10)
                            jump_command = 100 + 2 + attenuation * 10
                            intercom.memISet(jump_mode, jump_command)   # jump backward
                            self.local.coord_shift[0] = - abs(shift_x)
                    else:
                        if abs(shift_x) > self.glob.jump_forward_yield / 1000:
                            intercom.memISet(jump_mode, 101)   # jump forward
                            self.local.coord_shift[0] = self.glob.jump_forward_yield / 1000
                        else:
                            attenuation = int(abs(shift_x) /  self.glob.jump_forward_yield * 1000 * 10)
                            jump_command = 100 + 1 + attenuation * 10
                            intercom.memISet(jump_mode, jump_command)   # jump forward
                            self.local.coord_shift[0] =  abs(shift_x)
                    intercom.memISet(jump_motions.pitStop, 1)
                    time.sleep(0.5)
                    self.motion.jump_turn(0)
                if abs(shift_y) > 0.005:
                    self.glob.rcb.motionPlay(7)
                    while True:
                        ok, frameCount = intercom.memIGet(jump_motions.frameCount)
                        if ok: print('frameCount :', frameCount)
                        else: print(intercom.GetError())
                        if frameCount == 1: break
                        time.sleep(0.25)
                    if (shift_y) < 0:
                        if abs(shift_y) > self.glob.jump_right_yield / 1000:
                            intercom.memISet(jump_mode, 104)   # jump right
                            self.local.coord_shift[1] = - self.glob.jump_right_yield / 1000
                        else:
                            attenuation = int(abs(shift_y) /  self.glob.jump_right_yield * 1000 * 10)
                            jump_command = 100 + 4 + attenuation * 10
                            intercom.memISet(jump_mode, jump_command)   # jump right
                            self.local.coord_shift[1] = - abs(shift_y)
                    else:
                        if abs(shift_y) > self.glob.jump_left_yield / 1000:
                            intercom.memISet(jump_mode, 103)   # jump left
                            self.local.coord_shift[1] = self.glob.jump_left_yield / 1000
                        else:
                            attenuation = int(abs(shift_y) /  self.glob.jump_left_yield * 1000 * 10)
                            jump_command = 100 + 3 + attenuation * 10
                            intercom.memISet(jump_mode, jump_command)   # jump left
                            self.local.coord_shift[1] =  abs(shift_y)
                    intercom.memISet(jump_motions.pitStop, 1)
                    time.sleep(0.5)
                    self.motion.jump_turn(0)
                self.motion.refresh_Orientation()
                self.local.coordinate_record(odometry = True, shift = True)
                self.local.refresh_odometry()
                if abs(shift_y) < 0.005 and abs(shift_x) < 0.005: break
            # Basketball_PickUp start
            var = roki2met.roki2met.Basketball_PickUp_v2_S2
            self.glob.rcb.motionPlay(14)                                # Basketball_PickUp
            while True:
                ok, frameCount = intercom.memIGet(var.frameCount)
                if ok: print('frameCount :', frameCount)
                else: print(intercom.GetError())
                if frameCount == 1: break
                time.sleep(0.25)
            intercom.memISet(var.clamping, int(self.motion.params['BASKETBALL_CLAMPING']))         # clamping gap for ball gripping -50 best value
            intercom.memISet(var.steps, 0)    # side shift steps to provide 80mm shifting to right. 17 is the best value
            intercom.memISet(var.pitStop, 1)                                                       # ignition
            time.sleep(35)
            # Basketball_PickUp end

            return

        if pressed_button == 'start' or pressed_button == 'pick_up_test' :
            #self.motion.play_Soft_Motion_Slot( name = 'pickUp', motion_list = pickUp)
            var = roki2met.roki2met.Basketball_PickUp_v2_R2
            self.glob.rcb.motionPlay(10)                                # Basketball_PickUp
            while True:
                ok, frameCount = intercom.memIGet(var.frameCount)
                if ok: print('frameCount :', frameCount)
                else: print(intercom.GetError())
                if frameCount == 1: break
                time.sleep(0.25)
            intercom.memISet(var.clamping, int(self.motion.params['BASKETBALL_CLAMPING']))         # clamping gap for ball gripping -50 best value
            intercom.memISet(var.steps, int(self.motion.params['BASKETBALL_SIDE_SHIFT_STEPS']))    # side shift steps to provide 80mm shifting to right. 17 is the best value
            intercom.memISet(var.pitStop, 1)                                                       # ignition
            time.sleep(35)

        if pressed_button == 'start' or pressed_button == 'start_with_approach' or pressed_button == 'throw_test' or pressed_button == 'throw_control':
            var = roki2met.roki2met.Basketball_Throw
            
            #int_voltage = self.motion.stm_channel.read_voltage_from_body()[1]
            #voltage = int_voltage / 270.2
            voltage = 12
            #print("voltage = ", round(voltage, 2), " 'BASKETBALL_DISTANCE': ", 
            #      int(self.motion.params['BASKETBALL_DISTANCE']))
            #self.motion.play_Soft_Motion_Slot( name = 'throw', motion_list = throw)
            self.glob.rcb.motionPlay(9)                                # Basketball_Throw
            while True:
                ok, frameCount = intercom.memIGet(var.frameCount)
                if ok: print('frameCount :', frameCount)
                else: print(intercom.GetError())
                if frameCount == 1: break
                time.sleep(0.25)
            
            intercom.memISet(var.pitStop, 1)                                                       # ignition
            time.sleep(3)
            for i in range(10):
                result, displacement = self.glob.vision.detect_Basket_in_One_Shot_N()
                if result: break
            if result:
                os.system("espeak -ven-m1 -a"+ '200' + " " + "'I see basket'")
            else: 
                os.system("espeak -ven-m1 -a"+ '200' + " " + "'I don_t see basket'")
                displacement = 0
            time.sleep(3)
            corrected_direction = int(self.motion.params['BASKETBALL_DIRECTION']) + int(displacement/10 /360 * 16384)

            if pressed_button == 'start' or pressed_button == 'start_with_approach' or pressed_button == 'throw_control':
                if int(self.motion.params['BASKETBALL_BATTERY_NUMBER']) == 1:
                    voltage_correction = 10469.14829 + -1506.46893 * voltage + 55.11101 * voltage ** 2  
                elif int(self.motion.params['BASKETBALL_BATTERY_NUMBER']) == 3:
                    voltage_correction = 50435.37775 + -7951.56631 * voltage + 314.69345 * voltage ** 2
                else: voltage_correction = 0
                corrected_distance = int(self.motion.params['BASKETBALL_DISTANCE_CORRECTION'] + int(voltage_correction))
            else:
                corrected_distance = int(self.motion.params['BASKETBALL_DISTANCE'])
            intercom.memISet(var.distance, corrected_distance)         # start acceleration angle -350 best value
            intercom.memISet(var.direction, corrected_direction)       # direction to correct 200 best value
            intercom.memISet(var.startStop, 1)                                                       # ignition
            time.sleep(3)

            if pressed_button == 'throw_test':
                labels = [[], [], [], ['good', 'Bad'], []]
                pressed_button = self.motion.push_Button(labels, message = "'Give me feed back'")
                if pressed_button == 'good':
                    today = datetime.date.today()
                    try:
                        records = np.load("basketball_records.npy")
                    except Exception:
                        records = np.zeros((1,6), dtype = np.int16)
                    record = np.array([[int(today.year), int(today.month), int(today.day),
                                        self.motion.params['BASKETBALL_BATTERY_NUMBER'], int_voltage,
                                        int(self.motion.params['BASKETBALL_DISTANCE'])]], dtype = np.int16)
                    new_records = np.append(records, record, axis=0)
                    np.save("basketball_records.npy", new_records)


    def dance_main_cycle(self):
        #if self.glob.SIMULATION == 5:
            #self.motion.play_Soft_Motion_Slot( name = 'Basketball3')
        #else:
        self.motion.pause_in_ms(10000)  
        self.motion.play_Soft_Motion_Slot( name = 'record-2024-12-25', soft_factor = 3)
        self.motion.pause_in_ms(500)
        for _ in range(5):
            self.motion.play_Soft_Motion_Slot( name = 'record-2024-12-25rs', soft_factor = 3)
            self.motion.pause_in_ms(500)
            self.motion.play_Soft_Motion_Slot( name = 'record-2024-12-25s', soft_factor = 3)
            self.motion.pause_in_ms(500)
        self.motion.play_Soft_Motion_Slot( name = 'record-2024-12-25r', soft_factor = 3)
        return
            
        self.glob.record_motions = True
        self.motion.with_Vision = False
        self.motion.fr2 = 18
        stepLength = 45
        sideLength , rotation = 0, 0
        number_Of_Cycles = 2
        self.motion.first_Leg_Is_Right_Leg = True
        self.motion.walk_Initial_Pose()
        for cycle in range(number_Of_Cycles):
            stepLength1 = stepLength
            self.motion.refresh_Orientation()
            rotation = -self.motion.imu_body_yaw() * 1.0
            rotation = self.motion.normalize_rotation(rotation)
            rotation = 0
            if not self.motion.first_Leg_Is_Right_Leg: rotation *= -1
            self.motion.walk_Cycle(stepLength1,sideLength, rotation,cycle, number_Of_Cycles + 1)
        self.motion.walk_Cycle(stepLength1,sideLength, rotation,cycle, number_Of_Cycles + 1, half = True)

        record_name = 'record-'+ str(datetime.date.today().isoformat())
        record_dict = {}
        record_data = []
        pageNames = []
        for i in range(len(self.motion.motions_recorded)):
            record_line = [1]
            for j in range(len(self.motion.motions_recorded[i])):
                record_line.append(int(self.motion.motions_recorded[i][j] * 1698))
            record_data.append(record_line)
            pageNames.append('page ' + str(i))
        record_dict = {record_name: record_data, 'pageNames': pageNames}
        filename = self.glob.current_work_directory + record_name + ".json"
        with open(filename, "w") as f:
            json.dump(record_dict, f)





    def sprint(self, second_pressed_button):
        if self.glob.SIMULATION == 5:
            from Soccer.Vision import lookARUCO
            # Pipeline variables
            size = Value('i', 0)       # horizontal size of ARUCO code on picture
            side_shift = Value('i', 0) # horizontal shift of ARUCO code from center of picture 
            stopFlag = Value(c_bool, False)
            aruco_angle_horizontal = Value('d', 0)
            distance = Value('d', 0)
            id = self.glob.params["SPRINT_ARUCO_ID"]

            camera_thread = threading.Thread(
                target=lookARUCO.track_from_vision,
                args=(self.glob.vision, size, side_shift, aruco_angle_horizontal, distance, stopFlag, id),
                daemon=True,
            )
            camera_thread.start()
            self.motion.direction_To_Attack = 0
            self.motion.activation()
            self.motion.head_Return(0, -1000)
            labels = [[], [], [], ['start'], []]
            pressed_button = self.motion.push_Button(labels)
            self.motion.with_Vision = False
            sideLength = 0
            direction_by_imu = self.motion.imu_body_yaw()
            while True:
                direction = 0
                if self.motion.falling_Flag != 0: self.motion.falling_Flag = 0
                stepLength = 45
                number_Of_Cycles = 300
                self.motion.walk_Initial_Pose()
                reverseFlag = False
                countdown = 0
                for cycle in range(number_Of_Cycles):
                    if self.motion.falling_Flag != 0: break
                    stepLength1 = stepLength
                    if cycle ==0 : stepLength1 = stepLength/3
                    if cycle ==1 : stepLength1 = stepLength/3 * 2
                    self.motion.refresh_Orientation()
                    aruco_angle = aruco_angle_horizontal.value 
                    aruco_size = size.value / math.cos(aruco_angle)
                    if aruco_size == 0: corr = 1.1
                    else:
                        corr = aruco_size / self.glob.params['SPRINT_ARUCO_SIZE']  + 1.1 / aruco_size
                    corr = 1.1
                    if cycle > 0:  direction =  aruco_angle * corr
                    print('direction : ', direction )
                    if aruco_size != 0: 
                        direction_by_imu = aruco_angle + self.motion.imu_body_yaw()
                        rotation = direction
                    else:
                        rotation = (direction_by_imu - self.motion.imu_body_yaw()) * 1.1
                    rotation = self.motion.normalize_rotation(rotation)
                    self.motion.walk_Cycle(stepLength1,sideLength, rotation,cycle, number_Of_Cycles)
                    #aruco_dist = distance.value
                    #if 0 < aruco_dist < self.glob.params['SPRINT_REVERSE_DISTANCE_CM'] * 2 :
                    if aruco_size > self.glob.params['SPRINT_ARUCO_SIZE'] : reverseFlag = True
                    if reverseFlag: countdown += 1
                    if countdown == 2:
                        print('Reverse')
                        #self.motion.walk_Cycle(stepLength1,sideLength, rotation,cycle, cycle + 1)
                        #self.motion.walk_Final_Pose()
                        stepLength1 = stepLength/3 * 2
                        self.motion.refresh_Orientation()
                        rotation =  -self.motion.imu_body_yaw() * 1.1
                        rotation = self.motion.normalize_rotation(rotation)
                        self.motion.walk_Cycle(stepLength1,sideLength, rotation,cycle, number_Of_Cycles)
                        stepLength1 = stepLength/3
                        self.motion.refresh_Orientation()
                        rotation =  -self.motion.imu_body_yaw() * 1.1
                        rotation = self.motion.normalize_rotation(rotation)
                        self.motion.walk_Cycle(stepLength1,sideLength, rotation,cycle, number_Of_Cycles)
                        break
                if self.motion.falling_Flag != 0: continue
                number_Of_Cycles = 600
                stepLength = -20
                self.motion.stepHeight = 40
                #self.glob.params['BODY_TILT_AT_WALK'] -= 0.01
                #self.motion.walk_Initial_Pose()
                for cycle in range(number_Of_Cycles):
                    stepLength1 = stepLength
                    if cycle ==0 : stepLength1 = stepLength/3
                    if cycle ==1 : stepLength1 = stepLength/3 * 2
                    self.motion.refresh_Orientation()
                    rotation = - self.motion.imu_body_yaw() * 1.1
                    rotation = self.motion.normalize_rotation(rotation)
                    self.motion.walk_Cycle(stepLength1,sideLength, rotation,cycle +1, number_Of_Cycles +1) 
                self.motion.walk_Final_Pose()

        else:
            self.motion.direction_To_Attack = 0
            self.motion.activation()
            self.motion.head_Return(0, -1000)
            sideLength = 0
            while True:
                if self.motion.falling_Flag != 0: self.motion.falling_Flag = 0
                stepLength = 64
                number_Of_Cycles = 300
                self.motion.walk_Initial_Pose()
                for cycle in range(number_Of_Cycles):
                    if self.motion.falling_Flag != 0: break
                    stepLength1 = stepLength
                    if cycle ==0 : stepLength1 = stepLength/3
                    if cycle ==1 : stepLength1 = stepLength/3 * 2
                    self.motion.refresh_Orientation()
                    rotation =  -self.motion.imu_body_yaw() * 1.1
                    rotation = self.motion.normalize_rotation(rotation)
                    self.motion.walk_Cycle(stepLength1,sideLength, rotation,cycle, number_Of_Cycles)
                    if cycle == 6:
                        print('Reverse')
                        stepLength1 = stepLength/3 * 2
                        self.motion.refresh_Orientation()
                        rotation =  -self.motion.imu_body_yaw() * 1.1
                        rotation = self.motion.normalize_rotation(rotation)
                        self.motion.walk_Cycle(stepLength1,sideLength, rotation,cycle, number_Of_Cycles)
                        stepLength1 = stepLength/3
                        self.motion.refresh_Orientation()
                        rotation =  -self.motion.imu_body_yaw() * 1.1
                        rotation = self.motion.normalize_rotation(rotation)
                        self.motion.walk_Cycle(stepLength1,sideLength, rotation,cycle, number_Of_Cycles)
                        break
                if self.motion.falling_Flag != 0: continue
                number_Of_Cycles = 6
                stepLength = -50
                self.glob.params['BODY_TILT_AT_WALK'] -= 0.01
                for cycle in range(number_Of_Cycles):
                    stepLength1 = stepLength
                    if cycle ==0 : stepLength1 = stepLength/3
                    if cycle ==1 : stepLength1 = stepLength/3 * 2
                    self.motion.refresh_Orientation()
                    rotation = - self.motion.imu_body_yaw() * 1.1
                    rotation = self.motion.normalize_rotation(rotation)
                    self.motion.walk_Cycle(stepLength1,sideLength, rotation,cycle +1, number_Of_Cycles + 1)
                self.motion.walk_Final_Pose()

            
    def sprint_fast(self, second_pressed_button):
        self.motion.params['SPRINT_HIP_TILT'] = 400
        self.motion.params['SPRINT_STEP_LENGTH'] = 40
        self.motion.params['SPRINT_GAIT_HEIGHT'] = 135
        self.motion.params['SPRINT_STEP_HEIGHT'] = 30
        self.motion.params['SPRINT_FPS'] = 2
        self.motion.params['SPRINT_UGOL_TORSA'] = 0.1
        self.motion.params['SPRINT_IMU_FACTOR'] = -2
        self.motion.params['SPRINT_VISION_FACTOR'] = 1

        self.motion.params['SPRINT_HIP_TILT'] = 0
        self.motion.params['SPRINT_STEP_LENGTH'] = 60
        self.motion.params['SPRINT_GAIT_HEIGHT'] = 180
        self.motion.params['SPRINT_STEP_HEIGHT'] = 40
        self.motion.params['SPRINT_FPS'] = 4
        self.motion.params['SPRINT_UGOL_TORSA'] = 0.7
        self.motion.params['SPRINT_IMU_FACTOR'] = -2
        self.motion.params['SPRINT_VISION_FACTOR'] = 1

        if self.glob.SIMULATION == 5:
            from Soccer.Vision import lookARUCO

            # Pipeline variables
            size = Value('i', 0)       # horizontal size of ARUCO code on picture
            side_shift = Value('i', 0) # horizontal shift of ARUCO code from center of picture 
            stopFlag = Value(c_bool, False)
            #cx = Value('i', 0)
            #cy = Value('i', 0)
            aruco_angle_horizontal = Value('d', 0)

            distance = Value('d', 0)
            id = self.glob.params["SPRINT_ARUCO_ID"]
            camera_thread = threading.Thread(
                target=lookARUCO.track_from_vision,
                args=(self.glob.vision, size, side_shift, aruco_angle_horizontal, distance, stopFlag, id),
                daemon=True,
            )
            camera_thread.start()

            var = roki2met.roki2met.sprint_v4
            intercom = self.glob.stm_channel.zubr       # used for communication between head and zubr-controller with memIGet/memISet commands
            self.glob.rcb.motionPlay(23)

            while True:
                # wait until motion paramenetrs initiated 
                while True:
                    ok, restart_flag = intercom.memIGet(var.restart_flag)
                    if ok: print('restart_flag :', restart_flag)
                    else: print(intercom.GetError())
                    if restart_flag == 0: break
                    time.sleep(0.25)
            
                # write motion parameters to zubr-controller motion
                intercom.memISet(var.orderFromHead, 0)              #  0 - no order, 1 - straight forward, 2 - to left, 3- to right, 4 - reverse back
                intercom.memISet(var.cycle_number, 100)
                intercom.memISet(var.hipTilt, self.motion.params['SPRINT_HIP_TILT'])
                intercom.memISet(var.fps, self.motion.params['SPRINT_FPS'])
                intercom.memISet(var.stepLengthOrder, self.motion.params['SPRINT_STEP_LENGTH'])
                intercom.memISet(var.gaitHeight, self.motion.params['SPRINT_GAIT_HEIGHT'])
                intercom.memISet(var.stepHeight, self.motion.params['SPRINT_STEP_HEIGHT'])
                intercom.memFSet(var.imu_factor, self.motion.params['SPRINT_IMU_FACTOR'])
                intercom.memFSet(var.vision_factor, self.motion.params['SPRINT_VISION_FACTOR'])
                intercom.memISet(var.pitStop, 1)                    # 1 - go on, 0 - stop waiting
                #labels = [[], [], [], ['start'], []]
                #pressed_button = self.motion.push_Button(labels)
                #intercom.memISet(var.startStop, 1)
                while True:
                    ok, restart_flag = intercom.memIGet(var.restart_flag)
                    if ok: print('restart_flag :', restart_flag)
                    else: print(intercom.GetError())
                    if restart_flag == 1: break
                    time.sleep(0.25)
                while not stopFlag.value :
                    ok, restart_flag = intercom.memIGet(var.restart_flag)
                    if restart_flag == 0: break
                    aruco_size = size.value
                    aruco_shift = side_shift.value
                    #aruco_cx = cx.value
                    #aruco_cy = cy.value
                    #u, v = self.glob.vision.undistort_points(aruco_cx, aruco_cy)
                    #aruco_angle_horizontal = math.atan((self.glob.vision.undistort_cx -u)/ self.glob.vision.focal_length_horizontal)
                    rotationFromHead = aruco_angle_horizontal.value * 10
                    print('rotationFromHead: ', rotationFromHead)
                    intercom.memFSet(var.rotationFromHead, rotationFromHead)
                    if aruco_size > 90:
                        print('Reverse')
                        intercom.memISet(var.orderFromHead, 4)   
                    elif aruco_size == 0:
                        intercom.memISet(var.orderFromHead, 0) 
                        print('No marker')
                    else:
                        if aruco_shift > 0:
                            print('Go Left')
                            intercom.memISet(var.orderFromHead, 2)
                        elif aruco_shift < 0:
                            print('Go Right')
                            intercom.memISet(var.orderFromHead, 3)
                        else:
                            print('Go Straight')
                            intercom.memISet(var.orderFromHead, 1)
                    time.sleep(0.05)

            return
        self.motion.first_Leg_Is_Right_Leg == True
        timeStep = 1
        number_Of_Cycles = 10
        if timeStep == 1:                   # 10ms
            stepLength = 30 #100
            self.motion.ugol_torsa = 0.3 #0.65
            self.motion.gaitHeight = 135 # 150
            self.motion.stepHeight = 35  # 40
            self.motion.fr1 = 4  #3
            self.motion.fr2 = 9  #6
            self.motion.params['BODY_TILT_AT_WALK'] += 0.020
            self.motion.amplitude = 40
        if timeStep == 2:                      # 20ms
            stepLength = 80 # 200
            self.motion.ugol_torsa = 0.6            # 0.7
            self.motion.gaitHeight = 180
            self.motion.stepHeight = 40
            self.motion.fr1 = 4
            self.motion.fr2 = 6
            self.motion.params['BODY_TILT_AT_WALK'] += 0.020
            self.motion.amplitude = 32
        sideLength = 0
        #self.motion.first_Leg_Is_Right_Leg = False
        if self.motion.first_Leg_Is_Right_Leg: invert = 1
        else: invert = -1
        self.motion.walk_Initial_Pose()
        number_Of_Cycles += 1
        for cycle in range(number_Of_Cycles):
            stepLength1 = stepLength
            if cycle ==0 : stepLength1 = stepLength/3
            if cycle ==1 : stepLength1 = stepLength/3 * 2
            #self.motion.refresh_Orientation()
            #rotation = 0 - self.motion.imu_body_yaw() * 1.1
            #rotation = self.motion.normalize_rotation(rotation)
            self.motion.walk_Cycle_With_Tors_v3(stepLength1,sideLength, 0 ,cycle, number_Of_Cycles)
        self.motion.walk_Final_Pose()
        #self.motion.sim_Progress(10)

    def kick_test(self, second_pressed_button):
        if second_pressed_button == 'regular':
            self.motion.kick_power = 100
            self.motion.kick(True)
        else:
            self.motion.head_Return(0, -1500)
            kick_by_right = 1
            kick_offset = 0
            self.motion.kick_power = 100
            success_Code, napravl, dist, speed = self.motion.seek_Ball_In_Pose(fast_Reaction_On = True, with_Localization = False,
                                                                              very_Fast = False)
            if success_Code:
                if napravl < 0 : kick_by_right = 1
                else: kick_by_right = 0
                kick_offset = int(abs(math.sin(napravl) * dist * 1000)) - 62
            self.motion.hard_kick(kick_by_right, kick_offset = kick_offset)
            #self.test_walk_main_cycle()
            #self.motion.play_Soft_Motion_Slot(name ='Kick_Right_v3')
        if self.glob.SIMULATION == 1:
            self.motion.sim_Progress(10)
    
    def walk_straight_vision(self, number_Of_Cycles = 0, stepLength = 0, sideLength = 0, respect_body_tilt = False, direction = 0):
        self.motion.vith_Vision = True
        self.motion.walk_Initial_Pose()
        number_Of_Cycles += 2
        for cycle in range(number_Of_Cycles):
            sideLength = math.copysign(10, self.glob.ball_course) 
            stepLength1 = stepLength
            if cycle ==0 or cycle == number_Of_Cycles-1 : stepLength1 = stepLength/3
            if cycle ==1 or cycle == number_Of_Cycles-2 : stepLength1 = stepLength/3 * 2
            self.motion.refresh_Orientation()
            #self.motion.body_euler_angle_calc()
            rotation = direction - self.motion.body_euler_angle['yaw'] * 1.0
            rotation = self.motion.normalize_rotation(rotation)
            self.motion.walk_Cycle(stepLength1, sideLength, rotation,cycle, number_Of_Cycles)
        self.motion.walk_Final_Pose(respect_body_tilt = respect_body_tilt)

    def walk_straight(self, number_Of_Cycles = 0, stepLength = 0, sideLength = 0, respect_body_tilt = False, direction = 0):
        self.motion.walk_Initial_Pose()
        number_Of_Cycles += 2
        for cycle in range(number_Of_Cycles):
            stepLength1 = stepLength
            if cycle ==0 or cycle == number_Of_Cycles-1 : stepLength1 = stepLength/3
            if cycle ==1 or cycle == number_Of_Cycles-2 : stepLength1 = stepLength/3 * 2
            self.motion.refresh_Orientation()
            #self.motion.body_euler_angle_calc()
            rotation = direction - self.motion.body_euler_angle['yaw'] * 1.0
            rotation = self.motion.normalize_rotation(rotation)
            self.motion.walk_Cycle(stepLength1, sideLength, rotation,cycle, number_Of_Cycles)
        self.motion.walk_Final_Pose(respect_body_tilt = respect_body_tilt)
        
    def weight_lifting(self, pressed_button):
        self.motion.with_Vision = False
        #def walk_straight(number_Of_Cycles = 0, stepLength = 0, sideLength = 0, respect_body_tilt = False):
        #    self.motion.walk_Initial_Pose()
        #    number_Of_Cycles += 2
        #    for cycle in range(number_Of_Cycles):
        #        stepLength1 = stepLength
        #        if cycle ==0 or cycle == number_Of_Cycles-1 : stepLength1 = stepLength/3
        #        if cycle ==1 or cycle == number_Of_Cycles-2 : stepLength1 = stepLength/3 * 2
        #        self.motion.refresh_Orientation()
        #        #self.motion.body_euler_angle_calc()
        #        rotation = - self.motion.body_euler_angle['yaw'] * 1.0
        #        rotation = self.motion.normalize_rotation(rotation)
        #        self.motion.walk_Cycle(stepLength1, sideLength, rotation,cycle, number_Of_Cycles)
        #    self.motion.walk_Final_Pose(respect_body_tilt = respect_body_tilt)

        def walk_straight_slow(number_Of_Cycles = 0, stepLength = 0, sideLength = 0):
            amplitude = self.motion.amplitude
            fr1 = self.motion.fr1
            fr2 = self.motion.fr2
            self.motion.amplitude = 70 
            self.motion.fr1 = 50 
            self.motion.fr2 = 20 
            self.motion.walk_Initial_Pose()
            number_Of_Cycles += 2
            for cycle in range(number_Of_Cycles):
                stepLength1 = stepLength
                if cycle ==0 or cycle == number_Of_Cycles-1 : stepLength1 = stepLength/3
                if cycle ==1 or cycle == number_Of_Cycles-2 : stepLength1 = stepLength/3 * 2
                self.motion.refresh_Orientation()
                self.motion.body_euler_angle_calc()
                rotation = - self.motion.body_euler_angle['yaw'] * 1.2
                rotation = self.motion.normalize_rotation(rotation)
                self.motion.walk_Cycle_slow(stepLength1, sideLength, rotation,cycle, number_Of_Cycles)
            self.motion.walk_Final_Pose()
            self.motion.amplitude = amplitude 
            self.motion.fr1 = fr1 
            self.motion.fr2 = fr2 

        if pressed_button == 'start_simple':  
            #walk_straight(number_Of_Cycles = 9, stepLength = 32)
            self.walk_straight(number_Of_Cycles = self.motion.params['WEIGHTLIFTING_INITIAL_STEPS_NUMBER'],
                       stepLength = self.motion.params['WEIGHTLIFTING_INITIAL_STEPLENGTH'])
            self.motion.jump_turn(0)
            self.motion.play_Soft_Motion_Slot(name = 'Shtanga_1')
            #self.motion.play_Soft_Motion_Slot(name = 'Shtanga_1_3')
            #self.motion.play_Soft_Motion_Slot(name = 'Shtanga_1_2')

        if pressed_button == 'start':
            #var = roki2met.roki2met.jump_mode
            #intercom = self.glob.stm_channel.zubr       # used for communication between head and zubr-controller with memIGet/memISet commands
            self.walk_straight(number_Of_Cycles = self.motion.params['WEIGHTLIFTING_INITIAL_STEPS_NUMBER'],
                       stepLength = self.motion.params['WEIGHTLIFTING_INITIAL_STEPLENGTH'])
            self.motion.play_Soft_Motion_Slot(name = 'Initial_Pose')
            self.motion.head_Return(0, -2000)
            for _ in range(500):
                time.sleep(1)
                if self.glob.SIMULATION == 5: 
                    for _ in range(100):
                        result, course, distance = self.glob.vision.seek_Ball_In_Frame_N(with_Localization = False)
                        if result: break
                else: result, course, distance, ball_blob = self.glob.vision.seek_Ball_In_Frame(with_Localization = False)
                print('course :', course, 'distance :', distance)
                x = distance * math.cos(course) * 1000
                y = distance * math.sin(course) * 1000
                if abs(y) > 20:
                    if y > 0:
                        #intercom.memISet(var, 103)
                        fraction = min(1, abs(y) / self.glob.jump_left_yield)
                        self.motion.one_jump_left(fraction)
                    if y < 0:
                        #intercom.memISet(var, 104)
                        fraction = min(1, abs(y) / self.glob.jump_right_yield)
                        self.motion.one_jump_right(fraction)
                    #self.glob.rcb.motionPlay(7)
                    #time.sleep(0.5)
                    time.sleep(1)
                    self.motion.jump_turn(0)
                if x > self.motion.params['WEIGHTLIFTING_APPROACH_PROXIMITY']:
                    #intercom.memISet(var, 101)
                    #self.glob.rcb.motionPlay(7)
                    fraction = min(1, abs((x - self.motion.params['WEIGHTLIFTING_APPROACH_PROXIMITY']) / self.glob.jump_forward_yield))
                    self.motion.one_jump_forward(fraction)
                    #time.sleep(0.5)
                    time.sleep(1)
                    self.motion.jump_turn(0)
                if abs(y) < 20 and x < -self.motion.params['WEIGHTLIFTING_APPROACH_PROXIMITY']: 
                    break
            self.motion.play_Soft_Motion_Slot(name = 'Shtanga_1_1') 
        
        if pressed_button == 'start_lifting':
            self.motion.play_Soft_Motion_Slot(name = 'Shtanga_1_1')

        self.motion.keep_hands_up = True
        #self.motion.ztr0 = - 180
        #self.motion.ztl0 = - 180
        #self.motion.zt0 = - 180
        self.motion.gaitHeight = 170
        self.motion.params['BODY_TILT_AT_WALK'] += self.motion.params['WEIGHTLIFTING_NEXT_BODYTILT']  #22222222222222222222222222222222222
        self.motion.first_Leg_Is_Right_Leg = True
        self.motion.stepHeight = 20
        #self.motion.initPoses *= 50                                             # factor of extention of initial pose timing 
        #walk_straight(number_Of_Cycles = 16, stepLength = 30, respect_body_tilt = True)
        self.walk_straight(number_Of_Cycles = self.motion.params['WEIGHTLIFTING_NEXT_STEPS_NUMBER'],
                       stepLength = self.motion.params['WEIGHTLIFTING_NEXT_STEPLENGTH'])
        self.motion.params['BODY_TILT_AT_WALK'] -= self.motion.params['WEIGHTLIFTING_NEXT_BODYTILT']

        self.motion.play_Soft_Motion_Slot(name = 'Shtanga_2_2')          # Weight_Lift_2-2023
        self.motion.keep_hands_up = True
        #self.motion.ztr0 = - 170
        #self.motion.ztl0 = - 170
        #self.motion.zt0 = - 170
        self.motion.params['BODY_TILT_AT_WALK'] += self.motion.params['WEIGHTLIFTING_LAST_BODYTILT']  #333333333333333333333333333333333333333
        #if self.glob.SIMULATION != 5 :  self.motion.params['BODY_TILT_AT_WALK'] = 0
        self.motion.stepHeight = self.motion.params['WEIGHTLIFTING_LAST_STEPHEIGHT']
        self.walk_straight(number_Of_Cycles = self.motion.params['WEIGHTLIFTING_LAST_STEPS_NUMBER'],
                       stepLength = self.motion.params['WEIGHTLIFTING_LAST_STEPLENGTH'])
        return
    
    def triple_jump_main_cycle(self, pressed_button):
        self.motion.with_Vision = False
        if pressed_button == 'start_with_approach' or pressed_button == 'find_launch_bar':
            self.walk_straight(number_Of_Cycles = 3, stepLength = 30)
            self.motion.play_Soft_Motion_Slot(name = 'Initial_Pose')
            self.motion.head_Return(0, -1800)
            time.sleep(1)
            for _ in range(500):
                time.sleep(1)
                result, x, y = self.glob.vision.seek_Launch_Pad_In_Frame()
                if result:
                    print('x :', x, 'y :', y)
                    x = x - self.motion.params['TRIPLE_JUMP_LAUNCHBAR_DISTANCE']
                    if abs(y) > 10:
                        if y > 0:
                            fraction = min(1, abs(y) / self.glob.jump_left_yield)
                            self.motion.one_jump_left(fraction)
                        if y < 0:
                            fraction = min(1, abs(y) / self.glob.jump_right_yield)
                            self.motion.one_jump_right(fraction)
                        self.motion.jump_turn(0)
                    if x > 0:
                        fraction = min(1, (x) / self.glob.jump_forward_yield)
                        self.motion.one_jump_forward(fraction)
                        self.motion.jump_turn(0)
                    if abs(y) < 30 and x < 0: 
                        break
                else:
                    pass
        if pressed_button != 'find_launch_bar':
            self.motion.play_Soft_Motion_Slot(name = 'Initial_Pose')
            self.motion.head_Return(0, 0)
            if self.glob.SIMULATION == 5:
                os.system("espeak -ven-m1 -a"+ '200' + " " + "'I ready to jump'")
                var = roki2met.roki2met.TripleJumpForFIRA2023
                intercom = self.glob.stm_channel.zubr 
                #self.motion.params['TRIPLE_JUMP_TUNER'] = 0
                #self.motion.params['TRIPLE_JUMP_FACTOR'] = 0.9
                #self.glob.stm_channel.mb.SetBodyQueuePeriod(15)
                time.sleep(5)
                #self.motion.play_Soft_Motion_Slot(name = 'TripleJumpForFIRA2023')
                self.glob.rcb.motionPlay(11)
                while True:
                    ok, restart_flag = intercom.memIGet(var.restart_flag)
                    if ok: print('restart_flag :', restart_flag)
                    else: print(intercom.GetError())
                    if restart_flag == 0: break
                    time.sleep(0.25)
                intercom.memISet(var.tuner, self.motion.params['TRIPLE_JUMP_TUNER'])
                intercom.memISet(var.hip_at_landing, self.motion.params['TRIPLE_JUMP_HIP_AT_LANDING'])
                intercom.memFSet(var.factor, self.motion.params['TRIPLE_JUMP_FACTOR'])
                intercom.memISet(var.pitStop, 1)                    # 1 - go on, 0 - stop waiting


    def marathon_main_cycle_(self):
        self.motion.with_Vision = False
        self.motion.params['MARATHON_STEP_LENGTH'] = 0
        self.motion.params['MARATHON_GAIT_HEIGHT'] = 195
        self.motion.params['MARATHON_STEP_HEIGHT'] = 40
        self.motion.params['MARATHON_FPS'] = 4
        self.motion.params['MARATHON_UGOL_TORSA'] = 0
        self.motion.params['MARATHON_BODY_TILT_AT_WALK'] = 0.04
        if self.glob.SIMULATION == 6:
            from Soccer.Vision import lookAtLine

            # Pipeline variables
            turn_shift = Value('i', 0)       #  0 - no order, 1 - straight forward, 2 - to left, 3- to right, 4 - reverse back
                                             #  0X - no shift, 2X - shift to left, 3X - shift to right

            stopFlag = Value(c_bool, False)
            camera_thread = threading.Thread(
                target=lookAtLine.track_order_from_vision,
                args=(self.glob.vision, turn_shift, stopFlag),
                daemon=True,
            )
            camera_thread.start()

            var = roki2met.roki2met.marathon
            intercom = self.glob.stm_channel.zubr       # used for communication between head and zubr-controller with memIGet/memISet commands
            self.glob.rcb.motionPlay(24)

            while True:
                # wait until motion paramenetrs initiated 
                while True:
                    ok, restart_flag = intercom.memIGet(var.restart_flag)
                    if ok: print('restart_flag :', restart_flag)
                    else: print(intercom.GetError())
                    if restart_flag == 0: break
                    time.sleep(0.25)
            
                # write motion parameters to zubr-controller motion
                intercom.memISet(var.orderFromHead, 0)              #  0 - no order, 1 - straight forward, 2 - to left, 3- to right, 4 - reverse back
                                                                    #  0X - no shift, 2X - shift to left, 3X - shift to right
                intercom.memISet(var.cycle_number, 20000)
                intercom.memISet(var.hipTilt, 0)
                intercom.memISet(var.fps, self.motion.params['MARATHON_FPS'])
                intercom.memISet(var.stepLengthOrder, self.motion.params['MARATHON_STEP_LENGTH'])
                intercom.memISet(var.gaitHeight, self.motion.params['MARATHON_GAIT_HEIGHT'])
                intercom.memISet(var.stepHeight, self.motion.params['MARATHON_STEP_HEIGHT'])
                intercom.memFSet(var.ugol_torsa, self.motion.params['MARATHON_UGOL_TORSA'])
                intercom.memFSet(var.bodyTiltAtWalk, self.motion.params['MARATHON_BODY_TILT_AT_WALK'])
                intercom.memISet(var.pitStop, 1)                    # 1 - go on, 0 - stop waiting

                while True:
                    ok, restart_flag = intercom.memIGet(var.restart_flag)
                    if ok: print('restart_flag :', restart_flag)
                    else: print(intercom.GetError())
                    if restart_flag == 1: break
                    time.sleep(0.25)
                while True:
                    ok, restart_flag = intercom.memIGet(var.restart_flag)
                    if restart_flag == 0: break
                    intercom.memISet(var.orderFromHead, turn_shift.value)
                    time.sleep(0.05)
        # simulation:

        # Pipeline variables
        turn_shift = Value('i', 0)       #  0 - no order, 1 - straight forward, 2 - to left, 3- to right, 4 - reverse back
                                            #  0X - no shift, 2X - shift to left, 3X - shift to right
        direction_from_vision = Value('d', 0)

        while True:
            event = threading.Event()
            camera_thread = threading.Thread(target = self.glob.vision.detect_Line_Follow_Stream, args=(event, turn_shift, direction_from_vision))
            camera_thread.setDaemon(True)
            camera_thread.start()

            self.motion.head_Return(0, self.motion.neck_play_pose)
            stepLength = 50
            self.motion.gaitHeight = 190
            number_Of_Cycles = 100
            self.motion.amplitude = 32
            sideLength = 0
            self.motion.walk_Initial_Pose()
            direction = 0
            for cycle in range(number_Of_Cycles):
                if self.motion.falling_Flag != 0: break
                order_from_Head = turn_shift.value
                print('order_from_Head: ', order_from_Head)
                shift = order_from_Head // 10
                turn = order_from_Head % 10
                if shift == 2 : sideLength = -20
                elif shift == 3 : sideLength = 20
                else: sideLength = 0
                stepLength1 = stepLength
                if cycle ==0 : stepLength1 = stepLength/3
                if cycle ==1 : stepLength1 = stepLength/3 * 2
                self.motion.refresh_Orientation()
                rotation = 0
                if turn == 2: direction = 0.1
                elif turn == 3: direction = -0.1
                elif turn == 1: rotation = 0
                #rotation = direction #- self.motion.imu_body_yaw() * 1.1
                #if rotation > 0: rotation *= 1.5
                #rotation = -0.3
                #direction = direction * 0.8 + direction_from_vision.value * 0.2
                #rotation = direction - self.motion.imu_body_yaw() * 1.1
                #direction = direction_from_vision.value 
                #rotation = direction_from_vision.value
                #if rotation > 0.1: rotation = 0.1
                #if rotation < -0.1: rotation = -0.1
                #print("direction: ", direction)
                rotation = self.motion.normalize_rotation(rotation)
                print("rotatition: ", rotation)
                self.motion.walk_Cycle(stepLength1,sideLength, rotation,cycle, number_Of_Cycles)
                #time.sleep(3)
                if self.glob.camera_down_Flag == True:
                    stepLength1 = stepLength/3 * 2
                    self.motion.walk_Cycle(stepLength1,sideLength, rotation,cycle, cycle + 2)
                    stepLength1 = stepLength/3
                    self.motion.walk_Cycle(stepLength1,sideLength, rotation,cycle, cycle+1)
                    break
            self.motion.walk_Final_Pose()
            if self.motion.falling_Flag != 0: self.motion.falling_Flag = 0
            event.set()
            time.sleep(2)

            if self.glob.SIMULATION == 5:
                if self.glob.camera_down_Flag == True:
                    self.glob.camera_reset()

    def marathon_main_cycle(self):
        if self.glob.SIMULATION == 5:
            ntc1 = roki2met.roki2met.mixingRoki2met.right_knee_ntc
            ntc2 = roki2met.roki2met.mixingRoki2met.left_knee_ntc
            ntc3 = roki2met.roki2met.mixingRoki2met.right_knee_bot_ntc
            ntc4 = roki2met.roki2met.mixingRoki2met.left_knee_bot_ntc
            intercom = self.glob.stm_channel.zubr       # used for communication between head and zubr-controller with memIGet/memISet commands

        self.motion.with_Vision = True
        number_Of_Cycles = 600
        self.motion.amplitude = 32
        self.motion.refresh_Orientation()
        last_good_direction = self.motion.body_euler_angle['yaw']
        line_was_lost = False
        while True:
            self.motion.head_Return(0, -2300)
            stepLength = self.motion.params['MARATHON_STEP_LENGTH']
            self.motion.gaitHeight = 180
            sideLength = 0
            self.motion.walk_Initial_Pose()
            for cycle in range(number_Of_Cycles):
                if self.motion.falling_Flag != 0: break
                if self.glob.SIMULATION == 5:
                    ok1, ntc1_value = intercom.memIGet(ntc1)
                    ok2, ntc2_value = intercom.memIGet(ntc2)
                    ok3, ntc3_value = intercom.memIGet(ntc3)
                    ok4, ntc4_value = intercom.memIGet(ntc4)
                    temperature_is_bad = False
                    if ok1:
                        print('ntc1 : ', ntc1_value)
                        if ntc1_value < 1100 and ntc1_value > 0: temperature_is_bad = True
                    if ok2:
                        print('ntc2 : ', ntc2_value)
                        if ntc2_value < 1100 and ntc2_value > 0: temperature_is_bad = True
                    if ok3:
                        print('ntc3 : ', ntc3_value)
                        if ntc3_value < 1100 and ntc3_value > 0: temperature_is_bad = True
                    if ok4:
                        print('ntc4 : ', ntc4_value)
                        if ntc4_value < 1100 and ntc4_value > 0: temperature_is_bad = True
                    if temperature_is_bad : break
                    # if ok1 and ok2 and ok3 and ok4:
                    #     ntc = min(ntc1_value, ntc2_value, ntc3_value, ntc4_value)
                    #     print("Lowest NTC: ", ntc)
                    #     if ntc < 1100 and ntc != 0 : break                                    # risk temperature in knee servos
                stepLength1 = stepLength
                if cycle ==0 : stepLength1 = stepLength/3
                if cycle ==1 : stepLength1 = stepLength/3 * 2
                deflection = sum(self.glob.deflection[-2:]) / 2
                if self.glob.data_quality_is_good :
                    if line_was_lost:
                        if abs(self.norm_yaw(last_good_direction - self.motion.body_euler_angle['yaw'])) > math.pi/2:
                            stepLength1 = stepLength/3 * 2
                            self.motion.walk_Cycle(stepLength1,sideLength, rotation,cycle, cycle + 2)
                            stepLength1 = stepLength/3
                            self.motion.walk_Cycle(stepLength1,sideLength, rotation,cycle, cycle+1)
                            self.motion.falling_Flag = 5                                                # wrong direction
                            break
                    rotation =  math.radians(deflection)
                    limit = 0.1
                    last_good_direction = self.motion.body_euler_angle['yaw']
                    line_was_lost = False
                else:
                    stepLength1 /= 2
                    rotation = math.copysign(0.5, deflection) #0.5
                    limit = 0.2
                    line_was_lost = True
                self.motion.refresh_Orientation()
                rotation_imu = - self.motion.body_euler_angle['yaw'] * 1.1
                #rotation = rotation * 0.83 + rotation_imu * 0.17
                rotation = self.normalize_rotation(rotation, limit= limit)
                print("rotatition: ", rotation)
                #if self.glob.shift > 10 : sideLength = -min(self.glob.shift / 3, 20)
                #elif self.glob.shift < -10 : sideLength = -max(self.glob.shift / 3, -20)
                if self.glob.shift > 1 : sideLength = -min(self.glob.shift , 20)
                elif self.glob.shift < -1 : sideLength = -max(self.glob.shift , -20)
                else: sideLength = 0
                print('self.glob.shift', self.glob.shift)
                self.motion.walk_Cycle(stepLength1,sideLength, rotation,cycle, number_Of_Cycles)
                #time.sleep(3)
                if self.glob.camera_down_Flag == True:
                    stepLength1 = stepLength/3 * 2
                    self.motion.walk_Cycle(stepLength1,sideLength, rotation,cycle, cycle + 2)
                    stepLength1 = stepLength/3
                    self.motion.walk_Cycle(stepLength1,sideLength, rotation,cycle, cycle+1)
                    break
            self.motion.walk_Final_Pose()
            
            self.motion.play_Soft_Motion_Slot(name = 'Initial_Pose')
            if self.motion.falling_Flag != 0: 
                if self.motion.falling_Flag == 5:
                    self.motion.jump_turn(self.motion.body_euler_angle['yaw'] + math.pi)    # turn around due to wrong direction
                self.motion.falling_Flag = 0
            else:
                if self.glob.camera_down_Flag == True: self.glob.camera_reset()
                else:  
                    self.motion.rcb.motionPlay(32)
                    if self.glob.SIMULATION == 5: os.system("espeak -ven-m1 -a200 'I need cooling of knees'")
                    time.sleep(100)                           # cool down knees


    def normalize_rotation(self, yaw, limit= 0.3):
        if abs(yaw) > 2 * math.pi: yaw %= (2 * math.pi)
        if yaw > math.pi : yaw -= (2 * math.pi)
        if yaw < -math.pi : yaw += (2 * math.pi)
        if yaw > limit : yaw = limit
        if yaw < -limit : yaw = -limit
        return yaw

    def test_walk_main_cycle(self):
        self.motion.with_Vision = False
        #self.motion.fr1 = 40 #40 #50
        #self.motion.fr2 = 20 #12 #20
        #self.motion.amplitude = 110 #110
        stepLength = 64
        #self.motion.gaitHeight = 210
        #self.motion.stepHeight = 40
        number_Of_Cycles = 10
        #self.motion.amplitude = 32
        sideLength = 25
        #self.motion.first_Leg_Is_Right_Leg = False
        if self.motion.first_Leg_Is_Right_Leg: invert = -1
        else: invert = 1
        self.motion.walk_Initial_Pose()
        number_Of_Cycles += 1
        for cycle in range(number_Of_Cycles):
            stepLength1 = stepLength
            if cycle ==0 : stepLength1 = stepLength/3
            if cycle ==1 : stepLength1 = stepLength/3 * 2
            self.motion.refresh_Orientation()
            rotation = 0 + invert * self.motion.imu_body_yaw() * 1.1
            #if rotation > 0: rotation *= 1.5
            rotation = self.motion.normalize_rotation(rotation)
            #rotation = 0
            self.motion.walk_Cycle(stepLength1,sideLength, rotation,cycle, number_Of_Cycles)
        self.motion.walk_Final_Pose()
        



if __name__=="__main__":
    pass
