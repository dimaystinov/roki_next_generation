import json
import cv2
import math, time
from Soccer.Vision.reload import Image
from Soccer.rotation_math import Rotation
import numpy as np
from Soccer.Vision.ransac_line_segments_simulation import build_line_segments
import threading

class Vision_General:
    def __init__(self, glob):
        self.glob = glob
        self.glob.vision = self
        self.build_line_segments = build_line_segments
        self.pan = 0
        self.pitch = 0
        self.roll = 0
        self.camera_elevation = 410
        self.yaw = 0
        self.image = None
        if self.glob.camera_streaming:
            self.event = threading.Event()
            self.camera_thread = threading.Thread(target = self.detect_Ball_in_Stream, args=(self.event,))
            self.camera_thread.setDaemon(True)
        #self.camera_thread.start()
        self.last_seen_centroid = [0,0,0]

    def thinning(self, img, rows, columns, iterations_limit):
        return self.zhangSuen(img, rows, columns, iterations_limit)

    def detect_line_segments( self, img, rank_threshold = 30, line_num_limit = 5, upper_lines = False):
        line_segments = self.build_line_segments( img, rank_threshold, line_num_limit, upper_lines,
                        self.glob.params['CAMERA_HORIZONTAL_RESOLUTION'], self.glob.params['CAMERA_VERTICAL_RESOLUTION'])
        return line_segments

    def orange_ball_is_on_green_field(self, blob, img, distance_mm = 0):
        width = self.glob.params['CAMERA_HORIZONTAL_RESOLUTION']
        height = self.glob.params['CAMERA_VERTICAL_RESOLUTION']
        x , y , w , h = blob.rect()  # ball blob
        if distance_mm != 0:
            h = int(self.glob.params["DIAMETER_OF_BALL"] / distance_mm * self.focal_length_vertical )
            w = int(self.glob.params["DIAMETER_OF_BALL"] / distance_mm * self.focal_length_horizontal)
        x1 = x + w                      # x1, y1, w1, h1-right rectangle
        y1 = y
        if x1 + w <= width:
            w1 = w
        else:
            w1 = width - x1
        if x1 == width:
            w1 = 1
            x1 = width-1
        if y1 + 2 * h <= height:
            h1 = 2 * h
        else:
            h1 = height - y1
        if y1 + h == height:
            h1 = h
        y2 = y                         # x2, y2, w2, h2 - left rectangle
        if x - w > 0:
            x2 = x - w
            w2 = w
        else:
            x2 = 0
            w2 = x1 - x2
        if x1 == 0:
            x2 = 0
            w2 = 1
        y2 = y1
        h2 = h1
        x3 = x                          # x3, y3, w3, h3 - bottom rectangle
        y3 = y + h - 1
        w3 = w
        h3 = h1 - h + 1
        blob_p_g = []                     # right blobs
        blob_l_g = []                     # left blobs
        blob_n_g = []                     # bottom blobs
        blob_p_w = []                     # right blobs
        blob_l_w = []                     # left blobs
        blob_n_w = []                     # bottom blobs
        blob_p_g = img.find_blobs([self.TH['green field']['th']],roi = [x1 , y1 , w1 , h1], pixels_threshold=7, area_threshold=7, merge=True)
        blob_l_g = img.find_blobs([self.TH['green field']['th']],roi = [x2 , y1 , w2 , h1], pixels_threshold=7, area_threshold=7, merge=True)
        blob_n_g = img.find_blobs([self.TH['green field']['th']],roi = [x3 , y3 , w3 , h3], pixels_threshold=7, area_threshold=7, merge=True)
        blob_p_w = img.find_blobs([self.TH['white marking']['th']],roi = [x1 , y1 , w1 , h1], pixels_threshold=7, area_threshold=7, merge=True)
        blob_l_w = img.find_blobs([self.TH['white marking']['th']],roi = [x2 , y1 , w2 , h1], pixels_threshold=7, area_threshold=7, merge=True)
        blob_n_w = img.find_blobs([self.TH['white marking']['th']],roi = [x3 , y3 , w3 , h3], pixels_threshold=7, area_threshold=7, merge=True)
        return len(blob_p_g) > 0 or len( blob_l_g ) > 0  or len( blob_n_g ) > 0 or len(blob_p_w) > 0 or len( blob_l_w ) > 0  or len( blob_n_w ) > 0

    def seek_Ball_In_Frame(self, with_Localization = True):
        see_ball = 0
        timer1 = time.perf_counter()
        for number in range (2):
            camera_result, img1, pitch, roll, yaw, pan = self.snapshot()
            if camera_result:
                img = Image(img1)
                #self.display_camera_image(img1, window = 'Original')
                ball_column, ball_row = 0, 0
                for blob in img.find_blobs([self.TH['orange ball']['th']],
                                    pixels_threshold=self.TH['orange ball']['pixel'],
                                    area_threshold=self.TH['orange ball']['area'],
                                    merge=True):
                    if blob.cy() > ball_row:
                        ball_row = blob.cy()
                        ball_column = blob.cx()
                        ball_blob = blob
                        see_ball += 1
                if see_ball:
                    see_ball = 0
                    if self.orange_ball_is_on_green_field(ball_blob, img): 
                        result, self.camera_elevation = self.glob.motion.camera_elevation()
                        if result:
                            self.visible_reaction_ball()
                            img.draw_rectangle(ball_blob.rect())
                            self.display_camera_image(img.img, window = 'Ball')
                            see_ball += 1
                            break
        if with_Localization and number == 0 and camera_result: self.glob.local.read_Localization_marks(img1)
        #print('seek_Ball_In_Frame time:', time.perf_counter()-timer1)
        if see_ball == 0: return False, 0, 0, 0
        else:
            result, course, distance = self.get_course_and_distance_to_ball(ball_column, ball_row)
            if result: return True, course, distance, ball_blob
            else: return False, 0, 0, 0

    def seek_Ball_In_Frame_N(self, with_Localization = True):
        see_ball = 0
        timer1 = time.perf_counter()
        for number in range (2):
            camera_result, img, pitch, roll, yaw, pan = self.snapshot()
            if camera_result:
                ball_column, ball_row = self.glob.neural.object_detect_single(img, "ball")
                if not self.glob.neural.is_ready():
                    found, course, distance, _ = self.seek_Ball_In_Frame(with_Localization)
                    return found, course, distance
                if ball_column or ball_row: 
                    if self.glob.event_type == "FIRA":
                        self.camera_elevation = 410
                        result = True
                    else:
                        result, self.camera_elevation = self.glob.motion.camera_elevation()
                    if result:
                        self.visible_reaction_ball()
                        cv2.circle(img, (ball_column, ball_row), 20, (0, 255, 0), 5)
                        self.display_camera_image(img, window = 'Ball')
                        see_ball += 1
                        break
        if with_Localization and number == 0 and camera_result: self.glob.local.read_Localization_marks(img)
        #print('seek_Ball_In_Frame time:', time.perf_counter()-timer1)
        if see_ball == 0: return False, 0, 0
        else:
            result, course, distance = self.get_course_and_distance_to_ball(ball_column, ball_row)
            if result: return True, course, distance
            else: return False, 0, 0

    def detect_Ball_Speed_N(self):
        see_ball = 0
        position = []
        timer1 = time.perf_counter()
        if self.glob.event_type == "FIRA":
            self.camera_elevation = 410
            result = True
        else:
            result, self.camera_elevation = self.glob.motion.camera_elevation()
        if result:
            for number in range (2):
                camera_result, img, pitch, roll, yaw, pan = self.snapshot()
                if camera_result:
                    ball_column, ball_row = self.glob.neural.ball_detect_single(img)
                    if not self.glob.neural.is_ready():
                        return self.detect_Ball_Speed()
                    if ball_column or ball_row: 
                        result, course, distance = self.get_course_and_distance_to_ball(ball_column, ball_row)
                        if result:
                            self.visible_reaction_ball()
                            see_ball += 1
                            position.append([course,distance])
        #print('seek_Ball_In_Frame time:', time.perf_counter()-timer1)
        n = len(position)
        if n > 1:
            front_speed = ( position[n-1][1] - position[0][1])/ distance/n
            tangential_speed = ( position[n-1][0] - position[0][0]) * distance/n
            speed = [tangential_speed, front_speed ]
        if see_ball < 1: return False, 0, 0, [0,0]
        elif n < 2: return False, course, distance, [0,0]
        else: return True, course, distance, speed

    def get_course_and_distance_to_ball(self, ball_column, ball_row):                       # returns relative course from body in radians and relative distance in meters
        result, relative_x_on_floor, relative_y_on_floor = self.image_point_to_relative_coord_on_floor(
                                                    ball_column, ball_row, for_ball = True
                                                    )
        if not result:
            return False, 0, 0
        else:
            print('relative coord of ball on floor (x,y)= ', round(relative_x_on_floor), round(relative_y_on_floor))
            distance = math.sqrt(relative_x_on_floor**2 + relative_y_on_floor**2)
            course = math.atan2(relative_y_on_floor, relative_x_on_floor)
            return True, course, distance/1000

    def get_course_and_distance_to_post(self, blob_cx, blob_y_plus_h):   # returns absolute course in radians and relative distance in meters
        result, relative_x_on_floor, relative_y_on_floor = self.image_point_to_relative_coord_on_floor(
                                                    blob_cx, blob_y_plus_h, absolute = True
                                                    )
        if not result:
            return False, 0, 0
        else:
            relative_distance = math.sqrt(relative_x_on_floor**2 + relative_y_on_floor**2) / 1000
            absolute_course = math.atan2(relative_y_on_floor, relative_x_on_floor)
            return True, absolute_course, relative_distance

    def image_point_to_relative_coord_on_floor(self, column, row, for_ball = False, absolute = False):
        if for_ball: triangulation_base  = self.camera_elevation - self.glob.params['DIAMETER_OF_BALL'] / 2
        else: triangulation_base  = self.camera_elevation
        #print('triangulation_base = ', triangulation_base)
        u, v = self.undistort_points(column, row)
        point_angle_horizontal = (self.undistort_cx -u)/ self.focal_length_horizontal        # argument of math.atan
        point_angle_vertical = (self.undistort_cy -v)/ self.focal_length_vertical            # argument of math.atan
        point_vector = [1, point_angle_horizontal, point_angle_vertical]    # arguments of math.tan
        pitch_rotation = Rotation.from_euler('y', [self.pitch], degrees=False)
        roll_rotation = Rotation.from_euler('x', [self.roll], degrees=False)
        if absolute:
            pan_rotation = Rotation.from_euler('z', [self.yaw], degrees=False)
        else:
            pan_rotation = Rotation.from_euler('z', [self.pan], degrees=False)
        point_vector = pitch_rotation.apply(point_vector)
        point_vector = roll_rotation.apply(point_vector)
        point_vector = pan_rotation.apply(point_vector)
        if point_vector[0,2] >= 0: return False, 0, 0 
        relative_x_on_floor = point_vector[0,0] * triangulation_base / (- point_vector[0,2])
        relative_y_on_floor = point_vector[0,1] * triangulation_base / (- point_vector[0,2])
        return True, relative_x_on_floor, relative_y_on_floor

    #def detect_Ball_Speed(self):
    #    see_ball = 0
    #    position = []
    #    for number in range (2):
    #        camera_result, img1, pitch, roll, yaw, pan = self.snapshot()
    #        if camera_result:
    #            img = Image(img1)
    #            #self.display_camera_image(self.image, window = 'Original')
    #            ball_column, ball_row = 0, 0
    #            for blob in img.find_blobs([self.TH['orange ball']['th']],
    #                                pixels_threshold=self.TH['orange ball']['pixel'],
    #                                area_threshold=self.TH['orange ball']['area'],
    #                                merge=True):
    #                if blob.cy() > ball_row:
    #                    ball_row = blob.cy()
    #                    ball_column = blob.cx()
    #                    ball_blob = blob
    #                    see_ball += 1
    #            if see_ball:
    #                see_ball = 0
    #                if self.orange_ball_is_on_green_field(ball_blob, img): 
    #                    result, self.camera_elevation = self.glob.motion.camera_elevation()
    #                    if result:
    #                        self.visible_reaction_ball()
    #                        img.draw_rectangle(ball_blob.rect())
    #                        self.display_camera_image(img.img, 'Ball')
    #                        see_ball += 1
    #                        result, course, distance = self.get_course_and_distance_to_ball(ball_column, ball_row)
    #                        if result:
    #                            position.append([course,distance])
    #    n = len(position)
    #    if n > 1:
    #        front_speed = ( position[n-1][1] - position[0][1])/ distance/n
    #        tangential_speed = ( position[n-1][0] - position[0][0]) * distance/n
    #        speed = [tangential_speed, front_speed ]
    #    if see_ball < 1: return False, 0, 0, [0,0]
    #    elif n < 2: return False, course, distance, [0,0]
    #    else: return True, course, distance, speed

    def detect_Ball_Speed(self):
        see_ball = 0
        position = []
        result, self.camera_elevation = self.glob.motion.camera_elevation()
        if result:
            for number in range (2):
                camera_result, img1, self.pitch, self.roll, yaw, pan = self.snapshot()
                if camera_result:
                    img = Image(img1)
                    #self.display_camera_image(self.image, window = 'Original')
                    blobs = img.find_blobs([self.TH['orange ball']['th']],
                                        pixels_threshold=self.TH['orange ball']['pixel'],
                                        area_threshold=self.TH['orange ball']['area'],
                                        merge=True)
                    len_blobs = len(blobs)
                    height = self.glob.params["CAMERA_VERTICAL_RESOLUTION"]
                    order = []
                    for i in range(len_blobs):
                        order.append([height - blobs[i].cy(), i,0,0])
                    sorted_order = sorted(order)
                    new_order_len = min(10, len_blobs)
                    new_order = sorted_order[:new_order_len]
                    for i in range(new_order_len):
                        ball_column = blobs[new_order[i][1]].cx()
                        ball_row = blobs[new_order[i][1]].cy()
                        result, relative_x_on_floor, relative_y_on_floor = self.image_point_to_relative_coord_on_floor(ball_column, ball_row,
                                                                    for_ball = True)
                        new_order[i][0] = int(math.sqrt(relative_x_on_floor *relative_x_on_floor + relative_y_on_floor * relative_y_on_floor))
                        new_order[i][2] = int(relative_x_on_floor)
                        new_order[i][3] = int(relative_y_on_floor)
                    sorted_new_order = sorted(new_order)
                    if new_order_len != 0:
                        for i in range(new_order_len):
                            if sorted_new_order[i][0] == 0 : continue
                            if sorted_new_order[i][0] > 5000 : break            # getting off if distances more than 5m.
                            ball_blob = blobs[sorted_new_order[i][1]]
                            if self.orange_ball_is_on_green_field(ball_blob, img, distance_mm = sorted_new_order[i][0]): 
                                self.visible_reaction_ball()
                                print('relative coord of ball on floor (x,y)= ', sorted_new_order[i][2], sorted_new_order[i][3])
                                img.draw_rectangle(ball_blob.rect())
                                #self.display_camera_image(img.img, 'Ball')
                                distance = sorted_new_order[i][0] / 1000
                                course = math.atan2(sorted_new_order[i][3], sorted_new_order[i][2])
                                position.append([course, distance])
                                see_ball += 1
                                break
                    self.display_camera_image(self.image, window = 'Original')
        n = len(position)
        if n > 1:
            if distance == 0 : speed = [0,0]
            else:
                front_speed = ( position[n-1][1] - position[0][1])/ distance/n
                tangential_speed = ( position[n-1][0] - position[0][0]) * distance/n
                speed = [tangential_speed, front_speed ]
        if see_ball < 1: return False, 0, 0, [0,0]
        elif n < 2: return False, course, distance, [0,0]
        else: return True, course, distance, speed

    def detect_Ball_in_Stream(self, event):
        print('Ball detection in stream started')
        while True:
            result, course, distance, speed = self.detect_Ball_Speed()
            if result or distance != 0:
                self.glob.motion.refresh_Orientation()
                yaw = self.glob.local.coord_odometry[2] = self.glob.motion.imu_body_yaw()
                course_global_rad = course + yaw
                proforma_ball_coord = [distance * math.cos(course_global_rad) + self.glob.local.coord_odometry[0],
                                      distance * math.sin(course_global_rad) + self.glob.local.coord_odometry[1]]
                self.glob.local.ball_odometry = [self.glob.local.ball_odometry[0] * 0.5 + proforma_ball_coord[0] * 0.5,
                                                 self.glob.local.ball_odometry[1] * 0.5 + proforma_ball_coord[1] * 0.5]
                self.glob.ball_course = course
                self.glob.ball_distance = distance
                self.glob.ball_speed = speed
                self.glob.robot_see_ball = 5
                #self.glob.local.localisation_Complete()
                #course_global_rad = self.glob.ball_course + self.glob.pf_coord[2]
                #proforma_ball_coord = [self.glob.ball_distance * math.cos(course_global_rad) + self.glob.pf_coord[0],
                #                        self.glob.ball_distance * math.sin(course_global_rad) + self.glob.pf_coord[1]]
                #self.glob.ball_coord = [self.glob.ball_coord[0] * 0.8 + proforma_ball_coord[0] * 0.2,
                #                       self.glob.ball_coord[1] * 0.8 + proforma_ball_coord[1] * 0.2]
            else: self.glob.robot_see_ball -= 1
            time.sleep(self.camera_sleep)
            if event.is_set(): break
        print('Ball detection in stream terminated')

    def detect_Ball_in_One_Shot(self):
        result, course, distance, speed = self.detect_Ball_Speed()
        if result or distance != 0:
            self.glob.motion.refresh_Orientation()
            yaw = self.glob.local.coord_odometry[2] = self.glob.motion.imu_body_yaw()
            course_global_rad = course + yaw
            proforma_ball_coord = [distance * math.cos(course_global_rad) + self.glob.local.coord_odometry[0],
                                    distance * math.sin(course_global_rad) + self.glob.local.coord_odometry[1]]
            self.glob.local.ball_odometry = [self.glob.local.ball_odometry[0] * 0.2 + proforma_ball_coord[0] * 0.8,
                                                self.glob.local.ball_odometry[1] * 0.2 + proforma_ball_coord[1] * 0.8]
            self.glob.ball_course = course
            self.glob.ball_distance = distance
            self.glob.ball_speed = speed
            self.glob.robot_see_ball = 5
            self.glob.ball_coord[:] = self.glob.local.ball_odometry
        else: self.glob.robot_see_ball -= 1

    def detect_Basket_in_One_Shot_N(self):
        see_ball = 0
        position = []
        displacement = 0
        frame_number = 0
        #result, self.camera_elevation = self.glob.motion.camera_elevation()
        #if result:
            #result, img1, self.pitch, self.roll, yaw, pan = self.snapshot()
        img1, frame_number = self.camera.snapshot()
        if img1 is not None:
            result = True
            cx, cy = self.glob.neural.object_detect_single(img1, "basket")
            displacement = self.glob.params["CAMERA_HORIZONTAL_RESOLUTION"]/2 - cx
            result = not ( cx == 0 and cy == 0)
            if result: 
                self.visible_reaction_ball()
                print('relative displacement from cenetr of picture ', displacement)
            #img = Image(img1)
            ##self.display_camera_image(self.image, window = 'Original')
            #blobs = img.find_blobs([self.TH['orange ball']['th']],
            #                    pixels_threshold=self.TH['orange ball']['pixel'],
            #                    area_threshold=self.TH['orange ball']['area'],
            #                    merge=True, roi = (267, 217, 267, 217))
            #len_blobs = len(blobs)
            #height = self.glob.params["CAMERA_VERTICAL_RESOLUTION"]
            #order = []
            #for i in range(len_blobs):
            #    order.append([height - blobs[i].cy(), i,0,0])
            #sorted_order = sorted(order)
            #new_order_len = min(10, len_blobs)
            #new_order = sorted_order[:new_order_len]
            #if new_order_len != 0:
            #    result = True
            #    basket_blob = blobs[new_order[0][1]]
            #    basket_column = basket_blob.cx()
            #    displacement = 133 - basket_column
            #    self.visible_reaction_ball()
            #    print('relative displacement from cenetr of picture ', displacement)
            #    rect = [basket_blob.rect()[0]+267, basket_blob.rect()[1]+217, basket_blob.rect()[2], basket_blob.rect()[3]]
            #    img.draw_rectangle(rect)
            #    self.display_camera_image(img.img, 'Ball')
            #else: result = False
        else: result = False
        return result, displacement

    def detect_Basket_in_One_Shot(self):
        see_ball = 0
        position = []
        displacement = 0
        frame_number = 0
        #result, self.camera_elevation = self.glob.motion.camera_elevation()
        #if result:
            #result, img1, self.pitch, self.roll, yaw, pan = self.snapshot()
        img1, frame_number = self.camera.snapshot()
        if img1 is not None:
            result = True
            img = Image(img1, copy=False)
            #self.display_camera_image(self.image, window = 'Original')
            blobs = img.find_blobs([self.TH['orange ball']['th']],
                                pixels_threshold= 20,
                                area_threshold= 20,
                                merge=True, roi = (267, 217, 267, 217))
            len_blobs = len(blobs)
            biggest_blob_area = 0
            biggest_blob_number = 0
            for i in range(len_blobs):
                if blobs[i].pixels() > biggest_blob_area:
                    biggest_blob_area = blobs[i].pixels()
                    biggest_blob_number = i
            if len_blobs != 0:
                result = True
                basket_blob = blobs[biggest_blob_number]
                basket_column = basket_blob.cx()
                displacement = 133 - basket_column
                self.visible_reaction_ball()
                print('relative displacement from cenetr of picture ', displacement)
                rect = [basket_blob.rect()[0]+267, basket_blob.rect()[1]+217, basket_blob.rect()[2], basket_blob.rect()[3]]
                img.draw_rectangle(rect)
                self.display_camera_image(img.img, 'Basket')

            #height = self.glob.params["CAMERA_VERTICAL_RESOLUTION"]
            #order = []
            #for i in range(len_blobs):
            #    order.append([height - blobs[i].cy(), i,0,0])
            #sorted_order = sorted(order)
            #new_order_len = min(10, len_blobs)
            #new_order = sorted_order[:new_order_len]
            #if new_order_len != 0:
            #    result = True
            #    basket_blob = blobs[new_order[0][1]]
            #    basket_column = basket_blob.cx()
            #    displacement = 133 - basket_column
            #    self.visible_reaction_ball()
            #    print('relative displacement from cenetr of picture ', displacement)
            #    rect = [basket_blob.rect()[0]+267, basket_blob.rect()[1]+217, basket_blob.rect()[2], basket_blob.rect()[3]]
            #    img.draw_rectangle(rect)
            #    self.display_camera_image(img.img, 'Basket')
            else: result = False
        else: result = False
        return result, displacement

    def seek_Launch_Pad_In_Frame(self):
        see_pad = 0
        roi_x = 150
        roi_y = 150
        roi=(roi_x,roi_y,500,350)
        camera_result, img1, pitch, roll, yaw, pan = self.snapshot()
        if self.glob.event_type == "FIRA":
            self.camera_elevation = 410
        if camera_result:
            img = Image(img1, copy=False)
            #self.display_camera_image(img1, window = 'Original')
            left_x = right_x = int(self.glob.params['CAMERA_HORIZONTAL_RESOLUTION']/2)
            top_y = bottom_y =  int(self.glob.params['CAMERA_VERTICAL_RESOLUTION']/2)
            blobs = img.find_blobs([self.TH['blue posts']['th']],pixels_threshold=20, area_threshold=20, merge=True, roi=roi)
            for i in range(len(blobs)) :
                blob_rectangle = [blobs[i].x()+roi_x, blobs[i].y()+roi_y, blobs[i].w(), blobs[i].h()]
                img.draw_rectangle(blob_rectangle)
                if i == 0:
                    left_x = blobs[i].x()
                    right_x = blobs[i].x() + blobs[i].w()
                    top_y = blobs[i].y()
                    bottom_y = blobs[i].y() + blobs[i].h()
                else:
                    if blobs[i].x() < left_x : left_x = blobs[i].x()
                    if blobs[i].x() + blobs[i].w() > right_x: right_x = blobs[i].x() + blobs[i].w()
                    if blobs[i].y() < top_y: top_y = blobs[i].y()
                    if blobs[i].y() + blobs[i].h() > bottom_y: bottom_y = blobs[i].y() + blobs[i].h()
                see_pad += 1
            img.draw_rectangle(roi, color = (0, 0, 255))
            self.display_camera_image(img.img, window = 'Original')
        if see_pad == 0: return False, 0, 0
        else:
            self.visible_reaction_ball()
            column = int((left_x + right_x)/2) + roi_x
            row = int((top_y + bottom_y)/2) + roi_y
            result, relative_x_on_floor, relative_y_on_floor = self.image_point_to_relative_coord_on_floor( column, row, for_ball = False, absolute = False)
            if result: return True, relative_x_on_floor, relative_y_on_floor
            else: return False, 0, 0

    def detect_Line_Follow_Stream(self, event, turn_shift, direction_from_vision):
        x_size = 200
        y_size = 160
        turn_shift.value = 0
        while True:
            turn = 0
            shift = 0
            if event.is_set(): break
            #result, self.camera_elevation = self.glob.motion.camera_elevation()
            #camera_result, img1, self.pitch, self.roll, yaw, pan = self.snapshot()
            img1, frame_number = self.camera.snapshot()
            th = self.TH['orange ball']['th']
            low_th = (int(th[0] * 2.55), th[2] + 128, th[4] + 128)
            high_th = (math.ceil(th[1] * 2.55), th[3] + 128, th[5] + 128)
            if self.glob.SIMULATION == 5:
                img1 = cv2.resize(img1, (200,160))
            self.display_camera_image(img1, 'Track')
            img = Image(img1, copy=False)
            ROIS = [ # [ROI, weight]
                    (0, 140, 200, 20, 0.3), # You'll need to tweak the weights for your app
                    (0,  70, 200, 20, 0.4), # depending on how your robot is setup.
                    (0,   0, 200, 20, 0.3)
                    ]
            weight_sum = 0
            for r in ROIS: weight_sum += r[4]

            #image = cv2.resize(img1,(x_size, y_size))
            centroid_sum = 0
            blob_found = False
            for r in ROIS:
                blobs  = img.find_blobs([self.TH['orange ball']['th']],  pixels_threshold=self.TH['orange ball']['pixel'],
                                        area_threshold=self.TH['orange ball']['area'], merge=True, margin=10, roi=r[0:4])

                if (len (blobs) != 0):
                    blob_found = True

                if blobs:
                    # Find the blob with the most pixels.
                    largest_blob = max(blobs, key=lambda b: b.pixels())

                    # Draw a rect around the blob.
                    img.draw_rectangle(largest_blob.rect())

                    centroid_sum += largest_blob.cx() * r[4] # r[4] is the roi weight.
            # cv2.imshow('Track',img1)
            # cv2.waitKey(10)
            if (blob_found == False):
                self.glob.data_quality_is_good = False
                deflection_angle = 0
                return 0
            else:
                center_pos = (centroid_sum / weight_sum) # Determine center of line.

                deflection_angle = -math.atan((center_pos-80)/60)
                # Convert angle in radians to degrees.
                deflection_angle = math.degrees(deflection_angle)
            self.glob.deflection.append(deflection_angle)
            if len(self.glob.deflection) > 60:
                self.glob.deflection.pop(0)
            self.glob.data_quality_is_good = True

            time.sleep(0.1)

    def detect_Line_Follow_One_Shot(self):
        if self.glob.SIMULATION == 5:
            img1, frame_number = self.camera.snapshot()
        else:
            img1 = self.glob.motion.vision_Sensor_Get_Image()
        FRAME_X, FRAME_Y = 400, 320
        img1 = cv2.resize(img1, (FRAME_X, FRAME_Y))
        img = Image(img1, copy=False)
        ROIS = [ # [ROI, weight] by Osokin
                (140, 160, 120, 20, 0.2), # You'll need to tweak the weights for your app
                (130, 110, 140, 20, 0.5), # depending on how your robot is setup.
                (120,  65, 160, 20, 0.8)
                ]

        #ROIS = [ # [ROI, weight]
        #        (0, 140, 400, 20, 0.2), # You'll need to tweak the weights for your app
        #        (0, 110, 400, 20, 0.5), # depending on how your robot is setup.
        #        (0,  65, 400, 20, 0.8)
        #        ]

        weight_sum = 0
        weighted_y = 0

        for r in ROIS:
            weight_sum += r[4]
            weighted_y += (FRAME_Y - r[1]) * r[4]

        centroid_sum = 0
        blob_found = False
        count_ok_rois = 0
        for i, r in enumerate(ROIS):
            blobs1  = img.find_blobs([self.TH['line_follow_1']['th']], 
                                    pixels_threshold=20, 
                                    area_threshold=20, 
                                    merge=True, margin=10, roi=r[0:4])
            blobs2  = img.find_blobs([self.TH['line_follow_2']['th']], 
                                    pixels_threshold=20, 
                                    area_threshold=20, 
                                    merge=True, margin=10, roi=r[0:4])
            blobs3  = img.find_blobs([self.TH['line_follow_3']['th']], 
                                    pixels_threshold=20, 
                                    area_threshold=20, 
                                    merge=True, margin=10, roi=r[0:4])
            blobs4  = img.find_blobs([self.TH['line_follow_4']['th']], 
                                    pixels_threshold=20, 
                                    area_threshold=20, 
                                    merge=True, margin=10, roi=r[0:4])

            filtered_blobs1 =[]
            for blob in blobs1:
                if blob.w_ < 80:
                    filtered_blobs1.append(blob)
            filtered_blobs2 =[]
            for blob in blobs2:
                if blob.w_ < 80:
                    filtered_blobs2.append(blob)
            filtered_blobs3 =[]
            for blob in blobs3:
                if blob.w_ < 80:
                    filtered_blobs3.append(blob)
            filtered_blobs4 =[]
            for blob in blobs4:
                if blob.w_ < 80:
                    filtered_blobs4.append(blob)
            filtered_blobs = []
            if (len (filtered_blobs1) != 0 or len (filtered_blobs2) != 0 or len (filtered_blobs3) != 0):
                filtered_blobs = filtered_blobs1 + filtered_blobs2 + filtered_blobs3
            else:
                if (len (filtered_blobs4) != 0):
                    filtered_blobs = filtered_blobs4

            if (len (filtered_blobs) != 0):
                blob_found = True

            if filtered_blobs:
                count_ok_rois += 1
                # Find the blob with the most pixels.
                largest_blob = max(filtered_blobs, key=lambda b: b.pixels())

                print("largest blob found", ' roi = ', i, ' x_size: ', largest_blob.w_)

                # Draw a rect around the blob.
                img.draw_rectangle((largest_blob.rect()[0] + r[0], largest_blob.rect()[1] + r[1], largest_blob.rect()[2], largest_blob.rect()[3]))
                img.draw_rectangle(r[:-1], color=(255, 255, 255))

                centroid_sum += (largest_blob.cx() - r[2] / 2) * r[4] # r[4] is the roi weight.
                if i == 0: self.glob.shift = r[2] / 2 - largest_blob.cx()
                self.last_seen_centroid[i] = largest_blob.cx() - r[2] / 2
            else:
                if self.last_seen_centroid[i] > 0:
                    centroid_sum += r[2] / 2 * r[4] # r[4] is the roi weight.
                elif self.last_seen_centroid[i] < 0:
                    centroid_sum += - r[2] / 2 * r[4] # r[4] is the roi weight.

        
        img1 = cv2.resize(img.img, (800, 650))
        self.display_camera_image(img1, 'Track')
        if (blob_found == False):
            self.glob.data_quality_is_good = False
            deflection_angle = 0
            return 0
        
        else:
            center_pos = (centroid_sum / weight_sum) # Determine center of line.

            weighted_y /= weight_sum

            #deflection_angle = math.atan((center_pos - (FRAME_X / 2 + 30)) / weighted_y)#FRAME_Y * 2) # by Osokin
            deflection_angle = - math.atan(center_pos / weighted_y)
            # Convert angle in radians to degrees.
            deflection_angle = math.degrees(deflection_angle)

            print("center_pos, deflection, weight_y", center_pos, deflection_angle, weighted_y)
        
        self.glob.deflection.append(deflection_angle)
        if len(self.glob.deflection) > 60:
            self.glob.deflection.pop(0)
        self.glob.data_quality_is_good = (count_ok_rois == 3)

if __name__=="__main__":
    pass
