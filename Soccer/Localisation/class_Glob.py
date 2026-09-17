import json, array, os, time
from multiprocessing import Array 
from Soccer.config_paths import init_param_read_path, repo_root_str
from Soccer.Vision.display import create_display

class Glob:
    def __init__(self, simulation, current_work_directory, particles_number = 1000, event_type = 'Robocup'):
        self.hardcode_walking = False
        self.record_motions = False
        self.event_type = event_type
        self.neural_vision = False
        self.neural = None
        self.role = None
        self.monitor_is_on = False
        self.camera_streaming = False        # supply IMU data for camera as stream or as one-off
        self.with_Local = True
        self.data_quality_is_good = False
        self.deflection = []
        self.shift = 0
        if self.event_type == 'FIRA':
            self.COLUMNS = 20
            self.ROWS = 15
        elif event_type == 'Robocup':
            self.COLUMNS = 18
            self.ROWS = 13
        self.current_work_directory = repo_root_str(current_work_directory)
        self.particles_number = particles_number
        #self.pf_alloc1 = array.array('I',(0 for i in range(particles_number*4)))
        #self.pf_alloc2 = array.array('I',(0 for i in range(particles_number*4)))
        #self.weights = array.array('I',(0 for i in range(particles_number)))
        #self.new_p = array.array('I',(0 for i in range(particles_number)))
        self.strategy_data = array.array('b',(0 for i in range(self.COLUMNS * self.ROWS * 2)))
        self.SIMULATION = simulation             # 0/1/3 - simulation, 5 - live on RPi/CM4
        if self.SIMULATION == 2:
            raise ValueError("SIMULATION == 2 (OpenMV runtime) has been removed")
        self.ball_coord = Array('f', 2)
        self.ball_coord =[0.0, 0.0]                # global coordinate
        self.ball_course = 0                       # local course from robot body
        self.ball_distance = 0                     # local distance from robot body
        self.ball_speed = [0.0, 0.0]      # [tangential_speed, front_speed ]
        self.robot_see_ball = 0
        self.direction_to_guest = None
        self.kick_power_planned = None
        self.strategy_cell = None
        self.pf_coord = [0.0,0.0,0.0]
        self.obstacles = []
        self.motion = None
        self.local = None
        self.vision = None
        self.camera_down_Flag = False
        if self.SIMULATION == 1 or self.SIMULATION == 0 or self.SIMULATION == 3:
            import socket
            self.landmarks_filename = str(init_param_read_path(self.current_work_directory, "Sim/Sim_landmarks.json"))
            params_filename = str(init_param_read_path(self.current_work_directory, "Sim/Sim_params.json"))
            #params_filename = current_work_directory + "Init_params/Real/Real_params.json"
            params_2_filename = str(init_param_read_path(self.current_work_directory, "Sim/Sim_params_2.json"))
            with open(init_param_read_path(self.current_work_directory, "Sim/wifi_params.json"), "r") as f:
                self.wifi_params = json.loads(f.read())
            if self.wifi_params['WIFI_IS_ON']:
                self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self.target_wifi_address = (self.wifi_params['HOST'], self.wifi_params['PORT'])
        elif self.SIMULATION == 5 :
            from Soccer.Vision.camera import Camera
            from Soccer.Motion.class_stm_channel import STM_channel
            from Soccer.Vision.class_Vision_RPI import Vision_RPI
            #self.Camera = Camera
            self.Vision_RPI = Vision_RPI
            self.camera = Camera()
            self.STM_channel_class = STM_channel
            #import usocket, network
            if self.event_type == 'FIRA':
                self.landmarks_filename = str(init_param_read_path(self.current_work_directory, "Real/Real_landmarks_FIRA.json"))
            else:
                self.landmarks_filename = str(init_param_read_path(self.current_work_directory, "Real/Real_landmarks.json"))
            params_filename = str(init_param_read_path(self.current_work_directory, "Real/Real_params.json"))
            params_2_filename = str(init_param_read_path(self.current_work_directory, "Real/Real_params_2.json"))
            self.Roki = None
            self.stm_channel = STM_channel(self)
            self.rcb = self.stm_channel.rcb
        with open(self.landmarks_filename, "r") as f:
            landmarks = json.loads(f.read())
        with open(params_filename, "r") as f:
            self.params = json.loads(f.read())
        with open(params_2_filename, "r") as f:
            self.params.update(json.loads(f.read()))
        self.display = create_display(self.params, self.SIMULATION)
        self.use_particle_filter = self.params['USE_PARTICLE_FILTER']
        if self.SIMULATION == 5 : self.stm_channel.mb.SetBodyQueuePeriod(self.params['FRAME_DELAY'])
        self.first_step_yield = (19 * self.params['RUN_TEST_10_STEPS'] - 9 * self.params['RUN_TEST_20_STEPS']) / 10
        self.cycle_step_yield = ( self.params['RUN_TEST_20_STEPS'] - self.params['RUN_TEST_10_STEPS']) / 10
        self.side_step_right_yield = self.params['SIDE_STEP_RIGHT_TEST_RESULT'] / 20
        self.side_step_left_yield = self.params['SIDE_STEP_LEFT_TEST_RESULT'] / 20
        self.jump_forward_yield = self.params['JUMP_FORWARD_TEST'] / 10
        self.jump_backward_yield = self.params['JUMP_BACKWARD_TEST'] / 10
        self.jump_left_yield = self.params['JUMP_LEFT_TEST'] / 10
        self.jump_right_yield = self.params['JUMP_RIGHT_TEST'] / 10
        #print("self.first_step_yield", self.first_step_yield)
        #print("self.cycle_step_yield", self.cycle_step_yield)
        #print("self.side_step_right_yield", self.side_step_right_yield)
        #print(self.params)
        self.landmarks = landmarks
        self.import_strategy_data()
        self.obstacleAvoidanceIsOn = False
        self.imu_drift_correction = 0
        self.imu_drift_last_correction_time = 0
        if self.SIMULATION == 5:
            self.monitor_filename = '/dev/shm/monitor.json'
        else:
            self.monitor_filename = self.current_work_directory + "Soccer/log/monitor.json"

    def import_strategy_data(self):
        if self.event_type == 'FIRA':
            strategy_data_file = init_param_read_path(self.current_work_directory, "strategy_data_FIRA.json")
        else:
            strategy_data_file = init_param_read_path(self.current_work_directory, "strategy_data.json")
        with open(strategy_data_file, "r") as f:
            loaded_Dict = json.loads(f.read())
        if loaded_Dict.get('strategy_data') != None:
            strategy_data = loaded_Dict['strategy_data']
        for column in range(self.COLUMNS):
            for row in range(self.ROWS):
                index1 = column * self.ROWS + row
                power = strategy_data[index1][2]
                yaw = int(strategy_data[index1][3] * 40)  # yaw in radians multiplied by 40
                self.strategy_data[index1*2] = power
                self.strategy_data[index1*2+1] = yaw

    def monitor(self):
        report = {
            'time': time.time(),
            'role': self.role,
            'event_type': self.event_type,
            'ball': self.ball_coord,
            'ball_course': self.ball_course,
            'ball_distance': self.ball_distance,
            'ball_speed': self.ball_speed,
            'robot_see_ball': self.robot_see_ball,
            'direction_to_guest': self.direction_to_guest,
            'kick_power_planned': self.kick_power_planned,
            'strategy_cell': self.strategy_cell,
            'pf_coord': self.pf_coord,
            'coord_odometry': self.local.coord_odometry,
            'coord_visible': getattr(self.local, 'coord_visible', None),
            'ball_odometry': getattr(self.local, 'ball_odometry', None),
            'obstacles': self.obstacles,
            'post_data_in_pose_number': getattr(self.local, 'post_data_in_pose_number', 0),
            'data_quality_is_good': self.data_quality_is_good,
            'camera_down': self.camera_down_Flag,
            'use_particle_filter': self.use_particle_filter,
            'obstacle_avoidance': self.obstacleAvoidanceIsOn,
            'landmarks': {
                'FIELD_LENGTH': self.landmarks.get('FIELD_LENGTH'),
                'FIELD_WIDTH': self.landmarks.get('FIELD_WIDTH'),
            },
        }
        with open(self.monitor_filename, "w") as f:
            json.dump(report, f)

    def neural_vision_enable(self):
        from Soccer.Vision.neural import Neural
        if self.neural is not None:
            self.neural.close()
        self.neural = Neural(self.role, self.display)

    def camera_reset(self):
        print('Camera resetting')
        os.system("espeak -ven-m1 -a"+ '200' + " " + "'Camera resetting'")
        # Finish the old vision consumer before resetting the shared pipeline.
        event = getattr(self.vision, 'event', None)
        if event is not None:
            event.set()
        camera_thread = getattr(self.vision, 'camera_thread', None)
        if camera_thread is not None and camera_thread.is_alive():
            camera_thread.join(timeout=4)
            if camera_thread.is_alive():
                raise RuntimeError('Old vision thread did not stop before camera reset')
        self.camera.stop()
        # Vision_RPI resets the existing STM strobe history before camera.start.
        # Keep the same channel referenced by motion and localisation.
        new_vision = self.Vision_RPI(self)
        self.vision = new_vision
        self.motion.vision = self.vision
        if self.local is not None:
            self.local.vision = self.vision
        self.camera_down_Flag = False

class Variables_4_Walk:
    def __init__(self):
        self.stepLength_0 = 0.0
        self.sideLength_0 = 0.0
        self.direction_0 = 0
        self.stepLength_1 = 0.0
        self.sideLength_1 = 0.0
        self.direction_1 = 0
        self.stepLength_2 = 0.0
        self.sideLength_2 = 0.0
        self.direction_2 = 0
        self.stepLength_3 = 0.0
        self.sideLength_3 = 0.0
        self.direction_3 = 0
        self.rotation = 0.0
        self.gaitHeight = 0.0
        self.stepHeight = 0.0
        self.fr1 = 0
        self.fr2 = 0
        self.amplitude = 0.0
        self.body_tilt_at_walk = 0.0
        self.body_tilt_at_walk_backwards = 0.0
        self.sole_lading_skew = 0.0
        self.body_tilt_at_kick = 0.0
        self.number_Of_Cycles = 0
        self.SIMULATION = 0
        self.coord_shift_x = 0
        self.coord_shift_y = 0
        self.first_Leg_Is_Right_Leg = 0
        self.half = 0
        self.motion_shift_correction_x = 0
        self.motion_shift_correction_y = 0
        self.first_step_yield = 0
        self.cycle_step_yield = 0
        self.side_step_right_yield = 0
        self.side_step_left_yield = 0
        self.rotation_yield_right = 0
        self.rotation_yield_left = 0
        self.imu_drift_speed = 0
        
