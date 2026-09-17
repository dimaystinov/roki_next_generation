import Roki
import time
from Soccer.rotation_math import Rotation as R
import struct
import math
from math import pi

class STM_channel():
    def __init__(self, glob):
        self.glob = glob
        self.glob.Roki = Roki
        self.mb = Roki.Motherboard()
        conf = Roki.TTYConfig()
        conf.Port = "/dev/shm/motherboard"
        conf.Baudrate = 1250000
        conf.Stopbits = Roki.Stopbits.One
        conf.ParityBit = True
        conf.Timeout = 2
        self.mb.Configure(conf)
        self.rcb = Roki.Rcb4(self.mb)
        self.zubr = Roki.Zubr(self.mb)

    def read_quaternion_from_imu_in_head(self, frame_number = None):
        # A supplied Unicam sequence identifies the image, independently of
        # whether vision is running in a background thread. Zero is valid.
        if frame_number is not None:
            ok, fr = self.mb.GetIMUFrame(frame_number)
        else:
            ok, fr = self.mb.GetIMULatest()
        if not ok:
            print(self.mb.GetError())
            print ('strobe width :', self.mb.GetStrobeWidth())
            ok, info = self.mb.GetIMUContainerInfo()
            print('error : ', 'frame_number = ', frame_number)
            if ok:
                print("Imu info: " + "  First: " + str(info.First), "  NumAv: " + str(info.NumAv), "  MaxFr: " + str(info.MaxFrames))
                if frame_number is not None and not (info.First <= frame_number < info.First + info.NumAv):
                    self.glob.camera_down_Flag = True
            return (0,0,0,0,0,0)
        # try:
        #     if frame_number != None:
        #         fr = self.mb.GetOrientationBySeq(frame_number)
        #     else:
        #         fr = self.mb.GetCurrentOrientation()
        # except Exception: 
        #     info = self.mb.GetIMUInfo()
        #     print("Imu info: " + "  First: " + str(info.First), "  NumAv: " + str(info.NumAv), "  MaxFr: " + str(info.MaxFrames))
        #     print('error : ', 'frame_number = ', frame_number)
        #     if frame_number < info.First: self.glob.camera_down_Flag = True
        #     return (0,0,0,0,0,0)
        return fr.Orientation.X, fr.Orientation.Y, fr.Orientation.Z, fr.Orientation.W, fr.Timestamp.TimeS, fr.Timestamp.TimeNS
    
    def read_quaternion_from_imu_in_body(self):
        result, quat_bytes = self.rcb.moveRamToComCmdSynchronize(0x0060, 8)
        #print("read_quaternion_from_imu_in_body, quat_bytes: ", quat_bytes)
        try:
            quat = struct.unpack('<hhhh', bytes(quat_bytes))
        except Exception: return False, (0,0,0,1)
        x = quat[0]/16384
        y = quat[1]/16384
        z = quat[2]/16384
        w = quat[3]/16384
        #print("quaternion_control_sum = ", (x**2 + y**2 + z**2 + w**2), ' result: ', result)
        if 0.9 < (x**2 + y**2 + z**2 + w**2) < 1.1:
            return True, (x,y,z,w)
        else: return False, (0,0,0,1)
    
    def read_voltage_from_body(self):
        result, adc_bytes = self.rcb.moveRamToComCmdSynchronize(0xcc, 2)
        #print("read_voltage_from_body, adc_bytes: ", adc_bytes)
        try:
            adc = struct.unpack('<h', bytes(adc_bytes))
        except Exception: return False, 0
        return True, adc[0]

    def pitch_roll_yaw_from_imu_in_head(self, frame_number = None, degrees=False):
        x, y, z, w , timestamp_s, timestamp_ns = self.read_quaternion_from_imu_in_head(frame_number)
        if x*x + y*y + z*z + w*w == 0: return False,0,0,0
        head_rotation = R.from_quat([x, y, z, w])
        imu_pitch, imu_roll, imu_yaw = head_rotation.as_euler('xyz', degrees=degrees)
        imu_pitch -= pi/2                           # IMU Z axis is turned from upward to forward on PCB 
        imu_pitch = self.normalize_pitch(imu_pitch)
        return True, imu_pitch, -imu_roll, imu_yaw
    
    def pitch_roll_yaw_from_imu_in_body(self, degrees=False):
        result, quat  = self.read_quaternion_from_imu_in_body()
        if quat[0]*quat[0] + quat[1]*quat[1] + quat[2]*quat[2] + quat[3]*quat[3] == 0: return False,0,0,0
        head_rotation = R.from_quat(quat)
        imu_pitch, imu_roll, imu_yaw = head_rotation.as_euler('xyz', degrees=degrees)
        imu_pitch -= pi/2                           # IMU Z axis is turned from upward to forward on PCB
        imu_pitch = self.normalize_pitch(imu_pitch)
        return result, imu_pitch, -imu_roll, imu_yaw
    
    def normalize_pitch(self, pitch):       
        """
        function takes pitch value in radians and returns pitch value normalized in between -pi and +pi 
        """
        pitch %= (2 * pi)
        if pitch > pi: pitch -= 2 * pi
        if pitch < -pi: pitch += 2 * pi
        return pitch

if __name__ == '__main__':
    stm_channel = STM_channel(1)
    start_time = time.perf_counter()
    cycles = 100
    for _ in range(cycles):
        result, imu_pitch, imu_roll, imu_yaw = stm_channel.pitch_roll_yaw_from_imu_in_head(frame_number = None, degrees = True)
        print('\r', 'yaw = ', imu_yaw, 'pitch = ', imu_pitch, 'roll = ', imu_roll, end='')
    time_elapsed = time.perf_counter() - start_time
    print('\n Rate : ', int(cycles/ time_elapsed), ' FPS')
    
