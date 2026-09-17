#!/usr/bin/env python3
import argparse
import copy
import json
import math
import os
import queue
import socket
import sys
import threading
import time
import traceback
from pathlib import Path

import msgpack


REPO_ROOT = Path(__file__).resolve().parents[1]
SLOT_DIR = REPO_ROOT / "Soccer" / "Motion" / "motion_slots"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class DummyLocal:
    def __init__(self):
        self.coord_odometry = [0.0, 0.0, 0.0]
        self.coord_shift = [0.0, 0.0, 0.0]

    def coordinate_record(self, odometry=False, shift=False):
        return None

    def refresh_odometry(self):
        return None

    def correct_yaw_in_pf(self):
        return None


class UdpJpegCamera:
    def __init__(self, width, height, fps=0.0):
        self.width = width
        self.height = height
        self.fps = fps
        self.thread = None
        self.stop_event = threading.Event()
        self.error = None
        self.target = None
        self.raw_format = None
        self.raw_width = None
        self.raw_height = None

    def start(self, host, port, width=None, height=None, fps=None, raw_format=None, raw_width=None, raw_height=None):
        self.stop()
        if width is not None:
            self.width = int(width)
        if height is not None:
            self.height = int(height)
        if fps is not None:
            self.fps = float(fps)
        self.raw_format = str(raw_format) if raw_format else None
        self.raw_width = int(raw_width) if raw_width else None
        self.raw_height = int(raw_height) if raw_height else None
        self.error = None
        self.target = (host, port, self.width, self.height, self.fps, self.raw_format, self.raw_width, self.raw_height)
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run, args=(host, port), daemon=True)
        self.thread.start()

    def stop(self):
        if self.thread is None:
            return
        self.stop_event.set()
        self.thread.join(timeout=4)
        self.thread = None
        self.target = None

    def is_running(self):
        return self.thread is not None and self.thread.is_alive()

    def _run(self, host, port):
        camera = None
        pipeline = None
        try:
            import gi
            import numpy as np
            from Soccer.Vision.camera import Camera

            gi.require_version("Gst", "1.0")
            from gi.repository import Gst

            Gst.init(None)
            # The camera adapter supplies ISP-produced BGR pixels directly.
            caps = (
                f"video/x-raw,format=BGR,width={self.width},height={self.height},"
                f"framerate={self._fps_caps()}"
            )
            launch = (
                f'appsrc name=source is-live=true block=false format=time '
                f'do-timestamp=true caps="{caps}" '
                "! queue leaky=downstream max-size-buffers=1 max-size-time=0 max-size-bytes=0 "
                "! videoconvert ! v4l2jpegenc ! rtpjpegpay pt=26 "
                f"! udpsink host={host} port={port} sync=false async=false"
            )
            pipeline = Gst.parse_launch(launch)
            appsrc = pipeline.get_child_by_name("source")

            camera = Camera()
            camera.camera_lores = (self.width, self.height)
            sensor = {}
            if self.raw_format and self.raw_width and self.raw_height:
                import re
                match = re.search(r"(\d+)", self.raw_format)
                if match is None:
                    raise ValueError("Sensor format must specify its bit depth")
                sensor = {"sensor_size": (self.raw_width, self.raw_height),
                          "sensor_bit_depth": int(match.group(1))}
            frame_us = int(1_000_000 / self.fps) if self.fps > 0 else None
            camera.start(frame_duration_us=frame_us, neural=True, **sensor)
            pipeline.set_state(Gst.State.PLAYING)
            print(f"Manual UDP stream: RTP/JPEG to {host}:{port}", flush=True)

            base_time_ns = time.monotonic_ns()
            while not self.stop_event.is_set():
                frame, _ = camera.snapshot()
                data = np.ascontiguousarray(frame)
                buffer = Gst.Buffer.new_allocate(None, data.nbytes, None)
                buffer.fill(0, data.tobytes())
                pts = time.monotonic_ns() - base_time_ns
                buffer.pts = pts
                buffer.dts = pts
                result = appsrc.emit("push-buffer", buffer)
                if result != Gst.FlowReturn.OK:
                    print(f"Manual UDP stream push failed: {result}", flush=True)
                    time.sleep(0.05)
                if self.fps > 0:
                    time.sleep(max(0.0, (1.0 / self.fps) * 0.25))
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
        finally:
            if pipeline is not None:
                try:
                    pipeline.set_state(Gst.State.NULL)
                except Exception:
                    pass
            if camera is not None:
                try:
                    camera.stop()
                except Exception:
                    pass

    def _fps_caps(self):
        if self.fps <= 0:
            return "0/1"
        fps_int = int(round(self.fps))
        if abs(self.fps - fps_int) < 0.01:
            return f"{fps_int}/1"
        return f"{int(round(self.fps * 1000))}/1000"


class ManualRuntime:
    def __init__(self, args, send_event):
        from Soccer.Localisation.class_Glob import Glob
        from Soccer.Motion.class_Motion_real import Motion_real

        os.environ["ROKI_DISPLAY_BACKEND"] = "null"
        self.args = args
        self.send_event = send_event
        self.commands = queue.Queue()
        self.stop_event = threading.Event()
        self.drive_lock = threading.Lock()
        self.drive = {
            "x": 0.0,
            "y": 0.0,
            "yaw": 0.0,
            "speed": 0.5,
            "hold_crouch": True,
            "updated": 0.0,
        }
        self.state = {
            "ready": False,
            "busy": False,
            "pending": False,
            "pose": "unknown",
            "walking": False,
            "last_error": None,
            "last_command": None,
        }
        self.command_lock = threading.Lock()
        self.cancel_event = threading.Event()
        self.mixing_started = False

        self.glob = Glob(5, str(REPO_ROOT), particles_number=1, event_type="Robocup")
        self.glob.role = "manual_msgpack"
        self.glob.with_Local = False
        self.glob.monitor_is_on = False
        self.glob.camera_streaming = False
        self.glob.neural_vision = False

        self.motion = Motion_real(self.glob, vision=None)
        self.motion.with_Vision = False
        self.motion.local = DummyLocal()
        self.motion.falling_Flag = 0
        self.motion.direction_To_Attack = 0
        self.motion.activation()
        self._start_mixing_once()

        self.camera_modes = []
        self.camera = UdpJpegCamera(args.camera_width, args.camera_height, args.camera_fps)
        self.state["ready"] = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
        print("Manual MessagePack runtime initialized", flush=True)

    def _start_mixing_once(self):
        if self.mixing_started:
            return
        self.motion.rcb.motionPlay(3)
        self.mixing_started = True
        print("Manual runtime: controller mixing slot 3 started once", flush=True)

    def close(self):
        self.stop_event.set()
        self.camera.stop()
        self.thread.join(timeout=3)

    def caps(self):
        return {
            "motion": [
                "drive",
                "stop",
                "pose",
                "slot",
                "jump",
                "jump_turn",
                "kick",
                "hard_kick",
                "reset_queue",
                "head",
            ],
            "poses": ["initial", "final", "base_stand", "head_up", "head_field"],
            "slots": self.list_slots(),
            "stream": {
                "codec": "jpeg",
                "transport": "rtp-udp",
                "width": self.args.camera_width,
                "height": self.args.camera_height,
                "fps": self.args.camera_fps,
                "payload": 26,
                "modes": self.camera_modes,
            },
        }

    def list_slots(self):
        if not SLOT_DIR.exists():
            return []
        return sorted(path.stem for path in SLOT_DIR.glob("*.json"))

    def _detect_camera_modes_light(self):
        try:
            import re
            import libcamera
            manager = libcamera.CameraManager.singleton()
            camera = manager.cameras[0]
            camera.acquire()
            try:
                modes = []
                index = 0
                raw_config = camera.generate_configuration([libcamera.StreamRole.Raw])
                raw_formats = raw_config.at(0).formats
                for pix in raw_formats.pixel_formats:
                    fmt = str(pix)
                    match = re.search(r"(\d+)", fmt)
                    bit_depth = int(match.group(1)) if match else 0
                    for size in raw_formats.sizes(pix):
                        modes.append(
                            {
                                "index": index,
                                "width": int(size.width),
                                "height": int(size.height),
                                "fps": 0.0,
                                "format": fmt,
                                "bit_depth": bit_depth,
                            }
                        )
                        index += 1
                if modes:
                    self.camera_modes = modes
                    return modes
            finally:
                camera.release()
        except Exception as exc:
            print(f"light camera mode detection failed: {type(exc).__name__}: {exc}", flush=True)
        fallback = [
            {
                "index": 0,
                "width": int(self.args.camera_width),
                "height": int(self.args.camera_height),
                "fps": float(self.args.camera_fps),
                "format": "RGB888",
                "bit_depth": 0,
            }
        ]
        self.camera_modes = fallback
        return fallback

    def get_state(self):
        state = dict(self.state)
        state["stream"] = {
            "running": self.camera.is_running(),
            "target": self.camera.target,
            "error": self.camera.error,
        }
        try:
            info = self.glob.stm_channel.mb.GetBodyQueueInfo()
            state["body_queue"] = info[1].Size
        except Exception as exc:
            state["body_queue_error"] = f"{type(exc).__name__}: {exc}"
        return state

    def submit(self, seq, command, args):
        if command == "reset_queue":
            self.cancel_event.set()
            self._zero_drive()
            try:
                self.glob.stm_channel.mb.ResetBodyQueue()
                self.state["walking"] = False
                self.state["pending"] = False
                return {"queued": False, "reset": True}
            except Exception as exc:
                self.state["last_error"] = f"{type(exc).__name__}: {exc}"
                return {"queued": False, "accepted": False, "error": self.state["last_error"]}
        if command == "drive":
            self.set_drive(args)
            return {"queued": False, "accepted": True}
        if command == "stream_start":
            host = str(args.get("host") or self.args.default_client_host)
            port = int(args.get("port", self.args.video_port))
            mode = self._stream_mode(args)
            self.camera.start(
                host,
                port,
                mode["width"],
                mode["height"],
                mode["fps"],
                mode.get("raw_format"),
                mode.get("raw_width"),
                mode.get("raw_height"),
            )
            return {"queued": False, "streaming": True, "host": host, "port": port, "mode": mode}
        if command == "stream_stop":
            self.camera.stop()
            return {"queued": False, "streaming": False}
        if command == "get_caps":
            return self.caps()
        if command == "get_camera_modes":
            return {"modes": self._detect_camera_modes_light(), "probe": "light"}
        if command == "get_state":
            return self.get_state()
        with self.command_lock:
            if self.state["busy"] or self.state["pending"]:
                return {"queued": False, "accepted": False, "busy": True}
            self.state["pending"] = True
        self.commands.put((seq, command, args))
        return {"queued": True, "accepted": True}

    def _stream_mode(self, args):
        mode = None
        if "mode_index" in args:
            index = int(args["mode_index"])
            for candidate in self.camera_modes:
                if int(candidate.get("index", -1)) == index:
                    mode = dict(candidate)
                    break
        if mode is None:
            mode = {
                "index": -1,
                "width": int(args.get("width", self.args.camera_width)),
                "height": int(args.get("height", self.args.camera_height)),
                "fps": float(args.get("fps", self.args.camera_fps)),
                "format": str(args.get("format", "")),
                "bit_depth": int(args.get("bit_depth", 0) or 0),
                "raw_width": int(args.get("raw_width", 0) or 0),
                "raw_height": int(args.get("raw_height", 0) or 0),
                "raw_format": str(args.get("raw_format", args.get("format", "")) or ""),
            }
        mode["width"] = int(args.get("width", mode["width"]))
        mode["height"] = int(args.get("height", mode["height"]))
        mode["fps"] = float(args.get("fps", mode.get("fps", self.args.camera_fps)) or 0.0)
        mode["raw_width"] = int(args.get("raw_width", mode.get("raw_width", 0)) or 0)
        mode["raw_height"] = int(args.get("raw_height", mode.get("raw_height", 0)) or 0)
        mode["raw_format"] = str(args.get("raw_format", mode.get("raw_format", mode.get("format", ""))) or "")
        return mode

    def set_drive(self, args):
        with self.drive_lock:
            self.drive["x"] = self._clamp(float(args.get("x", 0.0)), -1.0, 1.0)
            self.drive["y"] = self._clamp(float(args.get("y", 0.0)), -1.0, 1.0)
            self.drive["yaw"] = self._clamp(float(args.get("yaw", 0.0)), -1.0, 1.0)
            self.drive["speed"] = self._clamp(float(args.get("speed", 0.5)), 0.1, 1.0)
            self.drive["hold_crouch"] = bool(args.get("hold_crouch", True))
            self.drive["updated"] = time.monotonic()

    def _worker(self):
        cycle = 0
        while not self.stop_event.is_set():
            try:
                try:
                    seq, command, args = self.commands.get_nowait()
                except queue.Empty:
                    seq = command = args = None

                if command is not None:
                    self._run_one_shot(seq, command, args)
                    continue

                active, drive = self._drive_snapshot()
                if active:
                    if not self.state["walking"]:
                        if self.state["pose"] != "crouched":
                            self.motion.walk_Initial_Pose(start_mixing=False)
                            self.state["pose"] = "crouched"
                        self.state["walking"] = True
                    self._walk_cycle(drive, cycle)
                    cycle += 1
                else:
                    if self.state["walking"]:
                        self.state["walking"] = False
                        cycle = 0
                        if drive["hold_crouch"]:
                            self._finish_walk_crouch()
                            self.state["pose"] = "crouched"
                        else:
                            self.motion.walk_Final_Pose()
                            self.state["pose"] = "standing"
                    time.sleep(0.02)
            except Exception as exc:
                self.state["last_error"] = f"{type(exc).__name__}: {exc}"
                traceback.print_exc()
                time.sleep(0.2)

    def _drive_snapshot(self):
        with self.drive_lock:
            drive = dict(self.drive)
        stale = time.monotonic() - drive["updated"] > self.args.drive_timeout
        active = (
            not stale
            and (
                abs(drive["x"]) > 0.05
                or abs(drive["y"]) > 0.05
                or abs(drive["yaw"]) > 0.05
            )
        )
        return active, drive

    def _walk_cycle(self, drive, cycle, number_of_cycles=1000000):
        speed = drive["speed"]
        step = drive["x"] * self.args.max_step * speed
        side_raw = drive["y"] * self.args.max_side * speed
        rotation = drive["yaw"] * self.args.max_rotation * speed
        if side_raw >= 0:
            self.motion.first_Leg_Is_Right_Leg = False
        else:
            self.motion.first_Leg_Is_Right_Leg = True
        self.motion.walk_Cycle(
            stepLength=step,
            sideLength=abs(side_raw),
            rotation=rotation,
            cycle=cycle,
            number_Of_Cycles=number_of_cycles,
        )

    def _finish_walk_crouch(self):
        settle_drive = {
            "x": 0.0,
            "y": 0.0,
            "yaw": 0.0,
            "speed": 1.0,
            "hold_crouch": True,
        }
        self._walk_cycle(settle_drive, cycle=0, number_of_cycles=1)

    def _run_one_shot(self, seq, command, args):
        self._zero_drive()
        self.cancel_event.clear()
        self.state["busy"] = True
        self.state["last_command"] = command
        try:
            result = self._execute(command, args)
            self.send_event("result", seq, {"ok": True, "cmd": command, "result": result})
        except Exception as exc:
            self.state["last_error"] = f"{type(exc).__name__}: {exc}"
            self.send_event(
                "result",
                seq,
                {"ok": False, "cmd": command, "error": self.state["last_error"]},
            )
            traceback.print_exc()
        finally:
            self.state["busy"] = False
            self.state["pending"] = False

    def _execute(self, command, args):
        if command == "stop":
            if bool(args.get("stand", False)):
                self.motion.walk_Final_Pose()
                self.state["pose"] = "standing"
            return {"stopped": True}

        if command == "pose":
            name = str(args.get("name", ""))
            if name == "initial":
                if self.state["pose"] == "crouched":
                    return {"pose": name, "already": True}
                self.motion.walk_Initial_Pose(start_mixing=False)
                self.state["pose"] = "crouched"
            elif name == "final":
                if self.state["pose"] in ("standing", "base_stand"):
                    return {"pose": name, "already": True}
                self.motion.walk_Final_Pose()
                self.state["pose"] = "standing"
            elif name == "base_stand":
                self._play_slot_cancellable(name="Initial_Pose", hands_on=True, soft_factor=1)
                self.motion.head_Return(0, 0)
                self.state["pose"] = "base_stand"
            elif name == "head_up":
                self.motion.head_Up()
            elif name == "head_field":
                self.motion.head_Return(0, self.motion.neck_play_pose)
            else:
                raise ValueError(f"unknown pose: {name}")
            return {"pose": name}

        if command == "slot":
            name = str(args["name"])
            hands_on = bool(args.get("hands_on", True))
            soft_factor = float(args.get("soft_factor", 1.0))
            cancelled = self._play_slot_cancellable(name=name, hands_on=hands_on, soft_factor=soft_factor)
            return {"slot": name, "cancelled": cancelled}

        if command == "jump_turn":
            direction = str(args.get("direction", "left"))
            angle = math.radians(float(args.get("angle_deg", self.args.jump_angle_deg)))
            self.motion.refresh_Orientation()
            sign = 1 if direction == "left" else -1
            target = self.motion.norm_yaw(self.motion.body_euler_angle["yaw"] + sign * angle)
            self.motion.jump_turn(target, jumps_limit=int(args.get("jumps_limit", 1)), hands_on=True)
            return {"direction": direction, "angle_deg": math.degrees(sign * angle), "target": target}

        if command == "jump":
            direction = str(args.get("direction", "forward"))
            fraction = float(args.get("fraction", 1.0))
            motion_map = {
                "forward": self.motion.jump_motion_forward,
                "backward": self.motion.jump_motion_backward,
                "left": self.motion.jump_motion_left,
                "right": self.motion.jump_motion_right,
            }
            if direction not in motion_map:
                raise ValueError(f"unknown jump direction: {direction}")
            motion_list = copy.deepcopy(motion_map[direction])
            self._scale_jump_motion(direction, motion_list, fraction)
            cancelled = self._play_slot_cancellable(motion_list=motion_list, hands_on=True, soft_factor=1.0)
            self.state["pose"] = "unknown"
            return {"direction": direction, "fraction": fraction, "cancelled": cancelled}

        if command == "kick":
            leg = str(args.get("leg", "right"))
            self.motion.kick_power = int(args.get("power", self.args.kick_power))
            self.motion.kick(first_Leg_Is_Right_Leg=(leg == "right"), kick_offset=int(args.get("offset", 0)))
            return {"leg": leg, "power": self.motion.kick_power}

        if command == "hard_kick":
            leg = str(args.get("leg", "right"))
            self.motion.kick_power = int(args.get("power", self.args.kick_power))
            self.motion.hard_kick(kick_by_right=1 if leg == "right" else 0, kick_offset=int(args.get("offset", 0)))
            return {"leg": leg, "power": self.motion.kick_power, "hard": True}

        if command == "head":
            pan = int(args.get("pan", 0))
            tilt = int(args.get("tilt", self.motion.neck_play_pose))
            self.motion.head_Return(pan, tilt)
            return {"pan": pan, "tilt": tilt}
        raise ValueError(f"unknown command: {command}")

    def _play_slot_cancellable(self, name="", motion_list=None, hands_on=True, soft_factor=1.0):
        print("playing : ", name, flush=True)
        self.motion.motion_slot_progress = True
        cancelled = False
        try:
            if motion_list is None:
                with open(REPO_ROOT / "Soccer" / "Motion" / "motion_slots" / f"{name}.json", "r") as f:
                    slots = json.loads(f.read())
                motion_list = slots[name]

            with open("Slot_log.txt", "a") as log_file:
                print("playing slot", file=log_file)
                for motion_num, motion in enumerate(motion_list, start=1):
                    if self.cancel_event.is_set():
                        cancelled = True
                        break
                    print("pose number = ", motion_num, file=log_file)
                    servo_datas = self._slot_servo_data(motion, hands_on)
                    frames_number = int(motion[0])
                    frames = int(round(frames_number * soft_factor))
                    pause = int(frames_number - 1)
                    self.motion.rcb.setServoPosAsync(servo_datas, frames, pause)
                    if self._sleep_cancelable(self.glob.params["FRAME_DELAY"] / 1000 * pause):
                        cancelled = True
                        break

            if cancelled:
                self.glob.stm_channel.mb.ResetBodyQueue()
            else:
                self.motion.wait_for_gueue_end(with_Vision=False)
            return cancelled
        finally:
            self.motion.motion_slot_progress = False

    def _slot_servo_data(self, motion, hands_on):
        servo_datas = []
        joint_number = len(motion) - 1
        for i in range(joint_number):
            if (self.motion.ACTIVESERVOS[i][0] in self.motion.hand_servo_ids) and (not hands_on):
                continue
            if self.motion.model == "Roki_2" and self.motion.ACTIVESERVOS[i][0] == 8:
                pos = int(motion[i + 1] * self.motion.ACTIVESERVOS[i][2] / 2 + 7500)
                servo_data = self.motion.Roki.Rcb4.ServoData()
                servo_data.Id, servo_data.Sio, servo_data.Data = 13, self.motion.ACTIVESERVOS[i][1], pos
                servo_datas.append(servo_data)
            else:
                pos = int(motion[i + 1] * self.motion.ACTIVESERVOS[i][2] + 7500)
            servo_data = self.motion.Roki.Rcb4.ServoData()
            servo_data.Id, servo_data.Sio, servo_data.Data = self.motion.ACTIVESERVOS[i][0], self.motion.ACTIVESERVOS[i][1], pos
            servo_datas.append(servo_data)
        return servo_datas

    @staticmethod
    def _scale_jump_motion(direction, motion_list, fraction):
        if abs(fraction - 1.0) < 0.001:
            return
        if direction in ("forward", "backward"):
            for row in (0, 1):
                motion_list[row][2] = int(motion_list[row][2] * fraction)
                motion_list[row][13] = int(motion_list[row][13] * fraction)
        elif direction == "left":
            value = int(-200 * fraction)
            motion_list[0][1] = motion_list[0][5] = motion_list[0][12] = motion_list[0][16] = value
            motion_list[1][5] = motion_list[1][16] = value
        elif direction == "right":
            value = int(200 * fraction)
            motion_list[0][1] = motion_list[0][5] = motion_list[0][12] = motion_list[0][16] = value
            motion_list[1][5] = motion_list[1][16] = value

    def _sleep_cancelable(self, seconds):
        deadline = time.monotonic() + max(0.0, seconds)
        while time.monotonic() < deadline:
            if self.cancel_event.is_set():
                return True
            time.sleep(min(0.02, deadline - time.monotonic()))
        return self.cancel_event.is_set()

    def _zero_drive(self):
        with self.drive_lock:
            self.drive["x"] = 0.0
            self.drive["y"] = 0.0
            self.drive["yaw"] = 0.0
            self.drive["updated"] = 0.0

    @staticmethod
    def _clamp(value, low, high):
        return max(low, min(high, value))


class ManualServer:
    def __init__(self, args):
        self.args = args
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((args.host, args.port))
        self.client_addr = None
        self.runtime = ManualRuntime(args, self.send_event)

    def close(self):
        self.runtime.close()
        self.sock.close()

    def send_event(self, msg_type, seq, payload, addr=None):
        target = addr or self.client_addr
        if target is None:
            return
        self._send(target, {"v": 1, "type": msg_type, "seq": seq, "time": time.time(), "payload": payload})

    def serve_forever(self):
        print(f"Manual MessagePack UDP control: {self.args.host}:{self.args.port}", flush=True)
        while True:
            data, addr = self.sock.recvfrom(65535)
            try:
                msg = msgpack.unpackb(data, raw=False)
                self._handle(addr, msg)
            except Exception as exc:
                traceback.print_exc()
                self._send(addr, self._message("result", None, {"ok": False, "error": str(exc)}))

    def _handle(self, addr, msg):
        msg_type = msg.get("type")
        seq = msg.get("seq")
        payload = msg.get("payload") or {}

        if msg_type == "hello":
            self.client_addr = addr
            print(f"manual client hello from {addr[0]}:{addr[1]}", flush=True)
            self._send(addr, self._message("welcome", seq, {"ok": True, "caps": self.runtime.caps()}))
            return

        if msg_type != "cmd":
            self._send(addr, self._message("result", seq, {"ok": False, "error": "unsupported type"}))
            return

        if self.client_addr is None:
            self.client_addr = addr

        command = str(payload.get("cmd", ""))
        args = payload.get("args") or {}
        if command != "drive":
            print(f"manual cmd from {addr[0]}:{addr[1]} seq={seq}: {command} {args}", flush=True)
        result = self.runtime.submit(seq, command, args)
        if result.get("accepted") is False:
            self._send(addr, self._message("result", seq, {"ok": False, "cmd": command, "error": result.get("error", "busy")}))
            return
        if command in ("drive", "stream_start", "stream_stop", "get_caps", "get_camera_modes", "get_state", "reset_queue"):
            if command == "drive":
                return
            self._send(addr, self._message("result", seq, {"ok": True, "cmd": command, "result": result}))

    def _send(self, addr, msg):
        self.sock.sendto(msgpack.packb(msg, use_bin_type=True), addr)

    @staticmethod
    def _message(msg_type, seq, payload):
        return {"v": 1, "type": msg_type, "seq": seq, "time": time.time(), "payload": payload}


def parse_args():
    parser = argparse.ArgumentParser(description="Manual Roki control over MessagePack/UDP.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8093)
    parser.add_argument("--default-client-host", default="100.0.0.2")
    parser.add_argument("--video-port", type=int, default=5004)
    parser.add_argument("--camera-width", type=int, default=800)
    parser.add_argument("--camera-height", type=int, default=650)
    parser.add_argument("--camera-fps", type=float, default=0.0)
    parser.add_argument("--drive-timeout", type=float, default=0.35)
    parser.add_argument("--max-step", type=float, default=24.0)
    parser.add_argument("--max-side", type=float, default=12.0)
    parser.add_argument("--max-rotation", type=float, default=0.0)
    parser.add_argument("--jump-angle-deg", type=float, default=18.0)
    parser.add_argument("--kick-power", type=int, default=80)
    return parser.parse_args()


def main():
    os.chdir(REPO_ROOT)
    server = ManualServer(parse_args())
    try:
        server.serve_forever()
    finally:
        server.close()


if __name__ == "__main__":
    main()
