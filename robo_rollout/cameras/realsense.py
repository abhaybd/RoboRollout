import numpy as np
import json
import time
import threading

import pyrealsense2 as rs
from google.protobuf.timestamp_pb2 import Timestamp

from robo_rollout.cameras.abstract import BaseCamera

class RealSenseCamera(BaseCamera):
    def __init__(self, camera_id, width=1280, height=720, frame_rate=30, buffer_size=10):
        self.width = width
        self.height = height
        self.frame_rate = frame_rate

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_device(camera_id)
        self.camera_id = camera_id
        self.initialize_cameras()

        self.max_frames = buffer_size
        self.bgr_stack = []
        self.depth_stack = []
        self.bgr_stack_lock = threading.Lock()
        self.depth_stack_lock = threading.Lock()
        self.stop_event = threading.Event()

        self.timestamp = Timestamp()

        camera_thread = threading.Thread(target=self.run)
        camera_thread.start()

        time.sleep(1.0)

    def get_frame_shape(self):
        return (self.height, self.width, 3)

    def save_intrinsics(self, profile):
        intr = (
            profile.get_stream(rs.stream.color)
            .as_video_stream_profile()
            .get_intrinsics()
        )
        intr_di = {
            "coeffs": intr.coeffs,
            "fx": intr.fx,
            "fy": intr.fy,
            "height": intr.height,
            "distortion_model": str(intr.model),
            "ppx": intr.ppx,
            "ppy": intr.ppy,
            "width": intr.width,
            "depth_scale": self.depth_scale,
        }
        with open(
            "camera_intrinsics_{}.txt".format(self.camera_id), "w"
        ) as camera_intr:
            camera_intr.write(json.dumps(intr_di))

    def initialize_cameras(self):
        self.config.enable_stream(
            rs.stream.depth, self.width, self.height, rs.format.z16, self.frame_rate
        )
        self.config.enable_stream(
            rs.stream.color, self.width, self.height, rs.format.bgr8, self.frame_rate
        )
        self.align = rs.align(rs.stream.color)
        profile = self.pipeline.start(self.config)
        self.depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        self.start_streaming = True
        self.save_intrinsics(profile)

    def close(self):
        self.stop_event.set()
        

    def update_images(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)

        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        # Ensure we only store the latest frame
        with self.bgr_stack_lock:
            self.bgr_stack = [np.asanyarray(color_frame.get_data())[:, :, ::-1]]
        with self.depth_stack_lock:
            self.depth_stack = [np.asanyarray(depth_frame.get_data())]
        self.timestamp.GetCurrentTime()
        time.sleep(1 / self.frame_rate)

    def run(self):
        while not self.stop_event.is_set():
            self.update_images()
        self.start_streaming = False
        self.pipeline.stop()

    def get_bgr_image(self):
        with self.bgr_stack_lock:
            if self.bgr_stack:
                return self.bgr_stack[-1].copy()
            else:
                print(f"could not get bgr frame for camera {self.camera_id}")
                return None

    def get_frame(self):
        return self.get_bgr_image()[..., ::-1]

    def get_depth_image(self):
        with self.depth_stack_lock:
            if self.depth_stack:
                return self.depth_stack[-1].copy()
            else:
                print(f"could not get depth frame for camera {self.camera_id}")
                return None
