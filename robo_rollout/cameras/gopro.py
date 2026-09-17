import numpy as np
import os
import time
import threading

from google.protobuf.timestamp_pb2 import Timestamp
import cv2

from robo_rollout.cameras.abstract import BaseCamera


class GoProCamera(BaseCamera):
    def __init__(self, elgato_device_id, width=640, height=480, frame_rate=30, buffer_size=7):
        assert width / height == 4 / 3, "Aspect ratio must be 4:3"

        self.width = width
        self.height = height
        self.frame_rate = frame_rate
        self.device_id = elgato_device_id

        # Assert that the Elgato device ID exists in /dev/v4l/by-id/ with index0
        base_video_path = "/dev/v4l/by-id/"
        all_video_paths = os.listdir(base_video_path)
        self.video_path = os.path.join(base_video_path, next(path for path in all_video_paths if elgato_device_id in path and path.endswith('-video-index0')))
        self.cap = cv2.VideoCapture(self.video_path, cv2.CAP_V4L2)

        # THIS IS LIKE THIS FOR A REASON. YES IT IS WEIRD.
        # The raw stream from the elagato when the camera is in 4:3 mode is, for reasons best known to
        # the developers and the devil, padded to be 16:9 and then squished to be 4:3 again. The stream has to
        # be set to 16:9 and then center cropped in order to get the raw 4:3 image.
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        print(f"Raw camera resolution: {self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")


        if not self.cap.isOpened():
            raise RuntimeError("Failed to open video capture")

        self.timestamp = Timestamp()
        self.max_frames = buffer_size
        self.frame_stack = []
        self.stack_lock = threading.Lock()
        self.stop_event = threading.Event()

        self.thread = threading.Thread(target=self._update_images)
        self.thread.start()

        time.sleep(0.2)

    def get_frame_shape(self):
        return (self.height, self.width, 3)

    def _update_images(self):
        while not self.stop_event.is_set():
            ret, frame = self.cap.read()

            if not ret: 
                continue
            # Get the dimensions of the frame
            h, w, _ = frame.shape

            # Calculate the target width for a 4:3 aspect ratio
            target_width = int(h * 4 / 3)

            # Calculate the number of pixels to remove from each side
            delta = (w - target_width) // 2

            # Extract the pixels that will be removed
            left_strip = frame[:, :delta]
            right_strip = frame[:, w - delta:]

            # Verify that the removed strips are all black pixels
            # (actually, just check that their mean is very low, close enough)
            if not np.mean(left_strip) < 0.01 and np.mean(right_strip) < 0.01:
            # if not (np.all(left_strip == 0) and np.all(right_strip == 0)):
                print("Warning: The removed pixels are not all black.")

            # Crop the frame to 4:3 aspect ratio
            cropped_frame = frame[:, delta:w - delta]

            # Resize the cropped frame to the desired width and height
            resized_frame = cv2.resize(cropped_frame, (self.width, self.height))

            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

            with self.stack_lock:
                self.frame_stack.append(rgb_frame)
                if len(self.frame_stack) > self.max_frames:
                    self.frame_stack.pop(0)
            self.timestamp.GetCurrentTime()
            time.sleep(1 / self.frame_rate)

    def get_frame(self):
        with self.stack_lock:
            if self.frame_stack:
                return self.frame_stack[-1].copy()
            else:
                print(f"Could not get_frame for device ID {self.device_id}")
                return None

    def close(self):
        self.stop_event.set()
        self.thread.join()
        self.cap.release()


    def __del__(self):
        self.close()
