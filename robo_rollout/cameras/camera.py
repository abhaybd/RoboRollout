from robo_rollout.cameras.abstract import BaseCamera
from robo_rollout.cameras.config import CameraConfig

def create_camera(camera_cfg: CameraConfig) -> BaseCamera:
    if camera_cfg.type == "zed":
        from robo_rollout.cameras.zed import ZedCamera
        return ZedCamera(camera_cfg)
    elif camera_cfg.type == "realsense":
        from robo_rollout.cameras.realsense import RealSenseCamera
        return RealSenseCamera(camera_cfg)
    elif camera_cfg.type == "gopro":
        from robo_rollout.cameras.gopro import GoProCamera
        return GoProCamera(camera_cfg)
    elif camera_cfg.type == "v4l":
        from robo_rollout.cameras.v4l import V4LCamera
        return V4LCamera(camera_cfg)
    else:
        raise ValueError(f"Invalid camera type: {camera_cfg.type}")
