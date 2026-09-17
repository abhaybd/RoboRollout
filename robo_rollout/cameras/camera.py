from omegaconf import DictConfig, OmegaConf

from robo_rollout.cameras.abstract import BaseCamera

def create_camera(camera_cfg: DictConfig) -> BaseCamera:
    camera_cfg_dict = OmegaConf.to_container(camera_cfg)

    cam_type = camera_cfg_dict.pop("type")
    cam_id = camera_cfg_dict.pop("id")
    if cam_type == "zed":
        from robo_rollout.cameras.zed import ZedCamera
        return ZedCamera(cam_id, **camera_cfg_dict)
    elif cam_type == "realsense":
        from robo_rollout.cameras.realsense import RealSenseCamera
        return RealSenseCamera(cam_id, **camera_cfg_dict)
    elif cam_type == "gopro":
        from robo_rollout.cameras.gopro import GoProCamera
        return GoProCamera(cam_id, **camera_cfg_dict)
    elif cam_type == "v4l":
        from robo_rollout.cameras.v4l import V4LCamera
        return V4LCamera(cam_id, **camera_cfg_dict)
    else:
        raise ValueError(f"Invalid camera type: {camera_cfg.type}")
