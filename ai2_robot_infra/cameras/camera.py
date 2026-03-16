from omegaconf import DictConfig, OmegaConf

from ai2_robot_infra.cameras.abstract import BaseCamera

def create_camera(camera_cfg: DictConfig) -> BaseCamera:
    camera_cfg_dict = OmegaConf.to_container(camera_cfg)

    cam_type = camera_cfg_dict.pop("type")
    cam_id = camera_cfg_dict.pop("id")
    if cam_type == "zed":
        from ai2_robot_infra.cameras.zed import ZedCamera
        return ZedCamera(cam_id, **camera_cfg_dict)
    elif cam_type == "realsense":
        from ai2_robot_infra.cameras.realsense import RealSenseCamera
        return RealSenseCamera(cam_id, **camera_cfg_dict)
    elif cam_type == "gopro":
        from ai2_robot_infra.cameras.gopro import GoProCamera
        return GoProCamera(cam_id, **camera_cfg_dict)
    elif cam_type == "v4l":
        from ai2_robot_infra.cameras.v4l import V4LCamera
        return V4LCamera(cam_id, **camera_cfg_dict)
    else:
        raise ValueError(f"Invalid camera type: {camera_cfg.type}")
