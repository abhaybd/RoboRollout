# Cameras

All cameras implement [`BaseCamera`][robo_rollout.cameras.abstract.BaseCamera], whose `get_frame()` returns the latest frame as an `(H, W, 3)` `uint8` RGB array. Cameras capture frames in the background, so `get_frame()` doesn't block on the camera.

## Configuring cameras

Cameras are configured under `robot.cameras`, as a dict from camera name to camera config. Each camera type has its own config class in [`robo_rollout.cameras.config`][robo_rollout.cameras.config], which sets its `type` and defaults:

```python
from robo_rollout.cameras.config import RealSenseCameraConfig, ZedCameraConfig

cameras = {
    "wrist_camera": ZedCameraConfig(id=12345678),
    "top_camera": RealSenseCameraConfig(id="123456789012", width=640, height=360),
}
```

All cameras have the following settings:

| Setting | Description |
| --- | --- |
| `type` | The camera type, set by the config class |
| `id` | Which camera to open (see below) |
| `width`, `height` | The resolution to capture at |
| `frame_rate` | The frame rate to capture at. Defaults to 30. |

## Supported cameras

| Type | Config class | `id` | Requirements |
| --- | --- | --- | --- |
| ZED | [`ZedCameraConfig`][robo_rollout.cameras.config.ZedCameraConfig] | Serial number | The ZED SDK and its python bindings (see [Installation](../installation.md#zed-cameras)) |
| Intel RealSense | [`RealSenseCameraConfig`][robo_rollout.cameras.config.RealSenseCameraConfig] | Serial number, as a string | The `realsense` extra |
| GoPro | [`GoProCameraConfig`][robo_rollout.cameras.config.GoProCameraConfig] | Device ID of the Elgato capture card, as found in `/dev/v4l/by-id/` | The `gopro` extra |
| Video4linux | [`V4LCameraConfig`][robo_rollout.cameras.config.V4LCameraConfig] | Device number `N` (opened as `/dev/videoN`), or a device path | None |

Some cameras have extra settings:

- **ZED**: `side` chooses which of the stereo cameras to capture (`left` by default).
- **RealSense** and **GoPro**: `buffer_size` sets how many frames to buffer.
- **GoPro**: the resolution must have a 4:3 aspect ratio.
- **Video4linux**: for stereo cameras that output both images side by side in one frame, `stereo_side` captures only the `left` or `right` half.

## Adding a new camera type

To add a new camera type:

1. Add a config dataclass to [`robo_rollout/cameras/config.py`](https://github.com/abhaybd/RoboRollout/blob/main/robo_rollout/cameras/config.py) that subclasses [`CameraConfig`][robo_rollout.cameras.config.CameraConfig] and sets a new `type`.
2. Subclass [`BaseCamera`][robo_rollout.cameras.abstract.BaseCamera], implementing `get_frame()`, `get_frame_shape()`, and `close()`. Its constructor takes the config. Raise [`CameraException`][robo_rollout.cameras.abstract.CameraException] if the camera fails.
3. Add the type to [`create_camera()`][robo_rollout.cameras.camera.create_camera]. Import the camera inside its branch, so its dependencies are only needed when it's used.

Keep `get_frame()` fast, since it's called each control step and by the real-time video recorder. Capturing frames on a background thread and returning the latest one works well.
