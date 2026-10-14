# Installation

## Supported robots

If you're using one of the supported robots, follow its getting started guide, which installs RoboRollout along with the robot's own dependencies:

- [Getting Started: Franka DROID](getting-started/droid.md)
- [Getting Started: Bimanual YAM](getting-started/yam.md)

## Custom robots

For custom use cases or robots, install RoboRollout as a project dependency:

```bash
pip install "robo_rollout @ git+https://github.com/abhaybd/RoboRollout.git"
```

Or, to develop RoboRollout itself, install it from a clone:

```bash
git clone https://github.com/abhaybd/RoboRollout.git
cd RoboRollout
pip install -e ".[test]"
```

RoboRollout writes videos by piping frames to `ffmpeg`, so it must be on your `PATH` and support the `h264` encoder.

## Optional dependencies

Camera-specific packages are included as optional dependencies, or extras. To install, use any combination of the following extras:

| Extra | Cameras |
| --- | --- |
| `gopro` | GoPro (through an Elgato capture card) |
| `realsense` | Intel RealSense |

For example:

```bash
pip install "robo_rollout[realsense] @ git+https://github.com/abhaybd/RoboRollout.git"
```

Video4linux cameras need no extra dependencies.

### ZED cameras

The ZED SDK's python bindings (`pyzed`) aren't on PyPI, so to use ZED cameras, install the [ZED SDK](https://www.stereolabs.com/developers/release) and then its python bindings into your environment by running:

```bash
python /usr/local/zed/get_python_api.py
```

## Building these docs

To build and preview these docs locally:

```bash
pip install -e ".[docs]"
mkdocs serve
```
