# Ai2 Robotics Infrastructure

This repository contains common robot infrastructure for Ai2 Robotics. This includes camera code, image/threading utilities, policy server clients, and more. The following cameras are supported:
- GoPro
- Intel RealSense
- ZED
- Generic video4linux devices

## Installation

To use, clone and install this repository:

```bash
git clone https://github.com/abhaybd/RoboRollout.git
cd robo_rollout
pip install -e .
```

### Optional dependencies

Camera-specific packages/libraries are included as optional dependencies, or extras. To install, use any combination of the following extras:

- `gopro`
- `realsense`
- `zed`
