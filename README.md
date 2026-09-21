# RoboRollout

RoboRollout is a lighweight framework for real-robot policy evaluation. It seeks to consolidate all robot-agnostic utilities and sensor code, and provides clean and easily extensible entrypoints for some robots.
This includes camera code, image/threading utilities, policy server clients, and more.

The following cameras are supported:
- GoPro
- Intel RealSense
- ZED
- Generic video4linux devices

The following robots are supported:
- Franka / DROID (via Polymetis)

## Installation

To use, clone and install this repository:

```bash
git clone https://github.com/abhaybd/RoboRollout.git
cd robo_rollout
pip install -e .
```

See [robot-specific documentation](/scripts) for more specific installation instructions.

### Optional dependencies

Camera-specific packages/libraries are included as optional dependencies, or extras. To install, use any combination of the following extras:

- `gopro`
- `realsense`
- `zed`
