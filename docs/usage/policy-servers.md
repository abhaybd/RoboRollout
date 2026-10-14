# Policy servers

Policies run in a separate process from the robot, behind a websocket server. This keeps the policy's dependencies out of the robot's environment, and lets the policy run on a separate (e.g. GPU) machine.

!!! tip "Off-the-shelf policies"
    [RoboRollout-VLA](https://github.com/abhaybd/RoboRollout_VLA) provides ready-made servers for openpi, MolmoBot, and LeRobot policies. Use those if they cover your policy, or as examples of how to write your own.

## Writing a policy server

Subclass [`InferencePolicy`][robo_rollout.policy.policy_server.InferencePolicy], implementing `get_action()` and optionally `reset()`, and serve it with [`WebsocketPolicyServer`][robo_rollout.policy.policy_server.WebsocketPolicyServer]:

```python
from robo_rollout.policy.policy_server import InferencePolicy, WebsocketPolicyServer


class MyPolicy(InferencePolicy):
    def __init__(self):
        self.model = load_my_model()

    def reset(self):
        # clear per-episode state, such as a queue of predicted actions
        ...

    def get_action(self, obs):
        action = self.model.predict(
            image=obs["exterior_image"],  # keys from policy_cameras
            state=obs["qpos"]["arm"],
            instruction=obs["task"],
        )
        return {"arm": action[:7], "gripper": action[7]}


if __name__ == "__main__":
    server = WebsocketPolicyServer(MyPolicy(), model_name="my_policy", port=8000)
    server.serve_forever()
```

Then point the robot at the server with `policy_host` and `policy_port`.

### Observations

Each observation is a dict containing:

- `task`: the task instruction, as a string
- The robot state from [`get_state()`][robo_rollout.robot.BaseRobot.get_state], which is often nested (e.g. `obs["qpos"]["arm"]`)
- One `(H, W, 3)` `uint8` RGB image per entry in `policy_cameras`, keyed by its observation key and resized as set by `image_input` (see [Configuration](configuration.md#cameras-and-policy-observations))

The robot state keys depend on the robot: see [Franka DROID](../getting-started/droid.md#observations-and-actions) and [Bimanual YAM](../getting-started/yam.md#observations-and-actions).

### Actions

`get_action()` must return a dict, which is passed to the robot's [`apply_action()`][robo_rollout.robot.BaseRobot.apply_action]. The keys it expects depend on the robot. You can include extra keys, such as debugging info, and they're saved to `actions.npz` along with the rest of the action.

The server adds a `server_timing` entry to each action, with the inference time in `infer_ms`, and the total time of the previous step (including receiving and sending) in `prev_total_ms`.

### Metadata

When a client connects, the server sends it metadata, which the robot side prints and logs to W&B under `policy_metadata`. It always includes `model_name`, and you can add more by passing `metadata` to the server, such as the checkpoint, so you can tell runs apart later:

```python
WebsocketPolicyServer(policy, model_name="my_policy", metadata={"checkpoint": ckpt_path})
```

## Protocol details

- Observations and actions are serialized with [msgpack-numpy](https://github.com/lebedov/msgpack-numpy), so they can contain numpy arrays.
- The robot reconnects at the start of each episode, and the server calls `reset()` whenever a client connects.
- Only one client is served at a time. Other clients wait until the current one disconnects.
- If `get_action()` raises, the server sends the traceback to the client, which raises it as a `RuntimeError`, and closes the connection.
- Inference runs on the server's event loop thread, so the server doesn't respond to anything else during inference.
- The server responds to HTTP requests to `/healthz` with `200 OK`, for health checks.
- [`PolicyClient`][robo_rollout.policy.policy_client.PolicyClient] can send an API key in the `Authorization` header, for servers behind an authenticating proxy. `WebsocketPolicyServer` doesn't check it.
