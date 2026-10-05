"""
End-to-end tests of WebsocketPolicyServer, served on a background thread and queried with PolicyClient.
"""

import socket
import threading
import time
import urllib.request

import numpy as np
import pytest

from robo_rollout.policy.policy_client import PolicyClient
from robo_rollout.policy.policy_server import InferencePolicy, WebsocketPolicyServer


class CountingPolicy(InferencePolicy):
    """Returns qpos + the number of steps since the last reset, and fails on obs with "fail" set."""

    def __init__(self):
        self.resets = 0
        self.steps = 0

    def reset(self):
        self.resets += 1
        self.steps = 0

    def get_action(self, obs):
        if obs.get("fail"):
            raise ValueError("bad observation")
        self.steps += 1
        return {"arm": obs["qpos"] + self.steps}


@pytest.fixture
def server():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    policy = CountingPolicy()
    server = WebsocketPolicyServer(policy, model_name="counting", host="127.0.0.1", port=port, metadata={"dt": 0.1})
    threading.Thread(target=server.serve_forever, daemon=True).start()
    # Wait for the server to come up, since PolicyClient waits 5s between connection attempts
    deadline = time.monotonic() + 5
    while True:
        try:
            socket.create_connection(("127.0.0.1", port)).close()
            break
        except ConnectionRefusedError:
            assert time.monotonic() < deadline, "server did not start"
            time.sleep(0.01)
    return policy, port


def test_infer_returns_action_metadata_and_timing(server):
    _, port = server
    client = PolicyClient(host="127.0.0.1", port=port)
    assert client.get_server_metadata() == {"dt": 0.1, "model_name": "counting"}

    first = client.infer({"qpos": np.zeros(3)})
    second = client.infer({"qpos": np.zeros(3)})
    client.close()

    np.testing.assert_array_equal(first["arm"], np.ones(3))
    np.testing.assert_array_equal(second["arm"], np.full(3, 2.0))
    assert set(first["server_timing"]) == {"infer_ms"}
    assert set(second["server_timing"]) == {"infer_ms", "prev_total_ms"}


def test_policy_is_reset_on_each_connection(server):
    policy, port = server
    for _ in range(2):
        client = PolicyClient(host="127.0.0.1", port=port)
        action = client.infer({"qpos": np.zeros(1)})
        client.close()
        np.testing.assert_array_equal(action["arm"], np.ones(1))
    assert policy.resets == 2


def test_second_client_waits_for_first_to_disconnect(server):
    _, port = server
    first = PolicyClient(host="127.0.0.1", port=port)
    first.infer({"qpos": np.zeros(1)})

    second_result = {}

    def connect_second():
        second = PolicyClient(host="127.0.0.1", port=port)
        second_result["action"] = second.infer({"qpos": np.zeros(1)})
        second.close()

    thread = threading.Thread(target=connect_second, daemon=True)
    thread.start()
    thread.join(timeout=0.5)
    assert thread.is_alive()  # blocked waiting for the server metadata

    first.close()
    thread.join(timeout=5)
    assert not thread.is_alive()
    np.testing.assert_array_equal(second_result["action"]["arm"], np.ones(1))


def test_policy_error_is_raised_in_client_and_server_keeps_serving(server):
    _, port = server
    client = PolicyClient(host="127.0.0.1", port=port)
    with pytest.raises(RuntimeError, match="bad observation"):
        client.infer({"qpos": np.zeros(1), "fail": True})
    client.close()

    client = PolicyClient(host="127.0.0.1", port=port)
    np.testing.assert_array_equal(client.infer({"qpos": np.zeros(1)})["arm"], np.ones(1))
    client.close()


def test_health_check(server):
    _, port = server
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz") as resp:
        assert resp.status == 200
