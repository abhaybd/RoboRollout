"""
Modified from: https://github.com/allenai/molmospaces/blob/main/molmo_spaces/evaluation/policy_server.py
"""

import asyncio
import http
import logging
import time
import traceback
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import msgpack_numpy
import websockets
import websockets.asyncio.server as _server
import websockets.frames

from robo_rollout.utils.misc import measure_elapsed

logger = logging.getLogger(__name__)


class InferencePolicy(ABC):
    """A policy that can be served by WebsocketPolicyServer."""

    def reset(self) -> None:
        """Clear any per-episode state. Called whenever a client connects, i.e. at the start of each episode."""
        pass

    @abstractmethod
    def get_action(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Compute an action from an observation sent by the client."""
        pass


class WebsocketPolicyServer:
    """
    Serves a policy using the websocket protocol. See PolicyClient for a client implementation.

    Only one client is served at a time; other clients wait until the current one disconnects.
    The policy is reset whenever a client connects.

    All policy methods are called on the thread that calls serve_forever(), which blocks the event loop during inference.
    """

    def __init__(
        self,
        policy: InferencePolicy,
        model_name: str,
        host: str = "0.0.0.0",
        port: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Args:
            policy: The policy to serve.
            model_name: The name of the model to serve. Will be included in the metadata.
            host: The host to serve the policy on.
            port: The port to serve the policy on.
            metadata: Additional metadata to send to clients when they connect.
        """
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = {**(metadata or {}), "model_name": model_name}
        self._client_lock: Optional[asyncio.Lock] = None  # created in _run(), since it must belong to the event loop

        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        """
        Starts the server.
        """
        asyncio.run(self._run())

    async def _run(self):
        self._client_lock = asyncio.Lock()
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        if self._client_lock.locked():
            logger.info(f"Another client is connected, {websocket.remote_address} is waiting")

        async with self._client_lock:
            packer = msgpack_numpy.Packer()
            prev_total_time = None
            try:
                self._policy.reset()
                await websocket.send(packer.pack(self._metadata))

                while True:
                    start_time = time.monotonic()
                    obs = msgpack_numpy.unpackb(await websocket.recv())

                    with measure_elapsed() as infer_time:
                        action = self._policy.get_action(obs)

                    action["server_timing"] = {"infer_ms": int(infer_time.value * 1000)}
                    if prev_total_time is not None:
                        # We can only record the last total time since we also want to include the send time.
                        action["server_timing"]["prev_total_ms"] = int(prev_total_time * 1000)

                    await websocket.send(packer.pack(action))
                    prev_total_time = time.monotonic() - start_time
            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
            except Exception:
                logger.exception("Error in policy server")
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> Optional[_server.Response]:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal request handling.
    return None
