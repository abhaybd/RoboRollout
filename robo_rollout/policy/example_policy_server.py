"""
Modified from: https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/serving/websocket_policy_server.py
"""


import asyncio
from contextlib import contextmanager
from dataclasses import dataclass
import http
import logging
import time
import traceback

import websockets.asyncio.server as _server
import websockets.frames
import msgpack_numpy

logger = logging.getLogger(__name__)


@dataclass
class MutableFloat:
    value: float | None = None


@contextmanager
def measure_elapsed():
    mf = MutableFloat()
    start = time.perf_counter()
    try:
        yield mf
    finally:
        mf.value = time.perf_counter() - start


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        self._prepared = False
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        if not self._prepared:
            self._policy.prepare_model()
            self._prepared = True
        asyncio.run(self.run())

    async def run(self):
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
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                obs = msgpack_numpy.unpackb(await websocket.recv())

                with measure_elapsed() as total_time:
                    with measure_elapsed() as preprocess_time:
                        model_input = self._policy.obs_to_model_input(obs)
                    with measure_elapsed() as infer_time:
                        model_output = self._policy.inference_model(model_input)
                    with measure_elapsed() as postprocess_time:
                        action = self._policy.model_output_to_action(model_output)

                action["server_timing"] = {
                    "infer_ms": int(infer_time.value * 1000),
                    "preprocess_ms": int(preprocess_time.value * 1000),
                    "postprocess_ms": int(postprocess_time.value * 1000),
                    "total_ms": int(total_time.value * 1000),
                }
                if prev_total_time is not None:
                    # We can only record the last total time since we also want to include the send time.
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal request handling.
    return None
