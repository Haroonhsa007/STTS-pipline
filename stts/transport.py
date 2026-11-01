"""Cancellable SSE reads from the inference processes owned by this app."""

import http.client
import queue
import socket
import threading
import time
from urllib.parse import urlsplit

from .config import PipelineError


def event_lines(request, check, abort, timeout=90):
    """Poll cancellation even while a local model has not sent its first byte.

    Shut down the request socket to release a blocked read without unloading the
    model. Stop the owned process only if its reader still fails to release.
    """
    messages = queue.Queue(maxsize=8)
    stopped = threading.Event()
    parsed = urlsplit(request.full_url)
    if parsed.scheme != "http" or parsed.hostname not in {
        "127.0.0.1",
        "localhost",
        "::1",
    }:
        raise PipelineError(
            "Inference streaming requires an owned localhost HTTP service."
        )
    connection = http.client.HTTPConnection(
        parsed.hostname, parsed.port, timeout=timeout
    )
    wire = []

    def publish(value):
        while not stopped.is_set():
            try:
                messages.put(value, timeout=0.1)
                return
            except queue.Full:
                pass

    def reader():
        try:
            connection.connect()
            wire.append(connection.sock)
            target = parsed.path + ("?" + parsed.query if parsed.query else "")
            connection.request(
                request.get_method(),
                target,
                body=request.data,
                headers=dict(request.header_items()),
            )
            with connection.getresponse() as response:
                if response.status != 200:
                    raise PipelineError(
                        f"Local inference HTTP{response.status}: {response.read(512).decode(errors='replace')}"
                    )
                for line in response:
                    if stopped.is_set():
                        break
                    publish(("line", line))
            publish(("end", None))
        except Exception as exc:  # noqa: BLE001 - propagate transport failures to caller
            publish(("error", exc))
        finally:
            connection.close()

    worker = threading.Thread(target=reader, name="stts-sse", daemon=True)
    worker.start()
    deadline = time.monotonic() + timeout
    try:
        while True:
            check()
            if time.monotonic() >= deadline:
                raise PipelineError("Local inference exceeded its response deadline.")
            try:
                kind, value = messages.get(timeout=0.1)
            except queue.Empty:
                continue
            if kind == "end":
                return
            if kind == "error":
                raise value
            yield value
    finally:
        stopped.set()
        if wire:
            try:
                wire[0].shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        worker.join(timeout=0.5)
        if worker.is_alive():
            abort()
