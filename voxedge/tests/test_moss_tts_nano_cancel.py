"""MOSS service cancellation reaches the native worker out of band."""

import threading
import time

from voxedge.backends.jetson.moss_tts_nano import (
    MossTtsNanoBackend,
    MossTtsNanoConfig,
)


def test_cancel_event_sends_protocol_and_drains_terminal(monkeypatch):
    backend = MossTtsNanoBackend(MossTtsNanoConfig())
    sent: list[dict] = []

    def _send(payload):
        sent.append(payload)
        if payload.get("type") == "cancel":
            request_id = payload["id"]
            with backend._queues_lock:
                request_queue = backend._request_queues[request_id]
            request_queue.put({
                "event": "cancel_ack",
                "id": request_id,
                "tripped": True,
            })
            request_queue.put({
                "event": "cancelled",
                "id": request_id,
                "ok": True,
                "reason": "cancelled",
            })

    monkeypatch.setattr(backend, "_send_request", _send)
    cancel_event = threading.Event()
    chunks: list[bytes] = []

    consumer = threading.Thread(
        target=lambda: chunks.extend(
            backend._generate_streaming_impl(
                "cancel me",
                cancel_event=cancel_event,
            )
        )
    )
    consumer.start()
    for _ in range(100):
        if sent:
            break
        time.sleep(0.005)

    assert sent and sent[0]["text"] == "cancel me"
    cancel_event.set()
    consumer.join(timeout=0.5)

    assert not consumer.is_alive()
    assert chunks == []
    assert sent[1] == {"type": "cancel", "id": sent[0]["id"]}


def test_false_cancel_ack_is_retried(monkeypatch):
    backend = MossTtsNanoBackend(MossTtsNanoConfig())
    sent: list[dict] = []

    def _send(payload):
        sent.append(payload)
        if payload.get("type") != "cancel":
            return
        request_id = payload["id"]
        with backend._queues_lock:
            request_queue = backend._request_queues[request_id]
        if sum(item.get("type") == "cancel" for item in sent) == 1:
            request_queue.put({
                "event": "cancel_ack",
                "id": request_id,
                "tripped": False,
            })
        else:
            request_queue.put({
                "event": "cancel_ack",
                "id": request_id,
                "tripped": True,
            })
            request_queue.put({
                "event": "cancelled",
                "id": request_id,
                "ok": True,
            })

    monkeypatch.setattr(backend, "_send_request", _send)
    cancel_event = threading.Event()
    cancel_event.set()

    assert list(backend._generate_streaming_impl(
        "retry cancel",
        cancel_event=cancel_event,
    )) == []
    assert sum(item.get("type") == "cancel" for item in sent) == 2
