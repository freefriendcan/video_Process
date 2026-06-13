import threading
import time

from capture.frame_producer import FrameProducer
from config import PipelineConfig


def test_release_interrupts_reconnect_backoff(monkeypatch):
    monkeypatch.setenv("GO2RTC_ENABLED", "false")

    cfg = PipelineConfig(
        rtsp_url="rtsp://camera.example/stream2",
        rtsp_reconnect_delay=30.0,
        rtsp_max_reconnect_delay=30.0,
    )
    producer = FrameProducer(cfg)
    connect_called = threading.Event()

    def fake_connect():
        connect_called.set()
        return None

    producer._connect = fake_connect

    producer.open()
    assert connect_called.wait(timeout=1.0)
    assert producer._thread is not None

    start = time.monotonic()
    producer.release()
    elapsed = time.monotonic() - start

    assert elapsed < 1.0
    assert not producer._thread.is_alive()
