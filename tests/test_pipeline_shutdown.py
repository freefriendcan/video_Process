import threading

from main import VisionPipeline


class FakeDispatcher:
    def __init__(self):
        self.offline_calls = 0
        self.shutdown_calls = 0

    def send_offline_signal(self):
        self.offline_calls += 1

    def shutdown(self):
        self.shutdown_calls += 1


class FakeProducer:
    def __init__(self):
        self.release_calls = 0

    def release(self):
        self.release_calls += 1


class FakeWSServer:
    def __init__(self):
        self.stop_calls = 0

    def stop(self):
        self.stop_calls += 1


def test_shutdown_is_idempotent():
    pipeline = VisionPipeline.__new__(VisionPipeline)
    dispatcher = FakeDispatcher()
    producer = FakeProducer()
    ws_server = FakeWSServer()

    pipeline._running = True
    pipeline._shutdown_lock = threading.Lock()
    pipeline._shutdown_started = False
    pipeline._dispatcher = dispatcher
    pipeline._producer = producer
    pipeline._ws_server = ws_server

    assert pipeline.shutdown() is True
    assert pipeline.shutdown() is False

    assert pipeline._running is False
    assert dispatcher.offline_calls == 1
    assert dispatcher.shutdown_calls == 1
    assert producer.release_calls == 1
    assert ws_server.stop_calls == 1
