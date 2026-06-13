import pytest

from config import PipelineConfig


def test_go2rtc_enabled_routes_rtsp_to_proxy(monkeypatch):
    monkeypatch.setenv("GO2RTC_ENABLED", "true")
    monkeypatch.setenv("GO2RTC_HOST", "localhost")
    monkeypatch.setenv("GO2RTC_RTSP_PORT", "8554")
    monkeypatch.setenv("GO2RTC_STREAM_NAME", "living_room")

    cfg = PipelineConfig()

    assert cfg.rtsp_url == "rtsp://localhost:8554/living_room_sd"
    assert cfg.rtsp_hires_url == "rtsp://localhost:8554/living_room_hd"
    assert cfg.go2rtc_dashboard_url == (
        "http://localhost:1984/stream.html?src=living_room_hd&mode=webrtc"
    )


def test_go2rtc_enabled_allows_explicit_stream_names(monkeypatch):
    monkeypatch.setenv("GO2RTC_ENABLED", "true")
    monkeypatch.setenv("GO2RTC_HOST", "localhost")
    monkeypatch.setenv("GO2RTC_RTSP_PORT", "8554")
    monkeypatch.setenv("GO2RTC_STREAM_NAME", "living_room")
    monkeypatch.setenv("GO2RTC_CAMERA_STREAM_NAME", "living_room_sd")
    monkeypatch.setenv("GO2RTC_HIRES_STREAM_NAME", "living_room_hd")
    monkeypatch.setenv("GO2RTC_DASHBOARD_STREAM_NAME", "living_room_hd")
    monkeypatch.setenv("GO2RTC_DASHBOARD_MODE", "webrtc/tcp")

    cfg = PipelineConfig()

    assert cfg.rtsp_url == "rtsp://localhost:8554/living_room_sd"
    assert cfg.rtsp_hires_url == "rtsp://localhost:8554/living_room_hd"
    assert cfg.go2rtc_dashboard_url == (
        "http://localhost:1984/stream.html?src=living_room_hd&mode=webrtc%2Ftcp"
    )


def test_go2rtc_disabled_keeps_direct_tapo_urls(monkeypatch):
    monkeypatch.setenv("GO2RTC_ENABLED", "false")
    monkeypatch.setenv("TAPO_IP", "192.168.1.50")
    monkeypatch.setenv("TAPO_USER", "alice")
    monkeypatch.setenv("TAPO_PASS", "secret")

    cfg = PipelineConfig()

    assert cfg.rtsp_url == "rtsp://alice:secret@192.168.1.50:554/stream2"
    assert cfg.rtsp_hires_url == "rtsp://alice:secret@192.168.1.50:554/stream1"


def test_go2rtc_preflight_fails_when_rtsp_port_is_unreachable(monkeypatch):
    monkeypatch.setenv("GO2RTC_ENABLED", "true")
    monkeypatch.setenv("GO2RTC_HOST", "localhost")
    monkeypatch.setenv("GO2RTC_RTSP_PORT", "8554")

    def fail_connect(address, timeout):
        raise ConnectionRefusedError("refused")

    monkeypatch.setattr("config.socket.create_connection", fail_connect)

    cfg = PipelineConfig()

    with pytest.raises(RuntimeError) as exc_info:
        cfg.preflight()

    assert (
        "go2rtc RTSP is not reachable at localhost:8554. "
        "Start it with: cd go2rtc && docker compose up -d"
    ) in str(exc_info.value)


def test_go2rtc_preflight_checks_api_after_rtsp(monkeypatch):
    monkeypatch.setenv("GO2RTC_ENABLED", "true")
    monkeypatch.setenv("GO2RTC_HOST", "localhost")
    monkeypatch.setenv("GO2RTC_RTSP_PORT", "8554")
    monkeypatch.setenv("GO2RTC_API_PORT", "1984")
    calls = []

    class FakeSocket:
        def close(self):
            pass

    def fake_connect(address, timeout):
        calls.append(address)
        if address == ("localhost", 1984):
            raise TimeoutError("timed out")
        return FakeSocket()

    monkeypatch.setattr("config.socket.create_connection", fake_connect)

    cfg = PipelineConfig()

    with pytest.raises(RuntimeError) as exc_info:
        cfg.preflight()

    assert calls == [("localhost", 8554), ("localhost", 1984)]
    assert "go2rtc API is not reachable at localhost:1984" in str(exc_info.value)


def test_go2rtc_preflight_is_skipped_when_disabled(monkeypatch):
    monkeypatch.setenv("GO2RTC_ENABLED", "false")

    def fail_connect(address, timeout):
        raise AssertionError("preflight should not open sockets")

    monkeypatch.setattr("config.socket.create_connection", fail_connect)

    cfg = PipelineConfig()

    cfg.preflight()
