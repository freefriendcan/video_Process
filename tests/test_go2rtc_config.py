from pathlib import Path

import yaml


def test_go2rtc_streams_are_sd_for_cv_and_hd_for_dashboard():
    cfg = yaml.safe_load(Path("go2rtc/go2rtc.yaml").read_text())

    assert set(cfg["streams"]) == {"living_room_sd", "living_room_hd"}
    assert cfg["streams"]["living_room_sd"].endswith(":554/stream2")
    assert cfg["streams"]["living_room_hd"].endswith(":554/stream1")


def test_go2rtc_webrtc_candidate_is_host_reachable():
    cfg = yaml.safe_load(Path("go2rtc/go2rtc.yaml").read_text())

    assert cfg["webrtc"]["listen"] == ":8555"
    assert cfg["webrtc"]["candidates"] == ["${GO2RTC_WEBRTC_CANDIDATE}"]
