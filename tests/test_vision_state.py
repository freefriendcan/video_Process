from streaming.vision_state import TrackedFace, VisionState


def test_vision_state_serializes_frontend_contract():
    state = VisionState(
        ts=123.4,
        fid=42,
        fps=14.8,
        pw=640,
        ph=480,
        faces=(
            TrackedFace(
                id=7,
                user="Ada",
                nx=0.1,
                ny=0.2,
                nw=0.3,
                nh=0.4,
            ),
        ),
        gesture="Victory",
        fall={
            "status": "FALL CONFIRMED",
            "confidence": 0.93,
            "verifying": False,
            "elapsed": 0.0,
        },
        ir=True,
    )

    payload = state.to_dict()

    assert list(payload.keys()) == ["ts", "fid", "fps", "pw", "ph", "faces", "gesture", "fall", "ir"]
    assert payload["faces"] == [
        {
            "id": 7,
            "user": "Ada",
            "nx": 0.1,
            "ny": 0.2,
            "nw": 0.3,
            "nh": 0.4,
        }
    ]
    assert payload["fall"] == {
        "status": "FALL CONFIRMED",
        "confidence": 0.93,
        "verifying": False,
        "elapsed": 0.0,
    }
    assert payload["ir"] is True
