from __future__ import annotations


def fall_region_px(
    frame_w: int,
    frame_h: int,
    *,
    aspect: float = 4.0 / 3.0,
) -> tuple[int, int, int, int]:
    """Central rect of the given aspect inside frame_w x frame_h."""
    if aspect <= 0 or frame_w <= 0 or frame_h <= 0:
        return 0, 0, frame_w, frame_h

    target_w = round(frame_h * aspect)
    if target_w <= frame_w:
        x_off = (frame_w - target_w) // 2
        return x_off, 0, target_w, frame_h

    target_h = round(frame_w / aspect)
    y_off = (frame_h - target_h) // 2
    return 0, y_off, frame_w, target_h
