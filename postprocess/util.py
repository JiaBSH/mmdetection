from PIL import ImageFont


def _load_font(size: int = 50):
    # Prefer common Linux fonts; fall back to default PIL bitmap font.
    candidates = [
        "arial.ttf",
        "DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for name in candidates:
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


# ================== 全局字体设置（跨平台字体回退） ==================
font = _load_font(50)

# ================== 通用工具函数 ==================
def angle_mod60(angle_deg: float) -> float:
    """把角度归一化到 [0, 60)"""
    return angle_deg % 60.0
