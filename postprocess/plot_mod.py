import matplotlib.pyplot as plt
import os


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None:
        return int(default)
    try:
        return int(str(v).strip())
    except Exception:
        return int(default)


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None:
        return float(default)
    try:
        return float(str(v).strip())
    except Exception:
        return float(default)


def plot_hist(data, bins, color, title, xlabel, ylabel, save_path, xlim=None):
    """统一风格绘制直方图"""
    font_size = _env_int("BL_GEOM_PLOT_FONT_SIZE", 14)
    if font_size <= 0:
        font_size = 14
    tick_size = max(1, int(round(font_size * 0.9)))

    # Minimum edge linewidth so histogram bin borders remain visible
    # even when the x-range is large.
    hist_edge_lw = _env_float("BL_GEOM_HIST_EDGE_LW_MIN", 1.2)
    if not (hist_edge_lw > 0):
        hist_edge_lw = 1.2

    plt.figure(figsize=(6,4))
    plt.hist(data, bins=bins, color=color, edgecolor="black", linewidth=float(hist_edge_lw))
    plt.title(title, fontsize=font_size)
    plt.xlabel(xlabel, fontsize=font_size)
    plt.ylabel(ylabel, fontsize=font_size)
    plt.tick_params(axis='both', labelsize=tick_size)
    plt.grid(axis='y', alpha=0.3)
    if xlim is not None:
        plt.xlim(xlim)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def zlplot(x, y, ylabel, save_dir,save_name):
    plt.figure(figsize=(8,4.5))
    plt.bar(x, y, edgecolor="black")
    plt.ylim(0, 1)
    plt.ylabel(ylabel)
    plt.title(ylabel + " Comparison Across Methods")
    for i, v in enumerate(y):
        plt.text(i, min(v + 0.03, 1.0), f"{v:.4f}", ha="center", fontsize=10)
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, save_name))
    plt.close()
