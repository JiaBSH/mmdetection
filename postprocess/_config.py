"""后处理配置数据类。

将原先散落在 analyze_domain_geometry 中的 20+ 个 BL_GEOM_* 环境变量开关
以及 process_one_image / evaluate_model 中的推理参数集中到 dataclass，
替代散乱的 kwargs 传递。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from ._shared import _env_flag, _env_int, _env_float, _env_str


# ---------------------------------------------------------------------------
# 推理配置
# ---------------------------------------------------------------------------

@dataclass
class InferenceConfig:
    """单张图推理参数。"""
    score_thresh: float = 0.5
    target_label: int = 0
    min_pixel_count: int = 10
    device: str = "cuda:0"
    sliding_window: bool = False
    patch_size: int = 1024
    patch_overlap_ratio: float = 0.0
    batch_size: int = 1


# ---------------------------------------------------------------------------
# 几何分析配置
# ---------------------------------------------------------------------------

@dataclass
class AnalysisConfig:
    """几何分析行为开关与参数。

    从 BL_GEOM_* 环境变量读取默认值；显式传参可覆盖。
    所有昂贵操作默认关闭（False），需显式开启。
    """

    # ── 行为开关 ──
    enable_plots: bool = False
    enable_gt: bool = False
    enable_gt_matching: bool = False
    enable_save_images: bool = False
    enable_polygon_metrics: bool = False
    timing: bool = True

    # ── 快速路径 ──
    only_iou_pred_vs_gt: bool = False
    save_pred_geom_hists: bool = False
    save_pred_doa_hists: bool = False
    save_diag_edge_overlay: bool = False

    # ── 并行 / 性能 ──
    parallel_hex: bool = True
    parallel_backend: str = "thread"
    geom_workers: int = 0
    progress_every: int = 500
    max_pts_per_instance: int = 0
    max_instances: int | None = None

    # ── 匹配 ──
    overlap_max_pairs: int = 500_000
    match_max_dist: float = 200.0
    strict_match_plots: bool = True

    # ── 绑图 ──
    scatter_metric: str = "mae"
    plot_font_size: int = 15
    pred_hist_bins: int = 30
    pred_hist_max_pts: int = 2000

    # ── 边界过滤 ──
    boundary_margin: int = 5

    # ── 物理缩放 ──
    scale_ratio: float | None = None
    scale_unit: str | None = None

    @classmethod
    def from_env(cls, **overrides) -> "AnalysisConfig":
        """从环境变量读取配置，kwargs 可覆盖任意字段。"""
        config = cls(
            enable_plots=_env_flag("BL_GEOM_PLOTS", False),
            enable_gt=_env_flag("BL_GEOM_GT", False),
            enable_gt_matching=_env_flag("BL_GEOM_GT_MATCH", False),
            enable_save_images=_env_flag("BL_GEOM_SAVE_IMAGES", False),
            enable_polygon_metrics=_env_flag("BL_GEOM_POLY_METRICS", False),
            timing=_env_flag("BL_GEOM_TIMING", True),
            only_iou_pred_vs_gt=_env_flag("BL_ONLY_PRED_VS_GT_IOU", False),
            save_pred_geom_hists=_env_flag("BL_GEOM_SAVE_PRED_HISTS", False),
            save_pred_doa_hists=_env_flag("BL_GEOM_SAVE_PRED_DOA_HISTS", False),
            save_diag_edge_overlay=_env_flag("BL_GEOM_SAVE_DIAG_EDGE", False),
            parallel_hex=_env_flag("BL_GEOM_PARALLEL_HEX", True),
            parallel_backend=os.getenv("BL_GEOM_PARALLEL_BACKEND", "thread").strip().lower(),
            geom_workers=_env_int("BL_GEOM_WORKERS", 0),
            progress_every=_env_int("BL_GEOM_PROGRESS_EVERY", 500),
            max_pts_per_instance=_env_int("BL_GEOM_MAX_PTS", 0),
            overlap_max_pairs=_env_int("BL_GEOM_OVERLAP_MAX_PAIRS", 500_000),
            match_max_dist=_env_float("BL_GEOM_MATCH_MAX_DIST", 200.0),
            strict_match_plots=_env_flag("BL_GEOM_STRICT_MATCH_PLOTS", True),
            scatter_metric=_env_str("BL_GEOM_SCATTER_METRIC", "mae").strip().lower(),
            plot_font_size=_env_int("BL_GEOM_PLOT_FONT_SIZE", 15),
            pred_hist_bins=_env_int("BL_GEOM_PRED_HIST_BINS", 30),
            pred_hist_max_pts=_env_int("BL_GEOM_PRED_HIST_MAX_PTS", 2000),
            boundary_margin=_env_int("BL_GEOM_BOUNDARY_MARGIN", 5),
        )
        # 应用显式覆盖
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config

    def update(self, **kwargs) -> None:
        """用 kwargs 更新已有字段（忽略不存在的字段）。"""
        for key, value in kwargs.items():
            if hasattr(self, key) and value is not None:
                setattr(self, key, value)


# ---------------------------------------------------------------------------
# 评估配置
# ---------------------------------------------------------------------------

@dataclass
class EvalConfig:
    """单模型评估完整配置。"""
    model_name: str
    config_path: str
    checkpoint_path: str
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    verbose: bool = True
