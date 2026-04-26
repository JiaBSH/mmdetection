#剔除边界
from .compute_metrics import Compute_metrics
import os
from typing import Optional
from .draw_polygons_overlay import draw_polygons_overlay
from .plot_mod import plot_hist, zlplot
import matplotlib
matplotlib.use('Agg')
from scipy.spatial import ConvexHull
from scipy.optimize import linear_sum_assignment
import numpy as np
import random
from PIL import Image, ImageDraw
import csv
import json
import matplotlib.pyplot as plt
from .fit_polygon import fit_polygon
#from fitting import fit_polygon
from .ex_coords import ex_c
from .angle_diag import angle_diag_reg
from .angle_edge import angle_edge_reg
from .lenght_diag import lenght_diag_reg
from .lenght_side import lenght_side_reg
from .diag_edge import diaglen_edgeangle_overlay
import time
from contextlib import contextmanager
from .util import angle_mod60
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed


def _compute_pred_basic_stats(
    global_instances,
    *,
    max_pts_for_orientation: int = 2000,
    progress_every: int = 0,
):
    """Fast per-instance stats without convex hull/hex fitting.

    Returns:
      widths_px, heights_px, areas_px2, orientations_deg_mod60

    - area uses pixel-count (len(coords)) as a fast proxy.
    - orientation uses PCA on subsampled coords.
    """
    widths = []
    heights = []
    areas = []
    orientations = []

    for i, inst in enumerate(global_instances):
        if progress_every and i > 0 and (i % progress_every == 0):
            print(f"... pred basic stats {i}/{len(global_instances)}")

        c = inst.get('coords', [])
        if isinstance(c, np.ndarray):
            pts = c
        else:
            pts = np.asarray(list(c))
        if pts.size == 0:
            continue
        if pts.ndim != 2 or pts.shape[1] != 2:
            continue

        ys = pts[:, 0]
        xs = pts[:, 1]
        widths.append(float(xs.max() - xs.min()))
        heights.append(float(ys.max() - ys.min()))
        areas.append(float(len(pts)))

        # PCA orientation (subsample for speed)
        pts_use = pts
        if max_pts_for_orientation and len(pts_use) > max_pts_for_orientation:
            step = max(1, int(len(pts_use) // max_pts_for_orientation))
            pts_use = pts_use[::step]
        if len(pts_use) >= 3:
            x = pts_use.astype(np.float64, copy=False)
            x = x - x.mean(axis=0, keepdims=True)
            cov = (x.T @ x) / max(1.0, float(len(x) - 1))
            try:
                w, v = np.linalg.eigh(cov)
                vec = v[:, int(np.argmax(w))]  # (y,x)
                dy, dx = float(vec[0]), float(vec[1])
                ang = angle_mod60(np.degrees(np.arctan2(dy, dx)))
                orientations.append(float(ang))
            except Exception:
                orientations.append(0.0)
        else:
            orientations.append(0.0)

    return widths, heights, areas, orientations


def _env_flag(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    v = str(v).strip().lower()
    if v in ("1", "true", "yes", "y", "on"):
        return True
    if v in ("0", "false", "no", "n", "off"):
        return False
    return default


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None:
        return default
    try:
        return int(str(v).strip())
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None:
        return float(default)
    try:
        return float(str(v).strip())
    except Exception:
        return float(default)


def _env_str(name: str, default: str) -> str:
    v = os.getenv(name)
    if v is None:
        return str(default)
    try:
        return str(v)
    except Exception:
        return str(default)


def safe_convex_hull(pts):
    pts = np.asarray(pts)

    # 点数不足
    if pts.shape[0] < 3:
        return None

    # 判断是否退化（共线）
    x_range = pts[:, 0].max() - pts[:, 0].min()
    y_range = pts[:, 1].max() - pts[:, 1].min()

    if x_range == 0 or y_range == 0:
        # 退化为 1D，不算 ConvexHull
        return None

    return ConvexHull(pts)


def _hull_hex_fit_from_pts(pts):
    """Compute convex hull vertices, area, centroid and hexagon fit from input points.

    Notes:
    - `pts` is expected to be (N,2) as (row=y, col=x).
    - This keeps the previous hexagon fitting logic: `fit_polygon(..., k=6)`.
    - Returns (hull_pts, hexagon, area, centroid) where hull_pts/centroid may be None.
    """
    try:
        pts = np.asarray(pts)
        if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] < 3:
            return None, None, 0.0, None

        hull = safe_convex_hull(pts)
        if hull is None:
            return None, None, 0.0, None

        hull_pts = pts[hull.vertices]
        if hull_pts is None or len(hull_pts) < 3:
            return None, None, 0.0, None

        area = float(getattr(hull, "volume", 0.0) or 0.0)
        centroid = hull_pts.mean(axis=0)
        hexagon = fit_polygon(None, hull_pts, k=6, use_convex_hull=False)
        return hull_pts, hexagon, area, centroid
    except Exception:
        return None, None, 0.0, None


def compute_instance_centroids(instances):
    """
    Compute centroids for a list of instances.

    Each instance is expected to be a dict with a 'coords' iterable of (row=y, col=x) points.
    Returns a numpy array of shape (N,2) where each row is [y, x].
    Empty or degenerate instances yield a centroid of [0.0, 0.0].
    This utility centralizes centroid computation to ensure consistent ordering and format
    when matching/visualizing predictions against ground-truth.
    """
    centroids = []
    for inst in instances:
        pts = np.array(list(inst.get('coords', [])))
        if pts.size == 0:
            centroids.append(np.array([0.0, 0.0]))
        else:
            centroids.append(pts.mean(axis=0))
    return np.array(centroids)


def filter_matches_by_distance(row_ind, col_ind, pred_centroids, gt_centroids, max_dist):
    """
    Filter matched index pairs (row_ind, col_ind) by the Euclidean distance
    between the corresponding centroids.

    - `row_ind`/`col_ind` are iterables of equal length representing matched pairs
      where `row_ind[i]` indexes GT entries and `col_ind[i]` indexes predicted entries.
    - `pred_centroids`: numpy array (M,2) with predicted centroids in (y,x) order,
      aligned with the prediction indices used when creating `col_ind`.
    - `gt_centroids`: numpy array (K,2) with GT centroids in (y,x) order,
      aligned with `row_ind` indices.
    - `max_dist`: maximum allowed distance (in pixels). Matches with greater
      centroid distance are discarded.

    Returns two lists: (filtered_row_ind, filtered_col_ind).
    """
    filtered_r = []
    filtered_c = []
    if pred_centroids is None or gt_centroids is None:
        return filtered_r, filtered_c
    for r, c in zip(row_ind, col_ind):
        if r < 0 or c < 0:
            continue
        if r >= len(gt_centroids) or c >= len(pred_centroids):
            continue
        gt_c = gt_centroids[r]
        pred_c = pred_centroids[c]
        dist = float(np.linalg.norm(pred_c - gt_c))
        if dist <= max_dist:
            filtered_r.append(r)
            filtered_c.append(c)
    return filtered_r, filtered_c


def draw_match_overlay(overlay_img, pred_centroids, gt_centroids, matches, save_path):
    """
    Draw an overlay image that visualizes matches between predictions and GT.

    - `overlay_img`: PIL Image to draw on (copied inside function).
    - `pred_centroids`: numpy array (M,2) of predicted centroids (y,x).
    - `gt_centroids`: numpy array (K,2) of GT centroids (y,x).
    - `matches`: iterable of (r,c) pairs indicating matched gt index -> pred index.
    - `save_path`: path where the resulting image will be saved.

    The function draws white connecting lines for each match and colored markers
    (orange for predictions, blue for GT). If `matches` is empty, it will draw
    centroids only (no connecting lines).
    """
    try:
        img = overlay_img.copy()
        draw = ImageDraw.Draw(img)

        # Ensure matches is a concrete list (zip/iterators may be consumed when checked)
        matches_list = list(matches) if matches is not None else []

        if len(matches_list) > 0:
            for r, c in matches_list:
                if r < 0 or c < 0:
                    continue
                if r >= len(gt_centroids) or c >= len(pred_centroids):
                    continue
                gt_c = gt_centroids[r]
                pred_c = pred_centroids[c]
                gt_xy = (float(gt_c[1]), float(gt_c[0]))
                pred_xy = (float(pred_c[1]), float(pred_c[0]))
                draw.line([pred_xy, gt_xy], fill='white', width=2)
                rads = 4
                draw.ellipse([pred_xy[0]-rads, pred_xy[1]-rads, pred_xy[0]+rads, pred_xy[1]+rads], outline='orange', width=2)
                draw.ellipse([gt_xy[0]-rads, gt_xy[1]-rads, gt_xy[0]+rads, gt_xy[1]+rads], outline='blue', width=2)
        else:
            # only draw centroids
            for pc in pred_centroids:
                pxy = (float(pc[1]), float(pc[0]))
                draw.ellipse([pxy[0]-3, pxy[1]-3, pxy[0]+3, pxy[1]+3], outline='orange', width=2)
            for gc in gt_centroids:
                gxy = (float(gc[1]), float(gc[0]))
                draw.ellipse([gxy[0]-3, gxy[1]-3, gxy[0]+3, gxy[1]+3], outline='blue', width=2)

        img.save(save_path)
    except Exception:
        # Drawing must not break the analysis pipeline; swallow errors silently
        pass

def analyze_domain_geometry(
    orig_image,
    global_instances,
    overlayed,
    save_dir,
    gt_json_path=None,
    max_instances=None,
    scale_ratio=None,
    scale_unit=None,
    *,
    timing: Optional[bool] = None,
    enable_plots: Optional[bool] = None,
    enable_gt: Optional[bool] = None,
    enable_gt_matching: Optional[bool] = None,
    enable_save_images: Optional[bool] = None,
    enable_polygon_metrics: Optional[bool] = None,
    progress_every: Optional[int] = None,
    only_iou_pred_vs_gt: Optional[bool] = None,
    save_pred_geom_hists: Optional[bool] = None,
    save_pred_doa_hists: Optional[bool] = None,
    save_diag_edge_overlay: Optional[bool] = None,
):
    """
    分析畴区几何 + 多边形拟合评估 + 面积/横纵尺寸统计
    【修改版】：自动剔除边界附近的畴区，不纳入统计和 CSV。
    """
    os.makedirs(save_dir, exist_ok=True)

    # ---- Feature toggles (env vars override defaults; kwargs override env) ----
    # Set env vars in shell if you want fast runs, e.g.:
    #   set BL_GEOM_PLOTS=0
    #   set BL_GEOM_GT=0
    #   set BL_GEOM_POLY_METRICS=0
    #   set BL_GEOM_SAVE_IMAGES=0
    #   set BL_GEOM_OVERLAP_MAX_PAIRS=0
    #   set BL_GEOM_MAX_PTS=2000
    timing = _env_flag("BL_GEOM_TIMING", True) if timing is None else bool(timing)
    enable_plots = _env_flag("BL_GEOM_PLOTS", True) if enable_plots is None else bool(enable_plots)
    enable_gt = _env_flag("BL_GEOM_GT", True) if enable_gt is None else bool(enable_gt)
    enable_gt_matching = _env_flag("BL_GEOM_GT_MATCH", True) if enable_gt_matching is None else bool(enable_gt_matching)
    enable_save_images = _env_flag("BL_GEOM_SAVE_IMAGES", True) if enable_save_images is None else bool(enable_save_images)
    enable_polygon_metrics = _env_flag("BL_GEOM_POLY_METRICS", True) if enable_polygon_metrics is None else bool(enable_polygon_metrics)
    progress_every = _env_int("BL_GEOM_PROGRESS_EVERY", 500) if progress_every is None else int(progress_every)
    max_pts_per_instance = _env_int("BL_GEOM_MAX_PTS", 0)
    if max_pts_per_instance < 0:
        max_pts_per_instance = 0

    # Parallelism knobs (keep hex fitting logic; only changes execution strategy)
    parallel_hex = _env_flag("BL_GEOM_PARALLEL_HEX", True)
    parallel_backend = os.getenv("BL_GEOM_PARALLEL_BACKEND", "thread").strip().lower()
    geom_workers = _env_int("BL_GEOM_WORKERS", 0)
    if geom_workers <= 0:
        geom_workers = min(32, (os.cpu_count() or 4))
    if geom_workers < 1:
        geom_workers = 1
    overlap_max_pairs = _env_int("BL_GEOM_OVERLAP_MAX_PAIRS", 500_000)
    if overlap_max_pairs < 0:
        overlap_max_pairs = 0

    # Matching knobs
    # - BL_GEOM_MATCH_MAX_DIST: centroid distance threshold (px) for keeping GT↔Pred matches
    # - BL_GEOM_STRICT_MATCH_PLOTS: when GT matching is enabled, hist/R^2 plots will use ONLY
    #   matched+filtered pairs; if no pairs exist, plots are skipped (prevents silent fallback).
    match_max_dist = _env_float("BL_GEOM_MATCH_MAX_DIST", 200.0)
    strict_match_plots = _env_flag("BL_GEOM_STRICT_MATCH_PLOTS", True)

    # Scatter title metric switch (for GT vs Pred scatter plots)
    # Values: 'mae' | 'r2' | 'both'
    scatter_metric = _env_str("BL_GEOM_SCATTER_METRIC", "mae").strip().lower()
    if scatter_metric not in ("mae", "r2", "both"):
        scatter_metric = "mae"

    # Unified plot font size for ALL hist/scatter plots produced in this run.
    # This is intentionally a single knob so titles/labels/ticks/legends scale together.
    # Example:
    #   set BL_GEOM_PLOT_FONT_SIZE=16
    plot_font_size = _env_int("BL_GEOM_PLOT_FONT_SIZE", 15)
    if plot_font_size <= 0:
        plot_font_size = 12
    tick_size = max(1, int(round(plot_font_size * 0.9)))
    legend_size = max(1, int(round(plot_font_size * 0.9)))
    try:
        matplotlib.rcParams.update({
            "font.size": plot_font_size,
            "axes.titlesize": plot_font_size,
            "axes.labelsize": plot_font_size,
            "xtick.labelsize": tick_size,
            "ytick.labelsize": tick_size,
            "legend.fontsize": legend_size,
        })
    except Exception:
        pass

    # Special fast-path: only generate IoU visualization for "原始预测 vs GT" (union)
    # Enable via:
    #   set BL_ONLY_PRED_VS_GT_IOU=1
    only_iou_pred_vs_gt = _env_flag("BL_ONLY_PRED_VS_GT_IOU", False) if only_iou_pred_vs_gt is None else bool(only_iou_pred_vs_gt)

    # Optional: also save predicted width/height/area/orientation histograms (fast, no hull/hex)
    # Enable via:
    #   set BL_PRED_GEOM_HISTS=1
    save_pred_geom_hists = _env_flag("BL_PRED_GEOM_HISTS", False) if save_pred_geom_hists is None else bool(save_pred_geom_hists)

    # Optional: save predicted Diagonal-length / Orientation / Area histograms
    # Enable via:
    #   set BL_PRED_DOA_HISTS=1
    save_pred_doa_hists = _env_flag("BL_PRED_DOA_HISTS", False) if save_pred_doa_hists is None else bool(save_pred_doa_hists)
    save_diag_edge_overlay = _env_flag("BL_SAVE_DIAG_EDGE_OVERLAY", False) if save_diag_edge_overlay is None else bool(save_diag_edge_overlay)
    pred_hist_bins = _env_int("BL_PRED_HIST_BINS", 30)
    pred_hist_max_pts = _env_int("BL_PRED_HIST_MAX_PTS", 2000)

    @contextmanager
    def _timer(label: str):
        if not timing:
            yield
            return
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt = time.perf_counter() - t0
            print(f"⏱️ {label}: {dt:.2f}s")
    # ---- Extra return values (counts + coverage) ----
    # pred_count: number of predicted instances kept for geometry analysis (after border filtering)
    # gt_count: number of GT polygons in the json (segmentation length >= 3)
    # coverage: union pixel coverage ratio over the whole image
    pred_count = 0
    gt_count = 0
    pred_coverage = 0.0
    gt_coverage = 0.0

    # ---- Early exit path: only one output image required ----
    if only_iou_pred_vs_gt:
        print("ℹ️ BL_ONLY_PRED_VS_GT_IOU=1: only IoU visualization is generated; GT-vs-Pred histogram/scatter plots are skipped in this mode")
        if gt_json_path is None or (not os.path.exists(gt_json_path)):
            print("⚠️ BL_ONLY_PRED_VS_GT_IOU=1 but gt_json_path missing; skip")
            return [], [], [], [], len(global_instances), 0, None, None

        image_size = (overlayed.height, overlayed.width)
        # Use contour extraction (same as existing "原始预测 vs GT" behavior)
        with _timer('Compute ex_c() for pred polygons'):
            ex_coor = ex_c(global_instances)
        with _timer('Compute_metrics: 原始预测 vs GT (union)'):
            m = Compute_metrics(
                '原始预测 vs GT',
                orig_image,
                gt_json_path,
                ex_coor,
                image_size,
                save_dir=save_dir,
                mode='union',
                save_bar=False,
                save_metrics=False,
                save_visualization=True,
                verbose=False,
            )

        if save_pred_geom_hists:
            # scaling (same convention as below)
            try:
                _sr = float(scale_ratio) if scale_ratio is not None else None
                if _sr is not None and _sr > 0:
                    length_scale = _sr
                    length_unit = (scale_unit or '').strip() or 'unit'
                else:
                    length_scale = 1.0
                    length_unit = 'px'
            except Exception:
                length_scale = 1.0
                length_unit = 'px'
            area_scale = float(length_scale) * float(length_scale)
            area_unit = f"{length_unit}^2" if length_unit != 'px' else 'px^2'

            with _timer('Pred geom histograms (fast stats)'):
                w_px, h_px, a_px2, ori = _compute_pred_basic_stats(
                    global_instances,
                    max_pts_for_orientation=pred_hist_max_pts,
                    progress_every=progress_every,
                )
                if len(w_px) > 0:
                    plot_hist(np.asarray(w_px) * length_scale, bins=pred_hist_bins, color='C0',
                              title='Pred Width Distribution', xlabel=f'Width ({length_unit})', ylabel='Count',
                              save_path=os.path.join(save_dir, 'pred_width_hist.png'))
                if len(h_px) > 0:
                    plot_hist(np.asarray(h_px) * length_scale, bins=pred_hist_bins, color='C1',
                              title='Pred Height Distribution', xlabel=f'Height ({length_unit})', ylabel='Count',
                              save_path=os.path.join(save_dir, 'pred_height_hist.png'))
                if len(a_px2) > 0:
                    plot_hist(np.asarray(a_px2) * area_scale, bins=pred_hist_bins, color='C2',
                              title='Pred Area Distribution', xlabel=f'Area ({area_unit})', ylabel='Count',
                              save_path=os.path.join(save_dir, 'pred_area_hist.png'))
                if len(ori) > 0:
                    plot_hist(np.asarray(ori), bins=pred_hist_bins, color='C3',
                              title='Pred Orientation Distribution (mod 60°)', xlabel='Angle (deg)', ylabel='Count',
                              save_path=os.path.join(save_dir, 'pred_orientation_hist.png'), xlim=(0, 60))

        if save_pred_doa_hists:
            # Use the original hexagon fitting logic for BOTH diagonal length and orientation.
            # This is heavier than PCA/bbox proxies, but matches the hex-fit pipeline.
            try:
                _sr = float(scale_ratio) if scale_ratio is not None else None
                if _sr is not None and _sr > 0:
                    length_scale = _sr
                    length_unit = (scale_unit or '').strip() or 'unit'
                else:
                    length_scale = 1.0
                    length_unit = 'px'
            except Exception:
                length_scale = 1.0
                length_unit = 'px'
            area_scale = float(length_scale) * float(length_scale)
            area_unit = f"{length_unit}^2" if length_unit != 'px' else 'px^2'

            W, H = overlayed.size
            BOUNDARY_MARGIN = _env_int("BL_GEOM_BOUNDARY_MARGIN", 5)

            with _timer('Pred DOA histograms (hex-fit pipeline)'):
                # Build tasks (border filter + optional subsample)
                pred_tasks = []
                for inst_i, inst in enumerate(global_instances):
                    if progress_every > 0 and inst_i > 0 and (inst_i % progress_every == 0):
                        print(f"... preparing DOA hex-fit {inst_i}/{len(global_instances)}")
                    c = inst.get('coords', [])
                    if isinstance(c, np.ndarray):
                        pts = c
                    else:
                        pts = np.asarray(list(c))
                    if pts is None or len(pts) < 3:
                        continue
                    if max_pts_per_instance and len(pts) > max_pts_per_instance:
                        step = max(1, int(len(pts) // max_pts_per_instance))
                        pts = pts[::step]

                    ys = pts[:, 0]
                    xs = pts[:, 1]
                    ymin, ymax = ys.min(), ys.max()
                    xmin, xmax = xs.min(), xs.max()
                    if (xmin < BOUNDARY_MARGIN or ymin < BOUNDARY_MARGIN or
                        xmax > (W - BOUNDARY_MARGIN) or ymax > (H - BOUNDARY_MARGIN)):
                        continue
                    pred_tasks.append((inst_i, pts))

                # Fit in parallel (same knobs as full path)
                pred_hexagons = []
                pred_areas = []
                pred_results = []

                if parallel_hex and geom_workers > 1 and len(pred_tasks) > 1:
                    Executor = ProcessPoolExecutor if parallel_backend in ("process", "proc", "mp", "multiprocess") else ThreadPoolExecutor
                    try:
                        with Executor(max_workers=geom_workers) as ex:
                            fut_to_idx = {ex.submit(_hull_hex_fit_from_pts, pts): inst_i for (inst_i, pts) in pred_tasks}
                            done = 0
                            for fut in as_completed(fut_to_idx):
                                done += 1
                                if progress_every > 0 and done > 0 and (done % progress_every == 0):
                                    print(f"... DOA hex-fit (parallel) {done}/{len(pred_tasks)}")
                                inst_i = fut_to_idx[fut]
                                hull_pts, hexagon, area, _centroid = fut.result()
                                if hexagon is None:
                                    continue
                                pred_results.append((inst_i, hexagon, float(area)))
                    except Exception:
                        pred_results = []

                if not pred_results:
                    for inst_i, pts in pred_tasks:
                        _hull_pts, hexagon, area, _centroid = _hull_hex_fit_from_pts(pts)
                        if hexagon is None:
                            continue
                        pred_results.append((inst_i, hexagon, float(area)))

                pred_results.sort(key=lambda x: x[0])
                for _inst_i, hexagon, area in pred_results:
                    pred_hexagons.append(hexagon)
                    pred_areas.append(area)

                if len(pred_hexagons) == 0:
                    print("⚠️ DOA hex-fit produced 0 valid hexagons; skip DOA histograms")
                else:
                    # Compute diag length / orientation using the original metric funcs; disable their own saving.
                    diag_vals = lenght_diag_reg(
                        pred_hexagons,
                        overlayed,
                        save_dir,
                        scale_ratio=length_scale,
                        unit=length_unit,
                        save_images=False,
                        save_hist=False,
                    )
                    edge_angles, *_ = angle_edge_reg(
                        pred_hexagons,
                        overlayed,
                        save_dir,
                        save_images=False,
                        save_hist=False,
                    )

                    if diag_vals is not None and len(diag_vals) > 0:
                        plot_hist(np.asarray(diag_vals, dtype=float), bins=pred_hist_bins, color='C0',
                                  title='Pred Diagonal Length Distribution',
                                  xlabel=f'Diagonal length ({length_unit})', ylabel='Count',
                                  save_path=os.path.join(save_dir, 'pred_diaglen_hist.png'))

                    if pred_areas is not None and len(pred_areas) > 0:
                        plot_hist(np.asarray(pred_areas, dtype=float) * area_scale, bins=pred_hist_bins, color='C2',
                                  title='Pred Area Distribution', xlabel=f'Area ({area_unit})', ylabel='Count',
                                  save_path=os.path.join(save_dir, 'pred_area_hist.png'))

                    if edge_angles is not None and len(edge_angles) > 0:
                        ori_mod = np.array([angle_mod60(float(a)) for a in np.asarray(edge_angles, dtype=float).ravel() if np.isfinite(a)], dtype=float)
                        if len(ori_mod) > 0:
                            plot_hist(ori_mod, bins=pred_hist_bins, color='C3',
                                      title='Pred Orientation Distribution (mod 60°)', xlabel='Angle (deg)', ylabel='Count',
                                      save_path=os.path.join(save_dir, 'pred_orientation_hist.png'), xlim=(0, 60))

        # Return lists compatible with callers that index [1]
        iou = float(m.get('IoU', float('nan'))) if isinstance(m, dict) else float('nan')
        p = float(m.get('Precision', float('nan'))) if isinstance(m, dict) else float('nan')
        r = float(m.get('Recall', float('nan'))) if isinstance(m, dict) else float('nan')
        f1 = float(m.get('F1-score', float('nan'))) if isinstance(m, dict) else float('nan')
        return [float('nan'), iou], [float('nan'), p], [float('nan'), r], [float('nan'), f1], len(global_instances), None, None, None
    def match_by_overlap(global_instances, gt_polygons_pts, overlayed_size):
        # global_instances: list of dict with 'coords' set of (y,x)
        # gt_polygons_pts: list of ndarray of (row=y, col=x) polygon vertices
        W, H = overlayed_size
        # build predicted sets
        pred_sets = [inst['coords'] for inst in global_instances]

        # build gt raster sets
        gt_sets = []
        from PIL import Image, ImageDraw
        for poly in gt_polygons_pts:
            mask_img = Image.new('L', (W, H), 0)
            if len(poly) < 3:
                gt_sets.append(set())
                continue
            # polygon expects (x,y) tuples
            poly_xy = [(int(p[1]), int(p[0])) for p in poly]
            ImageDraw.Draw(mask_img).polygon(poly_xy, outline=1, fill=1)
            mask_np = np.array(mask_img)
            ys, xs = np.where(mask_np > 0)
            gt_sets.append(set(zip(ys, xs)))

        if len(pred_sets) == 0 or len(gt_sets) == 0:
            return [], []

        # cost matrix: -overlap (we want to maximize overlap)
        cost = np.zeros((len(gt_sets), len(pred_sets)), dtype=float)
        for i, gset in enumerate(gt_sets):
            for j, pset in enumerate(pred_sets):
                inter = len(gset & pset)
                cost[i, j] = -inter

        try:
            row_ind, col_ind = linear_sum_assignment(cost)
        except Exception:
            # fallback greedy: for each GT pick best pred
            row_ind = []
            col_ind = []
            for i, gset in enumerate(gt_sets):
                best_j = -1
                best_inter = 0
                for j, pset in enumerate(pred_sets):
                    inter = len(gset & pset)
                    if inter > best_inter:
                        best_inter = inter
                        best_j = j
                if best_j >= 0:
                    row_ind.append(i)
                    col_ind.append(best_j)

        return row_ind, col_ind

    def match_by_centroid_nn(pred_centroids, gt_centroids, max_dist):
        """Fallback matching by nearest centroid within `max_dist`.

        Returns (row_ind, col_ind) where each GT index is matched to at most one Pred index.
        This is used only when overlap-based matching yields no usable pairs.
        """
        if pred_centroids is None or gt_centroids is None:
            return [], []
        pred_centroids = np.asarray(pred_centroids, dtype=float)
        gt_centroids = np.asarray(gt_centroids, dtype=float)
        if pred_centroids.ndim != 2 or gt_centroids.ndim != 2 or pred_centroids.shape[1] != 2 or gt_centroids.shape[1] != 2:
            return [], []
        if pred_centroids.shape[0] == 0 or gt_centroids.shape[0] == 0:
            return [], []

        max_dist = float(max_dist)
        if not np.isfinite(max_dist) or max_dist <= 0:
            return [], []

        # Build candidate pairs (gt_i, pred_j, dist)
        candidates = []
        for gi in range(gt_centroids.shape[0]):
            d = np.linalg.norm(pred_centroids - gt_centroids[gi], axis=1)
            if d.size == 0:
                continue
            pj = int(np.argmin(d))
            dist = float(d[pj])
            if np.isfinite(dist) and dist <= max_dist:
                candidates.append((gi, pj, dist))

        if not candidates:
            return [], []

        # Enforce one-to-one: pick smallest distances first
        candidates.sort(key=lambda x: x[2])
        used_pred = set()
        row_ind = []
        col_ind = []
        for gi, pj, _dist in candidates:
            if pj in used_pred:
                continue
            used_pred.add(pj)
            row_ind.append(int(gi))
            col_ind.append(int(pj))

        return row_ind, col_ind
    # 获取图像尺寸用于边界判断
    W, H = overlayed.size
    BOUNDARY_MARGIN = _env_int("BL_GEOM_BOUNDARY_MARGIN", 5)  # 距离边缘多少像素内视为边界畴区
    total_pixels = int(W) * int(H)

    # ---- Optional physical scaling ----
    # If provided, length values are converted from pixels to `scale_unit` using:
    #   length_physical = length_px * scale_ratio
    #   area_physical   = area_px2  * scale_ratio^2
    try:
        _sr = float(scale_ratio) if scale_ratio is not None else None
        if _sr is not None and _sr > 0:
            length_scale = _sr
            length_unit = (scale_unit or '').strip() or 'unit'
        else:
            length_scale = 1.0
            length_unit = 'px'
    except Exception:
        length_scale = 1.0
        length_unit = 'px'
    area_scale = float(length_scale) * float(length_scale)
    area_unit = f"{length_unit}^2" if length_unit != 'px' else 'px^2'

    o_image = Image.open(orig_image)
    if enable_save_images:
        o_image.save(os.path.join(save_dir, "original_image.png"))
    else:
        print("ℹ️ enable_save_images=False: skip saving original_image.png")

    img_hull = overlayed.copy()
    draw_hull = ImageDraw.Draw(img_hull)
    img_hex = overlayed.copy()
    draw_hex = ImageDraw.Draw(img_hex)
    img_ = overlayed.copy()
    draw_img_ = ImageDraw.Draw(img_)

    # ========== A. 可能的实例数量裁剪（按面积取前 N） ==========
    if max_instances is not None and len(global_instances) > max_instances:
        print(f"⚠️ 实例过多({len(global_instances)})，仅保留面积最大的 {max_instances} 个进行快速分析")
        areas_idx = []
        for idx, inst in enumerate(global_instances):
            pts = np.array(list(inst['coords']))
            hull = safe_convex_hull(pts)
            if hull is None:
                areas_idx.append((idx, 0.0))
            else:
                areas_idx.append((idx, float(hull.volume)))
        areas_idx.sort(key=lambda x: x[1], reverse=True)
        keep_idx = set([i for i, a in areas_idx[:max_instances]])
        global_instances = [inst for i, inst in enumerate(global_instances) if i in keep_idx]
        print(f"⚠️ 剩余实例数: {len(global_instances)}")

    # ---- Pred counts & coverage (DO NOT exclude border instances) ----
    # Count: number of predicted instances (after optional max_instances trimming)
    # Coverage: union of all predicted instance pixels / image pixels
    pred_count = len(global_instances)
    if total_pixels > 0 and pred_count > 0:
        with _timer("Pred coverage (raster union mask)"):
            try:
                pred_mask = np.zeros((H, W), dtype=np.bool_)
                for inst in global_instances:
                    c = inst.get('coords', [])
                    if isinstance(c, np.ndarray):
                        pts = c.astype(np.int64, copy=False)
                    else:
                        pts = np.asarray(list(c), dtype=np.int64)
                    if pts.size == 0:
                        continue
                    ys = pts[:, 0]
                    xs = pts[:, 1]
                    inb = (ys >= 0) & (ys < H) & (xs >= 0) & (xs < W)
                    ys = ys[inb]
                    xs = xs[inb]
                    if ys.size > 0:
                        pred_mask[ys, xs] = True
                pred_coverage = float(np.count_nonzero(pred_mask)) / float(total_pixels)
            except Exception:
                pred_coverage = 0.0

    # ========== B. 筛选有效实例（去除边界）并计算凸包 ==========
    # 创建新的列表存储经过筛选后的“内部”实例数据
    valid_instances = []

    widths, heights, areas = [], [], []
    predicted_centroids = []
    all_hulls = []
    all_hexagons = [] # 六边形数据也只存有效的

    print(f"正在分析几何特征 (总预测数: {len(global_instances)})...")

    skipped_border_count = 0

    with _timer("Pred instances loop (hull + hex fit + drawing)"):
        # 1) Pre-filter & prepare tasks (border filtering + optional subsampling)
        pred_tasks = []
        for inst_i, inst in enumerate(global_instances):
            if progress_every > 0 and inst_i > 0 and (inst_i % progress_every == 0):
                print(f"... preparing instances {inst_i}/{len(global_instances)}")

            c = inst.get('coords', [])
            if isinstance(c, np.ndarray):
                pts = c
            else:
                pts = np.asarray(list(c))

            if pts is None or len(pts) < 3:
                continue

            # Optional speed knob: cap points per instance (convex hull on all pixels is extremely slow).
            if max_pts_per_instance and len(pts) > max_pts_per_instance:
                step = max(1, int(len(pts) // max_pts_per_instance))
                pts = pts[::step]

            # Border check (pts is (y,x))
            ys = pts[:, 0]
            xs = pts[:, 1]
            ymin, ymax = ys.min(), ys.max()
            xmin, xmax = xs.min(), xs.max()

            if (xmin < BOUNDARY_MARGIN or ymin < BOUNDARY_MARGIN or
                xmax > (W - BOUNDARY_MARGIN) or ymax > (H - BOUNDARY_MARGIN)):
                skipped_border_count += 1
                continue

            pred_tasks.append((inst_i, inst, pts, xmin, xmax, ymin, ymax))

        # 2) Hull + hex fit (optionally parallel)
        pred_results = []
        if parallel_hex and geom_workers > 1 and len(pred_tasks) > 1:
            Executor = ProcessPoolExecutor if parallel_backend in ("process", "proc", "mp", "multiprocess") else ThreadPoolExecutor
            try:
                with Executor(max_workers=geom_workers) as ex:
                    fut_to_meta = {
                        ex.submit(_hull_hex_fit_from_pts, pts): (inst_i, inst, xmin, xmax, ymin, ymax)
                        for (inst_i, inst, pts, xmin, xmax, ymin, ymax) in pred_tasks
                    }
                    done = 0
                    for fut in as_completed(fut_to_meta):
                        done += 1
                        if progress_every > 0 and done > 0 and (done % progress_every == 0):
                            print(f"... fitting (parallel) {done}/{len(pred_tasks)}")
                        inst_i, inst, xmin, xmax, ymin, ymax = fut_to_meta[fut]
                        hull_pts, hexagon, area, centroid = fut.result()
                        if hull_pts is None:
                            continue
                        pred_results.append((inst_i, inst, xmin, xmax, ymin, ymax, hull_pts, hexagon, area, centroid))
            except Exception:
                # Fallback to single-thread if parallel executor fails (Windows spawn edge cases)
                pred_results = []

        if not pred_results:
            # Serial fallback / default when tasks are small
            for inst_i, inst, pts, xmin, xmax, ymin, ymax in pred_tasks:
                hull_pts, hexagon, area, centroid = _hull_hex_fit_from_pts(pts)
                if hull_pts is None:
                    continue
                pred_results.append((inst_i, inst, xmin, xmax, ymin, ymax, hull_pts, hexagon, area, centroid))

        # 3) Materialize results in a stable order (important for matching indices)
        pred_results.sort(key=lambda x: x[0])
        for _inst_i, inst, xmin, xmax, ymin, ymax, hull_pts, hexagon, area, centroid in pred_results:
            # Keep arrays aligned with `valid_instances` for downstream overlap matching
            valid_instances.append(inst)
            areas.append(area)
            widths.append(xmax - xmin)
            heights.append(ymax - ymin)
            predicted_centroids.append(centroid)
            all_hulls.append(1)

            if enable_save_images:
                #polygon_hull_xy = [(p[1], p[0]) for p in hull_pts]
                #draw_hull.polygon(polygon_hull_xy, outline="orange", width=3)
                for px, py in hull_pts:
                    draw_hull.ellipse((py-8, px-8, py+8, px+8), fill="red")

            if hexagon is not None:
                all_hexagons.append(hexagon)
                if enable_save_images:
                    polygon_hex_xy = [(p[1], p[0]) for p in hexagon]
                    draw_hex.polygon(polygon_hex_xy, outline="orange", width=8)
                    for px, py in hull_pts:
                        draw_hex.ellipse((py-8, px-8, py+8, px+8), fill="red")
            else:
                all_hexagons.append(None)

    print(f"🚫 已剔除边界接触实例: {skipped_border_count} 个")
    print(f"✅ 有效内部实例: {len(all_hulls)} 个")

    if enable_save_images:
        with _timer("Save pred hull/hex images"):
            img_hull.save(os.path.join(save_dir, "hull_outline.png"))
            img_.save(os.path.join(save_dir, "hexagon_outline_debug.png"))
            img_hex.save(os.path.join(save_dir, "hexagon_outline.png"))
    else:
        print("ℹ️ enable_save_images=False: skip saving hull/hex PNGs")

    with _timer("Pred geometry metrics (lenght_* / angle_*)"):
        # ========= 1. 最长对边间距 =========
        side_dists=lenght_side_reg(all_hexagons, overlayed, save_dir, scale_ratio=length_scale, unit=length_unit,
                                   save_images=enable_save_images, save_hist=enable_plots)
        # ========= 2. 最长对角线长度 =========
        diag_lens=lenght_diag_reg(all_hexagons, overlayed, save_dir, scale_ratio=length_scale, unit=length_unit,
                                  save_images=enable_save_images, save_hist=enable_plots)
        # ========= 3. 最长对角线方向角 =========
        diag_angles=angle_diag_reg(all_hexagons, overlayed, save_dir, save_images=enable_save_images, save_hist=enable_plots)
        # ========= 4. 单边方向 =========
        edge_angles, *_ = angle_edge_reg(all_hexagons, overlayed, save_dir, save_images=enable_save_images, save_hist=enable_plots)

        # ========= 4b. 合并可视化：DiagonalLength + EdgeAngle（同一张图） =========
        if enable_save_images or save_diag_edge_overlay:
            try:
                diaglen_edgeangle_overlay(all_hexagons, overlayed, save_dir)
            except Exception:
                pass

    # Optional: predicted DOA distributions (real values from hex-fit path)
    if save_pred_doa_hists:
        with _timer("Pred DOA histograms"):
            try:
                diag_vals = np.asarray(diag_lens, dtype=float).ravel() if 'diag_lens' in locals() else np.array([])
                if diag_vals.size > 0:
                    plot_hist(diag_vals, bins=pred_hist_bins, color='C0',
                              title='Pred Diagonal Length Distribution',
                              xlabel=f'Diagonal length ({length_unit})', ylabel='Count',
                              save_path=os.path.join(save_dir, 'pred_diaglen_hist.png'))

                area_vals = (np.asarray(areas, dtype=float).ravel() * float(area_scale)) if len(areas) > 0 else np.array([])
                if area_vals.size > 0:
                    plot_hist(area_vals, bins=pred_hist_bins, color='C2',
                              title='Pred Area Distribution', xlabel=f'Area ({area_unit})', ylabel='Count',
                              save_path=os.path.join(save_dir, 'pred_area_hist.png'))

                ori_src = np.asarray(edge_angles, dtype=float).ravel() if 'edge_angles' in locals() else np.array([])
                if ori_src.size > 0:
                    ori_mod = np.array([angle_mod60(float(a)) for a in ori_src if np.isfinite(a)], dtype=float)
                    if ori_mod.size > 0:
                        plot_hist(ori_mod, bins=pred_hist_bins, color='C3',
                                  title='Pred Orientation Distribution (mod 60°)', xlabel='Angle (deg)', ylabel='Count',
                                  save_path=os.path.join(save_dir, 'pred_orientation_hist.png'), xlim=(0, 60))
            except Exception:
                pass
    # ========= 5. 横向长度 / 纵向长度 / 面积 =========
    # ========= 汇总到 CSV =========
    csv_path = os.path.join(save_dir, "geometry_stats.csv")
    with _timer("Write geometry_stats.csv"):
        with open(csv_path, "w", newline='', encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Index",
                             f"SideDistance({length_unit})",
                             f"DiagonalLength({length_unit})",
                             "DiagonalAngle(deg)",
                             "OrientationAngle(deg)",
                             f"Width({length_unit})",
                             f"Height({length_unit})",
                             f"Area({area_unit})"])
            for idx in range(len(all_hulls)):
                row = [
                    idx,
                    side_dists[idx] if idx < len(side_dists) else 0,
                    diag_lens[idx] if idx < len(diag_lens) else 0,
                    diag_angles[idx] if idx < len(diag_angles) else 0,
                    edge_angles[idx] if idx < len(edge_angles) else 0,
                    (widths[idx] * length_scale) if idx < len(widths) else 0,
                    (heights[idx] * length_scale) if idx < len(heights) else 0,
                    (areas[idx] * area_scale) if idx < len(areas) else 0
                ]
                writer.writerow(row)

    print(f"✅ 几何分析完成，结果已保存到 {csv_path}")

    # ========= GT 取向尺寸识别 =========
    if enable_gt and gt_json_path is not None and os.path.exists(gt_json_path):
        print("\n========= 开始 GT 取向尺寸识别 =========")
        gt_analysis_dir = os.path.join(save_dir, "gt_analysis")
        os.makedirs(gt_analysis_dir, exist_ok=True)

        with _timer("GT: load json"):
            with open(gt_json_path, 'r', encoding='utf-8') as f:
                gt_data = json.load(f)

        # 从 GT JSON 提取多边形坐标
        gt_polygons = []
        for obj in gt_data.get('objects', []):
            seg = obj.get('segmentation', [])
            if seg and len(seg) >= 3:
                # GT segmentation is [x, y] pairs; convert to (row, col) = (y, x)
                pts = np.array([[p[1], p[0]] for p in seg])
                gt_polygons.append(pts)

        # Filter GT polygons touching the image border (same rule as pred)
        gt_polygons_valid = []
        skipped_gt_border_count = 0
        for poly in gt_polygons:
            try:
                pts = np.asarray(poly)
                if pts is None or pts.ndim != 2 or pts.shape[0] < 3:
                    continue
                ys = pts[:, 0]
                xs = pts[:, 1]
                ymin, ymax = ys.min(), ys.max()
                xmin, xmax = xs.min(), xs.max()
                if (xmin < BOUNDARY_MARGIN or ymin < BOUNDARY_MARGIN or
                    xmax > (W - BOUNDARY_MARGIN) or ymax > (H - BOUNDARY_MARGIN)):
                    skipped_gt_border_count += 1
                    continue
                gt_polygons_valid.append(pts)
            except Exception:
                continue
        if skipped_gt_border_count > 0:
            print(f"🚫 GT 边界接触实例已剔除(用于几何/对比图): {skipped_gt_border_count} 个")

        # ---- GT counts & coverage (union polygon pixels / image pixels) ----
        gt_count = len(gt_polygons)
        if total_pixels > 0 and gt_count > 0:
            with _timer("GT: coverage (raster union mask)"):
                try:
                    gt_mask_img = Image.new('L', (W, H), 0)
                    gt_draw = ImageDraw.Draw(gt_mask_img)
                    for poly in gt_polygons:
                        if poly is None or len(poly) < 3:
                            continue
                        poly_xy = [(int(p[1]), int(p[0])) for p in poly]
                        gt_draw.polygon(poly_xy, outline=1, fill=1)
                    gt_mask_np = np.array(gt_mask_img)
                    gt_coverage = float(np.count_nonzero(gt_mask_np > 0)) / float(total_pixels)
                except Exception:
                    gt_coverage = 0.0

        if gt_polygons_valid:
            img_gt_hex = overlayed.copy()
            draw_gt_hex = ImageDraw.Draw(img_gt_hex)
            img_gt_ = overlayed.copy()
            draw_gt_ = ImageDraw.Draw(img_gt_)

            # 对 GT 多边形进行六边形拟合
            gt_hexagons = []
            gt_widths, gt_heights, gt_areas = [], [], []
            gt_centroids = []

            with _timer("GT: hex fit loop"):
                gt_tasks = []
                for poly_idx, poly_pts in enumerate(gt_polygons_valid):
                    if poly_pts is None or len(poly_pts) < 3:
                        continue
                    pts = np.asarray(poly_pts)
                    if max_pts_per_instance and len(pts) > max_pts_per_instance:
                        step = max(1, int(len(pts) // max_pts_per_instance))
                        pts = pts[::step]
                    gt_tasks.append((poly_idx, pts))

                gt_results = []
                if parallel_hex and geom_workers > 1 and len(gt_tasks) > 1:
                    Executor = ProcessPoolExecutor if parallel_backend in ("process", "proc", "mp", "multiprocess") else ThreadPoolExecutor
                    try:
                        with Executor(max_workers=geom_workers) as ex:
                            fut_to_idx = {ex.submit(_hull_hex_fit_from_pts, pts): poly_idx for (poly_idx, pts) in gt_tasks}
                            done = 0
                            for fut in as_completed(fut_to_idx):
                                done += 1
                                if progress_every > 0 and done > 0 and (done % progress_every == 0):
                                    print(f"... GT fitting (parallel) {done}/{len(gt_tasks)}")
                                poly_idx = fut_to_idx[fut]
                                hull_pts, hexagon, area, centroid = fut.result()
                                if hull_pts is None or hexagon is None:
                                    continue
                                gt_results.append((poly_idx, hull_pts, hexagon, area, centroid))
                    except Exception:
                        gt_results = []

                if not gt_results:
                    for poly_idx, pts in gt_tasks:
                        hull_pts, hexagon, area, centroid = _hull_hex_fit_from_pts(pts)
                        if hull_pts is None or hexagon is None:
                            continue
                        gt_results.append((poly_idx, hull_pts, hexagon, area, centroid))

                gt_results.sort(key=lambda x: x[0])
                for _poly_idx, hull_pts, hexagon, area, centroid in gt_results:
                    # Keep GT arrays aligned with gt_hexagons indices (important for matching/plots)
                    xs, ys = hull_pts[:, 1], hull_pts[:, 0]
                    gt_widths.append(xs.max() - xs.min())
                    gt_heights.append(ys.max() - ys.min())
                    gt_areas.append(area)
                    gt_centroids.append(centroid)
                    gt_hexagons.append(hexagon)

                    if enable_save_images:
                        polygon_hex_xy = [(p[1], p[0]) for p in hexagon]
                        draw_gt_hex.polygon(polygon_hex_xy, outline="blue", width=3)

            if enable_save_images:
                with _timer("Save GT hex images"):
                    img_gt_.save(os.path.join(save_dir, "gt_hexagon_outline_debug.png"))
                    img_gt_hex.save(os.path.join(save_dir, "gt_hexagon_outline.png"))

            # 对 GT 进行相同的几何分析
            if gt_hexagons:
                with _timer("GT geometry metrics (lenght_* / angle_*)"):
                    gt_side_dists = lenght_side_reg(gt_hexagons, overlayed, gt_analysis_dir, scale_ratio=length_scale, unit=length_unit,
                                                    save_images=enable_save_images, save_hist=enable_plots)
                    gt_diag_lens = lenght_diag_reg(gt_hexagons, overlayed, gt_analysis_dir, scale_ratio=length_scale, unit=length_unit,
                                                   save_images=enable_save_images, save_hist=enable_plots)
                    gt_diag_angles = angle_diag_reg(gt_hexagons, overlayed, gt_analysis_dir, save_images=enable_save_images, save_hist=enable_plots)
                    gt_edge_angles, *_ = angle_edge_reg(gt_hexagons, overlayed, gt_analysis_dir, save_images=enable_save_images, save_hist=enable_plots)

                # 合并可视化：DiagonalLength + EdgeAngle（同一张图）
                if enable_save_images:
                    try:
                        diaglen_edgeangle_overlay(gt_hexagons, overlayed, gt_analysis_dir)
                    except Exception:
                        pass

                # 保存 GT 几何统计到 CSV
                gt_csv_path = os.path.join(save_dir, "gt_geometry_stats.csv")
                with _timer("Write gt_geometry_stats.csv"):
                    with open(gt_csv_path, "w", newline='', encoding="utf-8") as f:
                        writer = csv.writer(f)
                        writer.writerow(["Index",
                                         f"SideDistance({length_unit})",
                                         f"DiagonalLength({length_unit})",
                                         "DiagonalAngle(deg)",
                                         "OrientationAngle(deg)",
                                         f"Width({length_unit})",
                                         f"Height({length_unit})",
                                         f"Area({area_unit})"])
                        for idx in range(len(gt_hexagons)):
                            row = [
                                idx,
                                gt_side_dists[idx] if idx < len(gt_side_dists) else 0,
                                gt_diag_lens[idx] if idx < len(gt_diag_lens) else 0,
                                gt_diag_angles[idx] if idx < len(gt_diag_angles) else 0,
                                gt_edge_angles[idx] if idx < len(gt_edge_angles) else 0,
                                (gt_widths[idx] * length_scale) if idx < len(gt_widths) else 0,
                                (gt_heights[idx] * length_scale) if idx < len(gt_heights) else 0,
                                (gt_areas[idx] * area_scale) if idx < len(gt_areas) else 0
                            ]
                            writer.writerow(row)

                print(f"✅ GT 几何分析完成，结果已保存到 {gt_csv_path}")
                # ---------- 生成对比图：直方图与 R^2 散点图 ----------
                if not enable_plots:
                    print(
                        "ℹ️ enable_plots=False: skip GT vs Pred histogram/scatter plots "
                        "(set BL_GEOM_PLOTS=1 or pass enable_plots=True)"
                    )
                else:
                    # NOTE: intentionally no try/except wrapper here to avoid indentation issues;
                    # the internal blocks already have local try/except guards.
                    print(
                        "DEBUG: plot toggles: "
                        f"enable_plots={enable_plots}, enable_gt={enable_gt}, enable_gt_matching={enable_gt_matching}, "
                        f"strict_match_plots={strict_match_plots}, MATCH_MAX_DIST={float(match_max_dist)}, "
                        f"BOUNDARY_MARGIN={int(BOUNDARY_MARGIN)}, scatter_metric={scatter_metric}"
                    )
                    print(f"DEBUG: len(predicted_centroids)={len(predicted_centroids)}, len(gt_centroids)={len(gt_centroids)}")
                    print(f"DEBUG: len(widths)={len(widths)}, len(gt_widths)={len(gt_widths)}")
                    print(f"DEBUG: len(side_dists)={len(side_dists) if 'side_dists' in locals() else 0}, len(gt_side_dists)={len(gt_side_dists) if 'gt_side_dists' in locals() else 0}")
                    # Precompute overlap-based matching between GT polygons and valid predicted instances
                    row_ind, col_ind = [], []
                    try:
                        if enable_gt_matching and len(gt_hexagons) > 0 and len(valid_instances) > 0:
                            pairs = int(len(gt_hexagons)) * int(len(valid_instances))
                            if overlap_max_pairs and pairs > overlap_max_pairs:
                                print(f"⚠️ match_by_overlap skipped: pairs={pairs} > BL_GEOM_OVERLAP_MAX_PAIRS={overlap_max_pairs}")
                                row_ind, col_ind = [], []
                            else:
                                # Use original GT polygons (after border filtering) for overlap rasterization.
                                # This avoids using coarse hexagon vertices as GT polygons.
                                gt_polys = gt_polygons_valid
                                with _timer("GT matching: match_by_overlap"):
                                    row_ind, col_ind = match_by_overlap(valid_instances, gt_polys, overlayed.size)
                                print(f"DEBUG: match_by_overlap returned {len(row_ind)} matches")
                    except Exception:
                        row_ind, col_ind = [], []

                    # Build centroid arrays aligned with valid_instances and GT
                    pred_centroids_for_valid = compute_instance_centroids(valid_instances)
                    gt_cent_arr = np.array(gt_centroids) if len(gt_centroids) > 0 else np.zeros((0, 2))

                    # Filter matches by distance threshold and visualize the (filtered) matches
                    MATCH_MAX_DIST = float(match_max_dist)  # pixels
                    filtered_row_ind, filtered_col_ind = filter_matches_by_distance(
                        row_ind, col_ind, pred_centroids_for_valid, gt_cent_arr, MATCH_MAX_DIST
                    )

                    # draw overlay of filtered matches (white lines + markers). If no filtered
                    # matches are present, `draw_match_overlay` will just draw centroids.
                    if enable_gt_matching:
                        with _timer("GT matching: draw_match_overlay"):
                            draw_match_overlay(overlayed, pred_centroids_for_valid, gt_cent_arr, zip(filtered_row_ind, filtered_col_ind), os.path.join(save_dir, 'match_overlay.png'))

                    # Use filtered matches for subsequent pairing logic
                    if len(filtered_row_ind) > 0:
                        row_ind, col_ind = filtered_row_ind, filtered_col_ind
                    # Materialize match lists once and reuse for all histograms/R^2 plots
                    match_row = list(row_ind) if row_ind is not None else []
                    match_col = list(col_ind) if col_ind is not None else []

                    def _compute_r2_mae(paired_pred, paired_gt):
                        paired_pred = np.asarray(paired_pred, dtype=float).ravel()
                        paired_gt = np.asarray(paired_gt, dtype=float).ravel()
                        m = np.isfinite(paired_pred) & np.isfinite(paired_gt)
                        if not np.any(m):
                            return float('nan'), float('nan')
                        paired_pred = paired_pred[m]
                        paired_gt = paired_gt[m]
                        mae = float(np.mean(np.abs(paired_pred - paired_gt)))
                        ss_res = float(np.sum((paired_pred - paired_gt) ** 2))
                        ss_tot = float(np.sum((paired_gt - float(np.mean(paired_gt))) ** 2))
                        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else float('nan')
                        return r2, mae

                    def _format_scatter_metrics(r2: float, mae: float) -> str:
                        parts = []
                        if scatter_metric in ("r2", "both"):
                            parts.append(f"R2={r2:.3f}")
                        if scatter_metric in ("mae", "both"):
                            parts.append(f"MAE={mae:.3f}")
                        return " ".join(parts) if parts else f"MAE={mae:.3f}"

                    # Minimum histogram edge linewidth so bin borders stay visible.
                    hist_edge_lw = _env_float("BL_GEOM_HIST_EDGE_LW_MIN", 1.2)
                    if not (hist_edge_lw > 0):
                        hist_edge_lw = 1.2

                    def _safe_hist_range(a, b):
                        """Return a non-degenerate (lo, hi) range for plt.hist when min==max.

                        Matplotlib/numpy may produce an empty histogram if the inferred range
                        has zero width (e.g., a single sample or all samples identical).
                        """
                        try:
                            va = np.asarray(a, dtype=float).ravel() if a is not None else np.array([], dtype=float)
                            vb = np.asarray(b, dtype=float).ravel() if b is not None else np.array([], dtype=float)
                            v = np.concatenate([va, vb]) if (va.size + vb.size) > 0 else np.array([], dtype=float)
                            if v.size == 0:
                                return None
                            v = v[np.isfinite(v)]
                            if v.size == 0:
                                return None
                            vmin = float(np.min(v))
                            vmax = float(np.max(v))
                            if not (np.isfinite(vmin) and np.isfinite(vmax)):
                                return None
                            if vmin == vmax:
                                delta = max(1e-6, abs(vmin) * 0.01, 1.0)
                                return (vmin - delta, vmax + delta)
                        except Exception:
                            return None
                        return None

                    # When GT matching is enabled, ensure comparison plots use ONLY matched+filtered pairs
                    # (no silent fallback to unfiltered full arrays).
                    if enable_gt_matching and strict_match_plots and not (len(match_row) > 0 and len(match_col) > 0):
                        # Fallback: centroid NN matching (still constrained by MATCH_MAX_DIST)
                        try:
                            fb_r, fb_c = match_by_centroid_nn(pred_centroids_for_valid, gt_cent_arr, MATCH_MAX_DIST)
                        except Exception:
                            fb_r, fb_c = [], []
                        if len(fb_r) > 0 and len(fb_c) > 0:
                            match_row, match_col = list(fb_r), list(fb_c)
                            print(f"ℹ️ overlap+dist matching produced 0 pairs; using centroid-NN fallback pairs={len(match_row)} (MATCH_MAX_DIST={MATCH_MAX_DIST})")
                        else:
                            print(
                                "⚠️ GT matching enabled but 0 pairs after overlap+distance filtering "
                                f"(MATCH_MAX_DIST={MATCH_MAX_DIST}, BOUNDARY_MARGIN={BOUNDARY_MARGIN}); "
                                "skip GT vs Pred histogram/scatter plots. "
                                "Try increasing BL_GEOM_MATCH_MAX_DIST, reducing BL_GEOM_BOUNDARY_MARGIN, "
                                "or set BL_GEOM_STRICT_MATCH_PLOTS=0 to allow full-distribution fallback."
                            )
                            match_row, match_col = [], []

                    if enable_gt_matching and strict_match_plots:
                        print(f"DEBUG: matched pairs used for plots: {len(match_row)}")

                    # 宽度对比（使用匹配对进行分布对比；无匹配时退回到全量）
                    if len(gt_widths) > 0 and len(widths) > 0:
                        pred_w = (np.asarray(widths).astype(float).ravel() * length_scale)
                        gt_w = (np.asarray(gt_widths).astype(float).ravel() * length_scale)

                        # Matched pairs (strict when GT matching is enabled)
                        matched_pred_w = np.array([pred_w[c] for c in match_col if 0 <= c < len(pred_w)], dtype=float) if (len(match_col) > 0) else np.array([], dtype=float)
                        matched_gt_w = np.array([gt_w[r] for r in match_row if 0 <= r < len(gt_w)], dtype=float) if (len(match_row) > 0) else np.array([], dtype=float)

                        if enable_gt_matching and strict_match_plots:
                            if matched_pred_w.size == 0 or matched_gt_w.size == 0:
                                # No valid matched indices; skip width plots
                                pass
                        else:
                            # Legacy behavior: if no matches, compare full distributions
                            if matched_pred_w.size == 0 or matched_gt_w.size == 0:
                                matched_pred_w = pred_w
                                matched_gt_w = gt_w
                        # Save combined matched metrics to a single CSV
                        try:
                            if len(match_row) > 0 and len(match_col) > 0:
                                # Prepare prediction and GT arrays (safe fallbacks)
                                pred_h = np.asarray(heights).astype(float).ravel() if len(heights) > 0 else np.array([])
                                gt_h = np.asarray(gt_heights).astype(float).ravel() if len(gt_heights) > 0 else np.array([])
                                pred_area = np.asarray(areas).astype(float).ravel() if len(areas) > 0 else np.array([])
                                gt_area = np.asarray(gt_areas).astype(float).ravel() if len(gt_areas) > 0 else np.array([])

                                pred_side = np.asarray(side_dists).astype(float).ravel() if 'side_dists' in locals() and len(side_dists) > 0 else np.array([])
                                gt_side = np.asarray(gt_side_dists).astype(float).ravel() if 'gt_side_dists' in locals() and len(gt_side_dists) > 0 else np.array([])

                                pred_diag = np.asarray(diag_lens).astype(float).ravel() if 'diag_lens' in locals() and len(diag_lens) > 0 else np.array([])
                                gt_diag = np.asarray(gt_diag_lens).astype(float).ravel() if 'gt_diag_lens' in locals() and len(gt_diag_lens) > 0 else np.array([])

                                pred_diag_ang = np.asarray(diag_angles).astype(float).ravel() if 'diag_angles' in locals() and len(diag_angles) > 0 else np.array([])
                                gt_diag_ang = np.asarray(gt_diag_angles).astype(float).ravel() if 'gt_diag_angles' in locals() and len(gt_diag_angles) > 0 else np.array([])

                                pred_edge_ang = np.asarray(edge_angles).astype(float).ravel() if 'edge_angles' in locals() and len(edge_angles) > 0 else np.array([])
                                gt_edge_ang = np.asarray(gt_edge_angles).astype(float).ravel() if 'gt_edge_angles' in locals() and len(gt_edge_angles) > 0 else np.array([])

                                combined_csv = os.path.join(save_dir, 'matched_metrics.csv')
                                with open(combined_csv, 'w', newline='', encoding='utf-8') as cf:
                                    cw = csv.writer(cf)
                                    cw.writerow(['gt_index', 'pred_index',
                                                 'gt_Width', 'pred_Width',
                                                 'gt_Height', 'pred_Height',
                                                 'gt_Area', 'pred_Area',
                                                 'gt_SideDistance', 'pred_SideDistance',
                                                 'gt_DiagonalLength', 'pred_DiagonalLength',
                                                 'gt_DiagonalAngle', 'pred_DiagonalAngle',
                                                 'gt_EdgeAngle', 'pred_EdgeAngle'])
                                    for r, c in zip(match_row, match_col):
                                        row_vals = [int(r), int(c)]
                                        # Width
                                        row_vals.append(float(gt_w[r]) if 0 <= r < len(gt_w) else '')
                                        row_vals.append(float(pred_w[c]) if 0 <= c < len(pred_w) else '')
                                        # Height
                                        row_vals.append(float(gt_h[r]) if 0 <= r < len(gt_h) else '')
                                        row_vals.append(float(pred_h[c]) if 0 <= c < len(pred_h) else '')
                                        # Area
                                        row_vals.append(float(gt_area[r]) if 0 <= r < len(gt_area) else '')
                                        row_vals.append(float(pred_area[c]) if 0 <= c < len(pred_area) else '')
                                        # SideDistance
                                        row_vals.append(float(gt_side[r]) if 0 <= r < len(gt_side) else '')
                                        row_vals.append(float(pred_side[c]) if 0 <= c < len(pred_side) else '')
                                        # DiagonalLength
                                        row_vals.append(float(gt_diag[r]) if 0 <= r < len(gt_diag) else '')
                                        row_vals.append(float(pred_diag[c]) if 0 <= c < len(pred_diag) else '')
                                        # DiagonalAngle
                                        row_vals.append(float(gt_diag_ang[r]) if 0 <= r < len(gt_diag_ang) else '')
                                        row_vals.append(float(pred_diag_ang[c]) if 0 <= c < len(pred_diag_ang) else '')
                                        # EdgeAngle
                                        row_vals.append(float(gt_edge_ang[r]) if 0 <= r < len(gt_edge_ang) else '')
                                        row_vals.append(float(pred_edge_ang[c]) if 0 <= c < len(pred_edge_ang) else '')

                                        cw.writerow(row_vals)
                        except Exception:
                            pass

                        if matched_gt_w.size > 0 and matched_pred_w.size > 0:
                            plt.figure(figsize=(6,4))
                            _hr = _safe_hist_range(matched_gt_w, matched_pred_w)
                            if _hr is not None:
                                plt.hist(matched_gt_w, bins=30, range=_hr, alpha=0.5, label='GT', color='C0', edgecolor='black', linewidth=float(hist_edge_lw))
                                plt.hist(matched_pred_w, bins=30, range=_hr, alpha=0.5, label='Pred', color='C1', edgecolor='black', linewidth=float(hist_edge_lw))
                            else:
                                plt.hist(matched_gt_w, bins=30, alpha=0.5, label='GT', color='C0', edgecolor='black', linewidth=float(hist_edge_lw))
                                plt.hist(matched_pred_w, bins=30, alpha=0.5, label='Pred', color='C1', edgecolor='black', linewidth=float(hist_edge_lw))
                            plt.legend()
                            plt.xlabel(f'Width ({length_unit})')
                            plt.ylabel('Count')
                            plt.title('Width Distribution: GT vs Pred')
                            plt.tight_layout()
                            plt.savefig(os.path.join(save_dir, 'width_distribution_gt_vs_pred.png'))
                            plt.close()

                        # R^2 pairing (use matched+filtered pairs when GT matching is enabled)
                        if enable_gt_matching and strict_match_plots:
                            paired_gt = matched_gt_w
                            paired_pred = matched_pred_w
                        else:
                            paired_gt = matched_gt_w
                            paired_pred = matched_pred_w

                        if paired_gt is not None and paired_pred is not None and len(paired_gt) > 0 and len(paired_pred) > 0:
                            paired_pred = np.asarray(paired_pred, dtype=float)
                            paired_gt = np.asarray(paired_gt, dtype=float)
                            r2, mae = _compute_r2_mae(paired_pred, paired_gt)
                            metric_text = _format_scatter_metrics(r2, mae)

                            plt.figure(figsize=(5,5))
                            plt.scatter(paired_gt, paired_pred, alpha=0.7)
                            mn = min(paired_gt.min(), paired_pred.min())
                            mx = max(paired_gt.max(), paired_pred.max())
                            plt.plot([mn, mx], [mn, mx], 'k--')
                            plt.xlabel(f'GT Width ({length_unit})')
                            plt.ylabel(f'Pred Width ({length_unit})')
                            plt.title(f'Width {metric_text}')
                            plt.tight_layout()
                            plt.savefig(os.path.join(save_dir, 'width_r2_scatter.png'))
                            plt.close()

                    # ======= Area distribution and R^2 (same style as Width) =======
                    if len(gt_areas) > 0 and len(areas) > 0:
                        try:
                            pred_area = (np.asarray(areas).astype(float).ravel() * area_scale) if len(areas) > 0 else np.array([])
                            gt_area = (np.asarray(gt_areas).astype(float).ravel() * area_scale) if len(gt_areas) > 0 else np.array([])

                            matched_pred_area = np.array([pred_area[c] for c in match_col if 0 <= c < len(pred_area)], dtype=float) if (len(match_col) > 0) else np.array([], dtype=float)
                            matched_gt_area = np.array([gt_area[r] for r in match_row if 0 <= r < len(gt_area)], dtype=float) if (len(match_row) > 0) else np.array([], dtype=float)

                            if enable_gt_matching and strict_match_plots:
                                if matched_pred_area.size == 0 or matched_gt_area.size == 0:
                                    matched_pred_area = np.array([], dtype=float)
                                    matched_gt_area = np.array([], dtype=float)
                            else:
                                if matched_pred_area.size == 0 or matched_gt_area.size == 0:
                                    matched_pred_area = pred_area
                                    matched_gt_area = gt_area

                            if matched_gt_area.size > 0 and matched_pred_area.size > 0:
                                plt.figure(figsize=(6,4))
                                _hr = _safe_hist_range(matched_gt_area, matched_pred_area)
                                if _hr is not None:
                                    plt.hist(matched_gt_area, bins=30, range=_hr, alpha=0.5, label='GT', color='C0', edgecolor='black', linewidth=float(hist_edge_lw))
                                    plt.hist(matched_pred_area, bins=30, range=_hr, alpha=0.5, label='Pred', color='C1', edgecolor='black', linewidth=float(hist_edge_lw))
                                else:
                                    plt.hist(matched_gt_area, bins=30, alpha=0.5, label='GT', color='C0', edgecolor='black', linewidth=float(hist_edge_lw))
                                    plt.hist(matched_pred_area, bins=30, alpha=0.5, label='Pred', color='C1', edgecolor='black', linewidth=float(hist_edge_lw))
                                plt.legend()
                                plt.xlabel(f'Area ({area_unit})')
                                plt.ylabel('Count')
                                plt.title('Area Distribution: GT vs Pred')
                                plt.tight_layout()
                                plt.savefig(os.path.join(save_dir, 'area_distribution_gt_vs_pred.png'))
                                plt.close()

                                paired_pred_a = matched_pred_area
                                paired_gt_a = matched_gt_area

                                if paired_gt_a is not None and len(paired_gt_a) > 0:
                                    paired_pred_a = np.asarray(paired_pred_a, dtype=float)
                                    paired_gt_a = np.asarray(paired_gt_a, dtype=float)
                                    r2_a, mae_a = _compute_r2_mae(paired_pred_a, paired_gt_a)
                                    metric_text = _format_scatter_metrics(r2_a, mae_a)

                                    plt.figure(figsize=(5,5))
                                    plt.scatter(paired_gt_a, paired_pred_a, alpha=0.7)
                                    mn = min(paired_gt_a.min(), paired_pred_a.min())
                                    mx = max(paired_gt_a.max(), paired_pred_a.max())
                                    plt.plot([mn, mx], [mn, mx], 'k--')
                                    plt.xlabel(f'GT Area ({area_unit})')
                                    plt.ylabel(f'Pred Area ({area_unit})')
                                    plt.title(f'Area {metric_text}')
                                    plt.tight_layout()
                                    plt.savefig(os.path.join(save_dir, 'area_r2_scatter.png'))
                                    plt.close()
                        except Exception:
                            pass

                    # 其它指标：SideDistance, DiagonalLength, DiagonalAngle, EdgeAngle
                    comparisons = [
                        (side_dists, 'SideDistance', f'Length ({length_unit})', 'side_distance'),
                        (diag_lens, 'DiagonalLength', f'Length ({length_unit})', 'diagonal_length'),
                        (diag_angles, 'DiagonalAngle', 'Angle (deg)', 'diagonal_angle'),
                        (edge_angles, 'EdgeAngle', 'Angle (deg)', 'edge_angle')
                    ]

                    for pred_arr, name, xlabel, fname in comparisons:
                        gt_name = f'gt_{name.lower()}s' if name.endswith('s') == False else f'gt_{name.lower()}'
                        # map names to existing gt variables
                        if name == 'SideDistance':
                            gt_arr = np.asarray(gt_side_dists).astype(float).ravel() if 'gt_side_dists' in locals() else None
                        elif name == 'DiagonalLength':
                            gt_arr = np.asarray(gt_diag_lens).astype(float).ravel() if 'gt_diag_lens' in locals() else None
                        elif name == 'DiagonalAngle':
                            gt_arr = np.asarray(gt_diag_angles).astype(float).ravel() if 'gt_diag_angles' in locals() else None
                        elif name == 'EdgeAngle':
                            gt_arr = np.asarray(gt_edge_angles).astype(float).ravel() if 'gt_edge_angles' in locals() else None
                        else:
                            gt_arr = None

                        if gt_arr is None or len(gt_arr) == 0 or pred_arr is None or len(pred_arr) == 0:
                            continue

                        pred_arr_np = np.asarray(pred_arr).astype(float).ravel()
                        # If this metric is an angle, ensure arrays are in degrees
                        bin_s=30
                        if 'Angle' in name:
                            try:
                                if np.nanmax(np.abs(pred_arr_np)) <= 2 * np.pi:
                                    pred_arr_np = np.degrees(pred_arr_np)
                            except Exception:
                                pass
                            try:
                                if np.nanmax(np.abs(gt_arr)) <= 2 * np.pi:
                                    gt_arr = np.degrees(gt_arr)
                            except Exception:
                                pass
                            bin_s=np.linspace(0, 60, 31)
                        # Histogram: strict matched pairs when GT matching enabled
                        if len(match_row) > 0 and len(match_col) > 0:
                            pred_for_hist = np.array([pred_arr_np[c] for c in match_col if 0 <= c < len(pred_arr_np)], dtype=float)
                            gt_for_hist = np.array([gt_arr[r] for r in match_row if 0 <= r < len(gt_arr)], dtype=float)
                        else:
                            pred_for_hist = np.array([], dtype=float)
                            gt_for_hist = np.array([], dtype=float)

                        if not (enable_gt_matching and strict_match_plots):
                            if pred_for_hist.size == 0 or gt_for_hist.size == 0:
                                pred_for_hist = pred_arr_np
                                gt_for_hist = gt_arr

                        if gt_for_hist.size > 0 and pred_for_hist.size > 0:
                            plt.figure(figsize=(6,4))
                            _hr = None
                            if isinstance(bin_s, (int, np.integer)):
                                _hr = _safe_hist_range(gt_for_hist, pred_for_hist)
                            if _hr is not None:
                                plt.hist(gt_for_hist, bins=int(bin_s), range=_hr, alpha=0.5, label='GT', color='C0', edgecolor='black', linewidth=float(hist_edge_lw))
                                plt.hist(pred_for_hist, bins=int(bin_s), range=_hr, alpha=0.5, label='Pred', color='C1', edgecolor='black', linewidth=float(hist_edge_lw))
                            else:
                                plt.hist(gt_for_hist, bins=bin_s, alpha=0.5, label='GT', color='C0', edgecolor='black', linewidth=float(hist_edge_lw))
                                plt.hist(pred_for_hist, bins=bin_s, alpha=0.5, label='Pred', color='C1', edgecolor='black', linewidth=float(hist_edge_lw))
                            plt.legend()
                            plt.xlabel(xlabel)
                            plt.ylabel('Count')
                            plt.title(f'{name} Distribution: GT vs Pred')
                            if 'Angle' in name:
                                plt.xlim(0, 60)
                            plt.tight_layout()
                            plt.savefig(os.path.join(save_dir, f'{fname}_distribution_gt_vs_pred.png'))
                            plt.close()

                        # R^2 pairing: use matched pairs (strict when GT matching enabled)
                        paired_pred = pred_for_hist
                        paired_gt = gt_for_hist
                        if paired_pred is not None and paired_gt is not None and paired_pred.size > 0 and paired_gt.size > 0:
                            paired_pred = np.asarray(paired_pred, dtype=float)
                            paired_gt = np.asarray(paired_gt, dtype=float)
                            r2, mae = _compute_r2_mae(paired_pred, paired_gt)
                            metric_text = _format_scatter_metrics(r2, mae)

                            plt.figure(figsize=(5,5))
                            plt.scatter(paired_gt, paired_pred, alpha=0.7)
                            mn = min(paired_gt.min(), paired_pred.min())
                            mx = max(paired_gt.max(), paired_pred.max())
                            if 'Angle' in name:
                                plt.xlim(0, 60)
                                plt.ylim(0, 60)
                                mn = 0
                                mx = 60
                            plt.plot([mn, mx], [mn, mx], 'k--')
                            if name in ('SideDistance', 'DiagonalLength'):
                                plt.xlabel(f'GT {name} ({length_unit})')
                                plt.ylabel(f'Pred {name} ({length_unit})')
                            elif 'Angle' in name:
                                plt.xlabel(f'GT {name} (deg)')
                                plt.ylabel(f'Pred {name} (deg)')
                            else:
                                plt.xlabel(f'GT {name}')
                                plt.ylabel(f'Pred {name}')
                            plt.title(f'{name} {metric_text}')

                            plt.tight_layout()
                            plt.savefig(os.path.join(save_dir, f'{fname}_r2_scatter.png'))
                            plt.close()
                    # end enable_plots
    else:
        if gt_json_path is not None and os.path.exists(gt_json_path) and not enable_gt:
            print("ℹ️ enable_gt=False: skip GT analysis")

    # ================== (B) 多边形拟合评估（5/6/7/8/9/10边 + 凸包） ==================
    # 如果未提供 GT JSON，则只跑几何指标
    if gt_json_path is None or (not os.path.exists(gt_json_path)):
        print("⚠️ 未提供 gt_json_path，跳过 IoU/Precision/Recall/F1 评估。")
        print("✅ 所有几何指标和方向直方图已保存到", save_dir)
        # return empty metric lists so callers can safely unpack
        return [], [], [], [], pred_count, gt_count, pred_coverage, gt_coverage

    if not enable_polygon_metrics:
        print("ℹ️ enable_polygon_metrics=False: skip Compute_metrics IoU/Precision/Recall/F1")
        return [], [], [], [], pred_count, gt_count, pred_coverage, gt_coverage

    os.makedirs(save_dir, exist_ok=True)

    # ---------- 生成拟合多边形 ----------
    all_polygons = []
    for inst in global_instances:
        # Disable expensive debug drawing for speed
        poly = fit_polygon(None, inst["coords"], use_convex_hull=True)  # 用凸包
        if poly is not None and len(poly) >= 3:
            all_polygons.append(poly)

    # 叠加可视化
    draw_polygons_overlay(
        overlayed,
        all_polygons,
        save_path=os.path.join(save_dir, "polygons_overlay.png"),
        outline="orange",
        width=3,
        draw_hull_points=False
    )

    # 原始预测多边形
    all_pred_polygons = [inst["coords"] for inst in global_instances if len(inst["coords"]) >= 3]

    image_size = (overlayed.height, overlayed.width)

    # ---------- 三组比较 ----------
    summary = []
    print('拟合结果 vs GT')

    # 1) 拟合结果 vs GT
    metrics_hull_vs_gt = Compute_metrics('拟合结果 vs GT',orig_image, gt_json_path, all_polygons, image_size, save_dir=save_dir)
    summary.append({
        "Method": "Hull_vs_GT",
        "IoU": metrics_hull_vs_gt["IoU"],
        "Precision": metrics_hull_vs_gt["Precision"],
        "Recall": metrics_hull_vs_gt["Recall"],
        "F1-score": metrics_hull_vs_gt["F1-score"]
    })

    print('原始预测 vs GT')
    # 2) 原始预测 vs GT
    ex_coor=ex_c(global_instances)
    metrics_pred_vs_gt = Compute_metrics('原始预测 vs GT',orig_image, gt_json_path, ex_coor, image_size, save_dir=save_dir)
    summary.append({
        "Method": "Pred_vs_GT",
        "IoU": metrics_pred_vs_gt["IoU"],
        "Precision": metrics_pred_vs_gt["Precision"],
        "Recall": metrics_pred_vs_gt["Recall"],
        "F1-score": metrics_pred_vs_gt["F1-score"]
    })

    print('拟合结果 vs 原始预测')
    '''
    # 3) 拟合结果 vs 原始预测
    metrics_hull_vs_pred = compute_metrics('拟合结果 vs 原始预测',orig_image, ex_coor, all_polygons, image_size, save_dir=save_dir)
    summary.append({
        "Method": "Hull_vs_Pred",
        "IoU": metrics_hull_vs_pred["IoU"],
        "Precision": metrics_hull_vs_pred["Precision"],
        "Recall": metrics_hull_vs_pred["Recall"],
        "F1-score": metrics_hull_vs_pred["F1-score"]
    })
    '''
    # ---------- 保存 CSV ----------
    summary_csv = os.path.join(save_dir, "metrics_summary.csv")
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Method", "IoU", "Precision", "Recall", "F1-score"])
        writer.writeheader()
        for row in summary:
            writer.writerow(row)

    # ---------- 画图 ----------
    methods = [r["Method"] for r in summary]
    ious = [r["IoU"] for r in summary]
    precisions = [r["Precision"] for r in summary]
    recalls = [r["Recall"] for r in summary]
    f1_scores = [r["F1-score"] for r in summary]

    zlplot(methods, ious, "IoU", save_dir, "metrics_summary_IoU.png")
    zlplot(methods, precisions, "Precision", save_dir, "metrics_summary_Precision.png")
    zlplot(methods, recalls, "Recall", save_dir, "metrics_summary_Recall.png")
    zlplot(methods, f1_scores, "F1-score", save_dir, "metrics_summary_F1.png")

    print("📑 已保存方法对比表：", summary_csv)
    print("📊 已保存所有指标柱状图到：", save_dir)
    print("✅ 所有评估完成！")
    return ious, precisions, recalls, f1_scores, pred_count, gt_count, pred_coverage, gt_coverage
