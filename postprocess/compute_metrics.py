import os, json
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
def Compute_metrics(
    method,
    orig_image,
    json_path,
    all_polygons,
    image_size,
    save_dir="./output",
    mode="union",
    *,
    save_visualization: bool = True,
    save_metrics: bool = True,
    save_bar: bool = True,
    verbose: bool = True,
):
    """
    计算 IoU / Precision / Recall / F1 并保存结果与可视化

    :param method: 用于保存文件的标识
    :param orig_image: 原始图像路径 (PNG/JPG)
    :param json_path: 手动标注的json文件路径（ISAT风格，objects[].segmentation: [[x,y], ...]）
    :param all_polygons: 预测多边形顶点列表 (list of np.ndarray, 每个 (m,2)) 注意顶点是 (x,y)=(row,col)
    :param image_size: (H, W)
    :param save_dir: 保存目录
    :param mode: "instance" (逐实例, 默认) 或 "union" (合并所有实例为一类)
    :return: dict 指标
    """
    H, W = image_size
    os.makedirs(save_dir, exist_ok=True)
     # 1) 读取 GT 掩膜
    with open(json_path, "r", encoding="utf-8") as f:
        ann = json.load(f)
    # 初始化空白掩膜
    mask_gt = Image.new("L", (W, H), 0)
    draw_gt = ImageDraw.Draw(mask_gt)
    # 背景掩膜（可选）
    mask_bg = Image.new("L", (W, H), 0)
    draw_bg = ImageDraw.Draw(mask_bg)
    for obj in ann.get("objects", []):
        coords = obj.get("segmentation", [])
        if not coords:
            continue
        polygon = [(x, y) for x, y in coords]  # GT 是 (col,row)
        # 判断是否为背景实例
        if obj.get("category", "").strip() == "__background__":
            draw_bg.polygon(polygon, outline=1, fill=1)
        else:
            draw_gt.polygon(polygon, outline=1, fill=1)
    mask_gt_np = np.array(mask_gt, dtype=bool)
    mask_bg_np = np.array(mask_bg, dtype=bool)
    # 去掉背景区域：真正的畴区实例 = 实例掩膜 AND NOT 背景掩膜
    mask_gt_np = np.logical_and(mask_gt_np, ~mask_bg_np)
    # 2) 预测掩膜
    mask_pred = Image.new("L", (W, H), 0)
    draw_pred = ImageDraw.Draw(mask_pred)
    if mode == "instance":
        # 逐实例绘制
        for poly in all_polygons:
            if poly is None or len(poly) < 3:
                continue
            polygon = [(p[1], p[0]) for p in poly]  # (row,col) -> (col,row)
            draw_pred.polygon(polygon, outline=1, fill=1)
    elif mode == "union":
        # 所有实例合并为一类：直接把每个多边形绘制到同一个 mask
        for poly in all_polygons:
            if poly is None or len(poly) < 3:
                continue
            polygon = [(p[1], p[0]) for p in poly]
            draw_pred.polygon(polygon, outline=1, fill=1)
    mask_pred_np = np.array(mask_pred, dtype=bool)
    # 3) 计算指标
    TP = np.logical_and(mask_gt_np, mask_pred_np).sum()
    FP = np.logical_and(~mask_gt_np, mask_pred_np).sum()
    FN = np.logical_and(mask_gt_np, ~mask_pred_np).sum()
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    iou       = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0.0
    # 4) 可视化
    original_img = Image.open(orig_image).convert("RGB")
    base = original_img.convert("RGBA")
    vis = np.zeros((H, W, 3), dtype=np.uint8)
    vis[mask_gt_np] = [0, 0, 255]                 # 蓝色 GT
    vis[mask_pred_np] = [255, 0, 0]               # 红色预测
    vis[np.logical_and(mask_gt_np, mask_pred_np)] = [0, 255, 0]  # 绿色重叠
    mask_img = Image.fromarray(vis).convert("RGBA")
    mask_img.putalpha(128)  # 半透明
    overlayed = Image.alpha_composite(base, mask_img)
    vis_path = os.path.join(save_dir, f"iou_visualization_{method}_{mode}.png")
    if save_visualization:
        overlayed.save(vis_path)

    metrics_path = os.path.join(save_dir, f"metrics_{method}_{mode}.txt")
    if save_metrics:
        with open(metrics_path, "w", encoding="utf-8") as f:
            f.write(f"IoU: {iou:.4f}\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall: {recall:.4f}\n")
            f.write(f"F1-score: {f1:.4f}\n")

    if save_bar:
        names = ["IoU", "Precision", "Recall", "F1-score"]
        values = [iou, precision, recall, f1]
        plt.figure(figsize=(6,4))
        plt.bar(names, values, color=["orange", "red", "blue", "green"], edgecolor="black")
        plt.ylim(0, 1)
        plt.ylabel("Score")
        plt.title(f"Prediction vs GT ({mode})")
        for i, v in enumerate(values):
            plt.text(i, min(v + 0.03, 1.0), f"{v:.2f}", ha="center", fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"metrics_bar_{method}_{mode}.png"))
        plt.close()

    if verbose:
        print(f"✅ [{mode}] IoU={iou:.4f}, P={precision:.4f}, R={recall:.4f}, F1={f1:.4f}")
        if save_metrics:
            print(f"📌 指标已保存到 {metrics_path}")
        if save_visualization:
            print(f"📌 可视化已保存到 {vis_path}")
        if save_bar:
            print(f"📊 柱状图已保存到 {os.path.join(save_dir, f'metrics_bar_{method}_{mode}.png')}")
    return {"IoU": iou, "Precision": precision, "Recall": recall, "F1-score": f1}
