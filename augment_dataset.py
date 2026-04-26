"""
Augment dataset_root_512 → dataset_root_512_aug

Augmentations applied:
  - Horizontal / Vertical flip
  - Random rotation (±30°) with constant border fill
  - Random crop (75-100% of image retained)
  - ColorJitter: brightness, contrast, saturation, hue
  - Gaussian noise

Segmentation polygons are transformed accordingly.
Annotations whose visible area after crop/rotation drops below a
threshold are discarded automatically.

Usage:
    python augment_dataset.py
"""

import json, os, copy, random, shutil
from pathlib import Path
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon as MplPoly
from matplotlib.collections import PatchCollection

# ── Configuration ─────────────────────────────────────────────────────────────
SRC_ROOT   = "/hpcfs/fhome/sunxc/JiaBSH/mmdetection/dataset_root/dataset_1024"
DST_ROOT   = "/hpcfs/fhome/sunxc/JiaBSH/mmdetection/dataset_root/dataset_1024_aug"
VIS_DIR    = os.path.join(DST_ROOT, "visualizations")

AUG_FACTOR          = 4      # augmented copies per original image
INCLUDE_ORIGINAL    = True   # also copy originals into the new dataset
MIN_VISIBLE_RATIO   = 0.2    # discard annotation if < 20% original area remains
MIN_AREA_PX         = 10    # discard annotation if area < 10 px²
SEED                = 42
# ──────────────────────────────────────────────────────────────────────────────

random.seed(SEED)
np.random.seed(SEED)


# ═══════════════════════════════════════════════════════════════
# Geometry helpers
# ═══════════════════════════════════════════════════════════════

def poly_to_mask(flat_pts, h, w):
    """COCO flat polygon [x1,y1,x2,y2,...] → binary uint8 mask."""
    pts = np.array(flat_pts, dtype=np.float32).reshape(-1, 2).astype(np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 1)
    return mask


def mask_to_polys(mask):
    """Binary mask → list of COCO flat polygons (may return empty list)."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_AREA_PX:
            continue
        flat = c.flatten().tolist()
        if len(flat) >= 6:          # at least 3 points
            polys.append(flat)
    return polys


def poly_area(flat_pts):
    """Shoelace area of a flat polygon."""
    pts = np.array(flat_pts, dtype=np.float64).reshape(-1, 2)
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def polys_bbox(polys_list):
    """Compute COCO bbox [x, y, w, h] covering all polygons."""
    all_pts = np.vstack([
        np.array(p, dtype=np.float32).reshape(-1, 2) for p in polys_list
    ])
    x0, y0 = all_pts.min(axis=0)
    x1, y1 = all_pts.max(axis=0)
    return [float(x0), float(y0), float(x1 - x0), float(y1 - y0)]


# ═══════════════════════════════════════════════════════════════
# Individual augmentation functions
#   Each takes (image_rgb, masks_list) and returns the same types
# ═══════════════════════════════════════════════════════════════

def aug_hflip(image, masks):
    image = cv2.flip(image, 1)
    masks = [cv2.flip(m, 1) for m in masks]
    return image, masks


def aug_vflip(image, masks):
    image = cv2.flip(image, 0)
    masks = [cv2.flip(m, 0) for m in masks]
    return image, masks


def aug_rotate(image, masks, angle=None):
    """Rotate by random angle ∈ [-30, 30]° with constant border fill."""
    h, w = image.shape[:2]
    if angle is None:
        angle = random.uniform(-30, 30)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    image = cv2.warpAffine(image, M, (w, h),
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    masks = [cv2.warpAffine(m, M, (w, h),
                             flags=cv2.INTER_NEAREST,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
             for m in masks]
    return image, masks


def aug_crop(image, masks):
    """Random crop retaining 30-100% of each dimension."""
    h, w = image.shape[:2]
    ch = random.randint(int(h * 0.3), h)
    cw = random.randint(int(w * 0.3), w)
    y0 = random.randint(0, h - ch)
    x0 = random.randint(0, w - cw)
    image = image[y0:y0 + ch, x0:x0 + cw]
    masks = [m[y0:y0 + ch, x0:x0 + cw] for m in masks]
    return image, masks


def aug_color_jitter(image,
                     brightness=0.35,
                     contrast=0.35,
                     saturation=0.35,
                     hue=0.12):
    """Random brightness / contrast / saturation / hue shift (image only)."""
    img = image.astype(np.float32)

    # Brightness
    if random.random() < 0.8:
        beta = random.uniform(-brightness, brightness) * 255
        img = np.clip(img + beta, 0, 255)

    # Contrast
    if random.random() < 0.8:
        alpha = random.uniform(1 - contrast, 1 + contrast)
        img = np.clip(img * alpha, 0, 255)

    # Saturation & Hue via HSV
    if random.random() < 0.8:
        img_uint8 = img.astype(np.uint8)
        hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV).astype(np.float32)

        # Saturation
        s_scale = random.uniform(1 - saturation, 1 + saturation)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * s_scale, 0, 255)

        # Hue
        h_shift = random.uniform(-hue * 180, hue * 180)
        hsv[:, :, 0] = (hsv[:, :, 0] + h_shift) % 180

        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)

    return np.clip(img, 0, 255).astype(np.uint8)


def aug_gaussian_noise(image, var_limit=(10, 50)):
    """Add Gaussian noise (image only)."""
    var = random.uniform(*var_limit)
    noise = np.random.normal(0, var ** 0.5, image.shape).astype(np.float32)
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)


# ═══════════════════════════════════════════════════════════════
# Apply one complete augmentation pipeline
# ═══════════════════════════════════════════════════════════════

def apply_augmentation(image_rgb, annotations):
    """
    Apply a random augmentation pipeline to image + annotations.

    Returns:
        aug_image  (H', W', 3) uint8
        new_anns   list of updated annotation dicts (deep copies)
    """
    h, w = image_rgb.shape[:2]

    # Build per-instance combined mask (union over all polygons of an annotation)
    instance_masks = []
    orig_areas     = []
    for ann in annotations:
        m = np.zeros((h, w), dtype=np.uint8)
        for seg in ann["segmentation"]:
            m = np.maximum(m, poly_to_mask(seg, h, w))
        instance_masks.append(m)
        orig_areas.append(float(m.sum()))

    image = image_rgb.copy()

    # ── Geometric transforms (applied to image + masks) ──────
    if random.random() < 0.5:
        image, instance_masks = aug_hflip(image, instance_masks)
    if random.random() < 0.3:
        image, instance_masks = aug_vflip(image, instance_masks)
    if random.random() < 0.7:
        image, instance_masks = aug_rotate(image, instance_masks)
    if random.random() < 0.5:
        image, instance_masks = aug_crop(image, instance_masks)

    # ── Photometric transforms (image only) ──────────────────
    if random.random() < 0.8:
        image = aug_color_jitter(image)
    if random.random() < 0.3:
        image = aug_gaussian_noise(image)

    # ── Rebuild annotations from transformed masks ────────────
    new_anns = []
    for ann, mask, orig_area in zip(annotations, instance_masks, orig_areas):
        new_polys = mask_to_polys(mask)
        if not new_polys:
            continue

        new_area = sum(poly_area(p) for p in new_polys)

        # Drop annotations that mostly disappeared
        if orig_area > 0 and (new_area / orig_area) < MIN_VISIBLE_RATIO:
            continue
        if new_area < MIN_AREA_PX:
            continue

        new_ann = copy.deepcopy(ann)
        new_ann["segmentation"] = new_polys
        new_ann["bbox"]         = polys_bbox(new_polys)
        new_ann["area"]         = new_area
        new_anns.append(new_ann)

    return image, new_anns


# ═══════════════════════════════════════════════════════════════
# Process one split
# ═══════════════════════════════════════════════════════════════

def process_split(split):
    print(f"\n── Processing split: {split} ──────────────────────────────")
    src_img_dir  = os.path.join(SRC_ROOT, "images", split)
    src_ann_file = os.path.join(SRC_ROOT, "annotations", f"instances_{split}.json")

    dst_img_dir  = os.path.join(DST_ROOT, "images", split)
    dst_ann_file = os.path.join(DST_ROOT, "annotations", f"instances_{split}.json")
    os.makedirs(dst_img_dir,  exist_ok=True)
    os.makedirs(os.path.join(DST_ROOT, "annotations"), exist_ok=True)

    with open(src_ann_file) as f:
        coco = json.load(f)

    # index annotations by image_id
    img_to_anns = {}
    for ann in coco["annotations"]:
        img_to_anns.setdefault(ann["image_id"], []).append(ann)

    new_images, new_anns = [], []
    new_img_id = new_ann_id = 0

    for img_info in coco["images"]:
        src_path = os.path.join(src_img_dir, img_info["file_name"])
        if not os.path.exists(src_path):
            print(f"  [skip] {src_path} not found")
            continue

        image = cv2.imread(src_path)
        if image is None:
            print(f"  [skip] cannot read {src_path}")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        annots = img_to_anns.get(img_info["id"], [])
        stem   = Path(img_info["file_name"]).stem
        ext    = Path(img_info["file_name"]).suffix

        # ── Optionally keep original ──────────────────────────
        if INCLUDE_ORIGINAL:
            dst_fname = f"{stem}_orig{ext}"
            dst_path  = os.path.join(dst_img_dir, dst_fname)
            cv2.imwrite(dst_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

            h_orig, w_orig = image.shape[:2]
            rec = copy.deepcopy(img_info)
            rec["id"]        = new_img_id
            rec["file_name"] = dst_fname
            rec["height"]    = h_orig
            rec["width"]     = w_orig
            new_images.append(rec)

            for ann in annots:
                a = copy.deepcopy(ann)
                a["id"]         = new_ann_id
                a["image_id"]   = new_img_id
                if "image_name" in a:
                    a["image_name"] = dst_fname
                new_anns.append(a)
                new_ann_id += 1
            new_img_id += 1

        # ── Augmented copies ──────────────────────────────────
        for aug_idx in range(AUG_FACTOR):
            try:
                aug_img, aug_ann_list = apply_augmentation(image, annots)
            except Exception as exc:
                print(f"  [warn] {img_info['file_name']} aug{aug_idx}: {exc}")
                continue

            dst_fname = f"{stem}_aug{aug_idx}{ext}"
            dst_path  = os.path.join(dst_img_dir, dst_fname)
            cv2.imwrite(dst_path, cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))

            h_new, w_new = aug_img.shape[:2]
            rec = copy.deepcopy(img_info)
            rec["id"]        = new_img_id
            rec["file_name"] = dst_fname
            rec["height"]    = h_new
            rec["width"]     = w_new
            new_images.append(rec)

            for ann in aug_ann_list:
                a = copy.deepcopy(ann)
                a["id"]         = new_ann_id
                a["image_id"]   = new_img_id
                if "image_name" in a:
                    a["image_name"] = dst_fname
                new_anns.append(a)
                new_ann_id += 1
            new_img_id += 1

        print(f"  {img_info['file_name']}: {len(annots)} anns  →  "
              f"×{AUG_FACTOR + int(INCLUDE_ORIGINAL)} copies")

    new_coco = copy.deepcopy(coco)
    new_coco["images"]      = new_images
    new_coco["annotations"] = new_anns
    with open(dst_ann_file, "w") as f:
        json.dump(new_coco, f)

    print(f"\n  ✓ {len(new_images)} images, {len(new_anns)} annotations  →  {dst_ann_file}")
    return new_images, new_anns


# ═══════════════════════════════════════════════════════════════
# Visualisation
# ═══════════════════════════════════════════════════════════════

def visualize_split(split, num_samples=8):
    dst_img_dir  = os.path.join(DST_ROOT, "images", split)
    dst_ann_file = os.path.join(DST_ROOT, "annotations", f"instances_{split}.json")
    os.makedirs(VIS_DIR, exist_ok=True)

    with open(dst_ann_file) as f:
        coco = json.load(f)

    img_to_anns = {}
    for ann in coco["annotations"]:
        img_to_anns.setdefault(ann["image_id"], []).append(ann)

    imgs_with_anns = [im for im in coco["images"] if im["id"] in img_to_anns]
    if not imgs_with_anns:
        print(f"  [vis] no images with annotations for {split}")
        return

    random.seed(SEED)
    samples = random.sample(imgs_with_anns, min(num_samples, len(imgs_with_anns)))

    ncols = 4
    nrows = (len(samples) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4))
    axes = np.array(axes).flatten()

    for ax, img_info in zip(axes, samples):
        img_path = os.path.join(dst_img_dir, img_info["file_name"])
        bgr = cv2.imread(img_path)
        if bgr is None:
            ax.set_visible(False)
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        ax.imshow(rgb)

        anns = img_to_anns.get(img_info["id"], [])

        # Use a large set of perceptually distinct colors and cycle through them.
        # Offset by ann index using a large prime step so spatially adjacent
        # instances (which tend to have consecutive ids) get different colors.
        PALETTE = np.array([
            [0.902, 0.102, 0.294], [0.235, 0.706, 0.294], [0.000, 0.510, 0.784],
            [0.961, 0.510, 0.192], [0.569, 0.118, 0.706], [0.275, 0.941, 0.941],
            [0.941, 0.196, 0.902], [0.824, 0.961, 0.235], [0.980, 0.745, 0.745],
            [0.000, 0.502, 0.502], [0.902, 0.745, 1.000], [0.604, 0.388, 0.141],
            [1.000, 0.980, 0.784], [0.502, 0.000, 0.000], [0.667, 1.000, 0.765],
            [0.502, 0.502, 0.000], [1.000, 0.843, 0.706], [0.000, 0.000, 0.502],
            [0.502, 0.502, 0.502], [1.000, 1.000, 1.000],
        ], dtype=np.float32)
        STEP = 7  # large prime → non-adjacent colors for consecutive ids
        colors = [PALETTE[(i * STEP) % len(PALETTE)] for i in range(len(anns))]

        for ann, color in zip(anns, colors):
            # Segmentation polygons
            for seg in ann["segmentation"]:
                pts  = np.array(seg, dtype=np.float32).reshape(-1, 2)
                poly = MplPoly(pts, closed=True,
                               facecolor=(*color[:3], 0.25),
                               edgecolor=color[:3], linewidth=1.2)
                ax.add_patch(poly)
            # Bounding box
            x, y, bw, bh = ann["bbox"]
            rect = mpatches.Rectangle((x, y), bw, bh,
                                       linewidth=1, edgecolor=color[:3],
                                       facecolor="none", linestyle="--")
            ax.add_patch(rect)

        label = img_info["file_name"]
        # show aug type (orig / aug0 / aug1 / …)
        tag = label.split("_")[-1].replace(ext_of(label), "")
        ax.set_title(f"{tag}  ({len(anns)} anns)", fontsize=9)
        ax.axis("off")

    # Hide unused axes
    for ax in axes[len(samples):]:
        ax.set_visible(False)

    plt.suptitle(f"Augmented samples — {split}", fontsize=13, y=1.01)
    plt.tight_layout()
    out_path = os.path.join(VIS_DIR, f"check_{split}.png")
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Visualisation saved → {out_path}")


def ext_of(fname):
    return Path(fname).suffix


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Augment train and val
    for split in ["train", "val"]:
        process_split(split)

    # Copy test unchanged
    for part in ("images/test", ):
        src = os.path.join(SRC_ROOT, part)
        dst = os.path.join(DST_ROOT, part)
        if os.path.exists(src):
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
    for ann_name in ("instances_test.json", ):
        src = os.path.join(SRC_ROOT, "annotations", ann_name)
        dst = os.path.join(DST_ROOT, "annotations", ann_name)
        if os.path.exists(src):
            shutil.copy2(src, dst)
    print("\n  Test split copied unchanged.")

    # Visualise
    for split in ["train", "val"]:
        visualize_split(split)

    print("\n══ Done ══")
