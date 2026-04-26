import json
import os
import numpy as np
from collections import Counter

def check_split(json_path, image_dir):
    print(f"\n--- Checking {os.path.basename(json_path)} ---")
    if not os.path.exists(json_path):
        print("File not found.")
        return None
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    imgs = {img['id']: img for img in data['images']}
    anns = data['annotations']
    cats = {cat['id']: cat['name'] for cat in data['categories']}
    
    print(f"Images: {len(imgs)}, Annotations: {len(anns)}, Categories: {len(cats)}")
    print(f"Category IDs/Names: {cats}")
    
    # Missing images
    missing_imgs = [img['file_name'] for img in imgs.values() if not os.path.exists(os.path.join(image_dir, img['file_name']))]
    if missing_imgs:
        print(f"Missing images: {len(missing_imgs)} (e.g., {missing_imgs[:3]})")
    else:
        print("Missing images: 0")
        
    # Duplicates
    img_ids = [img['id'] for img in data['images']]
    img_files = [img['file_name'] for img in data['images']]
    ann_ids = [ann['id'] for ann in anns]
    
    if len(img_ids) != len(set(img_ids)): print(f"Duplicate image IDs: {len(img_ids) - len(set(img_ids))}")
    if len(img_files) != len(set(img_files)): print(f"Duplicate image file names: {len(img_files) - len(set(img_files))}")
    if len(ann_ids) != len(set(ann_ids)): print(f"Duplicate annotation IDs: {len(ann_ids) - len(set(ann_ids))}")

    cat_counts = Counter()
    issues = Counter()
    areas = []
    rel_areas = []
    tiny_boxes = 0
    extreme_aspect = 0
    img_ann_count = Counter()
    
    for ann in anns:
        img = imgs.get(ann['image_id'])
        if not img:
            issues['unknown_img_id'] += 1
            continue
        
        cat_counts[ann['category_id']] += 1
        img_ann_count[ann['image_id']] += 1
        
        if ann['category_id'] not in cats: issues['unknown_cat_id'] += 1
        
        # BBox issues
        x, y, w, h = ann['bbox']
        if w <= 0 or h <= 0: issues['non_positive_wh'] += 1
        if x < 0 or y < 0: issues['negative_coords'] += 1
        if x + w > img['width']: issues['bbox_out_w'] += 1
        if y + h > img['height']: issues['bbox_out_h'] += 1
        if w < 4 or h < 4: tiny_boxes += 1
        if w > 0 and h > 0:
            aspect = max(w/h, h/w)
            if aspect > 20: extreme_aspect += 1
            
        # Area
        area = ann.get('area', 0)
        if area <= 0: issues['area_le_0'] += 1
        areas.append(area)
        rel_areas.append(area / (img['width'] * img['height']))
        
        # Segmentation
        seg = ann.get('segmentation')
        if not seg:
            issues['missing_seg'] += 1
        elif isinstance(seg, list):
            if not seg:
                issues['empty_poly'] += 1
            else:
                for poly in seg:
                    if len(poly) % 2 != 0: issues['odd_poly'] += 1
                    if len(poly) < 6: issues['small_poly'] += 1
        elif isinstance(seg, dict):
             if not seg.get('counts'): issues['empty_rle'] += 1

    print(f"Annotations per category: {dict(cat_counts)}")
    if issues: print(f"Issues found: {dict(issues)}")
    print(f"Tiny boxes (<4px): {tiny_boxes}, Extreme aspect (>20): {extreme_aspect}")
    
    if areas:
        print(f"BBox Area - Min: {min(areas):.1f}, Median: {np.median(areas):.1f}, P95: {np.percentile(areas, 95):.1f}, Max: {max(areas):.1f}")
        print(f"Relative Area - Median: {np.median(rel_areas):.4f}")
        
    top_10 = sorted(img_ann_count.items(), key=lambda x: x[1], reverse=True)[:10]
    print(f"Top 10 images by ann count: {top_10}")
    
    return {cat: count/len(anns) for cat, count in cat_counts.items()}

root = "/hpcfs/fhome/sunxc/JiaBSH/mmdetection/dataset_root"
splits = [
    ("instances_train.json", "images/train"),
    ("instances_val.json", "images/val"),
    ("instances_test.json", "images/test")
]

distros = []
for j, i in splits:
    d = check_split(os.path.join(root, "annotations", j), os.path.join(root, i))
    if d: distros.append(d)

if len(distros) >= 2:
    print("\nClass distribution similarity (Train vs Val):")
    t, v = distros[0], distros[1]
    all_cats = set(t.keys()) | set(v.keys())
    diff = sum(abs(t.get(c, 0) - v.get(c, 0)) for c in all_cats)
    print(f"Total variation distance: {diff/2:.4f} (0 is identical)")

