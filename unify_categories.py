import json
import os
from collections import Counter

files = [
    "dataset_root/annotations/instances_train.json",
    "dataset_root/annotations/instances_val.json",
    "dataset_root/annotations/instances_test.json"
]

new_categories = [{"name": "畴区", "id": 1, "supercategory": None}]

for file_path in files:
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        continue
    
    backup_path = file_path + ".bak_unify_catid_20260426"
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Backup
    with open(backup_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    
    old_categories = data.get('categories', [])
    old_cat_counts = Counter(ann.get('category_id') for ann in data.get('annotations', []))
    
    # Modify
    data['categories'] = new_categories
    for ann in data.get('annotations', []):
        ann['category_id'] = 1
        
    new_cat_counts = Counter(ann.get('category_id') for ann in data.get('annotations', []))
    total_annotations = len(data.get('annotations', []))
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
        
    print(f"Summary for {file_path}:")
    print(f"  Old categories: {old_categories}")
    print(f"  New categories: {new_categories}")
    print(f"  Old category_id counts: {dict(old_cat_counts)}")
    print(f"  New category_id counts: {dict(new_cat_counts)}")
    print(f"  Total annotations: {total_annotations}")
    print("-" * 40)
