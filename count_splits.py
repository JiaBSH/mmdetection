import json, os

base = "/data/run01/scvi576/JiaBSH/mmdetection_para/dataset_root/mmdata_isat_1024"

print(f"Using base: {base}")
print(f"Exists: {os.path.exists(base)}")

for split in ['train', 'val', 'test']:
    path = os.path.join(base, 'annotations', f'instances_{split}.json')
    if os.path.exists(path):
        with open(path) as f:
            d = json.load(f)
        print(f"{split}: images={len(d['images'])}, annotations={len(d['annotations'])}")
    else:
        print(f"{split}: NOT FOUND at {path}")
