import json, os, sys

base = "/data/run01/scvi576/JiaBSH/mmdetection_para/dataset_root/mmdata_isat_1024"
out = []

for split in ['train', 'val', 'test']:
    path = os.path.join(base, 'annotations', f'instances_{split}.json')
    with open(path) as f:
        d = json.load(f)
    line = f"{split}: images={len(d['images'])}, annotations={len(d['annotations'])}"
    out.append(line)
    print(line)

# Also write to a small output file
with open("/data/run01/scvi576/JiaBSH/mmdetection_para/_split_counts.txt", "w") as f:
    f.write("\n".join(out) + "\n")
