import os, csv, glob

base = 'work_dirs/ablation_results'
experiments = {
    'E1_M1_noSW': ('M1', 'None', 0),
    'E2_M2_noSW': ('M2', 'None', 0),
    'E3_M2_fix1024': ('M2', 1024, 0.2),
    'E4_M3_fix1024': ('M3', 1024, 0.2),
    'E5_M2_adaptive': ('M2', 'adaptive', 0.2),
    'E6_M3_adaptive': ('M3', 'adaptive', 0.2),
}

print(f"{'Exp':<20} {'Image':<22} {'IoU':>8} {'F1':>8} {'Pred':>6} {'GT':>6}")
print("-" * 80)

all_data = {}
for exp_name, (model, window, overlap) in experiments.items():
    exp_dir = os.path.join(base, exp_name)
    csv_path = os.path.join(exp_dir, 'metrics_summary.csv')

    rows = []
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                rows.append(row)
    else:
        # Per-image dirs (E5, E6)
        for img_dir in sorted(glob.glob(os.path.join(exp_dir, '*'))):
            if not os.path.isdir(img_dir):
                continue
            sub_csv = os.path.join(img_dir, 'metrics_summary.csv')
            if os.path.exists(sub_csv):
                with open(sub_csv) as f:
                    for row in csv.DictReader(f):
                        rows.append(row)

    all_data[exp_name] = rows

    for row in rows:
        print(f"{exp_name:<20} {row['image']:<22} {float(row['iou']):8.4f} {float(row['f1']):8.4f} {row['pred_count']:>6} {row['gt_count']:>6}")

# Average per experiment
print("\n" + "=" * 80)
print(f"{'Exp':<20} {'Model':<5} {'Window':>8} {'Ov':>6} {'Avg IoU':>8} {'Avg F1':>8}")
print("-" * 60)
for exp_name, (model, window, overlap) in experiments.items():
    rows = all_data[exp_name]
    if rows:
        avg_iou = sum(float(r['iou']) for r in rows if r['iou'] != 'nan') / len(rows)
        avg_f1 = sum(float(r['f1']) for r in rows if r['f1'] != 'nan') / len(rows)
        print(f"{exp_name:<20} {model:<5} {str(window):>8} {overlap:>6} {avg_iou:8.4f} {avg_f1:8.4f}")
