import re
from collections import defaultdict
import numpy as np
import csv

file = "results_hidden.txt"
out_csv = "results_summary.csv"

pattern_title = re.compile(r"(ETTm2_[\w_]+)_([0-9]+)$")
pattern_metrics = re.compile(r"mse:([0-9.\-e]+), mae:([0-9.\-e]+)")

groups = defaultdict(list)
current_title = None

with open(file, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        m_title = pattern_title.match(line)
        if m_title:
            base_name = m_title.group(1) 
            current_title = base_name
            continue

        m_metrics = pattern_metrics.match(line)
        if m_metrics and current_title:
            mse = float(m_metrics.group(1))
            mae = float(m_metrics.group(2))

            groups[current_title].append((mse, mae))


results = []

for name, metrics in groups.items():
    mse_mean = np.mean([m[0] for m in metrics])
    mae_mean = np.mean([m[1] for m in metrics])

    
    parts = name.split("_")
    # Ex: ETTm2_96_96_ns_Transformer ...
    predict_window = parts[2]  

    # Ex: ... Exp_h96_l2  → hidden = 96
    hidden_match = re.search(r"Exp_h([0-9]+)", name)
    hidden = int(hidden_match.group(1)) if hidden_match else None

    results.append((int(predict_window), hidden, mse_mean, mae_mean, name))

results.sort(key=lambda x: (x[0], x[1]))


# Save to CSV
with open(out_csv, "w", newline="", encoding="utf-8") as cf:
    writer = csv.writer(cf)
    writer.writerow(["predict_window", "hidden", "mse_mean", "mae_mean", "base_name"])

    for r in results:
        writer.writerow(r)

print(f"CSV saved to {out_csv}")
# Impressão
print(f"{'Predict':>10} {'Hidden':>10} {'MSE':>15} {'MAE':>15}  Name")
for pw, hid, mse, mae, nm in results:
    print(f"{pw:>10} {hid:>10} {mse:>15.6f} {mae:>15.6f}  {nm}")
