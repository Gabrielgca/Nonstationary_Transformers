import re
from collections import defaultdict
import numpy as np
import csv

file = "result_extended.txt"
out_csv = "results_summary_extended.csv"

# Matches the title lines, capturing everything up to the last number
pattern_title = re.compile(r"(ETTm2_[A-Za-z0-9_]+)_([0-9]+)$")

pattern_metrics_normal = re.compile(r"mse:([0-9.\-e]+), mae:([0-9.\-e]+)")
pattern_metrics_abrupt = re.compile(r"mse_abrupt:([0-9.\-e]+), mae_abrupt:([0-9.\-e]+)")

groups = defaultdict(lambda: {
    "mse": [], "mae": [], "mse_abrupt": [], "mae_abrupt": []
})
current_title = None

with open(file, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        # Detect a title block
        m_title = pattern_title.match(line)
        if m_title:
            current_title = m_title.group(1)
            continue

        # Normal metrics
        m_norm = pattern_metrics_normal.match(line)
        if m_norm and current_title:
            groups[current_title]["mse"].append(float(m_norm.group(1)))
            groups[current_title]["mae"].append(float(m_norm.group(2)))
            continue

        # Abrupt metrics
        m_ab = pattern_metrics_abrupt.match(line)
        if m_ab and current_title:
            groups[current_title]["mse_abrupt"].append(float(m_ab.group(1)))
            groups[current_title]["mae_abrupt"].append(float(m_ab.group(2)))
            continue


def extract_pred_window(name):
    # name example:
    # ETTm2_96_96_ns_Transformer_ETTm2_ftM_sl96_ll48_pl96 ...
    parts = name.split("_")
    return int(parts[2])  # 2nd number after ETTm2


def extract_feature_type(name):
    # find _ft[S M]_
    m = re.search(r"_ft([SM])_", name)
    return m.group(1) if m else None


def extract_extended_and_iteration(name):
    # ends with ..._ext_[0-1]
    m = re.search(r"_ext_([0-1]+)", name)
    if m:
        return int(m.group(1))
    return None, None


rows = []
# print(groups)
for name, vals in groups.items():

    mse_mean = float(np.mean(vals["mse"])) if vals["mse"] else None
    mae_mean = float(np.mean(vals["mae"])) if vals["mae"] else None
    mse_ab_mean = float(np.mean(vals["mse_abrupt"])) if vals["mse_abrupt"] else None
    mae_ab_mean = float(np.mean(vals["mae_abrupt"])) if vals["mae_abrupt"] else None

    pred_window = extract_pred_window(name)
    feature_type = extract_feature_type(name)
    extended = extract_extended_and_iteration(name)

    rows.append([
        pred_window,
        extended,
        feature_type,
        np.round(mse_mean, 4),
        np.round(mse_ab_mean, 4),
        np.round(mae_mean, 4),
        np.round(mae_ab_mean, 4)
    ])

# Sort logically
rows.sort(key=lambda x: (x[0], x[1], x[2]))

# Write CSV
with open(out_csv, "w", newline="", encoding="utf-8") as cf:
    writer = csv.writer(cf)
    writer.writerow([
        "pred_window", "extended", "feature_type",
        "mse", "mse_abrupt", "mae", "mae_abrupt"
    ])
    writer.writerows(rows)

print(f"Saved to: {out_csv}")