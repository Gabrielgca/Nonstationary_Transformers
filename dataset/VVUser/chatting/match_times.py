import pandas as pd
import numpy as np

# --- Read files ---
df1 = pd.read_csv("simTest.csv")
df2 = pd.read_csv("0_VVUser.csv")

# --- Step 1: filter source and adjust time ---
df1_filtered = df1[df1["Source"] == "00:00:00_00:00:01"].copy()

df1_filtered["AdjTime"] = df1_filtered["Time"].astype(float) - 5

# --- Step 2: prepare df2 ---
df2["Timer"] = df2["Timer"].astype(float)

# --- Step 3: find closest Timer for each adjusted time per participant ---
matched_rows = []

for adj_time in df1_filtered["AdjTime"]:
    df2["Diff"] = (df2["Timer"] - adj_time).abs()
    closest_idx = df2.groupby("ParticipantID")["Diff"].idxmin()
    matched_rows.append(df2.loc[closest_idx])

# Combine
result = pd.concat(matched_rows)

# --- Remove duplicates based on ParticipantID + Timer ---
result = result.drop_duplicates(subset=["ParticipantID", "Timer"])

# Remove helper column
result = result.drop(columns=["Diff"], errors="ignore")

# --- Sort by ParticipantID then Timer ---
result = result.sort_values(by=["ParticipantID", "Timer"]).reset_index(drop=True)

# --- Save ---
result.to_csv("matched_output.csv", index=False)
