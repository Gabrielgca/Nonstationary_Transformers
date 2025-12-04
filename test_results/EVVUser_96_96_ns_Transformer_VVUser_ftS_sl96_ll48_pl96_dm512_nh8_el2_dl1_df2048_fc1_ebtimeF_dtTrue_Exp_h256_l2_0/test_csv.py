# import pandas as pd
# import ast
# import matplotlib.pyplot as plt
# import numpy as np

# csv_path = "participant_0_results.csv"

# # Load the CSV
# df = pd.read_csv(csv_path)

# # Convert list strings back to Python lists
# def parse_list(x):
#     return np.array(ast.literal_eval(x), dtype=float)

# df["input"] = df["input"].apply(parse_list)
# df["true"] = df["true"].apply(parse_list)
# df["pred"] = df["pred"].apply(parse_list)

# # -------- Select a row to visualize --------
# # For example: first row
# row = df.iloc[0]

# input_degree = row["input"]
# true_degree = row["true"]
# pred_degree = row["pred"]

# participant_id = int(row["participant_id"])
# iteration = int(row["iteration"])
# sub_sampling = int(row["sub_sampling"])

# # -------- Build the same sequences like your visual(gt, pd, ...) --------
# gt = np.concatenate((input_degree, true_degree), axis=0)
# pd = np.concatenate((input_degree, pred_degree), axis=0)

# # -------- Plot just like your code --------
# plt.figure(figsize=(10,4))
# plt.plot(gt, label="GT")
# plt.plot(pd, label="Pred")
# plt.title(f"Participant {participant_id}, Iteration {iteration}, sub_sampling={sub_sampling}")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()
# import pandas as pd
# import ast
# import numpy as np
# import matplotlib.pyplot as plt
# import os
# import glob

# # -------------------------------------------------------
# # Folder that contains many participant_X_results.csv
# # -------------------------------------------------------
# base_folder = "./test_results/EVVUser_96_96_ns_Transformer_VVUser_ftS_sl96_ll48_pl96_dm512_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_Exp_h256_l2_0"

# # Find all participant result files
# csv_files = glob.glob(os.path.join(base_folder, "participant_*_results.csv"))

# if len(csv_files) == 0:
#     print("No participant_*_results.csv files found.")
#     exit()

# # -------------------------------------------------------
# # Process each CSV file independently
# # -------------------------------------------------------
# for csv_path in csv_files:

#     print(f"\nProcessing: {csv_path}")

#     df = pd.read_csv(csv_path)

#     # Convert list strings to arrays
#     def parse_list(x):
#         return np.array(ast.literal_eval(x), dtype=float)

#     df["sub_sampling"] = df["sub_sampling"].map(
#         {"True": 1, "False": 0, True: 1, False: 0}
#     )

#     df["input"] = df["input"].apply(parse_list)
#     df["true"] = df["true"].apply(parse_list)
#     df["pred"] = df["pred"].apply(parse_list)
#     df["t_input"] = df["t_input"].apply(parse_list)
#     df["t_true"] = df["t_true"].apply(parse_list)
#     df["t_pred"] = df["t_pred"].apply(parse_list)

#     # Output folder = same folder of CSV
#     output_folder = os.path.dirname(csv_path)

#     # -------------------------------------------------------
#     # Process each participant + iteration that has both runs
#     # -------------------------------------------------------

#     participants = df["participant_id"].unique()

#     for pid in participants:
#         df_pid = df[df.participant_id == pid]

#         for iteration in df_pid["iteration"].unique():
#             subset = df_pid[df_pid.iteration == iteration]

#             # Need both sub_sampling=0 and 1
#             if len(subset) < 2:
#                 continue

#             if not (0 in subset.sub_sampling.values and 1 in subset.sub_sampling.values):
#                 continue

#             row0 = subset[subset.sub_sampling == 0].iloc[0]
#             row1 = subset[subset.sub_sampling == 1].iloc[0]

#             # ===== Build sequences (no subsampling) =====
#             input0 = row0["input"]
#             true0  = row0["true"]
#             pred0  = row0["pred"]

#             t0_input = row0["t_input"]
#             t0_true  = row0["t_true"]
#             t0_pred  = row0["t_pred"]

#             t0 = t0_true
#             gt0 = np.concatenate([input0, true0])

#             pred_start_idx0 = len(gt0) - len(pred0)
#             pd0 = np.concatenate([input0, pred0])

#             # ===== With subsampling =====
#             input1 = row1["input"]
#             true1  = row1["true"]
#             pred1  = row1["pred"]

#             t1_input = row1["t_input"]
#             t1_true  = row1["t_true"]
#             t1_pred  = row1["t_pred"]

#             t1 = t1_true
#             gt1 = np.concatenate([input1, true1])

#             pred_start_idx1 = len(gt1) - len(pred1)
#             pd1 = np.concatenate([input1, pred1])

#             # -------------------------------------------------------
#             # Create Plot
#             # -------------------------------------------------------
#             plt.figure(figsize=(12,5))

#             # ===== No subsampling =====
#             plt.plot(t0, gt0, label="GT (no subsampling)", linewidth=1.5)
#             plt.scatter(t0, gt0, marker='o', s=20)

#             plt.plot(t0[pred_start_idx0:], pd0[-len(pred0):],
#                      label="Pred (no subsampling)", linestyle="--")
#             plt.scatter(t0[pred_start_idx0:], pd0[-len(pred0):],
#                         marker='x', s=25)

#             # ===== With subsampling =====
#             plt.plot(t1, gt1, label="GT (sub_sampling=1)", linewidth=1.5)
#             plt.scatter(t1, gt1, marker='o', s=20)

#             plt.plot(t1[pred_start_idx1:], pd1[-len(pred1):],
#                      label="Pred (sub_sampling=1)", linestyle="--")
#             plt.scatter(t1[pred_start_idx1:], pd1[-len(pred1):],
#                         marker='x', s=25)

#             plt.title(f"Participant {pid}, Iteration {iteration}")
#             plt.xlabel("Timestamp")
#             plt.ylabel("Angle (deg)")
#             plt.legend()
#             plt.grid(True)
#             plt.tight_layout()

#             # -------------------------------------------------------
#             # Save figure next to the CSV file
#             # -------------------------------------------------------
#             save_path = os.path.join(
#                 output_folder,
#                 f"pid_{pid}_iter_{iteration}.png"
#             )

#             plt.savefig(save_path, dpi=200)
#             plt.close()

#             print(f"Saved: {save_path}")


# import pandas as pd
# import ast
# import numpy as np
# import matplotlib.pyplot as plt
# import os
# import glob

# # -------------------------------------------------------
# # Folder that contains many participant_X_results.csv
# # -------------------------------------------------------
# base_folder = "./test_results/EVVUser_96_96_ns_Transformer_VVUser_ftS_sl96_ll48_pl96_dm512_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_Exp_h256_l2_0"

# # Find all participant result files
# csv_files = glob.glob(os.path.join(base_folder, "participant_*_results.csv"))

# if len(csv_files) == 0:
#     print("No participant_*_results.csv files found.")
#     exit()

# # -------------------------------------------------------
# # Process each CSV file independently
# # -------------------------------------------------------
# for csv_path in csv_files:

#     print(f"\nProcessing: {csv_path}")

#     df = pd.read_csv(csv_path)

#     # Convert list strings to arrays
#     def parse_list(x):
#         return np.array(ast.literal_eval(x), dtype=float)

#     df["sub_sampling"] = df["sub_sampling"].map(
#         {"True": 1, "False": 0, True: 1, False: 0}
#     )

#     df["input"] = df["input"].apply(parse_list)
#     df["true"] = df["true"].apply(parse_list)
#     df["pred"] = df["pred"].apply(parse_list)
#     df["t_input"] = df["t_input"].apply(parse_list)
#     df["t_true"] = df["t_true"].apply(parse_list)
#     df["t_pred"] = df["t_pred"].apply(parse_list)

#     # Output folder = same folder of CSV
#     output_folder = os.path.dirname(csv_path)

#     # -------------------------------------------------------
#     # NEW LOGIC: pair by appearance order, not iteration
#     # -------------------------------------------------------
#     participants = df["participant_id"].unique()

#     for pid in participants:

#         df_pid = df[df.participant_id == pid].sort_index()

#         rows_no_sub = df_pid[df_pid.sub_sampling == 0].reset_index(drop=True)
#         rows_yes_sub = df_pid[df_pid.sub_sampling == 1].reset_index(drop=True)

#         pair_count = min(len(rows_no_sub), len(rows_yes_sub))

#         print(f"Participant {pid}: pairing {pair_count} sequences")

#         for idx in range(pair_count):

#             row0 = rows_no_sub.iloc[idx]   # sub_sampling = 0
#             row1 = rows_yes_sub.iloc[idx]  # sub_sampling = 1

#             iter0 = int(row0["iteration"])
#             iter1 = int(row1["iteration"])

#             print(f" → Pair {idx}: (0-sub iter={iter0})  with  (1-sub iter={iter1})")

#             # ===== Build sequences (no subsampling) =====
#             input0 = row0["input"]
#             true0  = row0["true"]
#             pred0  = row0["pred"]

#             t0 = row0["t_true"]
#             gt0 = np.concatenate([input0, true0])

#             pred_start_idx0 = len(gt0) - len(pred0)
#             pd0 = np.concatenate([input0, pred0])

#             # ===== With subsampling =====
#             input1 = row1["input"]
#             true1  = row1["true"]
#             pred1  = row1["pred"]

#             t1 = row1["t_true"]
#             gt1 = np.concatenate([input1, true1])

#             pred_start_idx1 = len(gt1) - len(pred1)
#             pd1 = np.concatenate([input1, pred1])

#             # -------------------------------------------------------
#             # PLOT
#             # -------------------------------------------------------
#             plt.figure(figsize=(12, 5))

#             # ===== No subsampling =====
#             plt.plot(t0, gt0, label="GT (no subsampling)", linewidth=1.5)
#             plt.scatter(t0, gt0, marker='o', s=20)

#             plt.plot(t0, pd0,
#                      label="Pred (no subsampling)", linestyle="--")
#             plt.scatter(t0, pd0,
#                         marker='x', s=25)

#             # ===== With subsampling =====
#             plt.plot(t1, gt1, label="GT (sub_sampling=1)", linewidth=1.5)
#             plt.scatter(t1, gt1, marker='o', s=20)

#             plt.plot(t1, pd1,
#                      label="Pred (sub_sampling=1)", linestyle="--")
#             plt.scatter(t1, pd1,
#                         marker='x', s=25)

#             plt.title(f"Participant {pid}, Pair {idx} → (iter {iter0} vs {iter1})")
#             plt.xlabel("Timestamp")
#             plt.ylabel("Angle (deg)")
#             plt.legend()
#             plt.grid(True)
#             plt.tight_layout()

#             # Save
#             save_path = os.path.join(
#                 output_folder,
#                 f"pid_{pid}_pair_{idx}_iter_{iter0}_vs_{iter1}.png"
#             )

#             plt.savefig(save_path, dpi=200)
#             plt.close()

#             print(f"Saved: {save_path}")



import pandas as pd
import ast
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

# -------------------------------------------------------
# Folder that contains many participant_X_results.csv
# -------------------------------------------------------
base_folder = "./test_results/EVVUser_96_96_ns_Transformer_VVUser_ftS_sl96_ll48_pl96_dm512_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_Exp_h256_l2_0"

# Find all participant result files
csv_files = glob.glob(os.path.join(base_folder, "participant_*_results.csv"))

if len(csv_files) == 0:
    print("No participant_*_results.csv files found.")
    exit()

# -------------------------------------------------------
# Process each CSV file independently
# -------------------------------------------------------
for csv_path in csv_files:

    print(f"\nProcessing: {csv_path}")

    df = pd.read_csv(csv_path)

    # Convert list strings to arrays
    def parse_list(x):
        return np.array(ast.literal_eval(x), dtype=float)

    df["sub_sampling"] = df["sub_sampling"].map(
        {"True": 1, "False": 0, True: 1, False: 0}
    )

    df["input"] = df["input"].apply(parse_list)
    df["true"] = df["true"].apply(parse_list)
    df["pred"] = df["pred"].apply(parse_list)
    df["t_input"] = df["t_input"].apply(parse_list)
    df["t_true"] = df["t_true"].apply(parse_list)
    df["t_pred"] = df["t_pred"].apply(parse_list)

    # Output folder = same folder of CSV
    output_folder = os.path.dirname(csv_path)

    # -------------------------------------------------------
    # NEW LOGIC: pair by closest starting t_true
    # -------------------------------------------------------
    participants = df["participant_id"].unique()

    for pid in participants:

        df_pid = df[df.participant_id == pid]

        rows_no_sub = df_pid[df_pid.sub_sampling == 0].reset_index(drop=True)
        rows_yes_sub = df_pid[df_pid.sub_sampling == 1].reset_index(drop=True)

        print(f"\nParticipant {pid}: matching by closest t_true start")

        for idx0 in range(len(rows_no_sub)):

            row0 = rows_no_sub.iloc[idx0]   # sub_sampling = 0
            t0_start = row0["t_true"][0]

            # ---- find the row with sub_sampling=1 with closest t_true[0] ----
            best_row1 = None
            best_dist = float("inf")
            best_idx1 = None

            for idx1 in range(len(rows_yes_sub)):
                row1 = rows_yes_sub.iloc[idx1]
                t1_start = row1["t_true"][0]
                dist = abs(t0_start - t1_start)

                if dist < best_dist:
                    best_dist = dist
                    best_row1 = row1
                    best_idx1 = idx1

            if best_row1 is None:
                continue

            iter0 = int(row0["iteration"])
            iter1 = int(best_row1["iteration"])

            print(f"  Pair: 0-sub iter={iter0} matched with 1-sub iter={iter1}, Δt = {best_dist}")

            # ===== Build sequences (no subsampling) =====
            input0 = row0["input"]
            true0  = row0["true"]
            pred0  = row0["pred"]
            t0 = row0["t_true"]

            gt0 = np.concatenate([input0, true0])
            pd0 = np.concatenate([input0, pred0])

            # ===== With subsampling =====
            input1 = best_row1["input"]
            true1  = best_row1["true"]
            pred1  = best_row1["pred"]
            t1 = best_row1["t_true"]

            gt1 = np.concatenate([input1, true1])
            pd1 = np.concatenate([input1, pred1])

            # -------------------------------------------------------
            # PLOT
            # -------------------------------------------------------
            plt.figure(figsize=(12, 5))

            # ---- no subsampling ----
            plt.plot(t0, gt0, label="GT (no subsampling)")
            plt.scatter(t0, gt0, s=12)

            plt.plot(t0, pd0, label="Pred (no subsampling)", linestyle="--")
            plt.scatter(t0, pd0, s=20)

            # ---- with subsampling ----
            plt.plot(t1, gt1, label="GT (sub_sampling=1)")
            plt.scatter(t1, gt1, s=12)

            plt.plot(t1, pd1, label="Pred (sub_sampling=1)", linestyle="--")
            plt.scatter(t1, pd1, s=20)

            plt.title(f"Participant {pid}, best match → iter {iter0} vs {iter1}")
            plt.xlabel("Timestamp")
            plt.ylabel("Angle (deg)")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()

            # Save
            save_path = os.path.join(
                output_folder,
                f"pid_{pid}_bestmatch_iter_{iter0}_vs_{iter1}.png"
            )

            plt.savefig(save_path, dpi=200)
            plt.close()

            print(f"  Saved: {save_path}")


