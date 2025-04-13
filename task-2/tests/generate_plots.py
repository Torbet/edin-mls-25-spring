import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

# === Font Settings ===
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["xtick.labelsize"] = 14
plt.rcParams["ytick.labelsize"] = 14
plt.rcParams["legend.fontsize"] = 14

# === Load CSVs ===
DATA_DIR = "test-results"
csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))

label_map = {
    "base": "Baseline",
    "batched": "Batched",
    "scaling": "Scaled-Balanced"
}
colors = ['tab:blue', 'tab:orange', 'tab:green']
markers = ['o', 's', '^']

# # === Part 1: Line Plots (Including Poisson Scaled-Balanced) ===
# def plot_line(metric, ylabel, filename):
#     plt.figure(figsize=(10, 6))

#     for idx, key in enumerate(label_map.keys()):
#         for mode in ["Ideal", "Poisson"]:  # Changed "Realistic" to "Poisson"
#             match_str = f"{key}{mode}.csv"
#             file_path = os.path.join(DATA_DIR, match_str)
#             if os.path.exists(file_path):
#                 df = pd.read_csv(file_path)
#                 label = f"{label_map[key]} - {mode}"
#                 plt.plot(df['rps'], df[metric],
#                          label=label,
#                          marker=markers[idx % len(markers)],
#                          color=colors[idx % len(colors)],
#                          linestyle='-' if mode == "Ideal" else '--')

#     # === Adding the Poisson Scaled-Balanced data ===
#     scaling_poisson_file = os.path.join(DATA_DIR, "scalingPoisson.csv")
#     if os.path.exists(scaling_poisson_file):
#         df_scaling_poisson = pd.read_csv(scaling_poisson_file)
#         plt.plot(df_scaling_poisson['rps'], df_scaling_poisson[metric],
#                  label="Scaled-Balanced - Poisson",
#                  marker='^', color='tab:green', linestyle='--')

#     plt.xlabel("Input RPS", fontsize=16)
#     plt.ylabel(ylabel, fontsize=16)
#     plt.title(f"{ylabel} vs Input RPS", fontsize=18)
#     plt.grid(True)
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(filename, dpi=300)
#     plt.close()

# # Generate plots
# plot_line("average_latency", "Mean Latency (s)", "latency_vs_rps.png")
# plot_line("throughput_rps", "Throughput (RPS)", "throughput_vs_rps.png")


# # Generate plots
# plot_line("average_latency", "Mean Latency (s)", "latency_vs_rps.png")
# plot_line("throughput_rps", "Throughput (RPS)", "throughput_vs_rps.png")

# # === Part 2: Bar Plots (Ideal vs Poisson) ===
# def plot_bars(metric, ylabel, filename_prefix):
#     for key in label_map.keys():
#         ideal_file = os.path.join(DATA_DIR, f"{key}Ideal.csv")
#         poisson_file = os.path.join(DATA_DIR, f"{key}Realistic.csv")  # Changed "Realistic" to "Poisson"

#         if not (os.path.exists(ideal_file) and os.path.exists(poisson_file)):
#             continue

#         df_ideal = pd.read_csv(ideal_file)
#         df_poisson = pd.read_csv(poisson_file)

#         rps_values = df_ideal['rps']
#         x = np.arange(len(rps_values))
#         bar_width = 0.35

#         plt.figure(figsize=(12, 6))
#         plt.bar(x - bar_width/2, df_ideal[metric], width=bar_width, label="Ideal", color="tab:blue")
#         plt.bar(x + bar_width/2, df_poisson[metric], width=bar_width, label="Poisson", color="tab:orange")

#         plt.xticks(x, rps_values)
#         plt.xlabel("Input RPS", fontsize=16)
#         plt.ylabel(ylabel, fontsize=16)
#         plt.title(f"{label_map[key]} - {ylabel} (Ideal vs Poisson)", fontsize=18)
#         plt.legend()
#         plt.grid(axis='y', linestyle='--', alpha=0.7)
#         plt.tight_layout()
#         plt.savefig(f"{filename_prefix}_{key}.png", dpi=300)
#         plt.close()

# # Generate bar plots
# plot_bars("average_latency", "Mean Latency (s)", "bar_latency")
# plot_bars("throughput_rps", "Throughput (RPS)", "bar_throughput")

# # === Part 3: Line Plot for Batched Ideal 2 vs Batched Ideal 1 vs Scaling Ideal ===
# def plot_batched_comparison():
#     # Files to compare
#     batched_ideal_1 = os.path.join(DATA_DIR, "batchedIdeal.csv")
#     batched_ideal_2 = os.path.join(DATA_DIR, "batchedIdeal2.csv")
#     scaling_ideal = os.path.join(DATA_DIR, "scalingIdeal.csv")

#     # Check if the files exist
#     if not (os.path.exists(batched_ideal_1) and os.path.exists(batched_ideal_2) and os.path.exists(scaling_ideal)):
#         print("One or more files are missing.")
#         return

#     # Read CSV files
#     df_batched_ideal_1 = pd.read_csv(batched_ideal_1)
#     df_batched_ideal_2 = pd.read_csv(batched_ideal_2)
#     df_scaling_ideal = pd.read_csv(scaling_ideal)

#     rps_values = df_batched_ideal_1['rps']

#     # Plot for Batched Ideal 1, Batched Ideal 2, and Scaling Ideal (Line Plot)
#     plt.figure(figsize=(12, 6))
#     plt.plot(rps_values, df_batched_ideal_1['average_latency'], label="Batched Ideal 1 (Batch Size 10)", color="tab:blue", marker='o')
#     plt.plot(rps_values, df_batched_ideal_2['average_latency'], label="Batched Ideal 2 (Batch Size 20)", color="tab:orange", marker='s')
#     plt.plot(rps_values, df_scaling_ideal['average_latency'], label="Scaling Ideal", color="tab:green", marker='^')

#     plt.xlabel("Input RPS", fontsize=16)
#     plt.ylabel("Mean Latency (s)", fontsize=16)
#     plt.title("Latency Comparison: Batch Size 10 Ideal vs Batch Size 20 Ideal 2 vs Scaling Ideal", fontsize=18)
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig("latency_batched_comparison.png", dpi=300)
#     plt.close()

# # Generate the line plot comparing batch sizes
# plot_batched_comparison()

def plot_poisson_latency_only():
    plt.figure(figsize=(10, 6))

    for idx, key in enumerate(label_map.keys()):
        file_path = os.path.join(DATA_DIR, f"{key}Realistic.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            label = f"{label_map[key]}"
            plt.plot(df['rps'], df['average_latency'],
                     label=label,
                     marker=markers[idx % len(markers)],
                     color=colors[idx % len(colors)],
                     linestyle='-')

    plt.xlabel("Input RPS Over 1000 Requests", fontsize=16)
    plt.ylabel("Mean Latency (s)", fontsize=16)
    plt.title("Mean Latency vs Input RPS - Poisson Distribution", fontsize=18)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("latency_vs_rps_poisson_only.png", dpi=300)
    plt.close()

# Generate Poisson-only latency plot
plot_poisson_latency_only()

