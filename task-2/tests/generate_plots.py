import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

# === Font Settings ===
# Customizing the font and label size for consistency across the plots
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["xtick.labelsize"] = 14
plt.rcParams["ytick.labelsize"] = 14
plt.rcParams["legend.fontsize"] = 14

# === Load CSVs ===
# Set the directory to where the result files are located
DATA_DIR = "test-results"
# Grab all CSV files in the directory
csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))

# Label mapping and plot styling configurations
label_map = {
    "base": "Baseline",
    "batched": "Batched",
    "scaling": "Scaled-Balanced"
}
colors = ['tab:blue', 'tab:orange', 'tab:green']  # Color scheme for different types
markers = ['o', 's', '^']  # Marker style for each line

# # === Part 1: Line Plots (Including Poisson Scaled-Balanced) ===
def plot_line(metric, ylabel, filename):
    """Plots the line graphs for the specified metric (e.g., latency or throughput)"""
    plt.figure(figsize=(10, 6))

    # Loop through different experiment types: base, batched, and scaling
    for idx, key in enumerate(label_map.keys()):
        for mode in ["Ideal", "Poisson"]:  # We plot both Ideal and Poisson results
            match_str = f"{key}{mode}.csv"
            file_path = os.path.join(DATA_DIR, match_str)
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                label = f"{label_map[key]} - {mode}"
                # Plot the data with the chosen markers, colors, and line styles
                plt.plot(df['rps'], df[metric],
                         label=label,
                         marker=markers[idx % len(markers)],
                         color=colors[idx % len(colors)],
                         linestyle='-' if mode == "Ideal" else '--')

    # === Adding the Poisson Scaled-Balanced data ===
    # Include the special scaled-balanced Poisson data
    scaling_poisson_file = os.path.join(DATA_DIR, "scalingPoisson.csv")
    if os.path.exists(scaling_poisson_file):
        df_scaling_poisson = pd.read_csv(scaling_poisson_file)
        plt.plot(df_scaling_poisson['rps'], df_scaling_poisson[metric],
                 label="Scaled-Balanced - Poisson",
                 marker='^', color='tab:green', linestyle='--')

    # Formatting the plot (labels, grid, title)
    plt.xlabel("Input RPS", fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.title(f"{ylabel} vs Input RPS", fontsize=18)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)  # Save the plot
    plt.close()

# Generate line plots for latency and throughput
plot_line("average_latency", "Mean Latency (s)", "latency_vs_rps.png")
plot_line("throughput_rps", "Throughput (RPS)", "throughput_vs_rps.png")


# === Part 2: Bar Plots (Ideal vs Poisson) ===
def plot_bars(metric, ylabel, filename_prefix):
    """Generates bar plots for Ideal vs Poisson comparisons"""
    for key in label_map.keys():
        ideal_file = os.path.join(DATA_DIR, f"{key}Ideal.csv")
        poisson_file = os.path.join(DATA_DIR, f"{key}Realistic.csv")  # "Realistic" is Poisson

        if not (os.path.exists(ideal_file) and os.path.exists(poisson_file)):
            continue

        df_ideal = pd.read_csv(ideal_file)
        df_poisson = pd.read_csv(poisson_file)

        rps_values = df_ideal['rps']
        x = np.arange(len(rps_values))
        bar_width = 0.35  # Set width of the bars for plotting

        plt.figure(figsize=(12, 6))
        # Plot bar charts for Ideal and Poisson comparison
        plt.bar(x - bar_width/2, df_ideal[metric], width=bar_width, label="Ideal", color="tab:blue")
        plt.bar(x + bar_width/2, df_poisson[metric], width=bar_width, label="Poisson", color="tab:orange")

        # Set labels and title for the plot
        plt.xticks(x, rps_values)
        plt.xlabel("Input RPS", fontsize=16)
        plt.ylabel(ylabel, fontsize=16)
        plt.title(f"{label_map[key]} - {ylabel} (Ideal vs Poisson)", fontsize=18)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)  # Gridlines for clarity
        plt.tight_layout()
        plt.savefig(f"{filename_prefix}_{key}.png", dpi=300)  # Save each plot
        plt.close()

# Generate bar plots for latency and throughput
plot_bars("average_latency", "Mean Latency (s)", "bar_latency")
plot_bars("throughput_rps", "Throughput (RPS)", "bar_throughput")

# === Part 3: Line Plot for Batched Ideal 2 vs Batched Ideal 1 vs Scaling Ideal ===
def plot_batched_comparison():
    """Plots the comparison between different batch sizes and scaling ideal"""
    # Files for comparison
    batched_ideal_1 = os.path.join(DATA_DIR, "batchedIdeal.csv")
    batched_ideal_2 = os.path.join(DATA_DIR, "batchedIdeal2.csv")
    scaling_ideal = os.path.join(DATA_DIR, "scalingIdeal.csv")

    # Check if all files exist before plotting
    if not (os.path.exists(batched_ideal_1) and os.path.exists(batched_ideal_2) and os.path.exists(scaling_ideal)):
        print("One or more files are missing.")
        return

    # Read CSV files for each condition
    df_batched_ideal_1 = pd.read_csv(batched_ideal_1)
    df_batched_ideal_2 = pd.read_csv(batched_ideal_2)
    df_scaling_ideal = pd.read_csv(scaling_ideal)

    rps_values = df_batched_ideal_1['rps']

    # Plot comparison for three conditions
    plt.figure(figsize=(12, 6))
    plt.plot(rps_values, df_batched_ideal_1['average_latency'], label="Batched Ideal 1 (Batch Size 10)", color="tab:blue", marker='o')
    plt.plot(rps_values, df_batched_ideal_2['average_latency'], label="Batched Ideal 2 (Batch Size 20)", color="tab:orange", marker='s')
    plt.plot(rps_values, df_scaling_ideal['average_latency'], label="Scaling Ideal", color="tab:green", marker='^')

    plt.xlabel("Input RPS", fontsize=16)
    plt.ylabel("Mean Latency (s)", fontsize=16)
    plt.title("Latency Comparison: Batch Size 10 Ideal vs Batch Size 20 Ideal 2 vs Scaling Ideal", fontsize=18)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("latency_batched_comparison.png", dpi=300)
    plt.close()

# Generate the batched comparison plot
plot_batched_comparison()

def plot_poisson_latency_only():
    """Generates the latency plot only for Poisson distribution"""
    plt.figure(figsize=(10, 6))

    for idx, key in enumerate(label_map.keys()):
        file_path = os.path.join(DATA_DIR, f"{key}Realistic.csv")  # Realistic data is Poisson
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
