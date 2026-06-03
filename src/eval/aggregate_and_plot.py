import argparse
import glob
import json
import os

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np


def plot_positional_analysis(results: dict, output_path: str = "GRAIL_positional.png"):
    steps = sorted(results.keys())
    if not steps:
        return

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    colours = cm.viridis(np.linspace(0.15, 0.9, len(steps)))

    ax = axes[0]
    for colour, step in zip(colours, steps):
        res = results[step]
        x, y, sem = (
            np.array(res["bin_centres"]),
            np.array(res["mean_weight"]),
            np.array(res["sem_weight"]),
        )
        mask = ~np.isnan(y)
        ax.plot(x[mask], y[mask], color=colour, label=f"step {step}", linewidth=1.8)
        ax.fill_between(
            x[mask], (y - sem)[mask], (y + sem)[mask], color=colour, alpha=0.12
        )

    ax.axhline(y=1.0, color="black", linestyle="--", linewidth=0.8, label="uniform")
    ax.set_xlabel("Normalised position in reasoning span")
    ax.set_ylabel("Mean GRAIL weight")
    ax.set_title("Positional Weight Profile over Training")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(alpha=0.3)

    ax = axes[1]
    step = steps[-1]
    res = results[step]
    x = np.array(res["bin_centres"])
    y_c, sem_c = np.array(res["mean_weight_correct"]), np.array(
        res["sem_weight_correct"]
    )
    y_w, sem_w = np.array(res["mean_weight_wrong"]), np.array(res["sem_weight_wrong"])
    mask_c, mask_w = ~np.isnan(y_c), ~np.isnan(y_w)

    ax.plot(
        x[mask_c],
        y_c[mask_c],
        color="steelblue",
        label="Correct rollouts",
        linewidth=1.8,
    )
    ax.fill_between(
        x[mask_c],
        (y_c - sem_c)[mask_c],
        (y_c + sem_c)[mask_c],
        color="steelblue",
        alpha=0.15,
    )
    ax.plot(
        x[mask_w], y_w[mask_w], color="tomato", label="Wrong rollouts", linewidth=1.8
    )
    ax.fill_between(
        x[mask_w],
        (y_w - sem_w)[mask_w],
        (y_w + sem_w)[mask_w],
        color="tomato",
        alpha=0.15,
    )
    ax.axhline(y=1.0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Normalised position in reasoning span")
    ax.set_ylabel("Mean GRAIL weight")
    ax.set_title(f"Correct vs Wrong Rollout Weights (step {step})")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[2]
    n_bins = len(results[steps[0]]["bin_centres"])
    heat = np.full((len(steps), n_bins), np.nan)
    for i, s in enumerate(steps):
        heat[i, :] = np.array(results[s]["mean_weight"])

    im = ax.imshow(
        heat,
        aspect="auto",
        origin="lower",
        cmap="RdBu_r",
        vmin=0.6,
        vmax=1.4,
        extent=[0, 1, 0, len(steps)],
    )
    ax.set_yticks(np.arange(len(steps)) + 0.5)
    ax.set_yticklabels([str(s) for s in steps], fontsize=7)
    ax.set_xlabel("Normalised position in reasoning span")
    ax.set_ylabel("Training step")
    ax.set_title("Weight Heatmap")
    plt.colorbar(im, ax=ax, label="Mean GRAIL weight")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Positional analysis plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stats_dir",
        type=str,
        required=True,
        help="Directory containing the per-checkpoint stats",
    )
    parser.add_argument("--output_dir", type=str, default="token_analysis_plots")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    full_pos_summary = {}

    pos_files = glob.glob(os.path.join(args.stats_dir, "checkpoint_*_pos_summary.json"))

    for f in pos_files:
        step = int(f.split("checkpoint_")[1].split("_pos_summary")[0])
        with open(f, "r") as fh:
            full_pos_summary[step] = json.load(fh)

    if full_pos_summary:
        pos_json_path = os.path.join(
            args.output_dir, "aggregated_positional_summary.json"
        )

        pos_str_summary = {str(k): v for k, v in full_pos_summary.items()}
        with open(pos_json_path, "w") as f:
            json.dump(pos_str_summary, f, indent=2)
        plot_positional_analysis(
            full_pos_summary, os.path.join(args.output_dir, "GRAIL_positional.png")
        )


if __name__ == "__main__":
    main()
