#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt

def load_training_curve(trainer_state_path: Path):
    with open(trainer_state_path, "r") as f:
        state = json.load(f)
    steps = []
    losses = []
    for entry in state.get("log_history", []):
        if "step" in entry and "loss" in entry:
            steps.append(entry["step"])
            losses.append(entry["loss"])
    return steps, losses

def load_validation_curve(mse_results_path: Path):
    ckpt_re = re.compile(r"checkpoint-(\d+):\s*([0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)")
    steps = []
    losses = []
    with open(mse_results_path, "r") as f:
        for line in f:
            m = ckpt_re.search(line.strip())
            if m:
                steps.append(int(m.group(1)))
                losses.append(float(m.group(2)))
    return steps, losses

def main():
    parser = argparse.ArgumentParser(description="Plot learning curve from trainer_state.json and mse_results.txt")
    parser.add_argument("--checkpoint-dir", type=str, required=True,
                        help="Directory containing trainer_state.json (e.g., outputs/.../checkpoint-5000)")
    parser.add_argument("--mse-results", type=str, required=True,
                        help="Path to mse_results.txt containing validation losses by checkpoint")
    parser.add_argument("--output", type=str, default="/workspace/learning_curve.png",
                        help="Output image path for the plot")
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    trainer_state_path = checkpoint_dir / "trainer_state.json"
    mse_results_path = Path(args.mse_results)

    if not trainer_state_path.exists():
        raise FileNotFoundError(f"trainer_state.json not found at {trainer_state_path}")
    if not mse_results_path.exists():
        raise FileNotFoundError(f"mse_results.txt not found at {mse_results_path}")

    train_steps, train_losses = load_training_curve(trainer_state_path)
    val_steps, val_losses = load_validation_curve(mse_results_path)

    plt.figure(figsize=(10, 6))
    if train_steps and train_losses:
        plt.plot(train_steps, train_losses, label="Training Loss", color="tab:blue", alpha=0.8)
    if val_steps and val_losses:
        plt.plot(val_steps, val_losses, label="Validation MSE", color="tab:orange", marker="o", linestyle="--", alpha=0.8)

    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Learning Curve")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    print(f"Saved plot to {out_path}")

if __name__ == "__main__":
    main()