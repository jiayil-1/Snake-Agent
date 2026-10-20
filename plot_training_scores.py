import argparse
import csv
from collections import deque
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_training_data(csv_path: Path):
    games = []
    scores = []

    with csv_path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            games.append(int(row['game']))
            scores.append(float(row['score']))

    return games, scores


def calculate_rolling_mean(scores, window_size=100):
    recent_scores = deque(maxlen=window_size)
    mean_scores = []
    for score in scores:
        recent_scores.append(score)
        mean_scores.append(sum(recent_scores) / len(recent_scores))
    return mean_scores


def plot_training_data(games, scores, output_path: Path):
    mean_scores = calculate_rolling_mean(scores)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(games, scores, label="Score", s=6, alpha=0.45, linewidths=0)
    ax.plot(games, mean_scores, label="Mean Score")

    ax.set_title("Snake Training Scores")
    ax.set_xlabel("Game")
    ax.set_ylabel("Score")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend()

    if games:
        ax.annotate(
            f"{scores[-1]:.2f}",
            (games[-1], scores[-1]),
            xytext=(-5, 8),
            textcoords="offset points",
            ha="right",
            va="bottom",
        )
        ax.annotate(
            f"{mean_scores[-1]:.2f}",
            (games[-1], mean_scores[-1]),
            xytext=(-5, -10),
            textcoords="offset points",
            ha="right",
            va="top",
        )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot training scores from CSV to PNG")
    parser.add_argument("--csv", default="training_scores.csv", help="Input CSV path")
    parser.add_argument(
        "--out",
        default="training_scores_plot.png",
        help="Output PNG path",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    games, scores = load_training_data(csv_path)
    if not games:
        raise ValueError(f"CSV has no rows: {csv_path}")

    plot_training_data(games, scores, Path(args.out))
    print(f"Saved plot to {args.out}")


if __name__ == "__main__":
    main()
