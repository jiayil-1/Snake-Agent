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
    start_scores = []

    with csv_path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            games.append(int(row['game']))
            scores.append(float(row['score']))
            # Games logged before start scores were recorded have no value.
            start_score = row.get('start_score')
            start_scores.append(float(start_score) if start_score else None)

    return games, scores, start_scores


def calculate_rolling_mean(scores, window_size=100):
    recent_scores = deque(maxlen=window_size)
    mean_scores = []
    for score in scores:
        recent_scores.append(score)
        mean_scores.append(sum(recent_scores) / len(recent_scores))
    return mean_scores


def plot_series(ax, games, values, label):
    mean_values = calculate_rolling_mean(values)
    ax.scatter(games, values, label=label, s=6, alpha=0.45, linewidths=0)
    ax.plot(games, mean_values, label=f"Mean {label}")
    ax.set_ylabel(label)
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax.annotate(
        f"{values[-1]:.2f}",
        (games[-1], values[-1]),
        xytext=(-5, 8),
        textcoords="offset points",
        ha="right",
        va="bottom",
    )
    ax.annotate(
        f"{mean_values[-1]:.2f}",
        (games[-1], mean_values[-1]),
        xytext=(-5, -10),
        textcoords="offset points",
        ha="right",
        va="top",
    )


def plot_training_data(games, scores, start_scores, output_path: Path):
    # Score gained is final minus starting score, so long starts are not
    # credited for length the agent never earned.
    gained = [
        (game, score - start)
        for game, score, start in zip(games, scores, start_scores)
        if start is not None
    ]
    if gained:
        fig, (ax, gained_ax) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    else:
        fig, ax = plt.subplots(figsize=(10, 6))

    plot_series(ax, games, scores, "Score")
    ax.set_title("Snake Training Scores")
    ax.set_ylim(bottom=min(0, *scores))

    if gained:
        gained_games, gained_scores = zip(*gained)
        plot_series(gained_ax, gained_games, gained_scores, "Score Gained")
        gained_ax.set_title("Score Gained (Final - Starting Score)")
        gained_ax.set_ylim(bottom=0)
        gained_ax.set_xlabel("Game")
    else:
        ax.set_xlabel("Game")

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

    games, scores, start_scores = load_training_data(csv_path)
    if not games:
        raise ValueError(f"CSV has no rows: {csv_path}")

    plot_training_data(games, scores, start_scores, Path(args.out))
    print(f"Saved plot to {args.out}")


if __name__ == "__main__":
    main()
