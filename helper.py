import matplotlib.pyplot as plt
import csv
import os

def plot(scores, mean_scores, csv_path=None, png_path=None, hidden=True):
    if hidden:
        plt.ioff()
        fig, ax = plt.subplots()
        ax.set_title('Training...')
        ax.set_xlabel('Number of Games')
        ax.set_ylabel('Score')
        ax.plot(scores)
        ax.plot(mean_scores)
        ax.set_ylim(ymin=0)
        ax.text(len(scores)-1, scores[-1], str(scores[-1]))
        ax.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    else:
        plt.ion()
        plt.clf()
        plt.title('Training...')
        plt.xlabel('Number of Games')
        plt.ylabel('Score')
        plt.plot(scores)
        plt.plot(mean_scores)
        plt.ylim(ymin=0)
        plt.text(len(scores)-1, scores[-1], str(scores[-1]))
        plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
        plt.show(block=False)
        plt.pause(.1)

    if csv_path and scores and mean_scores:
        file_exists = os.path.exists(csv_path)
        file_is_empty = (not file_exists) or os.path.getsize(csv_path) == 0

        with open(csv_path, 'a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            if file_is_empty:
                writer.writerow(['game', 'score', 'mean_score'])

            game_index = min(len(scores), len(mean_scores))
            writer.writerow([game_index, scores[-1], mean_scores[-1]])

    if png_path:
        if hidden:
            fig.savefig(png_path, dpi=150, bbox_inches='tight')
        else:
            plt.gcf().savefig(png_path, dpi=150, bbox_inches='tight')

    if hidden:
        plt.close(fig)
