import matplotlib.pyplot as plt
import csv
import os


def _ensure_end_reason_column(csv_path):
    csv_path = os.fspath(csv_path)
    with open(csv_path, newline='') as csv_file:
        rows = list(csv.reader(csv_file))

    if not rows or 'end_reason' in rows[0]:
        return

    rows[0].append('end_reason')
    rows[1:] = [row + [''] for row in rows[1:]]

    temp_path = f'{csv_path}.tmp.{os.getpid()}'
    try:
        with open(temp_path, 'w', newline='') as csv_file:
            csv.writer(csv_file).writerows(rows)
        os.chmod(temp_path, os.stat(csv_path).st_mode & 0o777)
        os.replace(temp_path, csv_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def plot(scores, mean_scores, csv_path=None, png_path=None, hidden=True,
         game_number_offset=0, end_reason=''):
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

    if csv_path and scores:
        file_exists = os.path.exists(csv_path)
        file_is_empty = (not file_exists) or os.path.getsize(csv_path) == 0

        if file_is_empty:
            with open(csv_path, 'a', newline='') as csv_file:
                csv.writer(csv_file).writerow(['game', 'score', 'end_reason'])
        else:
            _ensure_end_reason_column(csv_path)

        game_index = game_number_offset + len(scores)
        with open(csv_path, 'a', newline='') as csv_file:
            csv.writer(csv_file).writerow([game_index, scores[-1], end_reason])

    if png_path:
        if hidden:
            fig.savefig(png_path, dpi=150, bbox_inches='tight')
        else:
            plt.gcf().savefig(png_path, dpi=150, bbox_inches='tight')

    if hidden:
        plt.close(fig)
