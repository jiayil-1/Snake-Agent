import matplotlib.pyplot as plt
import csv
import os


SCORE_COLUMNS = ['game', 'score', 'end_reason', 'start_score']


def ensure_score_columns(csv_path):
    """Add any missing trailing columns to an older nonempty score CSV."""
    csv_path = os.fspath(csv_path)
    try:
        with open(csv_path, newline='') as csv_file:
            rows = list(csv.reader(csv_file))
    except FileNotFoundError:
        return

    if not rows:
        return
    missing = [column for column in SCORE_COLUMNS if column not in rows[0]]
    if not missing:
        return

    width = len(rows[0])
    rows[0].extend(missing)
    rows[1:] = [row + [''] * (width + len(missing) - len(row)) for row in rows[1:]]

    temp_path = f'{csv_path}.tmp.{os.getpid()}'
    try:
        with open(temp_path, 'w', newline='') as csv_file:
            csv.writer(csv_file).writerows(rows)
        os.chmod(temp_path, os.stat(csv_path).st_mode & 0o777)
        os.replace(temp_path, csv_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def append_score(csv_path, game_number, score, end_reason='', start_score=''):
    """Append one completed game, writing the header to a new or empty CSV."""
    write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    with open(csv_path, 'a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        if write_header:
            writer.writerow(SCORE_COLUMNS)
        writer.writerow([game_number, score, end_reason, start_score])


def plot(scores, mean_scores):
    """Show live training scores in an interactive window."""
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
