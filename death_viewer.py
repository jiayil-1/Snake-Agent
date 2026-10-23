"""Click through a training run's deaths, each drawn as the game showed its final frame.

Training appends every finished game's final board to ``runs/<run>/deaths.jsonl``.
The viewer follows that file, so games that finish while it is open join the list.
"""

import argparse
from pathlib import Path

import pygame

from configuration import REPO_ROOT, validate_name
from deaths import read_deaths, snake_cells
from game import BLACK, BLOCK_SIZE, BLUE1, BLUE2, HEAD, RED, WHITE

WINDOW = (18, 18, 22)
WALL = (52, 52, 60)
PANEL = (28, 28, 34)
SELECTED = (40, 70, 150)
HOVER = (44, 44, 54)
MUTED = (150, 150, 160)
RANDOM_MOVE = (90, 225, 120)
# Segments placed by an epsilon exploration: outer, inner, and head inner colours.
RANDOM_OUTER = (0, 170, 60)
RANDOM_INNER = (90, 235, 130)
RANDOM_HEAD = (190, 255, 205)
# The path through segment centres, translucent so segment colours still read through it.
PATH = (255, 255, 255, 150)
PATH_WIDTH = 4
REASONS = {
    'self-collision': ('self', 'Self-collision', (255, 160, 90)),
    'wall-collision': ('wall', 'Wall collision', (255, 105, 105)),
    'timeout': ('timeout', 'Timeout', (220, 210, 120)),
    'win': ('win', 'Win', (120, 225, 130)),
}

HEADER_HEIGHT = 70
FOOTER_HEIGHT = 48
PANEL_WIDTH = 390
ROW_HEIGHT = 22
LIST_TOP = 78
HINTS_HEIGHT = 52

# Column header label, sort key, and right edge (numbers) or left edge (text) within the panel.
COLUMNS = (
    ('Game', 'game', 84, 'right'),
    ('Score', 'score', 156, 'right'),
    ('ε', 'epsilon', 236, 'right'),
    ('Move', 'move', 256, 'left'),
    ('End', 'end', 320, 'left'),
)
SORT_KEYS = {
    'game': lambda r: r['game'],
    'score': lambda r: (r['score'], r['game']),
    'epsilon': lambda r: (effective_epsilon(r), r['game']),
    'move': lambda r: (move_label(r), r['game']),
    'end': lambda r: (r['end_reason'] or '', r['game']),
}
SORT_SHORTCUTS = {pygame.K_g: 'game', pygame.K_s: 'score', pygame.K_e: 'epsilon',
                  pygame.K_m: 'move'}
PATH_TOGGLE = pygame.K_p
# Text-sorted columns read alphabetically; numbers read best largest first.
TEXT_COLUMNS = {'move', 'end'}


def effective_epsilon(record):
    return record['epsilon'] * record['epsilon_scale']


def move_label(record):
    """Whether the final move was an epsilon exploration; logs from before tracking say '-'."""
    explored = record.get('random_move')
    return '-' if explored is None else 'random' if explored else 'greedy'


def segment_colors(is_head, random_move=False):
    """Colour segments as the game does; one placed by an epsilon exploration is green instead."""
    outer, inner, head = ((RANDOM_OUTER, RANDOM_INNER, RANDOM_HEAD) if random_move
                          else (BLUE1, BLUE2, HEAD))
    return outer, head if is_head else inner


def draw_segment(surface, x, y, outer, inner):
    pygame.draw.rect(surface, outer, pygame.Rect(x, y, BLOCK_SIZE, BLOCK_SIZE))
    pygame.draw.rect(surface, inner, pygame.Rect(x + 4, y + 4, 12, 12))


def _font(size):
    try:
        return pygame.font.Font(REPO_ROOT / 'arial.ttf', size)
    except (FileNotFoundError, OSError):
        return pygame.font.SysFont('arial', size)


class DeathViewer:
    """Sorted, scrollable list of recorded deaths beside the selected game's final board."""

    def __init__(self, path):
        self.path = Path(path)
        self.records = []
        self.order = []
        self.offset = 0
        self.sort_key, self.descending = 'game', True
        self.selected = None
        self.position = 0
        self.scroll = 0
        self.mouse = (-1, -1)
        self.show_path = True
        self.records, self.offset = read_deaths(self.path)

        columns, rows = self.records[0]['board'] if self.records else (32, 24)
        # A one-cell wall ring leaves room to draw a head that left the board.
        self.ring = pygame.Rect(0, HEADER_HEIGHT, (columns + 2) * BLOCK_SIZE, (rows + 2) * BLOCK_SIZE)
        self.board = self.ring.inflate(-2 * BLOCK_SIZE, -2 * BLOCK_SIZE)
        self.path_layer = pygame.Surface(self.ring.size, pygame.SRCALPHA)
        self.size = (self.ring.width + PANEL_WIDTH, self.ring.bottom + FOOTER_HEIGHT)
        self.panel = pygame.Rect(self.ring.right, 0, PANEL_WIDTH, self.size[1])
        self.visible_rows = (self.size[1] - LIST_TOP - HINTS_HEIGHT) // ROW_HEIGHT
        button_y = self.ring.bottom + (FOOTER_HEIGHT - 30) // 2
        self.prev_button = pygame.Rect(16, button_y, 84, 30)
        self.next_button = pygame.Rect(108, button_y, 84, 30)
        # Fonts need pygame initialised, which a previous pygame.quit() may have undone.
        pygame.init()
        self.title_font = _font(25)
        self.body_font = _font(16)
        self.small_font = _font(13)
        self._sort()

    # ── Data ─────────────────────────────────────────────────────────────
    def poll(self):
        """Read games appended since the last poll; start over if the file was replaced."""
        try:
            size = self.path.stat().st_size
        except FileNotFoundError:
            size = 0
        if size < self.offset:
            self.records, self.offset, self.selected = [], 0, None
        elif size == self.offset:
            return
        new, self.offset = read_deaths(self.path, self.offset)
        self.records.extend(new)
        self._sort()

    def sort_by(self, key):
        """Sort by ``key``, flipping the direction when it is already the sort key."""
        if key == self.sort_key:
            self.descending = not self.descending
        else:
            self.sort_key, self.descending = key, key not in TEXT_COLUMNS
        # A new order starts from its top; polling keeps the shown game instead.
        self.selected = None
        self.scroll = 0
        self._sort()

    def _sort(self):
        self.order = sorted(self.records, key=SORT_KEYS[self.sort_key], reverse=self.descending)
        if self.selected is None and self.order:
            self.selected = self.order[0]
        # Keep showing the same game; identity comparison skips equal-looking records.
        position = next((i for i, r in enumerate(self.order) if r is self.selected), 0)
        self.select(position)

    def select(self, position):
        if not self.order:
            return
        self.position = max(0, min(position, len(self.order) - 1))
        self.selected = self.order[self.position]
        if self.position < self.scroll:
            self.scroll = self.position
        elif self.position >= self.scroll + self.visible_rows:
            self.scroll = self.position - self.visible_rows + 1

    def _scroll(self, rows):
        self.scroll = max(0, min(self.scroll + rows, len(self.order) - self.visible_rows))

    # ── Input ────────────────────────────────────────────────────────────
    def handle(self, event):
        if event.type == pygame.KEYDOWN:
            steps = {pygame.K_DOWN: 1, pygame.K_RIGHT: 1, pygame.K_UP: -1, pygame.K_LEFT: -1,
                     pygame.K_PAGEDOWN: self.visible_rows, pygame.K_PAGEUP: -self.visible_rows}
            if event.key in steps:
                self.select(self.position + steps[event.key])
            elif event.key == pygame.K_HOME:
                self.select(0)
            elif event.key == pygame.K_END:
                self.select(len(self.order) - 1)
            elif event.key in SORT_SHORTCUTS:
                self.sort_by(SORT_SHORTCUTS[event.key])
            elif event.key == PATH_TOGGLE:
                self.show_path = not self.show_path
        elif event.type == pygame.MOUSEMOTION:
            self.mouse = event.pos
        elif event.type == pygame.MOUSEWHEEL:
            if self.panel.collidepoint(self.mouse):
                self._scroll(-3 * event.y)
            else:
                self.select(self.position - event.y)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self._click(event.pos)

    def _click(self, pos):
        if self.prev_button.collidepoint(pos):
            self.select(self.position - 1)
        elif self.next_button.collidepoint(pos):
            self.select(self.position + 1)
        elif (column := self._column_at(pos)) is not None:
            self.sort_by(column)
        elif (row := self._row_at(pos)) is not None:
            self.select(row)

    def _column_at(self, pos):
        x, y = pos[0] - self.panel.x, pos[1]
        if not LIST_TOP - 26 <= y < LIST_TOP - 2:
            return None
        bounds = [0] + [edge + 16 if align == 'right' else edge - 8
                        for _, _, edge, align in COLUMNS[:-1]] + [PANEL_WIDTH]
        for (_, key, _, _), left, right in zip(COLUMNS, bounds, bounds[1:]):
            if left <= x < right:
                return key
        return None

    def _row_at(self, pos):
        if not self.panel.collidepoint(pos) or pos[1] < LIST_TOP:
            return None
        row = (pos[1] - LIST_TOP) // ROW_HEIGHT
        if row >= self.visible_rows or self.scroll + row >= len(self.order):
            return None
        return self.scroll + row

    # ── Drawing ──────────────────────────────────────────────────────────
    def _text(self, surface, text, font, color, **anchor):
        image = font.render(text, True, color)
        surface.blit(image, image.get_rect(**anchor))
        return image.get_rect(**anchor)

    def draw(self, surface):
        surface.fill(WINDOW)
        self._draw_board(surface)
        self._draw_header(surface)
        self._draw_footer(surface)
        self._draw_panel(surface)

    def _draw_board(self, surface):
        pygame.draw.rect(surface, WALL, self.ring)
        pygame.draw.rect(surface, BLACK, self.board)
        record = self.selected
        if record is None:
            self._text(surface, 'No deaths recorded yet', self.title_font, WHITE,
                       center=(self.board.centerx, self.board.centery - 16))
            self._text(surface, f'Waiting for {self.path}', self.small_font, MUTED,
                       center=(self.board.centerx, self.board.centery + 16))
            return

        def corner(cell):
            return self.board.x + cell[0] * BLOCK_SIZE, self.board.y + cell[1] * BLOCK_SIZE

        # Food sits under the snake, where a winning head ends on it.
        if record['food'] is not None:
            pygame.draw.rect(surface, RED, pygame.Rect(*corner(record['food']), BLOCK_SIZE, BLOCK_SIZE))
        cells = snake_cells(record)
        # Logs from before segments were tracked have no random segments to show.
        random_segments = set(record.get('random_segments', ()))
        # Tail first, so a head that ran into the body is drawn over it.
        for index in reversed(range(len(cells))):
            draw_segment(surface, *corner(cells[index]),
                         *segment_colors(index == 0, index in random_segments))
        if self.show_path:
            self._draw_path(surface, cells)
        if record['end_reason'] in ('self-collision', 'wall-collision'):
            pygame.draw.rect(surface, RED, pygame.Rect(*corner(cells[0]), BLOCK_SIZE, BLOCK_SIZE), 2)
        self._text(surface, f'Score: {record["score"]}', self.title_font, WHITE,
                   topleft=(self.board.x + 8, self.board.y + 4))

    def _draw_path(self, surface, cells):
        """Trace the snake from tail to head through the centre of every segment."""
        # The ring-sized layer keeps a head that left the board on the line.
        self.path_layer.fill((0, 0, 0, 0))
        inset = (BLOCK_SIZE - PATH_WIDTH) // 2
        left, top = self.board.x - self.ring.x + inset, self.board.y - self.ring.y + inset
        # Moves are one cell along an axis, so each step is a rectangle centred on both segments.
        # Drawing writes alpha rather than blending, so overlapping corners stay evenly translucent.
        for (x1, y1), (x2, y2) in zip(cells, cells[1:]):
            x, y = min(x1, x2), min(y1, y2)
            step = pygame.Rect(left + x * BLOCK_SIZE, top + y * BLOCK_SIZE,
                               abs(x2 - x1) * BLOCK_SIZE + PATH_WIDTH,
                               abs(y2 - y1) * BLOCK_SIZE + PATH_WIDTH)
            self.path_layer.fill(PATH, step)
        surface.blit(self.path_layer, self.ring)

    def _draw_header(self, surface):
        record = self.selected
        if record is None:
            return
        _, label, color = REASONS.get(record['end_reason'],
                                      (None, str(record['end_reason']), WHITE))
        self._text(surface, f'Game {record["game"]:,}', self.title_font, WHITE, topleft=(16, 10))
        reason = self._text(surface, label, self.title_font, color, topright=(self.ring.right - 16, 10))
        epsilon = f'ε {record["epsilon"]:.4f}'
        if record['epsilon_scale'] != 1:
            epsilon += f' × ramp {record["epsilon_scale"]:.2f} = {effective_epsilon(record):.4f}'
        details = (f'Score {record["score"]} (start {record["start_score"]})    '
                   f'Length {len(record["body"]) + 1}    Steps {record["steps"]:,}    {epsilon}')
        drawn = self._text(surface, details, self.body_font, MUTED, topleft=(16, 44))
        if 'random_moves' in record:
            # A green segment before the count doubles as the legend for highlighted segments.
            count = self._text(surface, f'Random moves {record["random_moves"]:,}', self.body_font,
                               RANDOM_MOVE, midright=(reason.left - 24, reason.centery))
            draw_segment(surface, count.left - 8 - BLOCK_SIZE, reason.centery - BLOCK_SIZE // 2,
                         RANDOM_OUTER, RANDOM_INNER)
        elif move_label(record) == 'random':
            self._text(surface, '    Random final move', self.body_font, RANDOM_MOVE,
                       topleft=drawn.topright)

    def _draw_footer(self, surface):
        for button, label in ((self.prev_button, '< Prev'), (self.next_button, 'Next >')):
            hovered = button.collidepoint(self.mouse)
            pygame.draw.rect(surface, HOVER if hovered else PANEL, button, border_radius=4)
            pygame.draw.rect(surface, WALL, button, 1, border_radius=4)
            self._text(surface, label, self.body_font, WHITE, center=button.center)
        middle = self.prev_button.centery
        if self.order:
            self._text(surface, f'{self.position + 1:,} of {len(self.order):,}', self.body_font, MUTED,
                       midleft=(self.next_button.right + 16, middle))

    def _draw_panel(self, surface):
        pygame.draw.rect(surface, PANEL, self.panel)
        x0 = self.panel.x
        self._text(surface, 'Deaths', self.title_font, WHITE, topleft=(x0 + 16, 10))
        self._text(surface, f'{len(self.order):,}', self.body_font, MUTED,
                   topright=(self.panel.right - 16, 16))

        header_y = LIST_TOP - 14
        for label, key, edge, align in COLUMNS:
            active = key == self.sort_key
            anchor = {'midright' if align == 'right' else 'midleft': (x0 + edge, header_y)}
            drawn = self._text(surface, label, self.body_font, WHITE if active else MUTED, **anchor)
            if active:
                # The arrow sits beside the label so column text stays aligned with its values.
                arrow_x = x0 + edge + 4 if align == 'right' else drawn.right + 4
                self._text(surface, '▼' if self.descending else '▲', self.small_font, WHITE,
                           midleft=(arrow_x, header_y))
        pygame.draw.line(surface, WALL, (x0 + 8, LIST_TOP - 2), (self.panel.right - 8, LIST_TOP - 2))

        hovered = self._row_at(self.mouse)
        for row, record in enumerate(self.order[self.scroll:self.scroll + self.visible_rows]):
            position = self.scroll + row
            y = LIST_TOP + row * ROW_HEIGHT
            if position == self.position or position == hovered:
                pygame.draw.rect(surface, SELECTED if position == self.position else HOVER,
                                 pygame.Rect(x0 + 8, y, PANEL_WIDTH - 16, ROW_HEIGHT), border_radius=3)
            short, _, color = REASONS.get(record['end_reason'],
                                          (str(record['end_reason']), None, WHITE))
            middle = y + ROW_HEIGHT // 2
            move = move_label(record)
            cells = ((f'{record["game"]:,}', WHITE), (str(record['score']), WHITE),
                     (f'{effective_epsilon(record):.4f}', WHITE),
                     (move, RANDOM_MOVE if move == 'random' else MUTED), (short, color))
            for (_, _, edge, align), (text, text_color) in zip(COLUMNS, cells):
                anchor = {'midright' if align == 'right' else 'midleft': (x0 + edge, middle)}
                self._text(surface, text, self.body_font, text_color, **anchor)

        hints_y = self.size[1] - HINTS_HEIGHT + 12
        self._text(surface, '↑ ↓ or click to step    PgUp/PgDn/Home/End', self.small_font, MUTED,
                   topleft=(x0 + 16, hints_y))
        self._text(surface, 'Click a column or press G, S, E, M to sort    P toggles path', self.small_font, MUTED,
                   topleft=(x0 + 16, hints_y + 18))


def resolve_path(args):
    if args.deaths is not None:
        return Path(args.deaths)
    name = args.run or args.model
    if name is None:
        raise ValueError('Select a run with --run or --model, or pass --deaths PATH')
    run_dir = REPO_ROOT / 'runs' / validate_name(name)
    if not run_dir.is_dir():
        raise ValueError(f'Run does not exist: {run_dir}')
    return run_dir / 'deaths.jsonl'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', help='named preset in models.json (also the default run name)')
    parser.add_argument('--run', help='run directory under runs/ (default: --model)')
    parser.add_argument('--deaths', help='death log file path, instead of a run')
    args = parser.parse_args(argv)
    try:
        path = resolve_path(args)
    except ValueError as err:
        parser.error(str(err))

    viewer = DeathViewer(path)
    screen = pygame.display.set_mode(viewer.size)
    pygame.display.set_caption(f'Snake deaths - {path}')
    pygame.key.set_repeat(300, 35)
    clock = pygame.time.Clock()
    last_poll = pygame.time.get_ticks()
    try:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                        event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q)):
                    return
                viewer.handle(event)
            if pygame.time.get_ticks() - last_poll >= 1000:
                viewer.poll()
                last_poll = pygame.time.get_ticks()
            viewer.draw(screen)
            pygame.display.flip()
            clock.tick(30)
    finally:
        pygame.quit()


if __name__ == '__main__':
    main()
