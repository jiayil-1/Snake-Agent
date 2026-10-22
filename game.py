import pygame
import random
import math
from enum import Enum
from collections import namedtuple
import numpy as np
from reward_defaults import DEFAULT_REWARDS

pygame.init()
try:
    font = pygame.font.Font('arial.ttf', 25)
except (FileNotFoundError, OSError):
    font = pygame.font.SysFont('arial', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
HEAD = (150, 210, 255)
BLACK = (0,0,0)

BLOCK_SIZE = 20
SPEED = 30

STEP_PENALTY = DEFAULT_REWARDS['step_penalty']
PROGRESS_REWARD = DEFAULT_REWARDS['progress_reward']
PROGRESS_PENALTY = DEFAULT_REWARDS['progress_penalty']
FOOD_REWARD = DEFAULT_REWARDS['food_reward']
COLLISION_PENALTY = DEFAULT_REWARDS['collision_penalty']
TIMEOUT_PENALTY = DEFAULT_REWARDS['timeout_penalty']
WIN_REWARD = DEFAULT_REWARDS['win_reward']

def validate_start_length(start_length, w=640, h=480):
    """Accept None, a length, or an inclusive (min, max) range of lengths
    the random layout can place quickly."""
    if start_length is None:
        return None
    # Keep at least half the board open for food and manoeuvring.
    limit = (w // BLOCK_SIZE) * (h // BLOCK_SIZE) // 2
    is_range = isinstance(start_length, tuple)
    bounds = start_length if is_range else (start_length,)
    if ((is_range and len(bounds) != 2)
            or any(type(n) is not int or not 1 <= n <= limit for n in bounds)
            or bounds[0] > bounds[-1]):
        raise ValueError(f'start_length must be an integer or ascending (min, max) '
                         f'range from 1 to {limit}')
    return start_length


class SnakeGameAI:

    def __init__(self, w=640, h=480, render=True, rewards=None, start_length=None):
        self.w = w
        self.h = h
        # None keeps the classic centred three-segment start; an integer
        # starts each game coiled into a random solid block of that length, and
        # a (min, max) range draws each game's length uniformly from it.
        self.start_length = validate_start_length(start_length, w, h)
        self.render = render
        self.rewards = dict(DEFAULT_REWARDS)
        if rewards is not None:
            if not isinstance(rewards, dict):
                raise ValueError('rewards must be a mapping of reward names to numbers')
            unknown_rewards = rewards.keys() - DEFAULT_REWARDS.keys()
            if unknown_rewards:
                raise ValueError('Unknown rewards: ' + ', '.join(sorted(unknown_rewards)))
            for name, value in rewards.items():
                if type(value) not in (int, float) or not math.isfinite(value):
                    raise ValueError(f'{name} must be a finite number')
                self.rewards[name] = value
        # init display
        self.display = None
        if self.render:
            self.display = pygame.display.set_mode((self.w, self.h))
            pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()


    def reset(self):
        # init game state
        if self.start_length is None:
            self.direction = Direction.RIGHT
            self.head = Point(self.w/2, self.h/2)
            self.snake = [self.head,
                          Point(self.head.x-BLOCK_SIZE, self.head.y),
                          Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]
        elif isinstance(self.start_length, tuple):
            self._place_random_snake(random.randint(*self.start_length))
        else:
            self._place_random_snake(self.start_length)

        # Score is the length beyond the classic three-segment start, so a
        # longer random start is credited as though it had eaten its way there.
        self.score = len(self.snake) - 3
        self.start_score = self.score
        self.food = None
        self._place_food()
        self.frame_iteration = 0
        self.end_reason = None


    def _place_random_snake(self, length):
        """Coil the snake into a solid, hole-free block, tail to head.

        The block is filled row by row in alternating directions, so the head
        ends on its open edge. The serpentine is laid one cell past the head
        and that cell is left free, so moving straight on the first step is
        always safe. Width, orientation, mirroring, and position are random.
        """
        columns, rows = self.w // BLOCK_SIZE, self.h // BLOCK_SIZE
        cells = length + 1
        transpose = random.random() < 0.5
        across, down = (rows, columns) if transpose else (columns, rows)
        widths = [w for w in range(1, across + 1) if -(-cells // w) <= down]
        # Prefer blocks no more than twice as long as they are wide.
        compact = [w for w in widths if max(w, -(-cells // w)) <= 2 * min(w, -(-cells // w))]
        width = random.choice(compact or widths)
        height = -(-cells // width)

        path = []
        for i in range(cells):
            row, offset = divmod(i, width)
            path.append((offset if row % 2 == 0 else width - 1 - offset, row))
        flip_x, flip_y = random.random() < 0.5, random.random() < 0.5
        left = random.randrange(across - width + 1)
        top = random.randrange(down - height + 1)
        path = [(left + (width - 1 - x if flip_x else x), top + (height - 1 - y if flip_y else y))
                for x, y in path]
        if transpose:
            path = [(y, x) for x, y in path]

        steps = {Direction.RIGHT: (1, 0), Direction.LEFT: (-1, 0),
                 Direction.DOWN: (0, 1), Direction.UP: (0, -1)}
        ahead = (path[-1][0] - path[-2][0], path[-1][1] - path[-2][1])
        self.direction = next(d for d, step in steps.items() if step == ahead)
        body = path[-2::-1]
        self.snake = [Point(x * BLOCK_SIZE, y * BLOCK_SIZE) for x, y in body]
        self.head = self.snake[0]

    def _place_food(self):
        """Place food on a random free cell; return False when the snake fills the board."""
        occupied = set(self.snake)
        free = [Point(x * BLOCK_SIZE, y * BLOCK_SIZE)
                for y in range(self.h // BLOCK_SIZE)
                for x in range(self.w // BLOCK_SIZE)
                if Point(x * BLOCK_SIZE, y * BLOCK_SIZE) not in occupied]
        if not free:
            return False
        self.food = random.choice(free)
        return True


    def play_step(self, action):
        reward = self.rewards['step_penalty']
        self.frame_iteration += 1
        prev_distance = self._manhattan_distance(self.head, self.food)
        # 1. collect user input
        if self.render:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
        
        # 2. move
        self._move(action) # update the head
        self.snake.insert(0, self.head)

        ate_food = self.head == self.food
        if not ate_food:
            self.snake.pop()
        
        # 3. check if game over
        game_over = False
        if self.frame_iteration > 100*len(self.snake):
            game_over = True
            self.end_reason = 'timeout'
            reward += self.rewards['timeout_penalty']
            return reward, game_over, self.score
        if self.is_collision():
            game_over = True
            self.end_reason = (
                'self-collision' if self.head in self.snake[1:]
                else 'wall-collision'
            )
            reward += self.rewards['collision_penalty']
            return reward, game_over, self.score

        # 4. place new food or just move
        if ate_food:
            self.score += 1
            reward += self.rewards['food_reward']
            if not self._place_food():
                game_over = True
                self.end_reason = 'win'
                reward += self.rewards['win_reward']
                return reward, game_over, self.score
        else:
            reward += self._distance_delta_reward(prev_distance)
            
        
        # 5. update ui and clock
        if self.render:
            self._update_ui()
            self.clock.tick(SPEED)
        # 6. return game over and score
        return reward, game_over, self.score


    def _manhattan_distance(self, a, b):
        if a is None or b is None:
            return 0
        return abs(a.x - b.x) + abs(a.y - b.y)

    def _distance_delta_reward(self, prev_distance):
        if self.food is None:
            return 0

        current_distance = self._manhattan_distance(self.head, self.food)
        if current_distance < prev_distance:
            return self.rewards['progress_reward']
        if current_distance > prev_distance:
            return self.rewards['progress_penalty']

        return 0


    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # hits itself
        if pt in self.snake[1:]:
            return True

        return False


    def _update_ui(self):
        if self.display is None or self.food is None:
            return

        self.display.fill(BLACK)

        for index, pt in enumerate(self.snake):
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            # A lighter head shows which end leads on winding starts.
            inner = HEAD if index == 0 else BLUE2
            pygame.draw.rect(self.display, inner, pygame.Rect(pt.x+4, pt.y+4, 12, 12))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [8, 4])
        pygame.display.flip()


    def _move(self, action):
        # [straight, right, left]

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx] # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] # right turn r -> d -> l -> u
        else: # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] # left turn r -> u -> l -> d

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)
