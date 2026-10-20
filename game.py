import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

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
BLACK = (0,0,0)

BLOCK_SIZE = 20
SPEED = 1000000

STEP_PENALTY = -0.01
PROGRESS_REWARD = 0.1
PROGRESS_PENALTY = -0.1
FOOD_REWARD = 10
COLLISION_PENALTY = -10
TIMEOUT_PENALTY = -5

class SnakeGameAI:

    def __init__(self, w=640, h=480, render=True):
        self.w = w
        self.h = h
        self.render = render
        # init display
        self.display = None
        if self.render:
            self.display = pygame.display.set_mode((self.w, self.h))
            pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()


    def reset(self):
        # init game state
        self.direction = Direction.RIGHT

        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head,
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]

        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0
        self.end_reason = None


    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()


    def play_step(self, action):
        reward = STEP_PENALTY
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
        if self.frame_iteration > 50*len(self.snake):
            game_over = True
            self.end_reason = 'timeout'
            reward += TIMEOUT_PENALTY
            return reward, game_over, self.score
        if self.is_collision():
            game_over = True
            self.end_reason = (
                'self-collision' if self.head in self.snake[1:]
                else 'wall-collision'
            )
            reward += COLLISION_PENALTY
            return reward, game_over, self.score

        # 4. place new food or just move
        if ate_food:
            self.score += 1
            reward += FOOD_REWARD
            self._place_food()
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
            return PROGRESS_REWARD
        if current_distance > prev_distance:
            return PROGRESS_PENALTY

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

        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
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