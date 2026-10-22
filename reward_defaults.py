"""Default rewards shared by configuration validation and the game."""

DEFAULT_REWARDS = {
    'step_penalty': -0.01,
    'progress_reward': 0,
    'progress_penalty': 0,
    'food_reward': 10,
    'collision_penalty': -10,
    'timeout_penalty': -5,
    'win_reward': 100,
}
