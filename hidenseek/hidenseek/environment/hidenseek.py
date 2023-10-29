import functools
import random
from copy import copy

import numpy as np
from gymnasium.spaces import Discrete, MultiDiscrete
import time

from pettingzoo import ParallelEnv


# ParallelEnv - agents get rewards after the end of cycle
class HideAndSeekEnv(ParallelEnv):
    # The metadata holds environment constants
    metadata = {
        "name": "hide_and_seek_v1",
    }

    def __init__(self):
        # The init method takes in environment arguments.
        self.grid_size = 7
        self.hider_x = None
        self.hider_y = None
        self.seeker_x = None
        self.seeker_y = None
        self.wall_x = None
        self.wall_y = None
        self.possible_agents = ["hider", "seeker"]
        self.visibility_radius = 2
        self.total_game_time = 50  # Total game time (50 seconds)
        self.game_time = 50
        self.hider_time_limit = 30  # Hider has 30 seconds to hide
        self.seeker_time_limit = 20  # Seeker has 20 seconds to find the hider

    def reset(self):
        """Reset the environment to a starting point.

        It needs to initialize the following attributes:
        - agents
        - timestamp
        - hider, seeker and wall coordinates
        - observations

        And must set up the environment so that render(), step(), and observe() can be called without issues.
        """
        self.agents = copy(self.possible_agents)

        # Spawn the hider and seeker at opposite corners of the grid
        self.hider_x = 0
        self.hider_y = 0

        self.seeker_x = self.grid_size - 1
        self.seeker_y = self.grid_size - 1

        # Spawn the wall at a random location
        self.wall_x = random.randint(0, self.grid_size - 1)
        self.wall_y = random.randint(0, self.grid_size - 1)

        observations = {}

        self.game_time = self.total_game_time

        return observations

    def step(self, actions):
        """Takes in an action for the current agent specified by agent_selection.

        Needs to update:
        - hider and seeker coordinates
        - terminations
        - truncations
        - rewards
        - timestamp
        - infos

        And any internal state used by observe() or render()
        """

        if not self.hider_time_limit_exceeded():
            self.move_agent("hider", actions["hider"])
        if self.hider_time_limit_exceeded() and not self.seeker_time_limit_exceeded():
            self.move_agent("seeker", actions["seeker"])

        observations = self.get_observations()

        rewards, terminations = self.calculate_rewards_and_terminations()

        self.game_time -= 1  # Decrease the time left with each step

        return observations, rewards, terminations

    def hider_time_limit_exceeded(self):
        if self.game_time <= self.total_game_time - self.hider_time_limit:
            return True
        return False

    def seeker_time_limit_exceeded(self):
        if self.game_time == 0:
            return True
        return False

    def move_agent(self, agent, action):
        if action == 0:
            new_x, new_y = self.get_new_position(agent, -1, 0)  # Move left
        elif action == 1:
            new_x, new_y = self.get_new_position(agent, 1, 0)  # Move right
        elif action == 2:
            new_x, new_y = self.get_new_position(agent, 0, -1)  # Move up
        elif action == 3:
            new_x, new_y = self.get_new_position(agent, 0, 1)  # Move down
        elif action == 4:
            new_x, new_y = self.get_agent_position(agent)  # Don't move
        if agent == "hider":
            self.hider_x, self.hider_y = new_x, new_y
        elif agent == "seeker":
            self.seeker_x, self.seeker_y = new_x, new_y

    def get_new_position(self, agent, dx, dy):
        x, y = self.get_agent_position(agent)
        new_x = max(0, min(self.grid_size - 1, x + dx))
        new_y = max(0, min(self.grid_size - 1, y + dy))
        return new_x, new_y

    def get_agent_position(self, agent):
        if agent == "hider":
            return self.hider_x, self.hider_y
        elif agent == "seeker":
            return self.seeker_x, self.seeker_y

    def get_observations(self):
        hider_x, hider_y = self.hider_x, self.hider_y
        seeker_x, seeker_y = self.seeker_x, self.seeker_y
        # Limit hider's visibility
        if (
            abs(hider_x - seeker_x) > self.visibility_radius
            or abs(hider_y - seeker_y) > self.visibility_radius
        ):
            hider_x, hider_y = (
                -1,
                -1,
            )  # Set hider's position to -1, -1 if outside visibility radius
        observations = {
            "game_time": self.total_game_time,
            "time_left": self.game_time,
            "hider": [self.hider_x, self.hider_y],
            "seeker": [self.seeker_x, self.seeker_y],
            "wall": [self.wall_x, self.wall_y],
        }
        return observations

    def calculate_rewards_and_terminations(self):
        rewards = {"hider": 0.0, "seeker": 0.0}
        terminations = {"hider": False, "seeker": False}
        is_hider_behind_wall = (
            self.hider_x == self.wall_x and self.hider_y == self.wall_y
        )

        # If hider's time limit is exceeded terminate the hider
        if self.hider_time_limit_exceeded():
            terminations["hider"] = True

        # Terminate the seeker when their time is exhausted and calculate rewards
        if self.seeker_time_limit_exceeded():
            terminations["seeker"] = True

            if (
                abs(self.hider_x - self.seeker_x) > self.visibility_radius
                or abs(self.hider_y - self.seeker_y) > self.visibility_radius
            ):
                rewards["hider"] += +10.0
                rewards["hider"] += (
                    0.1 * self.seeker_time_limit
                )  # Time bonus for every second being hidden
                rewards[
                    "seeker"
                ] += (
                    -5.0
                )  # Negative penalty for the seeker if they don't find the hider within their time limit
            self.agents = []  # This ends the game

        # If hider is found by the seeker before the time runs out, terminate both and calculate rewards
        if (
            self.hider_time_limit_exceeded()
            and abs(self.hider_x - self.seeker_x) <= self.visibility_radius
            and abs(self.hider_y - self.seeker_y) <= self.visibility_radius
        ):
            terminations["hider"] = True
            terminations["seeker"] = True
            rewards["seeker"] += 10.0  # Discovery Reward for Seekers
            rewards["seeker"] += (
                0.1 * self.game_time
            )  # Time bonus for every second left
            rewards["hider"] += -5.0  # Negative penalty for the hider if they are found
            rewards["hider"] += 0.1 * (
                self.total_game_time - self.hider_time_limit - self.game_time
            )  # Positive reward for seconds spent hidden
            self.agents = []  # This ends the game
        elif self.hider_x == self.wall_x and self.hider_y == self.wall_y:
            rewards["hider"] += 5.0  # Positive reward for hider if they reach the wall

        return rewards, terminations

    def render(self):
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.str_)
        grid[self.hider_y, self.hider_x] = "H"
        grid[self.seeker_y, self.seeker_x] = "S"
        grid[self.wall_y, self.wall_x] = "W"
        print(f"{grid} \n")

    # @functools.lru_cache(maxsize=None)
    # def observation_space(self, agent):
    #     return MultiDiscrete([self.grid_size * self.grid_size - 1] * 3)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(5)
