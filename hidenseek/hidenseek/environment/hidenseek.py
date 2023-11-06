import functools
import random
from copy import copy

import numpy as np
from gymnasium.spaces import Discrete, MultiDiscrete
import time
from enum import Enum
from typing import List, Set
from pettingzoo import ParallelEnv


class AgentType(Enum):
    HIDER = 0
    SEEKER = 1


class Movement(Enum):
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3
    STAY = 4


class Agent:
    name: str
    type: AgentType
    x: int
    y: int

    def __init__(self, name, type: AgentType, x, y):
        self.name = name
        self.type = type
        self.x = x
        self.y = y

    def reset(self, x, y):
        self.x = x
        self.y = y


SEEKER_DISCOVERY_REWARD = 10.0
SEEKER_TIME_REWARD = 0.1
SEEKER_DISCOVERY_PENALTY = -5.0

HIDER_HIDDEN_REWARD = 10.0
HIDER_DISCOVERY_PENALTY = -5.0
HIDER_TIME_REWARD = 0.1
NEXT_TO_WALL_REWARD = 0.5


# ParallelEnv - agents get rewards after the end of cycle
class HideAndSeekEnv(ParallelEnv):
    # The metadata holds environment constants
    metadata = {
        "name": "hide_and_seek_v1",
    }

    seekers: List[Agent] = []
    hiders: List[Agent] = []
    wall: List[List[int]] = []

    def __init__(self, num_seekers=2, num_hiders=2, grid_size=7, wall=None):
        # Generate agents
        if num_seekers < 2 or num_hiders < 2:
            raise ValueError("Number of seekers and hiders must be at least 2")

        # Prepare seekers
        seekers_x = 0
        seekers_y = 0
        self.seekers = [
            Agent(f"seeker_{index}", AgentType.SEEKER, seekers_x, seekers_y)
            for index in range(num_seekers)
        ]

        # Prepare hiders
        hiders_x = grid_size - 1
        hiders_y = grid_size - 1
        self.hiders = [
            Agent(f"hider_{index}", AgentType.HIDER, hiders_x, hiders_y)
            for index in range(num_hiders)
        ]

        # Set possible agents
        self.possible_agents = [agent.name for agent in self.seekers + self.hiders]

        # The init method takes in environment arguments.
        self.grid_size = grid_size

        # Set wall
        if wall is None:
            self.wall = [[0] * grid_size] * grid_size

        if wall is not None:
            if len(wall) != grid_size:
                raise ValueError("Wall must be a square matrix")
            for row in wall:
                if len(row) != grid_size:
                    raise ValueError("Wall must be a square matrix")
                for column in row:
                    if column != 0 and column != 1:
                        raise ValueError("Wall must contain only 0 or 1")

            self.wall = wall

        self.visibility_radius = 2  # How far can the seeker see
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
        for seeker in self.seekers:
            seeker.reset(0, 0)

        for hider in self.hiders:
            hider.reset(self.grid_size - 1, self.grid_size - 1)

        observations = {}

        self.game_time = self.total_game_time

        return observations

    def step(self, hiders_actions, seekers_actions):
        """Takes in an action for the current agent specified by agent_selection.

        Needs to update:
        - hider and seeker coordinates
        - terminations
        - rewards
        - timestamp

        And any internal state used by observe() or render()
        """

        if not self.hider_time_limit_exceeded():
            for hidder_name, action in hiders_actions:
                self.move_agent(AgentType.HIDER, hidder_name, action)

        if self.hider_time_limit_exceeded() and not self.seeker_time_limit_exceeded():
            for seeker_name, action in seekers_actions:
                self.move_agent(AgentType.SEEKER, seeker_name, action)

        observations = self.get_observations()

        rewards, terminations = self.calculate_rewards_and_terminations(
            observations["found"]
        )

        self.game_time -= 1  # Decrease the time left with each step

        return observations, rewards, terminations

    def hider_time_limit_exceeded(self):
        # Return True if hider's time limit is exceeded (cant move anymore)
        if self.game_time <= self.total_game_time - self.hider_time_limit:
            return True
        return False

    def seeker_time_limit_exceeded(self):
        # Return True if seeker's time limit is exceeded (the game ends)
        if self.game_time == 0:
            return True
        return False

    def move_agent(self, agent_type: AgentType, agent_name: str, action: Movement):
        print(agent_type, agent_name, action)
        agent: Agent = None
        if agent_type == AgentType.HIDER:
            agent = next(filter(lambda x: x.name == agent_name, self.hiders))
        elif agent_type == AgentType.SEEKER:
            agent = next(filter(lambda x: x.name == agent_name, self.seekers))

        x = agent.x
        y = agent.y

        match action:
            case Movement.LEFT.value:
                x, y = self.get_new_position(x, y, -1, 0)  # Move left
            case Movement.RIGHT.value:
                x, y = self.get_new_position(x, y, 1, 0)  # Move right
            case Movement.UP.value:
                x, y = self.get_new_position(x, y, 0, -1)  # Move up
            case Movement.DOWN.value:
                x, y = self.get_new_position(x, y, 0, 1)  # Move down

        agent.x = x
        agent.y = y

    def get_new_position(self, x: int, y: int, dx: int, dy: int):
        # Check for wall
        new_x = max(0, min(self.grid_size - 1, x + dx))
        new_y = max(0, min(self.grid_size - 1, y + dy))
        if self.wall[new_x][new_y] == 1:
            return x, y
        return new_x, new_y

    def check_visibility(self, seeker_x, seeker_y, hider_x, hider_y):
        # TODO - check if the seeker can see the hider
        if seeker_x == hider_x and seeker_y == hider_y:
            return True
        return False

    def get_observations(self):
        found: Set[str] = set()

        for seeker in self.seekers:
            for hider in self.hiders:
                # Check for visibility and walls
                if self.check_visibility(seeker.x, seeker.y, hider.x, hider.y):
                    found.add(hider.name)

        observations = {
            "game_time": self.total_game_time,
            "time_left": self.game_time,
            "hider": [
                {"name": hider.name, "x": hider.x, "y": hider.y}
                for hider in self.hiders
            ],
            "seeker": [
                {"name": seeker.name, "x": seeker.x, "y": seeker.y}
                for seeker in self.seekers
            ],
            "found": found,
            "wall": self.wall,
        }
        return observations

    def game_over(self):
        self.agents = []

    def is_near_wall(self, x, y):
        # Check if the agent is near the wall (one block away)
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                # Check if in bounds
                if (
                    x + dx < 0
                    or x + dx >= self.grid_size
                    or y + dy < 0
                    or y + dy >= self.grid_size
                ):
                    continue
                if self.wall[x + dx][y + dy] == 1:
                    return True

    def calculate_rewards_and_terminations(self, found: Set[str]):
        rewards = {"hiders": 0.0, "seekers": 0.0}
        terminations = {"hiders": False, "seekers": False}
        hidden = len(self.hiders) - len(found)

        if self.hider_time_limit_exceeded():
            # If hider's time limit is exceeded terminate the hider
            terminations["hiders"] = True
        if not self.seeker_time_limit_exceeded():
            # Check if seeker found the hider
            if hidden == 0:  # Seekers won
                terminations["hiders"] = True
                terminations["seekers"] = True
                # Calculate rewards for the seekers
                rewards["seekers"] += SEEKER_DISCOVERY_REWARD * len(
                    self.hiders
                )  # Discovery Reward for Seekers
                rewards["seekers"] += SEEKER_TIME_REWARD * self.game_time

                # Calculate rewards for the hiders
                rewards["hiders"] += HIDER_DISCOVERY_PENALTY * len(
                    self.hiders
                )  # Negative penalty for the hider if they are found
                rewards["hiders"] += HIDER_TIME_REWARD * (
                    self.total_game_time - self.hider_time_limit - self.game_time
                )

                # Reward for hiding next to the wall
                for hider in self.hiders:
                    if self.is_near_wall(hider.x, hider.y):
                        rewards["hiders"] += NEXT_TO_WALL_REWARD

                self.game_over()
        else:  # Hiders won
            # Terminate the seeker when their time is exhausted and calculate rewards
            terminations["seekers"] = True

            # Calculate rewards for the hiders
            rewards["hiders"] += (
                HIDER_HIDDEN_REWARD * hidden
            )  # Positive reward for the hider if they are not found

            # Reward for hiding next to the wall
            for hider in self.hiders:
                if self.is_near_wall(hider.x, hider.y):
                    rewards["hiders"] += NEXT_TO_WALL_REWARD

            # Calculate rewards for the seekers
            rewards["seekers"] += (
                SEEKER_DISCOVERY_PENALTY * hidden
            )  # Negative reward for the seeker if they don't find the hider
            rewards["seekers"] += SEEKER_DISCOVERY_REWARD * len(found)

            self.game_over()

        return rewards, terminations

    def render(self):
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.str_)
        for hider in self.hiders:
            grid[hider.x][hider.y] = "H"
        for seeker in self.seekers:
            grid[seeker.x][seeker.y] = "S"

        for x in range(len(self.wall)):
            for y in range(len(self.wall[x])):
                if self.wall[x][y] == 1:
                    grid[x][y] = "#"

        print(f"{grid} \n")

    # @functools.lru_cache(maxsize=None)
    # def observation_space(self, agent):
    #     return MultiDiscrete([self.grid_size * self.grid_size - 1] * 3)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(5)
