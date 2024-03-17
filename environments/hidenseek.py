import functools
from copy import copy
import math

import numpy as np
from gymnasium import spaces
from enum import Enum
from typing import List, Dict
from pettingzoo import ParallelEnv

from rendering.renderer import Rewards, HiderRewards, SeekerRewards
from environments.models import Agent, AgentType, Movement

DISTANCE_COEFFICIENT = 0.1

SEEKER_DISCOVERY_REWARD = 40.0
SEEKER_DISCOVERY_BONUS = 2.5
SEEKER_TIME_REWARD = 0.1
SEEKER_DISCOVERY_PENALTY = -5.0

HIDER_HIDDEN_REWARD = 10.0
HIDER_HIDDEN_BONUS = 2.5
HIDER_DISCOVERY_PENALTY = -5.0
HIDER_TIME_REWARD = 0.1
NEXT_TO_WALL_REWARD = 0.5


class HideAndSeekEnv(ParallelEnv):
    # The metadata holds environment constants
    metadata = {
        "name": "hide_and_seek_v1",
    }

    seekers: List[Agent] = []
    hiders: List[Agent] = []
    wall: List[List[int]] = []
    found: Dict[str, str] = {}
    grid_size = 0
    static_seekers = False
    static_hiders = False
    max_distance = 0

    def __init__(
        self,
        num_seekers=2,
        num_hiders=2,
        grid_size=7,
        wall=None,
        total_time=50,
        hiding_time=30,
        visibility_radius=2,
        static_hiders=False,
        static_seekers=False,
    ):
        # Generate agents
        if num_seekers < 2 or num_hiders < 2:
            raise ValueError("Number of seekers and hiders must be at least 2")

        self.static_seekers = static_seekers
        self.static_hiders = static_hiders

        # Prepare seekers
        seekers_x = grid_size - 1 if static_seekers else 0
        seekers_y = grid_size - 1 if static_seekers else 0
        self.seekers = [
            Agent(f"seeker_{index}", AgentType.SEEKER, seekers_x, seekers_y)
            for index in range(num_seekers)
        ]

        # Prepare hiders
        hiders_x = grid_size - 1 if static_hiders else 0
        hiders_y = 0 if static_hiders else 0
        self.hiders = [
            Agent(f"hider_{index}", AgentType.HIDER, hiders_x, hiders_y)
            for index in range(num_hiders)
        ]

        self.found = {h.name: None for h in self.hiders}

        # Set possible agents
        self.possible_agents = [agent.name for agent in self.seekers + self.hiders]

        # The init method takes in environment arguments.
        self.grid_size = grid_size
        self.max_distance = math.sqrt(2 * (grid_size - 1) ** 2)

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
        self.walls_coords = self._get_walls_coordinates()

        self.visibility_radius = visibility_radius  # How far can the seeker see
        self.total_game_time = total_time  # Total game time
        self.game_time = total_time
        self.hider_time_limit = hiding_time  # Hider has 30 seconds to hide

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
        self.found = {h.name: None for h in self.hiders}

        seeker_x = self.grid_size - 1 if self.static_seekers else 0
        seeker_y = self.grid_size - 1 if self.static_seekers else 0

        for seeker in self.seekers:
            seeker.reset(seeker_x, seeker_y)

        hiders_x = self.grid_size - 1 if self.static_hiders else 0
        hiders_y = 0 if self.static_hiders else 0

        for hider in self.hiders:
            hider.reset(hiders_x, hiders_y)

        observations = self._get_observations()

        self.game_time = self.total_game_time

        return observations

    def step(self, hiders_actions, seekers_actions):
        """Takes in an action for the current agent specified by agent_selection.

        Needs to update:
        - hider and seeker coordinates
        - rewards
        - timestamp

        And any internal state used by observe() or render()
        """

        if not self._hider_time_limit_exceeded():
            for agent_name in hiders_actions.keys():
                self._move_agent(
                    AgentType.HIDER, agent_name, hiders_actions[agent_name]
                )

        if self._hider_time_limit_exceeded() and not self._seeker_time_limit_exceeded():
            for agent_name in seekers_actions.keys():
                self._move_agent(
                    AgentType.SEEKER, agent_name, seekers_actions[agent_name]
                )

        observations = self._get_observations()

        if self._hider_time_limit_exceeded() and not self._seeker_time_limit_exceeded():
            for seeker in self.seekers:
                for hider in self.hiders:
                    if self._check_found(seeker.x, seeker.y, hider.x, hider.y):
                        if self.found[hider.name] is None:
                            self.found[hider.name] = seeker.name

        rewards, won = self._get_cummulative_rewards()

        t_done = 1 if self.agents == [] else 0

        done = {
            "seekers": {agent.name: t_done for agent in self.seekers},
            "hiders": {agent.name: t_done for agent in self.hiders},
        }
        if self.game_time > 0:
            self.game_time -= 1  # Decrease the time left with each step

        return (
            observations,
            rewards,
            done,
            won,
            {f: self.found[f] for f in self.found},
        )

    def calculate_total_rewards(self):
        rewards = Rewards(
            hiders={h.name: HiderRewards(0.0, 0.0, 0.0, 0.0) for h in self.hiders},
            seekers={s.name: SeekerRewards(0.0, 0.0, 0.0) for s in self.seekers},
            hiders_total_reward=0.0,
            seekers_total_reward=0.0,
        )
        hidden = len(self.hiders) - len(
            [h for h in self.found if self.found[h] is not None]
        )

        if not self._seeker_time_limit_exceeded():
            # Check if seeker found the hider
            if hidden == 0:  # Seekers won
                # Calculate rewards for the seekers
                for s in self.seekers:
                    # Time Reward for Seekers
                    rewards.seekers[s.name].time_reward += (
                        SEEKER_TIME_REWARD * self.game_time
                    )

                for h in self.hiders:
                    # Time Reward for Hiders
                    rewards.hiders[h.name].time_reward += HIDER_TIME_REWARD * (
                        self.total_game_time - self.hider_time_limit - self.game_time
                    )

                    if self._is_near_wall(h.x, h.y):
                        rewards.hiders[
                            h.name
                        ].next_to_wall_reward += NEXT_TO_WALL_REWARD

        else:  # Hiders won
            # Calculate rewards for the hiders
            for h in self.hiders:
                # Reward for count of hidden hiders
                rewards.hiders[h.name].hidden_reward += HIDER_HIDDEN_REWARD * hidden

                if self.found[h.name] is None:
                    rewards.hiders[h.name].hidden_reward += HIDER_HIDDEN_BONUS

                # Reward for hiding next to the wall
                if self._is_near_wall(h.x, h.y):
                    rewards.hiders[h.name].next_to_wall_reward += NEXT_TO_WALL_REWARD

            for s in self.seekers:
                # Penalty for not finding hiders
                rewards.seekers[s.name].discovery_penalty += (
                    SEEKER_DISCOVERY_PENALTY * hidden
                )

        # Discovery Penalty for Hiders
        for f in self.found:
            if self.found[f] != None:
                rewards.hiders[f].discovery_penalty += HIDER_DISCOVERY_PENALTY
                rewards.seekers[
                    self.found[f]
                ].discovery_reward += SEEKER_DISCOVERY_BONUS
                for seeker in rewards.seekers:
                    rewards.seekers[seeker].discovery_reward += SEEKER_DISCOVERY_REWARD

        for h in self.hiders:
            rewards.hiders_total_reward += rewards.hiders[h.name].get_total_reward()

        for s in self.seekers:
            rewards.seekers_total_reward += rewards.seekers[s.name].get_total_reward()

        return rewards

    def render(self):
        grid = [[None for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        for hider in self.hiders:
            if grid[hider.x][hider.y] != None:
                grid[hider.x][hider.y].append({"type": "H", "name": hider.name})
            else:
                grid[hider.x][hider.y] = [{"type": "H", "name": hider.name}]
        for seeker in self.seekers:
            if grid[seeker.x][seeker.y] != None:
                grid[seeker.x][seeker.y].append({"type": "S", "name": seeker.name})
            else:
                grid[seeker.x][seeker.y] = [{"type": "S", "name": seeker.name}]

        for x in range(len(self.wall)):
            for y in range(len(self.wall[x])):
                if self.wall[x][y] == 1:
                    grid[x][y] = [{"type": "W", "name": f"wall_{x}_{y}"}]

        return grid

    def _hider_time_limit_exceeded(self):
        # Return True if hider's time limit is exceeded (cant move anymore)
        if self.game_time <= self.total_game_time - self.hider_time_limit:
            return True
        return False

    def _seeker_time_limit_exceeded(self):
        # Return True if seeker's time limit is exceeded (the game ends)
        if self.game_time == 0:
            return True
        return False

    def _is_near_wall(self, x, y):
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

    def _move_agent(self, agent_type: AgentType, name: str, action: int):
        agent: Agent = None
        if agent_type == AgentType.HIDER:
            agent = next(filter(lambda x: x.name == name, self.hiders))
        elif agent_type == AgentType.SEEKER:
            agent = next(filter(lambda x: x.name == name, self.seekers))

        x = agent.x
        y = agent.y

        match action:
            case Movement.LEFT.value:
                x, y = self._get_new_position(x, y, -1, 0)  # Move left
            case Movement.RIGHT.value:
                x, y = self._get_new_position(x, y, 1, 0)  # Move right
            case Movement.UP.value:
                x, y = self._get_new_position(x, y, 0, -1)  # Move up
            case Movement.DOWN.value:
                x, y = self._get_new_position(x, y, 0, 1)  # Move down

        agent.x = x
        agent.y = y

    def _get_new_position(self, x: int, y: int, dx: int, dy: int):
        # Check for wall
        new_x = max(0, min(self.grid_size - 1, x + dx))
        new_y = max(0, min(self.grid_size - 1, y + dy))
        if self.wall[new_x][new_y] == 1:
            return x, y
        return new_x, new_y

    def _get_walls_coordinates(self):
        walls = []
        for x in range(len(self.wall)):
            for y in range(len(self.wall[x])):
                if self.wall[x][y] == 1:
                    walls.append((x, y))
        return walls

    def _check_found(self, seeker_x, seeker_y, hider_x, hider_y):
        """
        Check if the seeker found the hider
        """
        if seeker_x == hider_x and seeker_y == hider_y:
            return True
        return False

    def _distance(self, x1, y1, x2, y2):
        dx = x1 - x2
        dy = y1 - y2
        return math.sqrt(dx**2 + dy**2)

    def _check_visibility(self, seeker_x, seeker_y, hider_x, hider_y):
        """
        We should provide observability of hiders only if they are within the specified radius
        """

        # Calculate Euclidean distance
        distance = self._distance(seeker_x, seeker_y, hider_x, hider_y)
        if distance == 0:
            return True

        # Check if the hider is within the specified radial radius
        if distance > self.visibility_radius:
            return False

        dx = hider_x - seeker_x
        dy = hider_y - seeker_y
        # Check for walls along the line of sight
        for t in range(int(distance) + 1):
            x = round(seeker_x + t * (dx / distance))
            y = round(seeker_y + t * (dy / distance))

            # Check if the cell contains a wall
            if (x, y) in self.walls_coords:
                return False

        # If no walls are encountered, the hider is within visibility radius
        return True

    def _get_observation(self, agent: Agent, type: AgentType):
        """
        Return the observation for the agent
        """
        m = [agent.x, agent.y]
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                if any([wx == x and wy == y for wx, wy in self.walls_coords]):
                    m.append(1)
                    continue
                hider = list(filter(lambda h: h.x == x and h.y == y, self.hiders))
                if type == AgentType.SEEKER and len(hider) > 0:
                    if self._check_visibility(agent.x, agent.y, x, y):
                        m.append(2)
                    else:
                        m.append(0)
                    continue
                m.append(0)

        # if type == AgentType.SEEKER:
        #     print(agent.name)
        #     self.print_grid(m)
        return np.array(m, dtype=np.float32)

    def _get_observations(self):
        """
        Return the observations for all agents
        """
        observations = {}
        for agent in self.seekers:
            observations[agent.name] = self._get_observation(agent, AgentType.SEEKER)
        for agent in self.hiders:
            observations[agent.name] = self._get_observation(agent, AgentType.HIDER)

        return observations

    def _get_reward(self, agent_name: str, type: AgentType):
        if type == AgentType.HIDER:
            agent = next(filter(lambda x: x.name == agent_name, self.hiders))
            return (
                min([self._distance(agent.x, agent.y, s.x, s.y) for s in self.seekers])
                * DISTANCE_COEFFICIENT
            )
        else:
            agent = next(filter(lambda x: x.name == agent_name, self.seekers))
            return (
                self.max_distance
                - min([self._distance(agent.x, agent.y, h.x, h.y) for h in self.hiders])
            ) * DISTANCE_COEFFICIENT

    def _get_cummulative_rewards(self):
        rewards = {
            "hiders": {
                h.name: self._get_reward(h.name, AgentType.HIDER) for h in self.hiders
            },
            "seekers": {
                s.name: self._get_reward(s.name, AgentType.SEEKER) for s in self.seekers
            },
        }
        rewards["hiders_total_reward"] = sum(rewards["hiders"].values())
        rewards["seekers_total_reward"] = sum(rewards["seekers"].values())
        won = {"hiders": False, "seekers": False}

        hidden = len(self.hiders) - len(
            [h for h in self.found if self.found[h] is not None]
        )

        if self._seeker_time_limit_exceeded():
            if hidden > 0:
                won["hiders"] = True
                self._game_over()
            else:
                won["seekers"] = True
                self._game_over()
        else:
            if hidden == 0:
                won["seekers"] = True
                self._game_over()

        return rewards, won

    def _game_over(self):
        self.agents = []

    def _print_grid(self, m):
        print(m[0], ":", m[1])
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                print(m[i * self.grid_size + j + 2], end=" ")
            print()

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent_name: str):
        return spaces.Discrete(5)
