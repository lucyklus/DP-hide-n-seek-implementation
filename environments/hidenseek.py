import functools
from copy import copy
import math

import numpy as np
from gymnasium import spaces
from typing import List, Dict, Tuple
from pettingzoo import ParallelEnv

from environments.models import (
    Agent,
    AgentType,
    Movement,
    Rewards,
    HiderRewards,
    SeekerRewards,
)

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
    """
    A parallel environment for a hide-and-seek game where multiple agents (hiders and seekers) interact in a grid world.
    The game progresses in discrete time steps, and agents move based on their actions.
    """

    metadata = {
        "name": "hide_and_seek_v1",
    }

    # Class attributes for agents and environment components
    seekers: List[Agent] = []  # List to store seeker agents
    hiders: List[Agent] = []  # List to store hider agents
    wall: List[List[int]] = []  # Grid representing wall placement
    found: Dict[str, str] = {}  # Dictionary to track which hiders have been found
    grid_size = 0  # Size of the grid
    static_seekers = False  # Indicates if seekers stay in their starting position
    static_hiders = False  # Indicates if hiders stay in their starting position
    max_distance = 0  # Maximum possible distance between any two points in the grid

    # Constructor to initialize the environment with given parameters
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
        # Agent generation logic - checks if enough agents are provided
        if num_seekers < 2 or num_hiders < 2:
            raise ValueError("Number of seekers and hiders must be at least 2")

        self.static_seekers = static_seekers
        self.static_hiders = static_hiders

        # Initialization of seekers
        # Places seekers at the bottom-right if static, else top-left
        seekers_x = grid_size - 1 if static_seekers else 0
        seekers_y = grid_size - 1 if static_seekers else 0
        self.seekers = [
            Agent(f"seeker_{index}", AgentType.SEEKER, seekers_x, seekers_y)
            for index in range(num_seekers)
        ]

        # Initialization of hiders
        # Places hiders at the bottom-left if static, else top-left
        hiders_x = grid_size - 1 if static_hiders else 0
        hiders_y = 0 if static_hiders else 0
        self.hiders = [
            Agent(f"hider_{index}", AgentType.HIDER, hiders_x, hiders_y)
            for index in range(num_hiders)
        ]

        # Initialize dictionary to track whether hiders have been found, starting with None (not found)
        self.found = {h.name: None for h in self.hiders}

        # Possible agents includes all seekers and hiders by their names
        self.possible_agents = [agent.name for agent in self.seekers + self.hiders]

        # The init method takes in environment arguments.
        self.grid_size = grid_size
        self.max_distance = math.sqrt(2 * (grid_size - 1) ** 2)

        # Wall configuration - setting up walls on the grid
        # A default wall configuration can be passed or an empty grid is initialized
        if wall is None:
            self.wall = [[0] * grid_size] * grid_size
        else:
            # Check if the wall configuration is valid (square matrix of 0s and 1s)
            if len(wall) != grid_size:
                raise ValueError("Wall must be a square matrix")
            for row in wall:
                if len(row) != grid_size:
                    raise ValueError("Wall must be a square matrix")
                for column in row:
                    if column != 0 and column != 1:
                        raise ValueError("Wall must contain only 0 or 1")

            self.wall = wall
        self.walls_coords = self._get_walls_coordinates()  # Get coordinates of walls

        # Initialize visibility radius and total game time
        self.visibility_radius = visibility_radius  # How far can the seeker see
        self.total_game_time = total_time
        self.game_time = total_time
        self.hider_time_limit = hiding_time

    def reset(self):
        """
        Resets the environment state for a new game episode.
        This includes resetting the positions of all agents, the 'found' status of hiders, and the remaining game time.

        Returns the initial observation for all agents.
        """
        # Copy the list of possible agents to the active agents list.
        # This 'resets' the agents so that all are considered active at the start of a new episode.
        self.agents = copy(self.possible_agents)
        # Reset the 'found' status of all hiders to None, indicating that no hiders have been found at the start.
        self.found = {h.name: None for h in self.hiders}

        # Determine the starting position for seekers based on whether they are static or not.
        # If static, they start at the bottom-right corner; otherwise, they start at the top-left.
        seeker_x = self.grid_size - 1 if self.static_seekers else 0
        seeker_y = self.grid_size - 1 if self.static_seekers else 0

        # Reset each seeker to their starting position.
        for seeker in self.seekers:
            seeker.reset(seeker_x, seeker_y)

        # Determine the starting position for hiders based on whether they are static or not.
        # If static, they start at the bottom-left corner; otherwise, they start at the top-left.
        hiders_x = self.grid_size - 1 if self.static_hiders else 0
        hiders_y = 0 if self.static_hiders else 0

        # Reset each hider to their starting position.
        for hider in self.hiders:
            hider.reset(hiders_x, hiders_y)

        # Get the initial observations for all agents.
        observations = self._get_observations()

        # Reset the game time to the total time allocated for the game.
        # This acts like a countdown timer for the duration of the game.
        self.game_time = self.total_game_time

        # Return the initial observations so agents can decide their first move.
        return observations

    def get_positions(self) -> Dict[str, Tuple[int, int]]:
        """
        Retrieves the current positions of all agents in the environment.

        Returns:
        - positions: A dictionary mapping agent names to their current x, y coordinates.
        """
        positions = {}
        for agent in self.seekers + self.hiders:
            positions[agent.name] = (agent.x, agent.y)
        return positions

    def step(self, hiders_actions, seekers_actions):
        """
        Advances the environment by one timestep. Updates agent states and computes new observations and rewards.

        Parameters:
        - hiders_actions: dict containing the actions for each hider agent
        - seekers_actions: dict containing the actions for each seeker agent

        Returns:
        - observations: a new snapshot of the environment after all agents have moved
        - rewards: the amount of reward or penalty received after the move
        - done: a flag indicating if the episode has ended
        - won: which group of agents won the game, if any
        - found: updated information on which hiders have been found
        """

        # Update the positions of the hiders if they are still allowed to move
        if not self._hider_time_limit_exceeded():
            for agent_name in hiders_actions.keys():
                self._move_agent(
                    AgentType.HIDER, agent_name, hiders_actions[agent_name]
                )

        # Update the positions of the seekers if the hiders' time has exceeded and the game is still ongoing
        if self._hider_time_limit_exceeded() and not self._seeker_time_limit_exceeded():
            for agent_name in seekers_actions.keys():
                self._move_agent(
                    AgentType.SEEKER, agent_name, seekers_actions[agent_name]
                )

        # Retrieve the new state of the environment after all agents have moved
        observations = self._get_observations()

        # If hiders' hiding time is up, check if seekers have found any hiders
        if self._hider_time_limit_exceeded():
            for seeker in self.seekers:
                for hider in self.hiders:
                    if self._check_found(seeker.x, seeker.y, hider.x, hider.y):
                        # If a hider is found for the first time, update the 'found' dictionary
                        if self.found[hider.name] is None:
                            self.found[hider.name] = seeker.name

        # Calculate the cumulative rewards based on the new state of the environment
        rewards, won = self._get_cummulative_rewards()

        # Determine if the game is done, which is the case if the list of active agents is empty
        t_done = 1 if self.agents == [] else 0

        # A dictionary that indicates whether the game is over for seekers and hiders
        done = {
            "seekers": {agent.name: t_done for agent in self.seekers},
            "hiders": {agent.name: t_done for agent in self.hiders},
        }

        # Decrement the game time unless it has run out
        if self.game_time > 0:
            self.game_time -= 1  # Decrease the time left with each step

        # Return the new observations, rewards, done status, winning status, and found hiders
        return (
            observations,
            rewards,
            done,
            won,
            {f: self.found[f] for f in self.found},
        )

    def calculate_total_rewards(self):
        """
        Calculates the accumulated rewards for hiders and seekers at the end of an episode.

        Rewards are based on various factors such as the remaining game time, whether hiders
        have been found or not, proximity to walls, and successful discoveries made by seekers.

        The method aggregates both individual rewards for each agent (hiders and seekers) and
        their total group rewards. A discovery bonus is granted to seekers for each hider found,
        while penalties are applied to hiders that have been discovered. Time rewards are given
        based on how long hiders remained hidden or how quickly seekers found them. Additional
        bonuses are granted for hiders that are close to walls, as this is a more strategic
        position in the game.

        The method updates the rewards structure which contains individual rewards, total rewards,
        and a record of which side won the game.

        Returns:
            Rewards: An object containing detailed reward breakdowns for both hiders and seekers.
        """
        # Initialize rewards structure with zero values for all agents.
        rewards = Rewards(
            hiders={h.name: HiderRewards(0.0, 0.0, 0.0, 0.0) for h in self.hiders},
            seekers={s.name: SeekerRewards(0.0, 0.0, 0.0) for s in self.seekers},
            hiders_total_reward=0.0,
            seekers_total_reward=0.0,
            hiders_total_penalty=0.0,
            seekers_total_penalty=0.0,
        )

        # Determine how many hiders are still hidden.
        hidden = len(self.hiders) - len(
            [h for h in self.found if self.found[h] is not None]
        )

        # Allocate rewards to seekers or hiders based on whether hiders have been found.
        if not self._seeker_time_limit_exceeded():
            # If game is ongoing, allocate time-based rewards.
            if hidden == 0:  # If all hiders have been found, seekers win.
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
                    # Bonus reward for hiders near walls.
                    if self._is_near_wall(h.x, h.y):
                        rewards.hiders[
                            h.name
                        ].next_to_wall_reward += NEXT_TO_WALL_REWARD

        else:  # If game is over, allocate hiding-based rewards.
            for h in self.hiders:
                # Reward for count of hidden hiders
                rewards.hiders[h.name].hidden_reward += HIDER_HIDDEN_REWARD * hidden

                # Additional bonus if the hider was never found.
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

        # Penalize hiders that have been found and reward seekers for each discovery.
        for f in self.found:
            if self.found[f] != None:
                rewards.hiders[f].discovery_penalty += HIDER_DISCOVERY_PENALTY
                rewards.seekers[
                    self.found[f]
                ].discovery_reward += SEEKER_DISCOVERY_BONUS
                for seeker in rewards.seekers:
                    rewards.seekers[seeker].discovery_reward += SEEKER_DISCOVERY_REWARD

        # Sum up total rewards for each group.
        for h in self.hiders:
            rewards.hiders_total_reward += rewards.hiders[h.name].get_total_reward()
            rewards.hiders_total_penalty += rewards.hiders[h.name].discovery_penalty

        for s in self.seekers:
            rewards.seekers_total_reward += rewards.seekers[s.name].get_total_reward()
            rewards.seekers_total_penalty += rewards.seekers[s.name].discovery_penalty

        return rewards

    def _hider_time_limit_exceeded(self):
        """
        Checks if the hiders' time to hide has elapsed, indicating they can no longer move.

        Returns:
            bool: True if the hiding time limit has been exceeded, otherwise False.
        """
        if self.game_time <= self.total_game_time - self.hider_time_limit:
            return True
        return False

    def _seeker_time_limit_exceeded(self):
        """
        Checks if the seekers' time to seek has run out, marking the end of the game.

        Returns:
            bool: True if the game time has reached zero, otherwise False.
        """
        return self.game_time == 0

    def _is_near_wall(self, x, y):
        """
        Determines if the specified coordinates are right next to block.

        Parameters:
        - x (int): The x-coordinate of the location to check.
        - y (int): The y-coordinate of the location to check.

        Returns:
        - bool: True if a wall is next to the specified coordinates, otherwise False.
        """
        if x > 0 and self.wall[x - 1][y] == 1:
            return True
        if x < self.grid_size - 1 and self.wall[x + 1][y] == 1:
            return True
        if y > 0 and self.wall[x][y - 1] == 1:
            return True
        if y < self.grid_size - 1 and self.wall[x][y + 1] == 1:
            return True
        return False

    def _move_agent(self, agent_type: AgentType, name: str, action: int):
        """
        Moves an agent based on the specified action. The movement is constrained by the grid boundaries
        and walls within the environment.

        Parameters:
        - agent_type (AgentType): The type of the agent, either HIDER or SEEKER.
        - name (str): The name identifier of the agent to move.
        - action (int): The action to take, represented by a value from the Movement enum.
        """

        # Find the agent to move by its name and type
        agent = next(
            filter(
                lambda x: x.name == name,
                self.hiders if agent_type == AgentType.HIDER else self.seekers,
            )
        )

        # Retrieve the agent's current position
        x, y = agent.x, agent.y

        # Update the agent's position based on the specified action
        match action:
            case Movement.LEFT.value:  # Move the agent left
                x, y = self._get_new_position(x, y, -1, 0)
            case Movement.RIGHT.value:  # Move the agent right
                x, y = self._get_new_position(x, y, 1, 0)
            case Movement.UP.value:  # Move the agent up
                x, y = self._get_new_position(x, y, 0, -1)
            case Movement.DOWN.value:  # Move the agent down
                x, y = self._get_new_position(x, y, 0, 1)

        # Update the agent's position in the environment
        agent.x = x
        agent.y = y

    def _get_new_position(self, x: int, y: int, dx: int, dy: int):
        """
        Calculates the new position of an agent considering movement and walls.

        Parameters:
        - x, y: Current coordinates of the agent.
        - dx, dy: Direction of movement.

        Returns:
        - (new_x, new_y): Updated coordinates after movement. If the movement is blocked by a wall, returns the original coordinates.
        """

        # Ensure the new position stays within the grid boundaries
        new_x = max(0, min(self.grid_size - 1, x + dx))
        new_y = max(0, min(self.grid_size - 1, y + dy))
        # If the new position is a wall, return the original coordinates
        if self.wall[new_x][new_y] == 1:
            return x, y
        return new_x, new_y

    def _get_walls_coordinates(self):
        """
        Collects the coordinates of all walls in the grid.

        Returns:
        - walls: A list of tuples (x, y) representing the coordinates of each wall.
        """
        walls = []
        for x in range(len(self.wall)):
            for y in range(len(self.wall[x])):
                if self.wall[x][y] == 1:
                    walls.append((x, y))
        return walls

    def _check_found(self, seeker_x, seeker_y, hider_x, hider_y):
        """
        Checks if a seeker has found a hider based on their coordinates.

        Parameters:
        - seeker_x, seeker_y: Coordinates of the seeker.
        - hider_x, hider_y: Coordinates of the hider.

        Returns:
        - bool: True if the seeker and hider are at the same coordinates, otherwise False.
        """
        if seeker_x == hider_x and seeker_y == hider_y:
            return True
        return False

    def _distance(self, x1, y1, x2, y2):
        """
        Calculates the Euclidean distance between two points.

        Parameters:
        - x1, y1: Coordinates of the first point.
        - x2, y2: Coordinates of the second point.

        Returns:
        - The Euclidean distance between the two points.
        """
        dx = x1 - x2
        dy = y1 - y2
        return math.sqrt(dx**2 + dy**2)

    def _check_visibility(self, seeker_x, seeker_y, hider_x, hider_y):
        """
        Determines if a hider is visible to a seeker, considering both distance and walls blocking the line of sight.

        Parameters:
        - seeker_x, seeker_y: Coordinates of the seeker.
        - hider_x, hider_y: Coordinates of the hider.

        Returns:
        - bool: True if the hider is within the seeker's visibility radius and not blocked by walls, otherwise False.
        """
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
        Constructs the observation vector for an agent, containing the agent's position, wall locations, and visible hiders (for seekers).

        Parameters:
        - agent: The agent for which to construct the observation.
        - type: The type of the agent (HIDER or SEEKER).

        Returns:
        - np.array: The observation vector for the agent.
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
        Generates observations for all agents in the environment. Observations may include
        the agent's own position, positions of walls, and positions of other agents depending
        on the agent type (hider or seeker).

        Returns:
            observations (Dict[str, np.array]): A dictionary mapping agent names to their
            respective observation arrays.
        """
        observations = {}
        # Generate observations for seekers based on their visibility
        for agent in self.seekers:
            observations[agent.name] = self._get_observation(agent, AgentType.SEEKER)
        # Generate observations for hiders
        for agent in self.hiders:
            observations[agent.name] = self._get_observation(agent, AgentType.HIDER)

        return observations

    def _get_reward(self, agent_name: str, type: AgentType):
        """
        Computes the reward for a given agent based on its type and current state.

        Parameters:
            agent_name (str): The name of the agent for which to calculate the reward.
            type (AgentType): The type of the agent (either HIDER or SEEKER).

        Returns:
            float: The calculated reward for the agent.
        """
        # Identify the agent from the list and calculate its reward based on distance to others
        agent = next(
            filter(
                lambda x: x.name == agent_name,
                self.hiders if type == AgentType.HIDER else self.seekers,
            )
        )
        if type == AgentType.HIDER:
            # Reward for hiders is based on the distance to the nearest seeker
            return (
                min([self._distance(agent.x, agent.y, s.x, s.y) for s in self.seekers])
                * DISTANCE_COEFFICIENT
            )
        else:
            # Reward for seekers is based on proximity to hiders
            return (
                self.max_distance
                - min([self._distance(agent.x, agent.y, h.x, h.y) for h in self.hiders])
            ) * DISTANCE_COEFFICIENT

    def _get_cummulative_rewards(self):
        """
        Calculates cumulative rewards for all agents based on the current game state, determining
        which team, hiders or seekers, has won if the game is concluded.

        Returns:
            tuple: Contains two elements; a dictionary of rewards for each agent, and a dictionary indicating if a team has won.
        """
        # Initialize rewards structure for hiders and seekers
        rewards = {
            "hiders": {
                h.name: self._get_reward(h.name, AgentType.HIDER) for h in self.hiders
            },
            "seekers": {
                s.name: self._get_reward(s.name, AgentType.SEEKER) for s in self.seekers
            },
        }

        # Sum total rewards for hiders and seekers
        rewards["hiders_total_reward"] = sum(rewards["hiders"].values())
        rewards["seekers_total_reward"] = sum(rewards["seekers"].values())

        # Determine win conditions
        won = {"hiders": False, "seekers": False}
        hidden = len(self.hiders) - len(
            [h for h in self.found if self.found[h] is not None]
        )

        # Determine the winning side based on remaining hiders and game state
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
        """
        Marks the game as over by clearing the list of active agents.
        """
        self.agents = []

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent_name: str):
        """
        Defines the action space for each agent. Here, all agents have the same action space size.

        Parameters:
            agent_name (str): The name of the agent.

        Returns:
            spaces.Discrete(5): The discrete action space with 5 actions for each agent.
        """
        return spaces.Discrete(5)
