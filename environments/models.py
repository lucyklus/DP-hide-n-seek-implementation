from enum import Enum
from dataclasses import dataclass
from typing import Self
from typing import List, Dict


class AgentType(Enum):
    """
    Enumerates the types of agents in the game: Hider or Seeker.
    """

    HIDER = 0
    SEEKER = 1

class Agent:
    """
    Represents an agent in the game, which can be either a Hider or a Seeker.

    Attributes:
        name (str): The name of the agent.
        type (AgentType): The type of the agent, either HIDER or SEEKER.
        x (int): The x-coordinate of the agent on the grid.
        y (int): The y-coordinate of the agent on the grid.
    """

    name: str
    type: AgentType
    x: int
    y: int

    def __init__(self, name, type: AgentType, x, y):
        """
        Initializes an Agent with a name, type, and starting coordinates.
        """
        self.name = name
        self.type = type
        self.x = x
        self.y = y

    def reset(self, x, y):
        """
        Resets the agent's position to the given coordinates. Used at the start of a new game or episode.
        """
        self.x = x
        self.y = y



class Movement(Enum):
    """
    Enumerates the possible movements for agents: left, right, up, down, or stay (no movement).
    """

    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3
    STAY = 4



@dataclass
class HiderRewards:
    """
    Represents the rewards accumulated by a hider during an episode.
    """

    time_reward: float
    next_to_wall_reward: float
    hidden_reward: float
    discovery_penalty: float

    def add(self, rew: Self):
        """
        Adds the rewards from another HiderRewards instance to this one, aggregating the total rewards.
        """
        self.time_reward += rew.time_reward
        self.next_to_wall_reward += rew.next_to_wall_reward
        self.hidden_reward += rew.hidden_reward
        self.discovery_penalty += rew.discovery_penalty

    def get_total_reward(self):
        """
        Calculates the total reward for a hider by summing all rewards and subtracting penalties.
        """
        return (
            self.time_reward
            + self.next_to_wall_reward
            + self.hidden_reward
            + self.discovery_penalty
        )


@dataclass
class SeekerRewards:
    """
    Represents the rewards accumulated by a seeker during an episode.

    Attributes describe various types of rewards and penalties specific to seeker agents.
    """
    time_reward: float
    discovery_reward: float
    discovery_penalty: float

    def add(self, rew: Self):
        """
        Adds the rewards from another SeekerRewards instance to this one, aggregating the total rewards.
        """
        self.time_reward += rew.time_reward
        self.discovery_reward += rew.discovery_reward
        self.discovery_penalty += rew.discovery_penalty

    def get_total_reward(self):
        """
        Calculates the total reward for a seeker by summing all rewards and subtracting penalties.
        """
        return self.time_reward + self.discovery_reward + self.discovery_penalty


@dataclass
class Rewards:
    """
    Holds the reward structures for both hiders and seekers, including total rewards for each group.

    Attributes:
        hiders (Dict[str, HiderRewards]): Rewards for each hider.
        hiders_total_reward (float): Total rewards for all hiders.
        seekers (Dict[str, SeekerRewards]): Rewards for each seeker.
        seekers_total_reward (float): Total rewards for all seekers.
    """
    hiders: Dict[str, HiderRewards]
    hiders_total_reward: float
    hiders_total_penalty: float
    seekers: Dict[str, SeekerRewards]
    seekers_total_reward: float
    seekers_total_penalty: float

    def add(self, rew: Self):
        """
        Aggregates rewards from another Rewards instance into this one, updating totals and individual rewards.
        """
        self.hiders_total_reward += rew.hiders_total_reward
        self.seekers_total_reward += rew.seekers_total_reward
        
        self.hiders_total_penalty += rew.hiders_total_penalty
        self.seekers_total_penalty += rew.seekers_total_penalty
        for hider in rew.hiders:
            if hider not in self.hiders:
                self.hiders[hider] = rew.hiders[hider]
            else:
                self.hiders[hider].add(rew.hiders[hider])
        for seeker in rew.seekers:
            if seeker not in self.seekers:
                self.seekers[seeker] = rew.seekers[seeker]
            else:
                self.seekers[seeker].add(rew.seekers[seeker])


@dataclass
class Frame:
    """
    Represents a single frame in an episode, capturing the state of the game, actions taken by agents,
    and the outcome of those actions.

    Attributes capture the state of the grid, actions performed, whether agents are done, if the game is won, and which agent was found.
    """
    state: List[List[Dict[str, str]]]
    actions: Dict[str, Dict[str, int]]
    done: Dict[str, Dict[str, int]]
    won: Dict[str, bool]
    found: Dict[str, str]


@dataclass
class Episode:
    """
    Represents an entire episode of gameplay, containing a sequence of frames, the accumulated rewards, and an identifier.

    Attributes:
        number (int): The episode number.
        rewards (Rewards): The accumulated rewards for the episode.
        frames (List[Frame]): A list of frames depicting the episode's progression.
    """
    number: int
    rewards: Rewards
    frames: List[Frame]
