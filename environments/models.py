from enum import Enum


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
