from dotenv import load_dotenv
from typing import Self
import os


class Config:
    config: Self = None
    N_HIDERS: int
    N_SEEKERS: int
    GRID_SIZE: int
    TOTAL_TIME: int
    HIDING_TIME: int
    VISIBILITY: int
    EPISODES: int
    EPISODE_PART_SIZE: int
    USE_CHECKPOINTS: bool

    def __init__(
        self,
        N_HIDERS: int,
        N_SEEKERS: int,
        GRID_SIZE: int,
        TOTAL_TIME: int,
        HIDING_TIME: int,
        VISIBILITY: int,
        EPISODES: int,
        EPISODE_PART_SIZE: int,
        USE_CHECKPOINTS: bool,
    ):
        self.N_HIDERS = N_HIDERS
        self.N_SEEKERS = N_SEEKERS
        self.GRID_SIZE = GRID_SIZE
        self.TOTAL_TIME = TOTAL_TIME
        self.HIDING_TIME = HIDING_TIME
        self.VISIBILITY = VISIBILITY
        self.EPISODES = EPISODES
        self.EPISODE_PART_SIZE = EPISODE_PART_SIZE
        self.USE_CHECKPOINTS = USE_CHECKPOINTS

    def _load_configurations():
        load_dotenv("./.env", verbose=True, override=True)
        return Config(
            N_HIDERS=int(os.getenv("N_HIDERS")),
            N_SEEKERS=int(os.getenv("N_SEEKERS")),
            GRID_SIZE=int(os.getenv("GRID_SIZE")),
            TOTAL_TIME=int(os.getenv("TOTAL_TIME")),
            HIDING_TIME=int(os.getenv("HIDING_TIME")),
            VISIBILITY=int(os.getenv("VISIBILITY")),
            EPISODES=int(os.getenv("EPISODES")),
            EPISODE_PART_SIZE=int(os.getenv("EPISODE_PART_SIZE")),
            USE_CHECKPOINTS=bool(os.getenv("USE_CHECKPOINTS")),
        )

    def get():
        if Config.config is None:
            Config.config = Config._load_configurations()
        return Config.config
