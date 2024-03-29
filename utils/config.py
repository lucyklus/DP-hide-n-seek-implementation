from dotenv import load_dotenv
from typing import Self
import os


class Config:
    """
    A configuration class for loading and accessing game settings from environment variables.

    Attributes:
        N_HIDERS (int): Number of hiders in the game.
        N_SEEKERS (int): Number of seekers in the game.
        GRID_SIZE (int): The size of the square grid.
        TOTAL_TIME (int): Total time allotted for each episode.
        HIDING_TIME (int): Time allotted for hiders to hide at the beginning of each episode.
        VISIBILITY (int): Visibility range for seekers.
        EPISODES (int): Number of episodes to run.
        EPISODE_PART_SIZE (int): Number of episodes after which to save a new file.
    """
    config: Self = None
    N_HIDERS: int
    N_SEEKERS: int
    GRID_SIZE: int
    TOTAL_TIME: int
    HIDING_TIME: int
    VISIBILITY: int
    EPISODES: int
    EPISODE_PART_SIZE: int

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
    ):
        """
        Initializes the configuration with specified values.
        """
        self.N_HIDERS = N_HIDERS
        self.N_SEEKERS = N_SEEKERS
        self.GRID_SIZE = GRID_SIZE
        self.TOTAL_TIME = TOTAL_TIME
        self.HIDING_TIME = HIDING_TIME
        self.VISIBILITY = VISIBILITY
        self.EPISODES = EPISODES
        self.EPISODE_PART_SIZE = EPISODE_PART_SIZE

    @staticmethod
    def _load_configurations():
        """
        Loads configuration values from a .env file into the Config class.

        Returns:
            A Config instance populated with settings from the environment variables.
        """
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
        )
        
    @staticmethod
    def get():
        """
        Returns a singleton instance of the Config class, loading the configuration from environment variables if not already done.

        Returns:
            The singleton Config instance with loaded settings.
        """
        if Config.config is None:
            Config.config = Config._load_configurations()
        return Config.config
