from typing import List, Dict, Self
from dataclasses import dataclass
import pygame


@dataclass
class HiderRewards:
    time_reward: float  # HIDER_TIME_REWARD * (self.total_game_time - self.hider_time_limit - self.game_time)
    next_to_wall_reward: float  # HIDER_NEXT_TO_WALL_REWARD
    hidden_reward: float  # HIDER_HIDDEN_REWARD * hidden
    discovery_penalty: float  # HIDER_DISCOVERY_PENALTY

    def add(self, rew: Self):
        self.time_reward += rew.time_reward
        self.next_to_wall_reward += rew.next_to_wall_reward
        self.hidden_reward += rew.hidden_reward
        self.discovery_penalty += rew.discovery_penalty

    def get_total_reward(self):
        return (
            self.time_reward
            + self.next_to_wall_reward
            + self.hidden_reward
            - self.discovery_penalty
        )


@dataclass
class SeekerRewards:
    time_reward: float  # SEEKER_TIME_REWARD * self.game_time
    discovery_reward: float  # SEEKER_DISCOVERY_REWARD
    discovery_penalty: float  # SEEKER_DISCOVERY_PENALTY * hidden

    def add(self, rew: Self):
        self.time_reward += rew.time_reward
        self.discovery_reward += rew.discovery_reward
        self.discovery_penalty += rew.discovery_penalty

    def get_total_reward(self):
        return self.time_reward + self.discovery_reward - self.discovery_penalty


@dataclass
class Rewards:
    hiders: Dict[str, HiderRewards]
    hiders_total_reward: float
    seekers: Dict[str, SeekerRewards]
    seekers_total_reward: float

    def add(self, rew: Self):
        self.hiders_total_reward += rew.hiders_total_reward
        self.seekers_total_reward += rew.seekers_total_reward
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
    state: List[List[Dict[str, str]]]
    actions: Dict[str, Dict[str, int]]
    done: Dict[str, Dict[str, int]]
    won: Dict[str, bool]
    found: Dict[str, str]


@dataclass
class Episode:
    number: int
    rewards: Rewards
    frames: List[Frame]


FRAMERATE = 30
CELL_SIZE = 100


class Hider:
    def __init__(self, name: str, images: List[pygame.Surface]):
        self.name = name
        self.images = images
        self.x = 0
        self.y = 0
        self.direction = 0

    def set_pos(self, x, y, direction):
        self.x = x
        self.y = y
        self.direction = direction

    def draw(self, screen, font):
        screen.blit(
            self.images[self.direction], (self.x * CELL_SIZE, self.y * CELL_SIZE)
        )
        screen.blit(
            font.render(self.name, True, (0, 0, 0)),
            (self.x * CELL_SIZE, self.y * CELL_SIZE),
        ),


class Seeker:
    def __init__(self, name: str, images: List[pygame.Surface]):
        self.name = name
        self.images = images
        self.x = 0
        self.y = 0
        self.direction = 0

    def set_pos(self, x, y, direction=4):
        self.x = x
        self.y = y
        self.direction = direction

    def draw(self, screen, font):
        screen.blit(
            self.images[self.direction], (self.x * CELL_SIZE, self.y * CELL_SIZE)
        )
        screen.blit(
            font.render(self.name, True, (0, 0, 0)),
            (self.x * CELL_SIZE, self.y * CELL_SIZE),
        ),


class Visibility:
    def __init__(self, seeker_name: str, radius: int):
        self.name = seeker_name
        self.x = 0
        self.y = 0
        self.direction = 0
        self.radius = radius
        self.visibility = False

    def set_pos(self, x, y):
        self.x = x
        self.y = y

    def set_visibility(self, visibility: bool):
        self.visibility = visibility

    def draw(self, screen):
        if self.visibility:
            pygame.draw.circle(
                screen,
                (255, 255, 0, 50),
                (
                    self.x * CELL_SIZE + CELL_SIZE / 2,
                    self.y * CELL_SIZE + CELL_SIZE / 2,
                ),
                self.radius * CELL_SIZE,
            )


class Wall:
    def __init__(self, image: pygame.Surface):
        self.image = image
        self.x = 0
        self.y = 0

    def set_pos(self, x, y):
        self.x = x
        self.y = y

    def draw(self, screen):
        screen.blit(self.image, (self.x * CELL_SIZE, self.y * CELL_SIZE))


class GameRenderer:
    """
    A class responsible for rendering the game environment and its entities (hiders, seekers, walls)
    on the screen using Pygame. It visualizes episodes, showing the movements and interactions
    between hiders and seekers.

    Attributes:
        episodes_data (List[Episode]): Data for all episodes to be rendered.
        grid_size (int): The size of the game grid.
        total_time (int): The total time allowed for each episode.
        hiding_time (int): The time allocated for hiders to hide before seekers start seeking.
        visibility (int): The visibility radius for seekers.
        n_seekers (int): The number of seekers in the game.
        n_hiders (int): The number of hiders in the game.
    """

    def __init__(
        self,
        episodes_data: List[Episode],
        grid_size: int,
        total_time: int,
        hiding_time: int,
        visibility: int,
        n_seekers: int,
        n_hiders: int,
    ):
        """
        Initializes the GameRenderer with the necessary configuration and loads the visual assets.

        The initialization process sets up the Pygame window, loads images for the entities, and prepares
        the game for rendering episodes.
        """
        self.episodes_data: List[Episode] = episodes_data
        self.grid_size = grid_size
        self.total_time = total_time
        self.hiding_time = hiding_time
        self.visibility = visibility
        self.n_seekers = n_seekers
        self.n_hiders = n_hiders

        # Load images
        self.hider_images = [
            pygame.transform.flip(
                pygame.image.load("./img/duck_right.png"), True, False
            ),  # left
            pygame.image.load("./img/duck_right.png"),  # right
            pygame.image.load("./img/duck_back.png"),  # up
            pygame.image.load("./img/duck_front.png"),  # down, stay, front
            pygame.image.load("./img/duck_front.png"),  # down, stay, front
            pygame.image.load("./img/duck_found_front.png"),  # found - always front
        ]

        self.seeker_images = [
            pygame.transform.flip(
                pygame.image.load("./img/programmer_side.png"), True, False
            ),  # left
            pygame.image.load("./img/programmer_side.png"),  # right
            pygame.image.load("./img/programmer_back.png"),  # up
            pygame.image.load("./img/programmer_front.png"),  # down, stay, front
            pygame.image.load("./img/programmer_front.png"),  # down, stay, front
        ]

        self.wall_image = pygame.image.load("./img/wall.png")

        self.hiders_group: Dict[str, Hider] = {}
        self.seekers_group: Dict[str, Seeker] = {}
        self.visibility_group: Dict[str, Visibility] = {}
        self.walls_group: Dict[str, Wall] = {}

        pygame.init()  # Initialize the Pygame library
        self.font = pygame.font.SysFont("Arial", 25)  # Font for drawing text
        pygame.display.set_caption("Episode 0")
        # Create a window of appropriate size
        self.screen = pygame.display.set_mode(
            (self.grid_size * CELL_SIZE, self.grid_size * CELL_SIZE)
        )
        self.clock = pygame.time.Clock()  # Clock to control frame rate

    def create_groups(self):
        """
        Initializes and organizes hiders, seekers, visibility zones, and walls into groups
        for more efficient rendering and updates during the game.
        """
        for i in range(self.n_hiders):
            hider = Hider(name=f"hider_{i}", images=self.hider_images)
            self.hiders_group[f"hider_{i}"] = hider

        for i in range(self.n_seekers):
            seeker = Seeker(name=f"seeker_{i}", images=self.seeker_images)
            self.seekers_group[f"seeker_{i}"] = seeker
            visibility_ring = Visibility(
                seeker_name=f"seeker_{i}", radius=self.visibility
            )
            self.visibility_group[f"seeker_{i}"] = visibility_ring

    def set_positions(
        self,
        frame: Frame,
        frame_i: int,
    ):
        """
        Sets the positions of all entities (hiders, seekers, walls) for a specific frame
        based on the game state at that moment.

        Parameters:
            frame (Frame): The current frame to be rendered.
            frame_i (int): The index of the frame within the episode.
        """
        for x, col in enumerate(frame.state):
            for y, cell in enumerate(col):
                if cell == None:
                    continue
                for entity in cell:
                    if entity["type"] == "W":
                        if self.walls_group.get(f"frame_{x}_{y}") is None:
                            self.walls_group[f"frame_{x}_{y}"] = Wall(
                                image=self.wall_image
                            )
                        self.walls_group[f"frame_{x}_{y}"].set_pos(x, y)
                    elif entity["type"] == "S":
                        if frame_i > self.hiding_time:
                            self.seekers_group[entity["name"]].set_pos(
                                x, y, frame.actions["seekers"][entity["name"]]
                            )
                            self.visibility_group[entity["name"]].set_pos(x, y)
                            self.visibility_group[entity["name"]].set_visibility(True)
                        else:
                            self.visibility_group[entity["name"]].set_visibility(False)
                            self.seekers_group[entity["name"]].set_pos(x, y, 4)

                    elif entity["type"] == "H":
                        if frame.found[entity["name"]] is not None:
                            self.hiders_group[entity["name"]].set_pos(x, y, 5)
                        else:
                            if frame_i < self.hiding_time:
                                self.hiders_group[entity["name"]].set_pos(
                                    x, y, frame.actions["hiders"][entity["name"]]
                                )
                            else:
                                self.hiders_group[entity["name"]].set_pos(x, y, 4)

    def render_frame(self, frame_index):
        """
        Renders a single frame of the episode, drawing all entities (hiders, seekers, walls)
        on the screen based on their positions and states.

        Parameters:
            frame_index (int): The index of the frame to render, used to determine
            the game state such as hiding or seeking phase.
        """
        if frame_index < self.hiding_time:
            self.screen.fill("white")
        else:
            self.screen.fill("black")

        for visibility in self.visibility_group.values():
            visibility.draw(self.screen)

        for wall in self.walls_group.values():
            wall.draw(self.screen)

        for hider in self.hiders_group.values():
            hider.draw(self.screen, self.font)

        for seeker in self.seekers_group.values():
            seeker.draw(self.screen, self.font)

    def render(self):
        """
        The main rendering loop that goes through each episode and each frame, calling 
        render_frame for each moment of the game. Handles user input for pausing and 
        quitting the game, and displays the outcome of each episode.
        """
        running = True  # Flag to keep the loop running
        self.create_groups()  # Prepare the groups for rendering

        for ep in self.episodes_data:
            pygame.display.set_caption(f"Episode {ep.number}")  # Window title
            for frame_i, frame in enumerate(ep.frames):
                paused = False  # Flag for pausing the game
                # Event handling loop
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:  # If window close button clicked
                        running = False  # Exit the main loop
                        pygame.quit()  # Shut down Pygame
                        return
                    if (
                        event.type == pygame.KEYDOWN
                    ):  # Additional event handling for pausing
                        if event.key == pygame.K_SPACE:
                            paused = True

                while paused:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            running = False
                            pygame.quit()
                            return
                        if event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_SPACE:
                                paused = False
                                break
                    self.clock.tick(FRAMERATE)
                if not running:
                    break

                self.set_positions(
                    frame,
                    frame_i,
                )  # Update positions for rendering
                self.render_frame(frame_i)  # Render the current frame
                pygame.display.flip()  # Update the full display
                self.clock.tick(FRAMERATE)  # Maintain the frame rate

            if not running:  # Exit the loop if the running flag is False
                break

            # Display win/loss message at the end of each episode
            if ep.frames[-1].won["seekers"]:
                text = self.font.render(
                    f"Seekers won: {round(ep.rewards.seekers_total_reward,2)} vs Hiders: {round(ep.rewards.hiders_total_reward,2)}",
                    True,
                    (0, 0, 0),
                )
                self.screen.fill("blue")
                self.screen.blit(
                    text, text.get_rect(center=self.screen.get_rect().center)
                )

            else:
                text = self.font.render(
                    f"Hiders won: {round(ep.rewards.hiders_total_reward,2)} vs Seekeers: {round(ep.rewards.seekers_total_reward,2)}",
                    True,
                    (0, 0, 0),
                )
                self.screen.fill("yellow")
                self.screen.blit(
                    text, text.get_rect(center=self.screen.get_rect().center)
                )

            pygame.time.wait(500)

            pygame.display.flip()
            pygame.time.wait(500)

        running = False  # Ensure the running flag is set to False after the loop
        pygame.quit()  # Deinitialize Pygame modules
