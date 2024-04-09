from typing import List, Dict
import pygame
from environments.models import Episode, Frame
from rendering.models import Hider, Seeker, Visibility, Wall

FRAMERATE = 30
CELL_SIZE = 100


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
        map_config: List[List[int]],
        init_positions: Dict[str, List[int]],
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
        self.map_config: List[List[int]] = map_config
        self.init_positions: Dict[str, List[int]] = init_positions
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

        for x, col in enumerate(self.map_config):
            for y, cell in enumerate(col):
                if cell == 1:
                    wall = Wall(image=self.wall_image)
                    wall.set_pos(x, y)
                    self.walls_group[f"frame_{x}_{y}"] = wall

    def can_move(self, x: int, y: int) -> bool:
        """
        Checks if a given position is valid for an entity to move to.
        Parameters:
            x (int): The x-coordinate of the position.
            y (int): The y-coordinate of the position.
        """
        if x < 0 or y < 0 or x >= self.grid_size or y >= self.grid_size:
            return False
        for wall in self.walls_group.values():
            if wall.x == x and wall.y == y:
                return False
        return True

    def get_new_position(self, x, y, action) -> tuple[int, int]:
        """
        Returns the new position for a given entity based on the action taken.
        
        Parameters:
            x (int): The current x-coordinate of the entity.
            y (int): The current y-coordinate of the entity.
            action (int): The action taken by the entity.
        """
        if action == 0 and self.can_move(x, y - 1):
            return x, y - 1
        elif action == 1 and self.can_move(x, y + 1):
            return x, y + 1
        elif action == 2 and self.can_move(x - 1, y):
            return x - 1, y
        elif action == 3 and self.can_move(x + 1, y):
            return x + 1, y
        else:
            return x, y

    def set_positions(
        self,
        frame: Frame,
        frame_i: int,
    ):
        """
        Sets the positions of hiders, seekers for a specific frame
        based on the game state at that moment.

        Parameters:
            frame (Frame): The current frame to be rendered.
            frame_i (int): The index of the frame within the episode.
        """

        for hider in frame.actions["hiders"]:
            last_x: int = self.hiders_group[hider].x
            last_y: int = self.hiders_group[hider].y
            if frame.found[hider] is not None:
                self.hiders_group[hider].set_pos(last_x, last_y, 5)
                continue
            if frame_i >= self.hiding_time:
                self.hiders_group[hider].set_pos(last_x, last_y, 4)
                continue
            new_action: int = frame.actions["hiders"][hider]
            new_x, new_y = self.get_new_position(last_x, last_y, new_action)
            self.hiders_group[hider].set_pos(new_x, new_y, new_action)

        for seeker in frame.actions["seekers"]:
            last_x: int = self.seekers_group[seeker].x
            last_y: int = self.seekers_group[seeker].y
            if frame_i <= self.hiding_time:
                self.visibility_group[seeker].set_visibility(False)
                self.seekers_group[seeker].set_pos(last_x, last_y, 4)
                continue
            new_action: int = frame.actions["seekers"][seeker]
            new_x, new_y = self.get_new_position(last_x, last_y, new_action)
            self.seekers_group[seeker].set_pos(new_x, new_y, new_action)
            self.visibility_group[seeker].set_pos(new_x, new_y)
            self.visibility_group[seeker].set_visibility(True)

    def reset_positions(self):
        """
        Resets the positions of hiders, seekers to their initial state at the beginning of an episode.
        """

        for hider in self.hiders_group.values():
            pos = self.init_positions[hider.name]
            hider.set_pos(pos[0], pos[1], 4)

        for seeker in self.seekers_group.values():
            pos = self.init_positions[seeker.name]
            seeker.set_pos(pos[0], pos[1], 4)
            self.visibility_group[seeker.name].set_visibility(False)

    def render_frame(self, frame_index: int):
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
            self.reset_positions()
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
