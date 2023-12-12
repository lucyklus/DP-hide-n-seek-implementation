from typing import List, Dict
from dataclasses import dataclass
import pygame
import numpy as np


@dataclass
class Frame:
    state: List[List[Dict[str, str]]]
    rewards: Dict[str, Dict[str, float]]
    actions: Dict[str, Dict[str, int]]
    terminations: Dict[str, bool]
    done: Dict[str, Dict[str, int]]
    won: Dict[str, bool]
    found: Dict[str, str]


Episode = List[Frame]

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
    Renders all episodes from episodes_data using pygame

    """

    # pygame setup
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

        pygame.init()
        self.font = pygame.font.SysFont("Arial", 25)
        pygame.display.set_caption("Episode 0")
        self.screen = pygame.display.set_mode(
            (self.grid_size * CELL_SIZE, self.grid_size * CELL_SIZE)
        )
        self.clock = pygame.time.Clock()

    def create_groups(self):
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
        for x, col in enumerate(frame.state):
            for y, cell in enumerate(col):
                if cell["type"] == "W":
                    if self.walls_group.get(f"frame_{x}_{y}") is None:
                        self.walls_group[f"frame_{x}_{y}"] = Wall(image=self.wall_image)
                    self.walls_group[f"frame_{x}_{y}"].set_pos(x, y)
                elif cell["type"] == "S":
                    if frame_i > self.hiding_time:
                        self.seekers_group[cell["name"]].set_pos(
                            x, y, frame.actions["seekers"][cell["name"]]
                        )
                        self.visibility_group[cell["name"]].set_pos(x, y)
                        self.visibility_group[cell["name"]].set_visibility(True)
                    else:
                        self.visibility_group[cell["name"]].set_visibility(False)
                        self.seekers_group[cell["name"]].set_pos(x, y, 4)

                elif cell["type"] == "H":
                    if frame.found[cell["name"]] is not None:
                        self.hiders_group[cell["name"]].set_pos(x, y, 5)
                    else:
                        if frame_i < self.hiding_time:
                            self.hiders_group[cell["name"]].set_pos(
                                x, y, frame.actions["hiders"][cell["name"]]
                            )
                        else:
                            self.hiders_group[cell["name"]].set_pos(x, y, 4)

    def render_frame(self, frame_index):
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
        running = True

        self.create_groups()

        for ep_index, ep in enumerate(self.episodes_data):
            pygame.display.set_caption(f"Episode {ep_index}")
            for frame_i, frame in enumerate(ep):
                paused = False
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_SPACE:
                            paused = True

                while paused:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            running = False
                            paused = False
                        if event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_SPACE:
                                paused = False
                                break
                    self.clock.tick(FRAMERATE)
                if running == False:
                    break

                self.set_positions(
                    frame,
                    frame_i,
                )

                self.render_frame(frame_i)

                pygame.display.flip()
                self.clock.tick(FRAMERATE)

            if running == False:
                break

            if ep[-1].won["seekers"]:
                total_rewards = sum(ep[-1].rewards["seekers"].values())
                text = self.font.render(
                    f"Seekers won with total rewards: {total_rewards}",
                    True,
                    (0, 0, 0),
                )
                self.screen.fill("blue")
                self.screen.blit(
                    text, text.get_rect(center=self.screen.get_rect().center)
                )
            else:
                total_rewards = sum(ep[-1].rewards["hiders"].values())
                text = self.font.render(
                    f"Hiders won with total rewards: {total_rewards}",
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

        running = False
        pygame.quit()
