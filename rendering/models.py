from typing import List
import pygame

CELL_SIZE = 100


class Hider:
    """
    Represents a hider in the game with a specific position and direction. It can be drawn on the screen with its designated sprite.
    """

    def __init__(self, name: str, images: List[pygame.Surface]):
        """
        Initializes a Hider with a name, a list of images for different directions, and default position and direction.
        """
        self.name = name
        self.images = images
        self.x = 0
        self.y = 0
        self.direction = 0

    def set_pos(self, x, y, direction):
        """
        Sets the position and direction of the Hider.
        """
        self.x = x
        self.y = y
        self.direction = direction

    def draw(self, screen, font):
        """
        Draws the Hider at its current position on the screen, including its name.
        """
        screen.blit(
            self.images[self.direction], (self.x * CELL_SIZE, self.y * CELL_SIZE)
        )
        screen.blit(
            font.render(self.name, True, (0, 0, 0)),
            (self.x * CELL_SIZE, self.y * CELL_SIZE),
        ),


class Seeker:
    """
    Represents a seeker in the game, similar to a Hider but with its own set of images and logic for rendering.
    """

    def __init__(self, name: str, images: List[pygame.Surface]):
        """
        Initializes a Seeker with a name, images for rendering, and default position and direction.
        """
        self.name = name
        self.images = images
        self.x = 0
        self.y = 0
        self.direction = 0

    def set_pos(self, x, y, direction=4):
        """
        Sets the position and direction of the Seeker.
        """
        self.x = x
        self.y = y
        self.direction = direction

    def draw(self, screen, font):
        """
        Draws the Seeker at its current position on the screen, including its name.
        """
        screen.blit(
            self.images[self.direction], (self.x * CELL_SIZE, self.y * CELL_SIZE)
        )
        screen.blit(
            font.render(self.name, True, (0, 0, 0)),
            (self.x * CELL_SIZE, self.y * CELL_SIZE),
        ),


class Visibility:
    """
    Represents the visibility range of a seeker. It is used to visually indicate how far a seeker can see on the grid.
    """

    def __init__(self, seeker_name: str, radius: int):
        """
        Initializes the visibility range with the name of the associated seeker, its radius, and default position.
        """
        self.name = seeker_name
        self.x = 0
        self.y = 0
        self.direction = 0
        self.radius = radius
        self.visibility = False

    def set_pos(self, x, y):
        """
        Sets the position of the visibility circle based on the seeker's position.
        """
        self.x = x
        self.y = y

    def set_visibility(self, visibility: bool):
        """
        Enables or disables the visibility circle to be drawn based on the seeker's ability to see.
        """
        self.visibility = visibility

    def draw(self, screen):
        """
        Draws the visibility circle on the screen if visibility is enabled.
        """
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
    """
    Represents a wall or obstacle in the game environment. Walls are static and do not move or change direction.
    """

    def __init__(self, image: pygame.Surface):
        """
        Initializes a Wall with a specific image for rendering.
        """
        self.image = image
        self.x = 0
        self.y = 0

    def set_pos(self, x, y):
        """
        Sets the position of the Wall on the grid.
        """
        self.x = x
        self.y = y

    def draw(self, screen):
        """
        Draws the Wall at its current position on the screen.
        """
        screen.blit(self.image, (self.x * CELL_SIZE, self.y * CELL_SIZE))
