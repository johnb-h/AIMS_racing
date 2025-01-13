import pygame
from typing import Callable

from constants import (
    WINDOW_WIDTH_IN_M,
    RESOLUTION
)

SteeringFunction = Callable[[float], float]


def meters_to_pixels(meters):
    return meters * (RESOLUTION[0] / WINDOW_WIDTH_IN_M)


class WorldRect(pygame.Rect):
    def __init__(self, left, top, width, height):
        # Assume we are given the position in meters
        left = meters_to_pixels(left)
        top = meters_to_pixels(top)
        width = meters_to_pixels(width)
        height = meters_to_pixels(height)
        super().__init__(left, top, width, height)

class WorldVector2(pygame.Vector2):
    def __init__(self, x, y):
        # Assume we are given the position in meters
        x = meters_to_pixels(x)
        y = meters_to_pixels(y)
        super().__init__(x, y)
