from typing import Callable, List, Tuple

import numpy as np
import pygame

from user_interface.car import Car
from user_interface.constants import SCREEN_CENTER_METERS
from user_interface.utils import SteeringFunction, WorldRect


def point_in_ellipse(point, center, radius_x, radius_y):
    """Check if the point is inside an ellipse defined by the given center and radii."""
    x, y = point
    cx, cy = center
    return ((x - cx) ** 2) / (radius_x**2) + ((y - cy) ** 2) / (radius_y**2) <= 1


class RaceTrack:
    def __init__(
        self,
        width_extent: float,
        length_extent: float,
        track_width: float,
        starting_line_thickness: float = 1,
    ):
        # All of these values are assumed to be in meters.
        # We also assume we will center at the center of the screen
        # We just need to create the outer and inner radius rectangles
        outer_left = SCREEN_CENTER_METERS[0] - width_extent / 2
        outer_top = SCREEN_CENTER_METERS[1] - length_extent / 2
        self.outer_radius = WorldRect(
            outer_left, outer_top, width_extent, length_extent
        )
        inner_left = SCREEN_CENTER_METERS[0] - (width_extent - track_width) / 2
        inner_top = SCREEN_CENTER_METERS[1] - (length_extent - track_width) / 2
        self.inner_radius = WorldRect(
            inner_left,
            inner_top,
            width_extent - track_width,
            length_extent - track_width,
        )

        self.starting_line_thickness = starting_line_thickness

    def starting_line(self) -> Tuple[pygame.Vector2, pygame.Vector2]:
        """Return vectors at the left and right of the starting line."""
        # Left is at the bottom middle of the inner radius
        left = pygame.Vector2(self.inner_radius.centerx, self.inner_radius.bottom)

        # Right is at the bottom middle of the outer radius
        right = pygame.Vector2(self.outer_radius.centerx, self.outer_radius.bottom)

        return left, right

    def spawn_cars_on_starting_line(
        self,
        colours: list[Tuple[int, int, int]],
        steering_funcs: list[SteeringFunction],
    ) -> list[Car]:
        """Spawn n_cars on the starting line."""
        n_cars = len(steering_funcs)
        left, right = self.starting_line()
        car_positions = [
            left.lerp(right, (i + 0.5) / n_cars) for i, _ in enumerate(steering_funcs)
        ]
        return [
            Car(pos, colour, steering_func)
            for pos, colour, steering_func in zip(
                car_positions, colours, steering_funcs
            )
        ]

    def is_on_track(self, position: pygame.Vector2) -> bool:
        """Check if the given position is on the track."""
        # Check if the position is within the inner radius and outside the outer radius
        is_in_inner_radius = point_in_ellipse(
            position,
            self.inner_radius.center,
            self.inner_radius.width / 2,
            self.inner_radius.height / 2,
        )
        is_in_outer_radius = point_in_ellipse(
            position,
            self.inner_radius.center,
            self.outer_radius.width / 2,
            self.outer_radius.height / 2,
        )
        is_on_track = is_in_outer_radius and not is_in_inner_radius
        return is_on_track

    def draw(self, screen):
        # Fill the background with green
        screen.fill((34, 139, 34))  # Green

        # Draw the outer ellipse (track boundary)
        pygame.draw.ellipse(screen, (169, 169, 169), self.outer_radius)

        # Draw the inner ellipse (track cutout)
        pygame.draw.ellipse(screen, (34, 139, 34), self.inner_radius)

        # Draw the starting line
        left, right = self.starting_line()
        pygame.draw.line(
            screen, (255, 255, 255), left, right, self.starting_line_thickness
        )
