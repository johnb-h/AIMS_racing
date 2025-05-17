from typing import Callable, List, Tuple

import numpy as np
import pygame

from user_interface.constants import (
    CAR_LENGTH,
    CAR_WIDTH,
    CONST_SPEED,
    OFF_TRACK_MULTIPLIER,
    WHEELBASE,
)
from user_interface.utils import SteeringFunction


class Car:
    def __init__(self, pixel_position, colour, steering_func: SteeringFunction):
        self.pixel_position = pixel_position
        self.colour = colour
        self.velocity = pygame.Vector2(0, 0)
        self.speed = 0
        self.direction_angle = 0
        self.steering_angle = 0
        self.wheelbase = WHEELBASE

        self.is_on_track = True

        # Expected to take in the current time and return the steering angle
        self.steering_func = steering_func

        self._t = 0.0

        self.past_positions = []

    def increment_time(self, dt):
        self._t += dt
        self.update_parameters()

    def update_parameters(self):
        self.update_speed()
        self.update_steering_angle()

    def update_speed(self):
        self.speed = CONST_SPEED
        if not self.is_on_track:
            self.speed *= OFF_TRACK_MULTIPLIER

    def update_steering_angle(self):
        self.steering_angle = self.steering_func(self._t)

    def update(self, dt):
        self.past_positions.append((self.pixel_position.x, self.pixel_position.y))
        # Update position based on speed and steering angle
        self.increment_time(dt)
