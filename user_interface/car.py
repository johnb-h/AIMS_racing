import math
import pygame

from utils import (
    SteeringFunction,
    meters_to_pixels,
    WorldVector2
)
from constants import (
    WHEELBASE,
    CONST_SPEED,
    OFF_TRACK_MULTIPLIER,
    CAR_LENGTH,
    CAR_WIDTH
)

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
        # Update position based on speed and steering angle
        self.increment_time(dt)

        if abs(self.steering_angle) > 1e-5:  # Turning
            # Compute the turning radius
            turning_radius = self.wheelbase / math.tan(self.steering_angle)
            # Compute angular velocity
            angular_velocity = self.speed / turning_radius
        else:  # Driving straight
            angular_velocity = 0

        # Update the car's heading direction
        self.direction_angle += angular_velocity * dt

        movement = WorldVector2(self.speed * dt, 0).rotate_rad(self.direction_angle)
        self.pixel_position += movement

    def draw(self, screen):
        rect_size = (meters_to_pixels(CAR_LENGTH), meters_to_pixels(CAR_WIDTH))

        # Create a surface for the car with the correct size
        car_surface = pygame.Surface(
            rect_size, pygame.SRCALPHA
        )  # Surface with alpha transparency
        car_surface.fill(self.colour)  # Fill the surface with a red color

        # Calculate the rear axle position (pivot point for rotation)
        # NOTE: The signs seem wrong here (looks like I'm rotating around the front wheels?) but it looks way
        # more natural IMO. Maybe change later.
        rear_axle_offset = WorldVector2(
            CAR_LENGTH / 2, 0
        )  # Rear axle is half the car's length behind the center
        rear_axle_offset = rear_axle_offset.rotate_rad(
            self.direction_angle
        )  # Rotate offset by car's direction
        rear_axle_position = self.pixel_position + rear_axle_offset

        # Rotate the car surface based on the direction angle
        rotated_car = pygame.transform.rotate(
            car_surface,
            -math.degrees(
                self.direction_angle
            ),  # Rotate by the car's direction angle (convert radians to degrees)
        )

        # Get the rectangle of the rotated car for correct positioning
        rotated_car_rect = rotated_car.get_rect()
        rotated_car_rect.center = (int(rear_axle_position.x), int(rear_axle_position.y))

        # Draw the rotated car onto the screen
        screen.blit(rotated_car, rotated_car_rect.topleft)

