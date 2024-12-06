import pygame
import math
from typing import Callable, Tuple
import matplotlib

RESOLUTION = (1200, 800)
WINDOW_WIDTH_IN_M = 400  # In meters

# Constants
CONST_SPEED = 50  # In m/s
CONST_STEER = -0.05  # In radians

WHEELBASE = 2.5
CAR_WIDTH = 1.5
CAR_LENGTH = 4

MAX_STEERING_RADIUS = 400

N_CARS = 5

OFF_TRACK_MULTIPLIER = 0.2


def pixels_to_meters(pixels):
    # Irrespective of resolution, we assume the track is a fixed length across
    return pixels * WINDOW_WIDTH_IN_M / RESOLUTION[0]


def meters_to_pixels(meters):
    return meters * (RESOLUTION[0] / WINDOW_WIDTH_IN_M)


WINDOW_HEIGHT_IN_M = pixels_to_meters(RESOLUTION[1])
SCREEN_CENTER_METERS = (WINDOW_WIDTH_IN_M / 2, WINDOW_HEIGHT_IN_M / 2)

SteeringFunction = Callable[[float], float]


class WorldVector2(pygame.Vector2):
    def __init__(self, x, y):
        # Assume we are given the position in meters
        x = meters_to_pixels(x)
        y = meters_to_pixels(y)
        super().__init__(x, y)


class WorldRect(pygame.Rect):
    def __init__(self, left, top, width, height):
        # Assume we are given the position in meters
        left = meters_to_pixels(left)
        top = meters_to_pixels(top)
        width = meters_to_pixels(width)
        height = meters_to_pixels(height)
        super().__init__(left, top, width, height)


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
        return self.inner_radius.collidepoint(position) and not self.outer_radius.collidepoint

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


class RacingGame:
    def __init__(self, track: RaceTrack, cars: list[Car]):
        self.track = track
        self.cars = cars

    def update(self, dt):
        for car in self.cars:
            #car.is_on_track = self.track.is_on_track(car.pixel_position)
            car.update(dt)

    def draw(self, screen):
        self.track.draw(screen)
        for car in self.cars:
            car.draw(screen)


# Main Game Loop
def main():
    pygame.init()
    screen = pygame.display.set_mode(RESOLUTION)
    clock = pygame.time.Clock()

    # car = Car(pygame.Vector2(400, 300), (255, 0, 0), lambda _: CONST_STEER)
    track = RaceTrack(
        width_extent=0.9 * WINDOW_WIDTH_IN_M,
        length_extent=0.9 * WINDOW_HEIGHT_IN_M,
        track_width=50,
    )

    # Assign colours according to the matplotlib colours
    car_colours = matplotlib.cm.get_cmap("tab10")(range(N_CARS))
    car_colours = [(int(r * 255), int(g * 255), int(b * 255)) for r, g, b, _ in car_colours]
    
    cars = track.spawn_cars_on_starting_line(
        car_colours, [lambda _: CONST_STEER for _ in range(N_CARS)]
    )

    racing_game = RacingGame(track, cars)

    running = True
    while running:
        dt = clock.tick(60) / 1000.0  # Delta time in seconds

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        racing_game.update(dt)

        # Rendering
        racing_game.draw(screen)
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
