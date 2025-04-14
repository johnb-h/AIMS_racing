import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import matplotlib
import numpy as np
import pygame

from evolution.evolution import CarEvolution
from user_interface.constants import (
    CAR_LENGTH,
    CAR_WIDTH,
    TARGET_FPS,
    WINDOW_HEIGHT_IN_M,
    WINDOW_WIDTH_IN_M,
    SPRITE_SCALE_MULTIPLIER
)
from user_interface.scene import Scene
from user_interface.states import State
from user_interface.utils import WorldVector2

BACKGROUND_COLOR = (0, 102, 16)
TRACK_COLOR = (50, 50, 50)


@dataclass
class Car:
    positions: List[Tuple[float, float]]
    selected: bool = False
    color: Tuple[int, int, int] = (255, 255, 255)
    last_position: Tuple[float, float] = (0, 0)


class GameScene(Scene):
    def __init__(
        self,
        num_visualised_cars: int = 10,
        car_colours: Optional[List[Tuple[int, int, int]]] = None,
    ):
        super().__init__()

        self.num_visualised_cars = num_visualised_cars
        self.car_colours = car_colours
        if self.car_colours is None:
            self.car_colours = matplotlib.cm.get_cmap("tab10")(
                range(self.num_visualised_cars)
            )
            self.car_colours = [
                (int(r * 255), int(g * 255), int(b * 255))
                for r, g, b, _ in self.car_colours
            ]
        self.crashed_time = None
        self.countdown = True
        self.countdown_time = pygame.time.get_ticks()
        self.countdown_text = ""
        self.font = pygame.font.Font("./assets/joystix_monospace.ttf", 36)
        self.font_go = pygame.font.Font("./assets/joystix_monospace.ttf", 72)

        # Load the F1 car sprite
        self.f1_car_sprite = pygame.image.load("assets/cleaned_f1.tiff").convert_alpha()

        self._init_display()
        self._init_track()
        self._init_evolution()
        self._init_state()
        self._init_visualization_cache()
        self.generate_new_population()

    def _init_display(self):
        """Initialize display settings based on current window."""
        info = pygame.display.Info()
        self.width = info.current_w
        self.height = info.current_h
        # Calculate aspect ratio preserving track dimensions
        self.track_scale = min(
            self.width / (WINDOW_WIDTH_IN_M * 1.2),  # Add margins
            self.height / (WINDOW_HEIGHT_IN_M * 1.2),
        )

    def _init_track(self):
        """Initialize track with dimensions relative to window size."""
        # Track setup with relative dimensions
        self.track_center = (self.width // 2, self.height // 2)

        # Scale track dimensions based on window size while maintaining aspect ratio
        track_height = (
            WINDOW_HEIGHT_IN_M * self.track_scale * 0.9
        )  # 90% of scaled height
        track_outer_width = (
            WINDOW_WIDTH_IN_M * self.track_scale * 0.9
        )  # 90% of scaled width
        track_inner_width = track_outer_width * 0.5  # Inner track is 50% of outer width
        track_inner_height = track_height * 0.5  # Inner track is 50% of outer height

        self.track_outer = self._generate_oval(
            self.track_center[0], self.track_center[1], track_outer_width, track_height
        )
        self.track_inner = self._generate_oval(
            self.track_center[0],
            self.track_center[1],
            track_inner_width,
            track_inner_height,
        )

        # Finish line setup - align vertically between inner and outer track
        outer_top = self.track_center[1] - track_height / 2
        inner_top = self.track_center[1] - track_inner_height / 2
        finish_line_center = (outer_top + inner_top) / 2
        track_gap = abs(outer_top - inner_top)
        self.finish_line = [
            (self.track_center[0], finish_line_center - track_gap / 2),
            (self.track_center[0], finish_line_center + track_gap / 2),
        ]

        # Store track dimensions for evolution
        self.track_outer_width = track_outer_width
        self.track_outer_height = track_height
        self.track_inner_width = track_inner_width
        self.track_inner_height = track_inner_height

    def _init_evolution(self):
        """Initialize evolution with track dimensions."""
        outer_top = self.track_center[1] - self.track_outer_height / 2
        inner_top = self.track_center[1] - self.track_inner_height / 2
        finish_line_center = (outer_top + inner_top) / 2

        self.evolution = CarEvolution(
            track_center=self.track_center,
            start_position=(self.track_center[0], finish_line_center),
            track_outer_width=self.track_outer_width,
            track_outer_height=self.track_outer_height,
            track_inner_width=self.track_inner_width,
            track_inner_height=self.track_inner_height,
            track_outer=self.track_outer,
            track_inner=self.track_inner,
            num_visualize=self.num_visualised_cars,
        )

    def resize(self, new_width: int, new_height: int):
        """Handle window resize events."""
        self.width = new_width
        self.height = new_height
        self._init_track()
        self._init_evolution()
        self._init_visualization_cache()
        self.generate_new_population()

    def _generate_oval(
        self, cx: float, cy: float, width: float, height: float
    ) -> List[Tuple[float, float]]:
        """Generate points for an oval track."""
        return [
            (cx + width / 2 * np.cos(angle), cy + height / 2 * np.sin(angle))
            for angle in np.linspace(0, 2 * np.pi, 100)
        ]

    def _create_track_surface(self) -> pygame.Surface:
        """Create the static track surface with boundaries and finish line."""
        surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)

        # Draw the filled track area
        pygame.draw.polygon(surface, TRACK_COLOR, self.track_outer)
        pygame.draw.polygon(
            surface, (0, 0, 0, 0), self.track_inner
        )  # Cut out the inner area

        # Draw track boundaries
        pygame.draw.lines(surface, (255, 0, 0), True, self.track_outer, 2)
        pygame.draw.lines(surface, (255, 0, 0), True, self.track_inner, 2)

        # Draw finish line
        stripe_height = 8
        total_height = self.finish_line[1][1] - self.finish_line[0][1]
        for i in range(int(total_height // stripe_height)):
            start_y = self.finish_line[0][1] + i * stripe_height
            end_y = min(start_y + stripe_height, self.finish_line[1][1])
            color = (255, 255, 255) if i % 2 == 0 else (0, 0, 0)
            pygame.draw.line(
                surface,
                color,
                (self.finish_line[0][0], start_y),
                (self.finish_line[0][0], end_y),
                4,
            )
        return surface

    def _check_finish_line_crossing(
        self, prev_pos: Tuple[float, float], current_pos: Tuple[float, float]
    ) -> bool:
        """Check if a line segment crosses the finish line."""

        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

        if self.current_step > 50 and prev_pos[0] < current_pos[0]:
            return ccw(prev_pos, self.finish_line[0], self.finish_line[1]) != ccw(
                current_pos, self.finish_line[0], self.finish_line[1]
            ) and ccw(prev_pos, current_pos, self.finish_line[0]) != ccw(
                prev_pos, current_pos, self.finish_line[1]
            )
        return False

    def generate_new_population(self):
        """Generate and initialize a new population of cars."""
        self.cars = [
            Car(positions=traj, color=color)
            for traj, color in zip(self.evolution.ask(), self.car_colours)
        ]
        self.current_step = 0
        self.crashed_time = None
        self.cars_driving = True
        self.finish_line_crossed = False
        self.finish_time = None

    def _update_mean_trajectory(self):
        """Update the cached mean trajectory visualization."""
        if self.generation == self.last_generation:
            return

        mean_traj = self.evolution.get_mean_trajectory()
        if mean_traj and len(mean_traj) > 1:
            self.mean_trajectory_surface = pygame.Surface(
                (self.width, self.height), pygame.SRCALPHA
            )
            points = mean_traj[::1]
            if len(points) >= 2:
                pygame.draw.lines(
                    self.mean_trajectory_surface,
                    (255, 215, 0, 180),
                    False,
                    points,
                    2,
                )
        self.last_generation = self.generation

    def _check_for_finish(self, up_to_step: int) -> bool:
        """Check if any car has crossed the finish line up to the given step."""
        if not self.finish_line_crossed:
            for car in self.cars:
                if len(car.positions) > 1:
                    positions = car.positions[: up_to_step + 1]
                    if len(positions) >= 2:
                        if self._check_finish_line_crossing(
                            positions[-2], positions[-1]
                        ):
                            self.finish_line_crossed = True
                            self.finish_time = up_to_step
                            self.cars_driving = False
                            return True
        return False

    def _draw_cars(self, screen):
        """Draw all car trajectories and current positions."""
        trajectory_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)

        # First check for finish line crossing
        self._check_for_finish(self.current_step)

        all_cars_crashed = True

        # Then draw cars and handle crashes
        for i, car in enumerate(self.cars):
            if len(car.positions) <= 1:
                continue

            # Get car status
            crashed = (
                self.evolution.displayed_crashed[i]
                if hasattr(self.evolution, "displayed_crashed")
                else False
            )
            crash_step = (
                self.evolution.displayed_crash_steps[i]
                if hasattr(self.evolution, "displayed_crash_steps")
                else len(car.positions)
            )
            all_cars_crashed = all_cars_crashed and crash_step < self.current_step

            # Determine how much of trajectory to draw
            max_step = self.current_step + 1
            max_step = min(max_step, crash_step)
            if self.finish_line_crossed:
                max_step = min(max_step, self.finish_time)

            car.last_position = car.positions[max_step - 1]
            positions = car.positions[:max_step]

            # Draw trajectory
            if len(positions) >= 2:
                pygame.draw.lines(trajectory_surface, car.color, False, positions, 2)

            # Draw current position marker
            if positions:
                current_pos = positions[-1]
                if (
                    0 <= current_pos[0] <= self.width
                    and 0 <= current_pos[1] <= self.height
                ):
                    if len(positions) > 1:
                        prev_pos = positions[-2]
                        dx, dy = (
                            current_pos[0] - prev_pos[0],
                            current_pos[1] - prev_pos[1],
                        )
                        angle = np.arctan2(dy, dx)

                        scaled_car = pygame.transform.scale_by(self.f1_car_sprite, self.track_scale * SPRITE_SCALE_MULTIPLIER)

                        # Create a colored version of the car
                        colored_car = scaled_car.copy()
                        colored_car.fill(car.color, special_flags=pygame.BLEND_RGBA_MULT)

                        # Calculate the rear axle position (pivot point for rotation)
                        rear_axle_offset = pygame.Vector2(
                            scaled_car.get_width() / 2, 0  # Use the actual scaled width
                        )
                        rear_axle_offset = rear_axle_offset.rotate_rad(angle)
                        rear_axle_position = current_pos - rear_axle_offset

                        # Rotate the car surface based on the direction angle
                        # Add 90 degrees to correct the initial orientation
                        rotated_car = pygame.transform.rotate(
                            colored_car,
                            -math.degrees(angle) + 90
                        )

                        # Get the rectangle of the rotated car for correct positioning
                        rotated_car_rect = rotated_car.get_rect()
                        rotated_car_rect.center = (
                            int(rear_axle_position.x),
                            int(rear_axle_position.y),
                        )

                        # Draw the rotated car onto the screen
                        screen.blit(rotated_car, rotated_car_rect.topleft)

                    # Draw crash marker
                    if crashed and max_step >= crash_step:
                        size = 15
                        pygame.draw.line(
                            trajectory_surface,
                            (255, 0, 0),
                            (current_pos[0] - size, current_pos[1] - size),
                            (current_pos[0] + size, current_pos[1] + size),
                            2,
                        )
                        pygame.draw.line(
                            trajectory_surface,
                            (255, 0, 0),
                            (current_pos[0] - size, current_pos[1] + size),
                            (current_pos[0] + size, current_pos[1] - size),
                            2,
                        )

            # Draw selection highlight
            if car.selected:
                pos = positions[-1] if positions else car.positions[0]
                pygame.draw.circle(trajectory_surface, (255, 255, 0), pos, 15, 2)

        if all_cars_crashed:
            self.cars_driving = False
            if self.crashed_time is None:
                self.crashed_time = self.current_step

        screen.blit(trajectory_surface, (0, 0))

    def _draw_ui(self, screen):
        """Draw UI elements including timer and instructions."""
        # Draw generation counter
        gen_text = self.font.render(f"Round: {self.generation}", True, (255, 255, 255))
        screen.blit(gen_text, (20, 20))

        # Draw timer
        if self.finish_line_crossed and self.finish_time is not None:
            elapsed_time = self.finish_time
        elif self.crashed_time is not None:
            elapsed_time = self.crashed_time
        else:
            elapsed_time = self.current_step

        elapsed_time = elapsed_time / TARGET_FPS
        timer_text = self.font.render(f"Time: {elapsed_time:.2f}s", True, (255, 255, 255))
        timer_rect = timer_text.get_rect(topright=(self.width - 20, 20))
        screen.blit(timer_text, timer_rect)

        # Draw instructions
        instructions = (
            "Racing..."
            # Press SPACE to end simulation"
            if self.cars_driving
            else "Which cars performed best?"
            # Press SPACE for next generation"
        )

        # Draw mean trajectory toggle instruction
        # mean_text = self.font.render("(m) show population mean", True, (255, 255, 255))
        # screen.blit(mean_text, (10, self.height - 80))

        # Draw restart instruction
        restart_text = self.font.render("(r) restart", True, (255, 255, 255))
        screen.blit(restart_text, (20, self.height - 120))

        # Draw exit instruction
        exit_text = self.font.render("(esc) exit", True, (255, 255, 255))
        screen.blit(exit_text, (20, self.height - 160))

        # Draw next instruction
        next_text = self.font.render("(space) next", True, (255, 255, 255))
        screen.blit(next_text, (20, self.height - 80))

        # Countdown from 3 to GO
        if self.countdown:
            countdown_text = self.font_go.render(self.countdown_text, True, (255, 255, 255), BACKGROUND_COLOR)
            countdown_rect = countdown_text.get_rect(
                center=(self.width // 2, self.height // 2)
            )
            screen.blit(countdown_text, countdown_rect)
            pygame.display.flip()
            if pygame.time.get_ticks() - self.countdown_time > 1000:
                self.countdown_time = pygame.time.get_ticks()
                if self.countdown_text == "":
                    self.countdown_text = "3"
                elif self.countdown_text == "3":
                    self.countdown_text = "2"
                elif self.countdown_text == "2":
                    self.countdown_text = "1"
                elif self.countdown_text == "1":
                    self.countdown_text = "GO!"
                elif self.countdown_text == "GO!":
                    self.countdown = False
        else:
            # Place instructions in the middle of the screen
            instructions_text = self.font.render(instructions, True, (255, 255, 255))
            instructions_rect = instructions_text.get_rect(
                center=(self.width // 2, self.height // 2)
            )
            screen.blit(instructions_text, instructions_rect)

    def handle_events(self, events):
        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()
                for car in self.cars:
                    if not car.positions:
                        continue
                    pos = car.last_position
                    if (
                        np.sqrt(
                            (pos[0] - mouse_pos[0]) ** 2 + (pos[1] - mouse_pos[1]) ** 2
                        )
                        < 15
                    ):
                        car.selected = not car.selected

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self._next_state = State.MAIN_MENU
                elif event.key == pygame.K_SPACE:
                    self._handle_space_key()
                elif event.key == pygame.K_m:
                    self.show_mean = not self.show_mean
                elif pygame.K_0 <= event.key <= pygame.K_9:
                    # TODO link this with button presses
                    car_index = event.key - pygame.K_0
                    if car_index < len(self.cars):
                        self.cars[car_index].selected = not self.cars[
                            car_index
                        ].selected
                elif event.key == pygame.K_r:
                    self.restart()

    def restart(self):
        self._init_state()
        self._init_evolution()
        self._init_visualization_cache()
        self.generation = 1
        self.generate_new_population()

    def _handle_space_key(self):
        if self.countdown:
            return
        if self.cars_driving:
            max_steps = max(len(car.positions) for car in self.cars)
            if hasattr(self.evolution, "displayed_crash_steps"):
                max_steps = max(self.evolution.displayed_crash_steps)

            while self.current_step < max_steps - 1:
                self.current_step += 1
                if self._check_for_finish(self.current_step):
                    break

            if not self.finish_line_crossed:
                self.cars_driving = False
        else:
            selected = [i for i, car in enumerate(self.cars) if car.selected]
            if selected:
                self.evolution.tell(selected)
                self.generation += 1
                self.generate_new_population()

    def update(self, dt):
        if self.countdown:
            return
        self.current_step += 1
        return self._next_state

    def draw(self, screen):
        screen.fill(BACKGROUND_COLOR)
        screen.blit(self.track_surface, (0, 0))

        if self.show_mean:
            self._update_mean_trajectory()
            if self.mean_trajectory_surface:
                screen.blit(self.mean_trajectory_surface, (0, 0))

        if not self.countdown:
            self._draw_cars(screen)
        self._draw_ui(screen)

    def reset(self):
        self._next_state = None
        self.generate_new_population()

    def _init_state(self):
        """Initialize state variables."""
        self._next_state = None
        self.cars: List[Car] = []
        self.current_step = 0
        self.cars_driving = True
        self.generation = 1
        self.show_mean = False
        self.finish_line_crossed = False
        self.finish_time = None

    def _init_visualization_cache(self):
        """Initialize visualization cache."""
        self.mean_trajectory_surface = None
        self.last_generation = -1
        self.track_surface = self._create_track_surface()
