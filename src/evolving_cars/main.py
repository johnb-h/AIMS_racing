from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pygame

from .evolution import CarEvolution


@dataclass
class Car:
    positions: List[Tuple[float, float]]  # List of positions forming the trajectory
    fitness: float = 0.0
    selected: bool = False
    color: Tuple[int, int, int] = (255, 255, 255)  # White by default


class TrackVisualizer:
    def __init__(self, width=800, height=600):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Evolutionary Cars")
        self.clock = pygame.time.Clock()

        # Track boundaries (simple oval track)
        self.track_outer = self._generate_oval(
            width // 2, height // 2, width * 0.8, height * 0.7
        )
        self.track_inner = self._generate_oval(
            width // 2, height // 2, width * 0.4, height * 0.35
        )

        # Initialize evolution
        self.evolution = CarEvolution(
            track_outer=self.track_outer,
            track_inner=self.track_inner,
        )

        self.cars: List[Car] = []
        self.current_step = 0
        self.simulation_running = True
        self.generation = 0
        self.show_mean = False
        self.start_time = pygame.time.get_ticks()
        self.finish_time = None

        # Define finish line coordinates (vertical line)
        center_x, center_y = width // 2, height // 2
        self.finish_line = [
            (center_x, center_y - 175 - 40),  # Top point
            (center_x, center_y - 175 + 40),  # Bottom point
        ]
        self.finish_line_crossed = False

        # Cache for mean trajectory visualization
        self.mean_trajectory_surface = None
        self.last_generation = -1

        # Create cached track surface
        self.track_surface = pygame.Surface((width, height), pygame.SRCALPHA)
        self._draw_static_track()

        # Generate initial population
        self.generate_new_population()

    def _generate_oval(
        self, cx: float, cy: float, width: float, height: float
    ) -> List[Tuple[float, float]]:
        points = []
        for angle in np.linspace(0, 2 * np.pi, 100):
            x = cx + width / 2 * np.cos(angle)
            y = cy + height / 2 * np.sin(angle)
            points.append((x, y))
        return points

    def _draw_static_track(self):
        """Draw the static track elements to a cached surface."""
        # Draw outer boundary
        pygame.draw.lines(self.track_surface, (255, 0, 0), True, self.track_outer, 2)
        # Draw inner boundary
        pygame.draw.lines(self.track_surface, (255, 0, 0), True, self.track_inner, 2)
        # Draw finish line (black and white striped)
        stripe_height = 8
        total_height = self.finish_line[1][1] - self.finish_line[0][1]
        num_stripes = total_height // stripe_height
        for i in range(int(num_stripes)):
            start_y = self.finish_line[0][1] + i * stripe_height
            end_y = start_y + stripe_height
            if end_y > self.finish_line[1][1]:
                end_y = self.finish_line[1][1]
            color = (255, 255, 255) if i % 2 == 0 else (0, 0, 0)
            pygame.draw.line(
                self.track_surface,
                color,
                (self.finish_line[0][0], start_y),
                (self.finish_line[0][0], end_y),
                4,
            )

    def _check_finish_line_crossing(self, prev_pos, current_pos):
        """Check if a line segment crosses the finish line."""

        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

        def intersect(A, B, C, D):
            return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

        # Only check if car is moving rightward (to avoid false positives when starting)
        if self.current_step > 50 and prev_pos[0] < current_pos[0]:  # Moving rightward
            return intersect(
                prev_pos, current_pos, self.finish_line[0], self.finish_line[1]
            )
        return False

    def generate_new_population(self):
        # Generate new population
        self.cars.clear()
        trajectories = self.evolution.ask()

        for trajectory in trajectories:
            car = Car(
                positions=trajectory,
                color=(
                    np.random.randint(100, 255),
                    np.random.randint(100, 255),
                    np.random.randint(100, 255),
                ),
            )
            self.cars.append(car)

        self.current_step = 0
        self.simulation_running = True
        self.finish_line_crossed = False
        self.finish_time = None
        self.start_time = pygame.time.get_ticks()

    def evolve_next_generation(self):
        # Get indices of selected cars
        selected_indices = [i for i, car in enumerate(self.cars) if car.selected]

        if len(selected_indices) > 0:
            # Update evolution strategy with selected cars
            self.evolution.tell(selected_indices)
            self.generation += 1

            # Generate new population
            self.generate_new_population()
        else:
            print("Please select at least one car before evolving!")

    def _update_mean_trajectory(self):
        """Update cached mean trajectory surface if generation changed."""
        if self.generation == self.last_generation:
            return

        mean_traj = self.evolution.get_mean_trajectory()
        if mean_traj and len(mean_traj) > 1:
            # Create new surface for mean trajectory
            self.mean_trajectory_surface = pygame.Surface(
                (self.width, self.height), pygame.SRCALPHA
            )

            step = 1
            for i in range(0, len(mean_traj) - step, step):
                start = mean_traj[i]
                end = mean_traj[i + step]
                pygame.draw.line(
                    self.mean_trajectory_surface, (255, 215, 0, 180), start, end, 2
                )

        self.last_generation = self.generation

    def add_car(self, car: Car):
        self.cars.append(car)

    def draw_track(self):
        """Draw the cached track surface."""
        self.screen.blit(self.track_surface, (0, 0))

    def _get_last_visible_position(self, positions):
        """Get the last position that was within screen bounds."""
        for pos in reversed(positions):
            x, y = pos
            if 0 <= x <= self.width and 0 <= y <= self.height:
                return pos
        return positions[0]  # Fallback to start position if none visible

    def draw_cars(self):
        """Draw cars with optimized rendering."""
        # Draw trajectories first
        trajectory_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)

        for i, car in enumerate(self.cars):
            if len(car.positions) > 1:
                # Get crash status for this car
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

                # Draw trajectory up to crash point or current step
                max_step = (
                    min(crash_step, self.current_step + 1)
                    if crashed
                    else self.current_step + 1
                )

                # Sample trajectory points for smoother rendering
                positions = car.positions[:max_step]
                if len(positions) >= 2:
                    # Check for finish line crossing
                    if not self.finish_line_crossed and max_step > 1:
                        prev_pos = positions[-2]
                        current_pos = positions[-1]
                        if self._check_finish_line_crossing(prev_pos, current_pos):
                            self.finish_line_crossed = True
                            self.finish_time = pygame.time.get_ticks()
                            self.simulation_running = False

                    step = 1
                    sampled_positions = positions[::step]
                    if len(sampled_positions) >= 2:
                        pygame.draw.lines(
                            trajectory_surface, car.color, False, sampled_positions, 2
                        )

                # Draw current position
                if max_step > 0:
                    current_pos = positions[-1]
                    if (
                        0 <= current_pos[0] <= self.width
                        and 0 <= current_pos[1] <= self.height
                    ):
                        # Draw direction indicator only for current position
                        if len(positions) > 1:
                            prev_pos = positions[-2]
                            dx = current_pos[0] - prev_pos[0]
                            dy = current_pos[1] - prev_pos[1]
                            length = 15
                            angle = np.arctan2(dy, dx)
                            tip = current_pos
                            left = (
                                tip[0] - length * np.cos(angle + 2.5),
                                tip[1] - length * np.sin(angle + 2.5),
                            )
                            right = (
                                tip[0] - length * np.cos(angle - 2.5),
                                tip[1] - length * np.sin(angle - 2.5),
                            )
                            pygame.draw.polygon(
                                trajectory_surface, car.color, [tip, left, right]
                            )

                        # Draw red X only if car has crashed and we've reached the crash step
                        if crashed and self.current_step >= crash_step:
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

                # Highlight selected cars
                if car.selected:
                    if self.simulation_running and max_step > 0:
                        pos = positions[-1]
                        if 0 <= pos[0] <= self.width and 0 <= pos[1] <= self.height:
                            pygame.draw.circle(
                                trajectory_surface, (255, 255, 0), pos, 15, 2
                            )
                    else:
                        pos = self._get_last_visible_position(positions)
                        pygame.draw.circle(
                            trajectory_surface, (255, 255, 0), pos, 15, 2
                        )

        # Blit all trajectories at once
        self.screen.blit(trajectory_surface, (0, 0))

    def draw_ui(self):
        font = pygame.font.Font(None, 36)

        # Draw generation counter
        gen_text = font.render(f"Generation: {self.generation}", True, (255, 255, 255))
        self.screen.blit(gen_text, (10, 10))

        # Draw timer in top right with hundredths of a second
        if self.finish_time is None:
            elapsed_time = (pygame.time.get_ticks() - self.start_time) / 1000.0
        else:
            elapsed_time = (self.finish_time - self.start_time) / 1000.0
        timer_text = font.render(f"Time: {elapsed_time:.2f}s", True, (255, 255, 255))
        timer_rect = timer_text.get_rect()
        timer_rect.topright = (self.width - 10, 10)
        self.screen.blit(timer_text, timer_rect)

        # Update instructions to reflect all controls
        if self.simulation_running:
            instructions = font.render(
                "Click crashed cars to select, press SPACE to end simulation",
                True,
                (255, 255, 255),
            )
        else:
            instructions = font.render(
                "Click cars to select, press SPACE for next generation",
                True,
                (255, 255, 255),
            )
        self.screen.blit(instructions, (10, self.height - 40))

        # Add mean trajectory toggle instruction
        mean_instructions = font.render(
            "(m) show population mean", True, (255, 255, 255)
        )
        self.screen.blit(mean_instructions, (10, self.height - 80))

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Handle car selection during simulation (for crashed cars) or when simulation is finished
                mouse_pos = pygame.mouse.get_pos()
                for i, car in enumerate(self.cars):
                    if len(car.positions) > 0:
                        # During simulation, only allow selecting crashed cars
                        if self.simulation_running:
                            if (
                                hasattr(self.evolution, "displayed_crashed")
                                and self.evolution.displayed_crashed[i]
                            ):
                                car_pos = car.positions[
                                    min(self.current_step, len(car.positions) - 1)
                                ]
                                distance = np.sqrt(
                                    (car_pos[0] - mouse_pos[0]) ** 2
                                    + (car_pos[1] - mouse_pos[1]) ** 2
                                )
                                if distance < 15:
                                    car.selected = not car.selected
                        else:
                            # When simulation is finished, allow selecting any car
                            car_pos = self._get_last_visible_position(car.positions)
                            distance = np.sqrt(
                                (car_pos[0] - mouse_pos[0]) ** 2
                                + (car_pos[1] - mouse_pos[1]) ** 2
                            )
                            if distance < 15:
                                car.selected = not car.selected
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    if self.simulation_running:
                        # End simulation immediately
                        self.simulation_running = False
                        # Set current_step to end of trajectories
                        self.current_step = max(len(car.positions) for car in self.cars)
                    else:
                        self.evolve_next_generation()
                elif event.key == pygame.K_m:
                    self.show_mean = not self.show_mean
                    if self.show_mean:
                        self._update_mean_trajectory()
        return True

    def run(self):
        """Main game loop."""
        running = True
        while running:
            # Handle events
            running = self.handle_events()

            self.screen.fill((0, 0, 0))  # Black background

            # Draw track first
            self.draw_track()

            # Update and draw mean trajectory if enabled
            if self.show_mean:
                self._update_mean_trajectory()
                if self.mean_trajectory_surface:
                    self.screen.blit(self.mean_trajectory_surface, (0, 0))

            # Draw cars on top
            self.draw_cars()
            self.draw_ui()

            if self.simulation_running:
                # Update simulation step
                if self.current_step < max(len(car.positions) for car in self.cars) - 1:
                    self.current_step += 1

            pygame.display.flip()
            self.clock.tick(60)  # Limit to 60 FPS

        pygame.quit()
