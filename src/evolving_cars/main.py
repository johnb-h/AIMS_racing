from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pygame

from .evolution import CarEvolution


@dataclass
class Car:
    positions: List[Tuple[float, float]]
    selected: bool = False
    color: Tuple[int, int, int] = (255, 255, 255)


class TrackVisualizer:
    def __init__(self, width=800, height=600):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Evolutionary Cars")
        self.clock = pygame.time.Clock()

        # Track setup
        self.track_center = (width // 2, height // 2)
        self.track_outer = self._generate_oval(
            self.track_center[0], self.track_center[1], width * 0.8, height * 0.7
        )
        self.track_inner = self._generate_oval(
            self.track_center[0], self.track_center[1], width * 0.4, height * 0.35
        )

        # Finish line setup
        self.finish_line = [
            (self.track_center[0], self.track_center[1] - 175 - 40),  # Top
            (self.track_center[0], self.track_center[1] - 175 + 40),  # Bottom
        ]

        # Evolution setup
        self.evolution = CarEvolution(
            track_outer=self.track_outer,
            track_inner=self.track_inner,
            slow_down=True,
        )

        # State variables
        self.cars: List[Car] = []
        self.current_step = 0
        self.simulation_running = True
        self.generation = 0
        self.show_mean = False
        self.finish_line_crossed = False
        self.start_time = pygame.time.get_ticks()
        self.finish_time = None

        # Visualization cache
        self.mean_trajectory_surface = None
        self.last_generation = -1
        self.track_surface = self._create_track_surface()

        # Start simulation
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

        # Only check if moving rightward after initial delay
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
            Car(positions=traj, color=tuple(np.random.randint(100, 255, 3)))
            for traj in self.evolution.ask()
        ]
        self.current_step = 0
        self.simulation_running = True
        self.finish_line_crossed = False
        self.finish_time = None
        self.start_time = pygame.time.get_ticks()

    def _update_mean_trajectory(self):
        """Update the cached mean trajectory visualization."""
        if self.generation == self.last_generation:
            return

        mean_traj = self.evolution.get_mean_trajectory()
        if mean_traj and len(mean_traj) > 1:
            self.mean_trajectory_surface = pygame.Surface(
                (self.width, self.height), pygame.SRCALPHA
            )
            points = mean_traj[::1]  # Can increase step for optimization
            if len(points) >= 2:
                pygame.draw.lines(
                    self.mean_trajectory_surface,
                    (255, 215, 0, 180),
                    False,
                    points,
                    2,
                )
        self.last_generation = self.generation

    def draw_cars(self):
        """Draw all car trajectories and current positions."""
        trajectory_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)

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

            # Determine how much of trajectory to draw
            max_step = (
                min(crash_step, self.current_step + 1)
                if crashed
                else self.current_step + 1
            )
            positions = car.positions[:max_step]

            # Check finish line crossing
            if not self.finish_line_crossed and len(positions) >= 2:
                if self._check_finish_line_crossing(positions[-2], positions[-1]):
                    self.finish_line_crossed = True
                    self.finish_time = pygame.time.get_ticks()
                    self.simulation_running = False

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
                    # Draw direction arrow
                    if len(positions) > 1:
                        prev_pos = positions[-2]
                        dx, dy = (
                            current_pos[0] - prev_pos[0],
                            current_pos[1] - prev_pos[1],
                        )
                        angle = np.arctan2(dy, dx)

                        # Arrow points
                        length = 15
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

                    # Draw crash marker
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

            # Draw selection highlight
            if car.selected:
                pos = positions[-1] if positions else car.positions[0]
                pygame.draw.circle(trajectory_surface, (255, 255, 0), pos, 15, 2)

        self.screen.blit(trajectory_surface, (0, 0))

    def draw_ui(self):
        """Draw UI elements including timer and instructions."""
        font = pygame.font.Font(None, 36)

        # Draw generation counter
        gen_text = font.render(f"Generation: {self.generation}", True, (255, 255, 255))
        self.screen.blit(gen_text, (10, 10))

        # Draw timer
        elapsed_time = (
            self.finish_time or pygame.time.get_ticks() - self.start_time
        ) / 1000.0
        timer_text = font.render(f"Time: {elapsed_time:.2f}s", True, (255, 255, 255))
        timer_rect = timer_text.get_rect(topright=(self.width - 10, 10))
        self.screen.blit(timer_text, timer_rect)

        # Draw instructions
        instructions = (
            "Click crashed cars to select, press SPACE to end simulation"
            if self.simulation_running
            else "Click cars to select, press SPACE for next generation"
        )
        instructions_text = font.render(instructions, True, (255, 255, 255))
        self.screen.blit(instructions_text, (10, self.height - 40))

        # Draw mean trajectory toggle instruction
        mean_text = font.render("(m) show population mean", True, (255, 255, 255))
        self.screen.blit(mean_text, (10, self.height - 80))

    def handle_events(self):
        """Handle user input events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()
                for i, car in enumerate(self.cars):
                    if not car.positions:
                        continue

                    # Only allow selecting crashed cars during simulation
                    if self.simulation_running and not (
                        hasattr(self.evolution, "displayed_crashed")
                        and self.evolution.displayed_crashed[i]
                    ):
                        continue

                    # Get current car position
                    pos = car.positions[min(self.current_step, len(car.positions) - 1)]
                    if (
                        np.sqrt(
                            (pos[0] - mouse_pos[0]) ** 2 + (pos[1] - mouse_pos[1]) ** 2
                        )
                        < 15
                    ):
                        car.selected = not car.selected

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    if self.simulation_running:
                        self.simulation_running = False
                        self.current_step = max(len(car.positions) for car in self.cars)
                    else:
                        selected = [
                            i for i, car in enumerate(self.cars) if car.selected
                        ]
                        if selected:
                            self.evolution.tell(selected)
                            self.generation += 1
                            self.generate_new_population()
                elif event.key == pygame.K_m:
                    self.show_mean = not self.show_mean
                    if self.show_mean:
                        self._update_mean_trajectory()
        return True

    def run(self):
        """Main game loop."""
        running = True
        while running:
            running = self.handle_events()

            # Update
            if self.simulation_running:
                max_steps = max(len(car.positions) for car in self.cars)
                if self.current_step < max_steps - 1:
                    self.current_step += 1

            # Draw
            self.screen.fill((0, 0, 0))
            self.screen.blit(self.track_surface, (0, 0))

            if self.show_mean and self.mean_trajectory_surface:
                self.screen.blit(self.mean_trajectory_surface, (0, 0))

            self.draw_cars()
            self.draw_ui()

            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()
