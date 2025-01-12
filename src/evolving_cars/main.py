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
            population_size=5,
            track_outer=self.track_outer,
            track_inner=self.track_inner,
        )

        self.cars: List[Car] = []
        self.current_step = 0
        self.simulation_running = True
        self.generation = 0

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

    def generate_new_population(self):
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

    def add_car(self, car: Car):
        self.cars.append(car)

    def draw_track(self):
        # Draw outer boundary
        pygame.draw.lines(self.screen, (255, 0, 0), True, self.track_outer, 2)
        # Draw inner boundary
        pygame.draw.lines(self.screen, (255, 0, 0), True, self.track_inner, 2)
        # Draw reference trajectory
        ref_traj = self.evolution.get_reference_trajectory()
        if len(ref_traj) > 1:
            pygame.draw.lines(self.screen, (100, 100, 100), True, ref_traj, 1)

    def _get_last_visible_position(self, positions):
        """Get the last position that was within screen bounds."""
        for pos in reversed(positions):
            x, y = pos
            if 0 <= x <= self.width and 0 <= y <= self.height:
                return pos
        return positions[0]  # Fallback to start position if none visible

    def draw_cars(self):
        for car in self.cars:
            if len(car.positions) > 1:
                # Draw trajectory
                positions = car.positions[: self.current_step + 1]
                if len(positions) >= 2:
                    pygame.draw.lines(self.screen, car.color, False, positions, 3)

                # Draw current position with a larger circle
                if self.simulation_running:
                    if self.current_step < len(car.positions):
                        current_pos = car.positions[self.current_step]
                        # Only draw if on screen
                        if (
                            0 <= current_pos[0] <= self.width
                            and 0 <= current_pos[1] <= self.height
                        ):
                            # Draw direction indicator
                            if self.current_step > 0:
                                prev_pos = car.positions[self.current_step - 1]
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
                                    self.screen, car.color, [tip, left, right]
                                )
                else:
                    # During selection, show last visible position
                    visible_pos = self._get_last_visible_position(car.positions)
                    # Draw a larger circle for easier selection
                    pygame.draw.circle(self.screen, car.color, visible_pos, 10)
                    # Draw an X to indicate if car went offscreen
                    if visible_pos != car.positions[-1]:
                        size = 15
                        pygame.draw.line(
                            self.screen,
                            car.color,
                            (visible_pos[0] - size, visible_pos[1] - size),
                            (visible_pos[0] + size, visible_pos[1] + size),
                            2,
                        )
                        pygame.draw.line(
                            self.screen,
                            car.color,
                            (visible_pos[0] - size, visible_pos[1] + size),
                            (visible_pos[0] + size, visible_pos[1] - size),
                            2,
                        )

                # Highlight selected cars with a bright outline
                if car.selected:
                    if self.simulation_running:
                        pos = car.positions[self.current_step]
                        if 0 <= pos[0] <= self.width and 0 <= pos[1] <= self.height:
                            pygame.draw.circle(self.screen, (255, 255, 0), pos, 15, 2)
                    else:
                        pos = self._get_last_visible_position(car.positions)
                        pygame.draw.circle(self.screen, (255, 255, 0), pos, 15, 2)

    def draw_ui(self):
        # Draw generation counter
        font = pygame.font.Font(None, 36)
        gen_text = font.render(f"Generation: {self.generation}", True, (255, 255, 255))
        self.screen.blit(gen_text, (10, 10))

        # Draw steering parameters for each car
        params = self.evolution.get_current_params()
        if params is not None:
            y_offset = 50
            small_font = pygame.font.Font(None, 24)
            for i, (car, param_set) in enumerate(zip(self.cars, params)):
                # Show first 5 steering values
                param_text = f"Car {i+1} steering: "
                param_text += " ".join([f"{p:.2f}" for p in param_set[:5]])
                color = car.color if car.selected else (150, 150, 150)
                text = small_font.render(param_text, True, color)
                self.screen.blit(text, (10, y_offset))
                y_offset += 25

        # Draw instructions
        if not self.simulation_running:
            instructions = font.render(
                "Click cars to select, press SPACE for next generation",
                True,
                (255, 255, 255),
            )
            self.screen.blit(instructions, (10, self.height - 40))

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Handle car selection when simulation is finished
                if not self.simulation_running:
                    mouse_pos = pygame.mouse.get_pos()
                    for car in self.cars:
                        if len(car.positions) > 0:
                            car_pos = self._get_last_visible_position(car.positions)
                            distance = np.sqrt(
                                (car_pos[0] - mouse_pos[0]) ** 2
                                + (car_pos[1] - mouse_pos[1]) ** 2
                            )
                            if distance < 15:
                                car.selected = not car.selected
                                print(f"Car selected: {car.selected}")
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and not self.simulation_running:
                    self.evolve_next_generation()
        return True

    def run(self):
        running = True
        while running:
            self.screen.fill((0, 0, 0))  # Black background

            running = self.handle_events()

            self.draw_track()
            self.draw_cars()
            self.draw_ui()

            if self.simulation_running:
                self.current_step += 1
                # Check if all cars have finished their trajectories
                if all(self.current_step >= len(car.positions) for car in self.cars):
                    self.simulation_running = False

            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()


if __name__ == "__main__":
    visualizer = TrackVisualizer()
    visualizer.run()
