from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pygame

from evolution import CarEvolution


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

    def draw_cars(self):
        for car in self.cars:
            if len(car.positions) > 1:
                # Draw trajectory up to current step
                positions = car.positions[: self.current_step + 1]
                if len(positions) >= 2:
                    pygame.draw.lines(self.screen, car.color, False, positions, 2)

                # Draw current position
                if self.current_step < len(car.positions):
                    current_pos = car.positions[self.current_step]
                    pygame.draw.circle(self.screen, car.color, current_pos, 5)

                # Highlight selected cars
                if car.selected:
                    if self.current_step < len(car.positions):
                        pygame.draw.circle(self.screen, (0, 255, 0), current_pos, 8, 2)

    def draw_ui(self):
        # Draw generation counter
        font = pygame.font.Font(None, 36)
        gen_text = font.render(f"Generation: {self.generation}", True, (255, 255, 255))
        self.screen.blit(gen_text, (10, 10))

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
                        if self.current_step < len(car.positions):
                            car_pos = car.positions[self.current_step]
                            # Check if click is near the car
                            distance = np.sqrt(
                                (car_pos[0] - mouse_pos[0]) ** 2
                                + (car_pos[1] - mouse_pos[1]) ** 2
                            )
                            if distance < 10:
                                car.selected = not car.selected
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
