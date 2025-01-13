from typing import List, Optional, Tuple

import numpy as np


class CarEvolution:
    def __init__(
        self,
        track_center: Tuple[float, float],
        start_position: Tuple[float, float],
        track_outer_width: float,
        track_outer_height: float,
        track_inner_width: float,
        track_inner_height: float,
        track_outer: List[Tuple[float, float]],
        track_inner: List[Tuple[float, float]],
        population_size: int = 20,
        mutation_rate: float = 0.1,
        mutation_strength: float = 0.1,
        n_points: int = 100,
        slow_down: bool = False,
    ):
        self.track_center = track_center
        self.start_position = start_position
        self.track_outer_width = track_outer_width
        self.track_outer_height = track_outer_height
        self.track_inner_width = track_inner_width
        self.track_inner_height = track_inner_height
        self.track_outer = track_outer
        self.track_inner = track_inner
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.n_points = n_points
        self.slow_down = slow_down

        # Evolution state
        self.generation = 0
        self.population = None
        self.mean_trajectory = None
        self.displayed_crashed = None
        self.displayed_crash_steps = None

        # Initialize population
        self._initialize_population()

    def _initialize_population(self):
        """Initialize the population with random trajectories."""
        self.population = []
        for _ in range(self.population_size):
            trajectory = self._generate_random_trajectory()
            self.population.append(trajectory)

    def _generate_random_trajectory(self) -> List[Tuple[float, float]]:
        """Generate a random trajectory starting from the start position."""
        trajectory = [self.start_position]
        current_pos = self.start_position

        # Parameters for trajectory generation
        max_step = 10.0  # Maximum step size
        angle_range = np.pi / 2  # Maximum turning angle

        # Previous direction (start moving right)
        prev_direction = np.array([1.0, 0.0])

        for _ in range(self.n_points - 1):
            # Generate random angle change
            angle = np.random.uniform(-angle_range, angle_range)

            # Rotate previous direction
            direction = np.array(
                [
                    prev_direction[0] * np.cos(angle)
                    - prev_direction[1] * np.sin(angle),
                    prev_direction[0] * np.sin(angle)
                    + prev_direction[1] * np.cos(angle),
                ]
            )
            direction = direction / np.linalg.norm(direction)

            # Generate step size
            step_size = np.random.uniform(0, max_step)
            if self.slow_down:
                step_size *= 0.5

            # Calculate new position
            new_pos = (
                current_pos[0] + direction[0] * step_size,
                current_pos[1] + direction[1] * step_size,
            )

            trajectory.append(new_pos)
            current_pos = new_pos
            prev_direction = direction

        return trajectory

    def _mutate_trajectory(
        self, trajectory: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        """Apply mutation to a trajectory."""
        mutated = [self.start_position]  # Keep start position fixed

        for i in range(1, len(trajectory)):
            if np.random.random() < self.mutation_rate:
                # Add random offset to point
                offset = np.random.normal(0, self.mutation_strength, 2)
                new_point = (
                    trajectory[i][0] + offset[0] * self.track_outer_width,
                    trajectory[i][1] + offset[1] * self.track_outer_height,
                )
                mutated.append(new_point)
            else:
                mutated.append(trajectory[i])

        return mutated

    def _crossover(
        self, parent1: List[Tuple[float, float]], parent2: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        """Perform crossover between two parent trajectories."""
        # Single point crossover
        crossover_point = np.random.randint(1, len(parent1))
        child = parent1[:crossover_point] + parent2[crossover_point:]
        return child

    def _check_collision(
        self, trajectory: List[Tuple[float, float]]
    ) -> Tuple[bool, Optional[int]]:
        """Check if trajectory collides with track boundaries."""

        def point_in_oval(point, center, width, height):
            x = (point[0] - center[0]) / (width / 2)
            y = (point[1] - center[1]) / (height / 2)
            return x * x + y * y

        for i, point in enumerate(trajectory):
            # Check outer boundary
            if (
                point_in_oval(
                    point,
                    self.track_center,
                    self.track_outer_width,
                    self.track_outer_height,
                )
                > 1
            ):
                return True, i

            # Check inner boundary
            if (
                point_in_oval(
                    point,
                    self.track_center,
                    self.track_inner_width,
                    self.track_inner_height,
                )
                < 1
            ):
                return True, i

        return False, None

    def ask(self) -> List[List[Tuple[float, float]]]:
        """Get current population trajectories for evaluation."""
        self.displayed_crashed = []
        self.displayed_crash_steps = []

        for trajectory in self.population:
            crashed, crash_step = self._check_collision(trajectory)
            self.displayed_crashed.append(crashed)
            self.displayed_crash_steps.append(
                crash_step if crash_step is not None else len(trajectory)
            )

        return self.population

    def tell(self, selected_indices: List[int]):
        """Update population based on selection."""
        # Create new population from selected individuals
        selected = [self.population[i] for i in selected_indices]

        new_population = []

        # Keep selected individuals
        new_population.extend(selected)

        # Fill rest with mutations and crossovers
        while len(new_population) < self.population_size:
            if len(selected) >= 2 and np.random.random() < 0.5:
                # Crossover
                parent1, parent2 = np.random.choice(selected, 2, replace=False)
                child = self._crossover(parent1, parent2)
                child = self._mutate_trajectory(child)  # Apply mutation after crossover
                new_population.append(child)
            else:
                # Mutation
                parent = np.random.choice(selected)
                child = self._mutate_trajectory(parent)
                new_population.append(child)

        self.population = new_population
        self.generation += 1

        # Update mean trajectory
        self._update_mean_trajectory(selected)

    def _update_mean_trajectory(self, selected: List[List[Tuple[float, float]]]):
        """Update the mean trajectory from selected individuals."""
        if not selected:
            self.mean_trajectory = None
            return

        # Convert to numpy array for easier computation
        trajectories = np.array(selected)
        self.mean_trajectory = list(map(tuple, np.mean(trajectories, axis=0)))

    def get_mean_trajectory(self) -> Optional[List[Tuple[float, float]]]:
        """Get the mean trajectory of selected individuals."""
        return self.mean_trajectory
