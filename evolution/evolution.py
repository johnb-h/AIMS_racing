from typing import List, NamedTuple, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from evosax import SimpleES

from user_interface.constants import CONST_SPEED


class TrajectoryInfo(NamedTuple):
    positions: List[Tuple[float, float]]
    crashed: bool
    crash_step: int


class CarEvolution:
    def __init__(
        self,
        track_center: Tuple[float, float],
        start_position: Tuple[float, float],
        track_outer_width: float,
        track_outer_height: float,
        track_inner_width: float,
        track_inner_height: float,
        population_size: int = 100,
        num_visualize: int = 10,
        n_steps: int = 1000,
        track_outer: List[Tuple[float, float]] = None,
        track_inner: List[Tuple[float, float]] = None,
        slow_down: bool = False,
    ):
        seed = np.random.randint(0, 1000000)
        self.rng = jax.random.PRNGKey(seed)
        self.n_steps = n_steps
        self.slow_down = slow_down

        # Track setup
        self.start_position = np.array(start_position)
        self.track_center = np.array(track_center)
        self.track_outer_width = track_outer_width
        self.track_outer_height = track_outer_height
        self.track_inner_width = track_inner_width
        self.track_inner_height = track_inner_height
        self.track_outer = np.array(track_outer) if track_outer else None
        self.track_inner = np.array(track_inner) if track_inner else None

        # Basis function setup
        self.basis_functions = [(0.02, 40), (0.01, 100)]  # (width, count) pairs
        self.n_params = sum(count for _, count in self.basis_functions)
        self._setup_basis_functions()

        # Evolution strategy setup
        self.num_visualize = num_visualize
        self.strategy = self._setup_evolution_strategy(population_size)

    def _setup_basis_functions(self):
        """Set up basis functions for trajectory generation."""
        # Precompute centers and widths
        centers, widths = [], []
        for width, count in self.basis_functions:
            centers.extend(np.linspace(0, 1, count))
            widths.extend([width] * count)
        self.centers = np.array(centers)
        self.widths = np.array(widths)

        # Precompute basis matrix
        t = np.linspace(0, 1, self.n_steps)
        self.basis_matrix = np.array(
            [
                np.exp(-((t - center) ** 2) / (2 * width**2))
                for center, width in zip(self.centers, self.widths)
            ]
        ).T

    def _setup_evolution_strategy(self, population_size: int):
        """Initialize the evolution strategy."""
        strategy = SimpleES(
            popsize=population_size,
            num_dims=self.n_params,
            maximize=True,
            sigma_init=0.2,
            mean_decay=0.01,
        )
        del strategy.elite_ratio

        self.es_params = strategy.default_params.replace(c_m=0.5, c_sigma=0.3)
        self.rng, rng_init = jax.random.split(self.rng)
        self.state = strategy.initialize(rng_init, self.es_params)

        return strategy

    def _compute_steering_angles(self, params: np.ndarray) -> np.ndarray:
        """Compute steering angles using basis functions."""
        angles = np.dot(self.basis_matrix, params)
        return 2.0 / (1.0 + np.exp(-0.2 * angles)) - 1.0  # Sigmoid

    def _is_inside_track(self, position: np.ndarray) -> bool:
        """Check if a position is inside the track boundaries."""
        delta = position - self.track_center
        ow, oh = self.track_outer_width / 2, self.track_outer_height / 2
        outer_ellipse = (delta[0] / ow) ** 2 + (delta[1] / oh) ** 2
        iw, ih = self.track_inner_width / 2, self.track_inner_height / 2
        inner_ellipse = (delta[0] / iw) ** 2 + (delta[1] / ih) ** 2
        return outer_ellipse <= 1.0 and inner_ellipse >= 1.0

    def _generate_trajectory(
        self, params: np.ndarray, check_crashes: bool = True
    ) -> TrajectoryInfo:
        """Generate a car trajectory from parameters."""
        # Initial state
        pos = np.array(self.start_position, dtype=float)
        angle = 0.0
        angular_velocity = 0.0
        base_speed = CONST_SPEED

        # Get steering angles
        target_velocities = self._compute_steering_angles(params)
        # Generate trajectory
        positions = []
        for i in range(self.n_steps):
            positions.append((float(pos[0]), float(pos[1])))

            if check_crashes and not self._is_inside_track(pos):
                return TrajectoryInfo(
                    positions + [positions[-1]] * (self.n_steps - i - 1), True, i
                )

            # Update steering
            target_velocity = target_velocities[i]
            angular_velocity = target_velocity * (np.pi / 12)  # Max 15-degree turn
            angle += angular_velocity

            # Calculate speed
            speed = base_speed
            if self.slow_down:
                turn_ratio = min(1, abs(angular_velocity) * 10)
                speed *= 1.0 - 0.7 * turn_ratio

            # Update position
            pos += speed * np.array([np.cos(angle), np.sin(angle)])

        return TrajectoryInfo(positions, False, self.n_steps)

    def ask(self) -> List[List[Tuple[float, float]]]:
        """Get a batch of trajectories to evaluate."""
        # Generate parameters
        self.rng, rng_ask = jax.random.split(self.rng)
        self.current_x, self.state = self.strategy.ask(
            rng_ask, self.state, self.es_params
        )

        # Generate trajectories
        trajectories = [self._generate_trajectory(p) for p in self.current_x]

        # Store information
        self.trajectories = [t.positions for t in trajectories]
        self.crashed = [t.crashed for t in trajectories]
        self.crash_steps = [t.crash_step for t in trajectories]

        # Select display trajectories
        scores = [t.crash_step if t.crashed else self.n_steps + 1 for t in trajectories]
        sorted_indices = np.argsort(scores)

        # Get indices for display
        num_best = num_worst = self.num_visualize // 3
        best_indices = sorted_indices[-num_best:]
        worst_indices = sorted_indices[:num_worst]
        self.rng, rng_sample = jax.random.split(self.rng)
        random_indices = jax.random.choice(
            rng_sample, sorted_indices[num_worst:-num_best], shape=(4,), replace=False
        )

        # Store display information
        self.displayed_indices = np.concatenate(
            [best_indices, worst_indices, random_indices]
        )
        self.displayed_crashed = [self.crashed[i] for i in self.displayed_indices]
        self.displayed_crash_steps = [
            self.crash_steps[i] for i in self.displayed_indices
        ]

        return [self.trajectories[i] for i in self.displayed_indices]

    def tell(self, selected_indices: List[int]):
        """Update evolution strategy based on selection."""
        # Map display indices to full population indices
        full_indices = jnp.array([self.displayed_indices[i] for i in selected_indices])

        # Create fitness array
        fitness = jnp.zeros(self.strategy.popsize).at[full_indices].set(1.0)

        # Update weights and strategy
        weights = (
            jnp.zeros(self.strategy.popsize)
            .at[: len(full_indices)]
            .set(1.0 / len(full_indices))
        )
        self.state = self.state.replace(weights=weights)
        self.state = self.strategy.tell(
            self.current_x, fitness, self.state, self.es_params
        )

    def get_mean_trajectory(self) -> List[Tuple[float, float]]:
        """Get trajectory for mean parameters."""
        if not hasattr(self, "state"):
            return None
        return self._generate_trajectory(self.state.mean, check_crashes=False).positions
