from typing import List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from evosax import SNES


class CarEvolution:
    def __init__(
        self,
        population_size: int = 5,
        n_steps: int = 500,
        track_outer: List[Tuple[float, float]] = None,
        track_inner: List[Tuple[float, float]] = None,
    ):
        self.n_steps = n_steps

        # Parameters define a grid of steering angles
        self.grid_size = 8  # 8x8 grid of control points
        self.param_size = self.grid_size * self.grid_size

        # Track boundaries for interpolation
        self.track_bounds = (
            (150, 650),  # x bounds
            (125, 475),  # y bounds
        )

        # Initialize SNES strategy with maximization
        self.strategy = SNES(
            popsize=population_size,
            num_dims=self.param_size,
            maximize=True,
        )
        self.es_params = self.strategy.default_params
        self.es_params = self.es_params.replace(
            sigma_init=1.0,
        )  # Increased initial variance

        # Initialize state
        rng = jax.random.PRNGKey(0)
        self.state = self.strategy.initialize(rng)

        # Store track boundaries
        self.track_outer = np.array(track_outer) if track_outer else None
        self.track_inner = np.array(track_inner) if track_inner else None

        # Generate reference trajectory
        self.reference_trajectory = self._generate_reference_trajectory()

    def _generate_reference_trajectory(self) -> List[Tuple[float, float]]:
        """Generate a perfect oval trajectory between inner and outer track."""
        positions = []
        center_x, center_y = 400, 300  # Center of the track
        a, b = 250, 175  # Oval parameters (horizontal and vertical radii)

        # Generate points along an oval
        for t in np.linspace(0, 2 * np.pi, self.n_steps):
            x = center_x + a * np.cos(t)
            y = center_y + b * np.sin(t)
            positions.append((float(x), float(y)))

        return positions

    def _get_steering_angle(self, x: float, y: float, params: np.ndarray) -> float:
        """Get steering angle for a given position by interpolating the steering field."""
        # Convert position to grid coordinates
        x_min, x_max = self.track_bounds[0]
        y_min, y_max = self.track_bounds[1]

        # Normalize position to [0, 1]
        x_norm = (x - x_min) / (x_max - x_min)
        y_norm = (y - y_min) / (y_max - y_min)

        # Clamp to grid bounds
        x_norm = np.clip(x_norm, 0, 1)
        y_norm = np.clip(y_norm, 0, 1)

        # Convert to grid indices
        x_grid = x_norm * (self.grid_size - 1)
        y_grid = y_norm * (self.grid_size - 1)

        # Get surrounding grid points
        x0, x1 = int(x_grid), min(int(x_grid) + 1, self.grid_size - 1)
        y0, y1 = int(y_grid), min(int(y_grid) + 1, self.grid_size - 1)

        # Get weights for bilinear interpolation
        wx = x_grid - x0
        wy = y_grid - y0

        # Get steering values at grid points
        v00 = params[y0 * self.grid_size + x0]
        v01 = params[y1 * self.grid_size + x0]
        v10 = params[y0 * self.grid_size + x1]
        v11 = params[y1 * self.grid_size + x1]

        # Bilinear interpolation
        steering = (
            v00 * (1 - wx) * (1 - wy)
            + v10 * wx * (1 - wy)
            + v01 * (1 - wx) * wy
            + v11 * wx * wy
        )

        # Convert to angle using sigmoid and scale
        max_turn = np.pi / 12  # Maximum 15-degree turn per step
        return self._sigmoid(steering) * max_turn

    def _sigmoid(self, x):
        """Convert raw parameter to steering angle using sigmoid."""
        x = x * 0.1  # Scale input for smoother response
        return 2.0 / (1.0 + np.exp(-x)) - 1.0

    def _generate_trajectory(self, params):
        """Generate a car trajectory from steering field parameters."""
        # Start at the top of the track
        center_x, center_y = 400, 300  # Track center
        x = center_x  # At the vertical midpoint
        y = center_y - 175  # Start at the top of the oval
        angle = 0  # Facing right
        speed = 4.0  # Constant speed

        # Apply momentum physics to steering
        positions = []
        angular_momentum = 0.9  # High momentum for smooth turns
        angular_velocity = 0.0

        for i in range(self.n_steps):
            # Update position
            positions.append((float(x), float(y)))

            # Get steering angle from field
            target_angular_velocity = self._get_steering_angle(x, y, params)

            # Update angle with momentum
            angular_velocity = (
                angular_momentum * angular_velocity
                + (1 - angular_momentum) * target_angular_velocity
            )
            angle += angular_velocity

            # Move in current direction with constant speed
            vx = speed * np.cos(angle)
            vy = speed * np.sin(angle)

            # Update position
            x += vx
            y += vy

        return positions

    def ask(self):
        """Get a batch of parameters to evaluate."""
        rng = jax.random.PRNGKey(0)
        x, self.state = self.strategy.ask(rng, self.state, self.es_params)
        self.current_x = x
        trajectories = [self._generate_trajectory(p) for p in x]
        self.current_params = x
        return trajectories

    def tell(self, selected_indices: List[int]):
        """Update the evolution strategy based on selected solutions."""
        # Convert selections to fitness scores (now maximizing)
        # Selected cars get high (good) fitness, others get low (bad) fitness
        fitness = jnp.array([0.0] * self.strategy.popsize)  # Bad fitness
        selected_indices = jnp.array(selected_indices)

        # Add regularization to penalize large steering values
        reg_weight = 1  # Weight of regularization term
        for i in range(self.strategy.popsize):
            # L2 regularization on steering parameters
            steering_penalty = -reg_weight * jnp.mean(self.current_x[i] ** 2)
            fitness = fitness.at[i].set(steering_penalty)

        # Add selection reward
        fitness = fitness.at[selected_indices].add(100.0)  # Good fitness for selected

        # Update the strategy
        self.state = self.strategy.tell(
            self.current_x, fitness, self.state, self.es_params
        )

    def get_reference_trajectory(self):
        """Return the reference trajectory for visualization."""
        return self.reference_trajectory

    def get_current_params(self):
        """Return current parameters for visualization."""
        return self.current_params if hasattr(self, "current_params") else None
