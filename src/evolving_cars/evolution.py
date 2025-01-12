from typing import List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from evosax import SimpleES


class CarEvolution:
    def __init__(
        self,
        population_size: int = 100,
        n_steps: int = 500,
        track_outer: List[Tuple[float, float]] = None,
        track_inner: List[Tuple[float, float]] = None,
        slow_down: bool = False,
    ):
        self.rng = jax.random.PRNGKey(0)
        self.n_steps = n_steps
        self.slow_down = slow_down
        # Parameters for multi-scale Gaussian basis functions
        # List of (width, count) tuples, from broad to narrow
        self.basis_functions = [
            (0.02, 40),  # Very fine control
            (0.01, 100),  # Very very fine control
        ]

        # Calculate total number of parameters
        self.n_params = sum(count for _, count in self.basis_functions)

        # Precompute centers for each scale
        self.centers = []
        self.widths = []
        for width, count in self.basis_functions:
            centers = np.linspace(0, 1, count)
            self.centers.extend(centers)
            self.widths.extend([width] * count)

        self.centers = np.array(self.centers)
        self.widths = np.array(self.widths)

        # Precompute basis function matrix (n_steps × n_params)
        self.basis_matrix = self._precompute_basis_matrix()

        # Initialize SimpleES strategy
        self.strategy = SimpleES(
            popsize=population_size,
            num_dims=self.n_params,
            maximize=True,
            sigma_init=0.2,  # Small noise for fine-tuning
            mean_decay=0.005,
        )
        del self.strategy.elite_ratio
        self.es_params = self.strategy.default_params
        self.es_params = self.es_params.replace(
            c_m=0.5,
            c_sigma=0.3,
        )

        # Initialize state
        self.rng, rng_init = jax.random.split(self.rng)
        self.state = self.strategy.initialize(rng_init, self.es_params)

        # Store track boundaries
        self.track_outer = np.array(track_outer) if track_outer else None
        self.track_inner = np.array(track_inner) if track_inner else None

    def _precompute_basis_matrix(self) -> np.ndarray:
        """Precompute the basis function matrix for all time steps."""
        # Time points
        t = np.linspace(0, 1, self.n_steps)

        # Initialize matrix for all basis functions
        basis_matrix = np.zeros((self.n_steps, self.n_params))

        # Fill in all Gaussian basis functions
        for i, (center, width) in enumerate(zip(self.centers, self.widths)):
            activation = np.exp(-((t - center) ** 2) / (2 * width**2))
            basis_matrix[:, i] = activation

        return basis_matrix

    def _compute_steering_angles(self, params: np.ndarray) -> np.ndarray:
        """Compute steering angles for all time steps using matrix multiplication."""
        # Multiply basis matrix by parameters
        angles = np.dot(self.basis_matrix, params)

        # Apply sigmoid to bound the steering angles
        return self._sigmoid(angles)

    def _sigmoid(self, x):
        """Convert raw parameter to steering angle using sigmoid."""
        x = x * 0.2  # Scale input for smoother response
        return 2.0 / (1.0 + np.exp(-x)) - 1.0

    def _is_inside_track(self, position):
        """Check if a position is inside the track boundaries."""
        # Convert position to numpy array for calculations
        pos = np.array(position)
        center = np.array([400, 300])  # Track center

        # Convert to ellipse coordinates
        dx = pos[0] - center[0]
        dy = pos[1] - center[1]

        # Check if inside outer ellipse (a=320, b=210 for outer track)
        outer_ellipse = (dx / 320) ** 2 + (dy / 210) ** 2
        # Check if outside inner ellipse (a=160, b=105 for inner track)
        inner_ellipse = (dx / 160) ** 2 + (dy / 105) ** 2

        # Point is valid if it's inside outer ellipse AND outside inner ellipse
        return outer_ellipse <= 1.0 and inner_ellipse >= 1.0

    def _generate_trajectory(self, params, check_crashes=True):
        """Generate a car trajectory from basis function weights."""
        # Start at the top of the track
        center_x, center_y = 400, 300  # Track center
        x = center_x  # At the vertical midpoint
        y = center_y - 175  # Start at the top of the oval
        angle = 0  # Facing right
        base_speed = 4.0  # Base speed when going straight

        # Precompute all steering angles
        target_angular_velocities = self._compute_steering_angles(params)

        # Apply momentum physics to steering
        positions = []
        angular_momentum = 0.0  # High momentum for smooth turns
        angular_velocity = 0.0
        crashed = False
        crash_step = self.n_steps

        for i in range(self.n_steps):
            current_pos = (float(x), float(y))
            positions.append(current_pos)

            # Check for collision if needed
            if check_crashes and not self._is_inside_track(current_pos):
                crashed = True
                crash_step = i
                # Fill remaining positions with last valid position
                positions.extend([current_pos] * (self.n_steps - i - 1))
                break

            # Get precomputed steering angle
            target_angular_velocity = target_angular_velocities[i]

            # Apply momentum to steering
            momentum_term = angular_momentum * angular_velocity
            target_term = (1 - angular_momentum) * target_angular_velocity
            angular_velocity = momentum_term + target_term

            # Scale the angular velocity to reasonable range
            max_turn = np.pi / 12  # Maximum 15-degree turn per step
            angular_velocity = angular_velocity * max_turn

            # Update angle
            angle += angular_velocity

            current_speed = base_speed
            if self.slow_down:
                # Calculate speed based on steering angle
                # Speed reduction factor: 1.0 when straight, down to 0.3 at maximum turn
                # Use quadratic reduction for more aggressive slowdown when turning
                turn_ratio = min(1, abs(angular_velocity) * 10)
                speed_factor = 1.0 - 0.7 * turn_ratio
                current_speed = base_speed * speed_factor

            # Move in current direction with variable speed
            vx = current_speed * np.cos(angle)
            vy = current_speed * np.sin(angle)

            # Update position
            x += vx
            y += vy

        return positions, crashed, crash_step

    def ask(self):
        """Get a batch of parameters to evaluate."""
        self.rng, rng_ask = jax.random.split(self.rng)
        x, self.state = self.strategy.ask(rng_ask, self.state, self.es_params)
        self.current_x = x

        # Generate all trajectories
        trajectories_and_info = [self._generate_trajectory(p) for p in x]

        # Store full information
        self.trajectories = [t[0] for t in trajectories_and_info]
        self.crashed = [t[1] for t in trajectories_and_info]
        self.crash_steps = [t[2] for t in trajectories_and_info]
        self.current_params = x

        # Calculate trajectory lengths (longer is better if not crashed)
        scores = []
        for i, (crashed, crash_step) in enumerate(zip(self.crashed, self.crash_steps)):
            if crashed:
                scores.append(crash_step)
            else:
                scores.append(self.n_steps + 1)  # Non-crashed trajectories are best

        # Get indices of best and worst trajectories
        sorted_indices = np.argsort(scores)
        best_indices = sorted_indices[-3:]  # 3 best
        worst_indices = sorted_indices[:3]  # 3 worst

        # Get 4 random unique indices excluding best and worst
        middle_indices = sorted_indices[3:-3]
        self.rng, rng_sample = jax.random.split(self.rng)
        random_indices = jax.random.choice(
            rng_sample, middle_indices, shape=(4,), replace=False
        )

        # Combine indices and get selected trajectories
        selected_indices = np.concatenate([best_indices, worst_indices, random_indices])
        selected_trajectories = [self.trajectories[i] for i in selected_indices]
        self.displayed_indices = selected_indices
        self.displayed_crashed = [self.crashed[i] for i in selected_indices]
        self.displayed_crash_steps = [self.crash_steps[i] for i in selected_indices]

        return selected_trajectories

    def tell(self, selected_indices: List[int]):
        """Update the evolution strategy based on selected solutions."""
        # Map displayed indices back to full population indices
        selected_full_indices = [self.displayed_indices[i] for i in selected_indices]

        fitness = jnp.array([0.0] * self.strategy.popsize)
        selected_full_indices = jnp.array(selected_full_indices)

        # Add selection reward
        fitness = fitness.at[selected_full_indices].add(1.0)

        # Update the strategy with masked parameters
        elite_popsize = len(selected_full_indices)
        weights = jnp.zeros(self.strategy.popsize)
        weights = weights.at[:elite_popsize].set(1 / elite_popsize)
        self.state = self.state.replace(weights=weights)

        self.state = self.strategy.tell(
            self.current_x, fitness, self.state, self.es_params
        )

    def get_current_params(self):
        """Return current parameters for visualization."""
        return self.current_params if hasattr(self, "current_params") else None

    def get_sigma_vector(self):
        """Return the current sigma vector from the strategy."""
        return self.state.sigma

    def get_mean_trajectory(self):
        """Get the complete trajectory generated from the strategy mean."""
        if not hasattr(self, "state"):
            return None
        positions, _, _ = self._generate_trajectory(
            self.state.mean, check_crashes=False
        )
        return positions
