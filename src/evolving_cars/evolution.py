from typing import List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from evosax import OpenES


class CarEvolution:
    def __init__(
        self,
        population_size: int = 10,
        n_steps: int = 500,
        track_outer: List[Tuple[float, float]] = None,
        track_inner: List[Tuple[float, float]] = None,
    ):
        self.rng = jax.random.PRNGKey(0)
        self.n_steps = n_steps

        # Parameters for multi-scale Gaussian basis functions
        self.n_centers = 100  # More time offsets for finer control
        self.n_widths = 3  # Fewer widths, focusing on local control
        self.n_params = self.n_centers * self.n_widths  # Only Gaussians, no constant

        # Precompute Gaussian centers and widths
        self.centers = np.linspace(0, 1, self.n_centers)  # Time offsets
        # Widths focusing on local control
        self.widths = np.array([0.01, 0.05, 0.1])

        # Precompute basis function matrix (n_steps × n_params)
        self.basis_matrix = self._precompute_basis_matrix()

        # Initialize OpenES strategy with maximization
        self.strategy = OpenES(
            popsize=population_size,
            num_dims=self.n_params,
            maximize=True,
            opt_name="sgd",
        )
        self.es_params = self.strategy.default_params
        # Set higher initial variance for better exploration
        self.es_params = self.es_params.replace(
            sigma_init=0.05,  # Lower initial std for finer control
            opt_params=self.es_params.opt_params.replace(
                lrate_init=0.003,
            ),
        )

        # Initialize state
        self.rng, rng_init = jax.random.split(self.rng)
        self.state = self.strategy.initialize(rng_init, self.es_params)

        # Store track boundaries
        self.track_outer = np.array(track_outer) if track_outer else None
        self.track_inner = np.array(track_inner) if track_inner else None

        # Generate reference trajectory
        self.reference_trajectory = self._generate_reference_trajectory()

    def _precompute_basis_matrix(self) -> np.ndarray:
        """Precompute the basis function matrix for all time steps."""
        # Time points
        t = np.linspace(0, 1, self.n_steps)

        # Initialize matrix for Gaussian bases
        basis_matrix = np.zeros((self.n_steps, self.n_params))

        # Fill in Gaussian basis functions
        col_idx = 0
        for width in self.widths:
            for center in self.centers:
                # Compute Gaussian activation for all time points at once
                activation = np.exp(-((t - center) ** 2) / (2 * width**2))
                basis_matrix[:, col_idx] = activation
                col_idx += 1

        return basis_matrix

    def _compute_steering_angles(self, params: np.ndarray) -> np.ndarray:
        """Compute steering angles for all time steps using matrix multiplication."""
        # Multiply basis matrix by parameters
        angles = np.dot(self.basis_matrix, params)

        # Apply sigmoid to bound the steering angles
        return self._sigmoid(angles)

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

    def _sigmoid(self, x):
        """Convert raw parameter to steering angle using sigmoid."""
        x = x * 0.2  # Scale input for smoother response
        return 2.0 / (1.0 + np.exp(-x)) - 1.0

    def _get_min_distance_to_reference(self, position: np.ndarray) -> float:
        """Calculate minimum distance from a point to reference trajectory."""
        ref_points = np.array(self.reference_trajectory)
        distances = np.linalg.norm(ref_points - position, axis=1)
        return np.min(distances)

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
        speed = 4.0  # Constant speed

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

            # Move in current direction with constant speed
            vx = speed * np.cos(angle)
            vy = speed * np.sin(angle)

            # Update position
            x += vx
            y += vy

        return positions, crashed, crash_step

    def ask(self):
        """Get a batch of parameters to evaluate."""
        self.rng, rng_ask = jax.random.split(self.rng)
        x, self.state = self.strategy.ask(rng_ask, self.state, self.es_params)
        print(f"x: {x}, es_params: {self.es_params}")
        self.current_x = x

        # Generate trajectories from control points
        trajectories_and_info = [self._generate_trajectory(p) for p in x]

        # Store trajectories and crash information
        self.trajectories = [
            t[0] for t in trajectories_and_info
        ]  # Store just the positions
        self.crashed = [t[1] for t in trajectories_and_info]  # Store crash status
        self.crash_steps = [t[2] for t in trajectories_and_info]  # Store crash steps
        self.current_params = x

        return self.trajectories

    def tell(self, selected_indices: List[int]):
        """Update the evolution strategy based on selected solutions."""
        fitness = jnp.array([0.0] * self.strategy.popsize)
        selected_indices = jnp.array(selected_indices)

        # Zero out parameters after crashes
        masked_params = np.array(self.current_x)
        for i in range(self.strategy.popsize):
            if self.crashed[i]:
                # Calculate which parameters correspond to time steps after the crash
                crash_time = self.crash_steps[i] / self.n_steps
                for j in range(self.n_params):
                    # Zero out parameters if their Gaussian's center is after the crash
                    center_idx = j % self.n_centers
                    if self.centers[center_idx] > crash_time:
                        masked_params[i, j] = 0.0

        # Add selection reward
        fitness = fitness.at[selected_indices].add(10.0)

        # Debug print for final fitness
        for i in range(self.strategy.popsize):
            selection_reward = 100.0 if i in selected_indices else 0.0
            print(
                f"Car {i}: Final fitness = {float(fitness[i]):.3f} "
                f"(Selection: {selection_reward:.1f})"
            )

        # Update the strategy with masked parameters
        self.state = self.strategy.tell(
            masked_params, fitness, self.state, self.es_params
        )

    def get_reference_trajectory(self):
        """Return the reference trajectory for visualization."""
        return self.reference_trajectory

    def get_mean_trajectory(self):
        """Get the complete trajectory generated from the strategy mean."""
        if not hasattr(self, "state"):
            return None
        positions, _, _ = self._generate_trajectory(
            self.state.mean, check_crashes=False
        )
        return positions

    def get_current_params(self):
        """Return current parameters for visualization."""
        return self.current_params if hasattr(self, "current_params") else None

    def get_sigma_vector(self):
        """Return the current sigma vector from the strategy."""
        return self.state.sigma
