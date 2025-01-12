from typing import List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from evosax import CMA_ES


class CarEvolution:
    def __init__(
        self,
        population_size: int = 5,
        n_steps: int = 200,
        track_outer: List[Tuple[float, float]] = None,
        track_inner: List[Tuple[float, float]] = None,
    ):
        self.n_steps = n_steps
        # Parameters are just control points for steering
        n_controls = 20  # Number of control points
        self.param_size = n_controls

        # Initialize CMA-ES strategy with maximization
        self.strategy = CMA_ES(
            popsize=population_size,
            num_dims=self.param_size,
            elite_ratio=0.5,
            maximize=True,  # Now we'll maximize fitness
        )
        self.es_params = self.strategy.default_params
        self.es_params = self.es_params.replace(sigma_init=1.0)

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

    def _sigmoid(self, x):
        """Convert raw parameter to steering angle using sigmoid."""
        x = x * 0.2  # Scale input for smoother response
        return 2.0 / (1.0 + np.exp(-x)) - 1.0

    def _generate_trajectory(self, params):
        """Generate a car trajectory from steering parameters."""
        # Start at the top of the track
        center_x, center_y = 400, 300  # Track center
        x = center_x  # At the vertical midpoint
        y = center_y - 175  # Start at the top of the oval (using b=175 from reference)
        angle = 0  # Facing right
        speed = 4.0  # Constant speed

        # Interpolate control points for smooth steering
        t = np.linspace(0, 1, len(params))
        t_fine = np.linspace(0, 1, self.n_steps)
        raw_steering = np.interp(t_fine, t, params)

        # Convert to steering angles using sigmoid
        max_turn = np.pi / 12  # Maximum 15-degree turn per step
        steering = self._sigmoid(raw_steering) * max_turn

        # Apply momentum physics to steering with reference trajectory bias
        positions = []
        angular_momentum = 0.9  # High momentum for smooth turns
        angular_velocity = 0.0
        reference_bias = 0.1  # Reduced bias towards reference trajectory

        for i in range(self.n_steps):
            # Update position
            positions.append((float(x), float(y)))

            # Find nearest point on reference trajectory
            current_pos = np.array([x, y])
            ref_points = np.array(self.reference_trajectory)
            distances = np.linalg.norm(ref_points - current_pos, axis=1)
            nearest_idx = np.argmin(distances)

            # Get direction to reference point
            ref_point = ref_points[nearest_idx]
            to_ref = ref_point - current_pos
            ref_angle = np.arctan2(to_ref[1], to_ref[0])

            # Blend evolved steering with reference bias
            angle_diff = (ref_angle - angle + np.pi) % (2 * np.pi) - np.pi
            target_angular_velocity = (1 - reference_bias) * steering[
                i
            ] + reference_bias * (angle_diff * 0.1)

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
        fitness = fitness.at[selected_indices].set(100.0)  # Good fitness for selected

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
