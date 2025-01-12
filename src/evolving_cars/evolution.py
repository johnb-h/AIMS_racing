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

        # Parameters define steering angles at control points
        self.n_control_points = 20  # Number of control points for steering
        self.param_size = self.n_control_points  # One steering angle per control point

        # Initialize SNES strategy with maximization
        self.strategy = SNES(
            popsize=population_size,
            num_dims=self.param_size,
            maximize=True,
        )
        self.es_params = self.strategy.default_params
        # Set much higher learning rate and initial variance
        self.es_params = self.es_params.replace(
            sigma_init=2.0,  # Much higher initial variance
            lrate_sigma=0.01,  # Much higher learning rate (default is 0.01)
        )

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

    def _generate_trajectory(self, params):
        """Generate a car trajectory from sequence of steering angles."""
        # Start at the top of the track
        center_x, center_y = 400, 300  # Track center
        x = center_x  # At the vertical midpoint
        y = center_y - 175  # Start at the top of the oval
        angle = 0  # Facing right
        speed = 4.0  # Constant speed

        # Interpolate steering angles for smooth control
        t_control = np.linspace(0, 1, self.n_control_points)
        t_fine = np.linspace(0, 1, self.n_steps)
        raw_steering = np.interp(t_fine, t_control, params)

        # Convert to steering angles using sigmoid
        max_turn = np.pi / 12  # Maximum 15-degree turn per step
        steering = self._sigmoid(raw_steering) * max_turn

        # Apply momentum physics to steering
        positions = []
        angular_momentum = 0.9  # High momentum for smooth turns
        angular_velocity = 0.0
        crashed = False
        crash_step = self.n_steps

        for i in range(self.n_steps):
            current_pos = (float(x), float(y))
            positions.append(current_pos)

            # Check for collision
            if not self._is_inside_track(current_pos):
                crashed = True
                crash_step = i
                break

            # Update angle with momentum
            target_angular_velocity = steering[i]
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

        # If crashed, fill remaining positions with crash position
        if crashed:
            crash_pos = positions[-1]
            positions.extend([crash_pos] * (self.n_steps - crash_step))

        return positions, crashed, crash_step

    def ask(self):
        """Get a batch of parameters to evaluate."""
        rng = jax.random.PRNGKey(0)
        x, self.state = self.strategy.ask(rng, self.state, self.es_params)
        self.current_x = x

        # Generate trajectories and store crash information
        trajectories_and_info = [self._generate_trajectory(p) for p in x]
        self.trajectories = [t[0] for t in trajectories_and_info]
        self.crashed = [t[1] for t in trajectories_and_info]
        self.crash_steps = [t[2] for t in trajectories_and_info]

        self.current_params = x
        return self.trajectories

    def tell(self, selected_indices: List[int]):
        """Update the evolution strategy based on selected solutions."""
        fitness = jnp.array([0.0] * self.strategy.popsize)
        selected_indices = jnp.array(selected_indices)

        # Calculate regularization penalties
        reg_weight = 1  # Base regularization weight
        crash_penalty_multiplier = 10.0  # Stronger regularization after crash

        for i in range(self.strategy.popsize):
            params = self.current_x[i]

            if self.crashed[i]:
                # For crashed cars, apply heavy regularization to unused parameters
                crash_idx = int(
                    self.crash_steps[i] * self.n_control_points / self.n_steps
                )
                used_params = params[:crash_idx]
                unused_params = params[crash_idx:]

                # Normal regularization for used parameters
                used_penalty = -reg_weight * np.mean(used_params**2)
                # Heavy regularization for unused parameters
                unused_penalty = (
                    -reg_weight * crash_penalty_multiplier * np.mean(unused_params**2)
                )

                distance_penalty = used_penalty + unused_penalty
            else:
                # Normal regularization for successful cars
                distance_penalty = -reg_weight * np.mean(params**2)

            fitness = fitness.at[i].set(distance_penalty)
            print(f"Car {i}: Distance penalty = {distance_penalty:.3f}")

        # Add selection reward
        fitness = fitness.at[selected_indices].add(100.0)

        # Debug print for final fitness
        for i in range(self.strategy.popsize):
            selection_reward = 100.0 if i in selected_indices else 0.0
            print(
                f"Car {i}: Final fitness = {float(fitness[i]):.3f} (Reg: {float(fitness[i] - selection_reward):.3f}, Selection: {selection_reward:.1f})"
            )

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

    def get_sigma_vector(self):
        """Return the current sigma vector from the strategy."""
        return self.state.sigma

    def get_mean_trajectory(self):
        """Get the trajectory generated from the strategy mean."""
        if not hasattr(self, "state"):
            return None

        # Get mean parameters
        x = self.state.mean

        # Interpolate steering angles for smooth control
        t_control = np.linspace(0, 1, self.n_control_points)
        t_fine = np.linspace(0, 1, self.n_steps)
        raw_steering = np.interp(t_fine, t_control, x)

        # Generate full trajectory without crash detection
        positions = []
        angle = 0.0
        pos = (400, 125)  # Start at top of track
        speed = 4.0

        # Convert to steering angles using sigmoid
        max_turn = np.pi / 12  # Maximum 15-degree turn per step
        steering = self._sigmoid(raw_steering) * max_turn

        # Apply momentum physics to steering
        angular_momentum = 0.9
        angular_velocity = 0.0

        for t in range(self.n_steps):
            positions.append(pos)

            # Update angle with momentum
            target_angular_velocity = steering[t]
            angular_velocity = (
                angular_momentum * angular_velocity
                + (1 - angular_momentum) * target_angular_velocity
            )
            angle += angular_velocity

            # Move in current direction
            dx = speed * np.cos(angle)
            dy = speed * np.sin(angle)
            pos = (pos[0] + dx, pos[1] + dy)

        return positions
