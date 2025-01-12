from typing import List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from evosax import CMA_ES


class CarEvolution:
    def __init__(
        self,
        population_size: int = 5,
        n_steps: int = 100,
        track_outer: List[Tuple[float, float]] = None,
        track_inner: List[Tuple[float, float]] = None,
    ):
        # Initialize evolution strategy
        # We'll evolve parameters that control car movement:
        # Each timestep will have (dx, dy) control, so param_size = n_steps * 2
        self.n_steps = n_steps
        self.param_size = n_steps * 2
        self.strategy = CMA_ES(
            population_size=population_size, num_dims=self.param_size, elite_ratio=0.5
        )

        self.state = self.strategy.initialize(jax.random.PRNGKey(0))
        self.track_outer = np.array(track_outer) if track_outer else None
        self.track_inner = np.array(track_inner) if track_inner else None

    def _generate_trajectory(self, params):
        # Convert parameters to dx, dy pairs
        controls = params.reshape(-1, 2)

        # Generate trajectory
        positions = []
        x, y = 400, 500  # Starting position

        for dx, dy in controls:
            x += dx
            y += dy
            positions.append((float(x), float(y)))

        return positions

    def _compute_fitness(self, params):
        # Generate trajectory
        positions = self._generate_trajectory(params)
        positions = np.array(positions)

        # Compute fitness based on:
        # 1. Distance traveled (encourage exploration)
        # 2. Staying within track boundaries (penalty for going outside)
        # 3. Smoothness of trajectory

        # Distance traveled
        distances = np.sqrt(np.sum(np.diff(positions, axis=0) ** 2, axis=1))
        total_distance = np.sum(distances)

        # Boundary penalties
        penalty = 0
        if self.track_outer is not None and self.track_inner is not None:
            # Simple penalty based on distance to track boundaries
            # In a real implementation, you'd want a more sophisticated check
            for pos in positions:
                # Find closest points on boundaries
                dist_outer = np.min(
                    np.sqrt(np.sum((self.track_outer - pos) ** 2, axis=1))
                )
                dist_inner = np.min(
                    np.sqrt(np.sum((self.track_inner - pos) ** 2, axis=1))
                )

                if dist_outer > 50 or dist_inner < 30:  # Arbitrary thresholds
                    penalty += 100

        # Smoothness (penalize sharp turns)
        smoothness_penalty = np.sum(np.abs(np.diff(distances)))

        fitness = total_distance - penalty - 0.1 * smoothness_penalty
        return float(fitness)

    def ask(self):
        """Get a batch of parameters to evaluate"""
        params = self.strategy.ask(self.state, self.strategy.default_params)
        return [self._generate_trajectory(p) for p in params]

    def tell(self, selected_indices: List[int]):
        """Update the evolution strategy based on selected solutions"""
        # Convert selected indices to fitness scores
        # Selected solutions get high fitness, others get low fitness
        fitness = jnp.array([-1000.0] * self.strategy.population_size)
        fitness = fitness.at[selected_indices].set(1000.0)

        # Update the strategy
        self.state = self.strategy.tell(
            self.state, fitness, self.strategy.default_params
        )
