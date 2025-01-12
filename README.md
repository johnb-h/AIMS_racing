# Evolving Cars

A visualization of evolutionary algorithms using cars racing around a track. The visualization shows cars learning to navigate a track through evolution, with user interaction to select the best performers for breeding the next generation.

## Installation

This project uses PDM for dependency management. To install:

1. Install PDM if you haven't already:
```bash
pip install pdm
```

2. Install project dependencies:
```bash
pdm install
```

## Usage

Run the visualization:
```bash
pdm run python main.py
```

### Controls
- Watch the cars navigate the track
- When they finish, click on the best performing cars to select them
- Press SPACE to evolve the next generation based on your selection

## How it Works

The project uses:
- evosax for evolutionary computation
- JAX for efficient numerical operations
- Pygame for visualization

Cars are evolved using CMA-ES (Covariance Matrix Adaptation Evolution Strategy), with their trajectories optimized based on:
1. Distance traveled
2. Staying within track boundaries
3. Movement smoothness 
