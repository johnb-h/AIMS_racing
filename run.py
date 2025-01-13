#!/usr/bin/env python3
"""Entry point for the evolving cars visualization."""

from src.evolving_cars.main import TrackVisualizer


def main():
    visualizer = TrackVisualizer()
    visualizer.run()


if __name__ == "__main__":
    main()
