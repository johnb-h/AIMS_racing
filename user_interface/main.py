import pygame
import sys
import argparse

from manager import Manager
from scene import Scene
from states import State
from constants import RESOLUTION


# Main Function
def main(args):
    pygame.init()
    screen = pygame.display.set_mode(RESOLUTION)
    pygame.display.set_caption("Evolving Cars")

    # Initialize Scene Manager with the starting scene
    manager = Manager(State.MAIN_MENU)
    # manager.current_scene.manager = manager  # Pass manager reference to the scene

    clock = pygame.time.Clock()

    running = True
    while running:
        dt = clock.tick(60) / 1000.0  # Delta time in seconds
        events = pygame.event.get()

        for event in events:
            if event.type == pygame.QUIT:
                running = False

        manager.handle_events(events)
        manager.update(dt)
        manager.draw(screen)

        pygame.display.flip()


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--x_resolution", type=int, default=1200)
    # parser.add_argument("--y_resolution", type=int, default=800)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
