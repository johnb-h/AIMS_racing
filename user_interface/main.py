import pygame
import sys

from manager import Manager
from scene import Scene
from states import State


# Main Function
def main():
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Scene Manager Example")

    # Initialize Scene Manager with the starting scene
    manager = Manager(State.MAIN_MENU)
    # manager.current_scene.manager = manager  # Pass manager reference to the scene

    clock = pygame.time.Clock()

    while True:
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        manager.handle_events(events)
        manager.update()
        manager.draw(screen)

        pygame.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    main()
