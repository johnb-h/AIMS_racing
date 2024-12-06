import pygame

from scene import Scene
from states import State

# Main Menu Scene
class MainMenuScene(Scene):
    def __init__(self):
        super().__init__()
        self.font = pygame.font.Font(None, 74)

    def handle_events(self, events):
        for ev in events:
            if ev.type == pygame.MOUSEBUTTONDOWN:
                self._next_state = State.INSTRUCTIONS

    def update(self, dt):
        return self._next_state

    def draw(self, screen):
        screen.fill((0, 0, 128))  # Blue background
        title_surface = self.font.render("Main Menu", True, (255, 255, 255))
        screen.blit(title_surface, (250, 250))

    def reset(self):
        print("Resetting Main Menu")
        self._next_state = None