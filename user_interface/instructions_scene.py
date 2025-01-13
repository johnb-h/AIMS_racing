import pygame

from user_interface.scene import Scene
from user_interface.states import State


# Instructions Scene
class InstructionsScene(Scene):
    def __init__(self):
        super().__init__()
        self.font = pygame.font.Font(None, 36)

    def handle_events(self, events):
        for ev in events:
            if ev.type == pygame.MOUSEBUTTONDOWN:
                self._next_state = State.GAME

    def update(self, dt):
        return self._next_state

    def draw(self, screen):
        screen.fill((128, 128, 128))  # Gray background
        instructions = [
            "Use arrow keys to move.",
            "Press space to shoot.",
            "Click to continue.",
        ]
        for i, line in enumerate(instructions):
            text_surface = self.font.render(line, True, (0, 0, 0))
            screen.blit(text_surface, (50, 50 + i * 40))

    def reset(self):
        self._next_state = None
