import pygame

from scene import Scene
from states import State

# Name Entry Scene
class NameEntryScene(Scene):
    def __init__(self):
        super().__init__()
        self.font = pygame.font.Font(None, 36)
        self.name = ""

    def handle_events(self, events):
        for ev in events:
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_RETURN:
                    self._next_state = State.MAIN_MENU
                elif ev.key == pygame.K_BACKSPACE:
                    self.name = self.name[:-1]
                else:
                    self.name += ev.unicode

    def update(self):
        return self._next_state

    def draw(self, screen):
        screen.fill((128, 0, 128))  # Purple background
        prompt = self.font.render("Enter your name:", True, (255, 255, 255))
        name_surface = self.font.render(self.name, True, (255, 255, 255))
        screen.blit(prompt, (50, 50))
        screen.blit(name_surface, (50, 100))

    def reset(self):
        print("Resetting Name Entry")
        self._next_state = None
        self.name = ""
