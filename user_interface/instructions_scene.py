import pygame
from itertools import cycle

from user_interface.scene import Scene
from user_interface.states import State

# Instructions Scene
class InstructionsScene(Scene):
    def __init__(self):
        super().__init__()
        self.title_font = pygame.font.Font("./assets/joystix_monospace.ttf", 144)
        self.body_font = pygame.font.Font("./assets/joystix_monospace.ttf", 50)
        self.start_font = pygame.font.Font("./assets/joystix_monospace.ttf", 36)
        self.screen = pygame.display.set_mode((1920, 1080))

    def handle_events(self, events):
        for ev in events:
            if ev.type == pygame.MOUSEBUTTONDOWN:
                self._next_state = State.GAME

    def update(self, dt):
        return self._next_state

    def draw(self, screen):
        screen_rect = self.screen.get_rect()

        background = pygame.image.load("assets/Background2_1920_1080.png")
        heading = self.title_font.render("INSTRUCTIONS", True, (0, 0, 0))
        heading_background = self.title_font.render("INSTRUCTIONS", True, (65, 26, 64))
        body3 = self.start_font.render("[PRESS ANY BUTTON TO START]", True, (0, 0, 0))

        blink_rect = body3.get_rect()
        blink_rect.center = (screen_rect.centerx, 800)
        off_body3 = self.start_font.render("", True, (0, 0, 0))
        blink_surfaces = cycle([body3, off_body3])
        blink_surface = next(blink_surfaces)
        pygame.time.set_timer(pygame.USEREVENT, 500)

        clock = pygame.time.Clock()

        while True:
            screen.blit(background, (0, 0))
            screen.blit(heading, (screen_rect.centerx - heading.get_width() // 2, 100))
            screen.blit(heading_background, (screen_rect.centerx - heading_background.get_width() // 2 + 10, 100))

            instructions = [
                "1. Look at the different cars racing.",
                "2. Choose the best-performing car.",
                "3. Watch them evolve and finish the track!"
            ]
            for i, line in enumerate(instructions):
                text_surface = self.body_font.render(line, True, (0, 0, 0))
                screen.blit(text_surface, (screen_rect.centerx - text_surface.get_width() // 2, 400 + i * 100))

            for event in pygame.event.get():
                if event.type == pygame.MOUSEBUTTONDOWN:
                    self._next_state = State.GAME
                    return
                if event.type == pygame.USEREVENT:
                    blink_surface = next(blink_surfaces)

            screen.blit(blink_surface, blink_rect)
            pygame.display.update()
            clock.tick(60)

    def reset(self):
        self._next_state = None
