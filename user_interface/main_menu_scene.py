from typing import Optional
from itertools import cycle

import pygame

from user_interface.constants import WINDOW_HEIGHT_IN_M, WINDOW_WIDTH_IN_M
from user_interface.scene import Scene
from user_interface.states import State


# Main Menu Scene
class MainMenuScene(Scene):
    def __init__(self):
        super().__init__()
        self.title_font = pygame.font.Font("./assets/joystix_monospace.ttf", 144)
        self.body_font = pygame.font.Font("./assets/joystix_monospace.ttf", 72)
        self.start_font = pygame.font.Font("./assets/joystix_monospace.ttf", 36)

    def handle_events(self, events):
        for ev in events:
            if ev.type == pygame.MOUSEBUTTONDOWN:
                self._next_state = State.INSTRUCTIONS

    def update(self, dt) -> Optional[State]:
        return self._next_state

    def draw(self, screen):
        screen = pygame.display.set_mode((1920, 1080))
        screen_rect = screen.get_rect()

        background = pygame.image.load("assets/Background1_1920_1080.png")
        cup = pygame.transform.scale(pygame.image.load("assets/Prize.png"), (300, 300))
        car = pygame.transform.scale(pygame.image.load("assets/Ferrari.png"), (400, 400))
        flag = pygame.transform.scale(pygame.image.load("assets/Flag2_pixel.png"), (400, 400))

        title = self.title_font.render("EVORACER", True, (0, 0, 0))
        title_background = self.title_font.render("EVORACER", True, (65, 26, 64))
        body1 = self.body_font.render("TRAIN THE BEST CAR!", True, (0, 0, 0))
        body2 = self.body_font.render("WIN A PRIZE!", True, (0, 0, 0))
        body3 = self.start_font.render("[PRESS ANY BUTTON TO START]", True, (0, 0, 0))

        blink_rect = body3.get_rect()
        blink_rect.center = (screen_rect.centerx, 700)
        off_body3 = self.start_font.render("", True, (0, 0, 0))
        blink_surfaces = cycle([body3, off_body3])
        blink_surface = next(blink_surfaces)
        pygame.time.set_timer(pygame.USEREVENT, 500)

        clock = pygame.time.Clock()

        while True:
            screen.blit(background, (0, 0))
            screen.blit(title_background, (screen_rect.centerx - title_background.get_width() // 2 + 10, 150))
            screen.blit(title, (screen_rect.centerx - title.get_width() // 2, 150))
            screen.blit(body1, (screen_rect.centerx - body1.get_width() // 2, 400))
            screen.blit(body2, (screen_rect.centerx - body2.get_width() // 2, 500))

            screen.blit(cup, (50, 700))
            screen.blit(car, (1450, 725))
            screen.blit(flag, (screen_rect.centerx - flag.get_width() // 2, 700))

            for event in pygame.event.get():
                if event.type == pygame.MOUSEBUTTONDOWN:
                    self._next_state = State.INSTRUCTIONS
                    return
                if event.type == pygame.USEREVENT:
                    blink_surface = next(blink_surfaces)

            screen.blit(blink_surface, blink_rect)
            pygame.display.update()
            clock.tick(60)

    def reset(self):
        self._next_state = None
