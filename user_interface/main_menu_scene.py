import pygame
from typing import Optional
from user_interface.scene import Scene
from user_interface.states import State
from user_interface.constants import MENU_FPS

class MainMenuScene(Scene):
    """Main menu scene that displays title, messages, and images."""
    def __init__(self, shared_data: dict) -> None:
        super().__init__(shared_data)
        self.title_font = pygame.font.Font("./assets/joystix_monospace.ttf", 144)
        self.body_font = pygame.font.Font("./assets/joystix_monospace.ttf", 72)
        self.start_font = pygame.font.Font("./assets/joystix_monospace.ttf", 36)

        # Pre-render text surfaces.
        self.title_surface = self.title_font.render("EVORACER", True, (0, 0, 0))
        self.title_shadow_surface = self.title_font.render("EVORACER", True, (65, 26, 64))
        self.body1_surface = self.body_font.render("TRAIN THE BEST CAR!", True, (0, 0, 0))
        self.body2_surface = self.body_font.render("WIN A PRIZE!", True, (0, 0, 0))
        self.body3_surface = self.start_font.render("[PRESS ANY BUTTON TO START]", True, (0, 0, 0))
        self.off_body3_surface = self.start_font.render("", True, (0, 0, 0))

        # Load images.
        self.background = pygame.image.load("assets/Background1_1920_1080.png").convert()
        self.cup = pygame.transform.scale(
            pygame.image.load("assets/Prize.png").convert_alpha(), (300, 300)
        )
        self.car = pygame.transform.scale(
            pygame.image.load("assets/Ferrari.png").convert_alpha(), (400, 400)
        )
        self.flag = pygame.transform.scale(
            pygame.image.load("assets/Flag2_pixel.png").convert_alpha(), (400, 400)
        )

        self._blink_visible = True
        self._blink_elapsed = 0.0
        self._update_accumulator = 0.0
        self._update_interval = 1.0 / MENU_FPS 

    def handle_events(self, events) -> None:
        for ev in events:
            if ev.type == pygame.MOUSEBUTTONDOWN:
                self._next_state = State.INSTRUCTIONS

    def update(self, dt: float) -> Optional[State]:
        self._update_accumulator += dt
        if self._update_accumulator < self._update_interval:
            return None

        self._update_accumulator -= self._update_interval

        # Update blinking text.
        self._blink_elapsed += self._update_interval
        if self._blink_elapsed >= 1.0:
            self._blink_visible = not self._blink_visible
            self._blink_elapsed -= 1.0

        return self._next_state

    def draw(self, screen: pygame.Surface) -> None:
        screen.blit(self.background, (0, 0))
        title_shadow_x = (1920 - self.title_shadow_surface.get_width()) // 2 + 10
        title_y = 150
        screen.blit(self.title_shadow_surface, (title_shadow_x, title_y))
        title_x = (1920 - self.title_surface.get_width()) // 2
        screen.blit(self.title_surface, (title_x, title_y))

        body1_x = (1920 - self.body1_surface.get_width()) // 2
        screen.blit(self.body1_surface, (body1_x, 400))
        body2_x = (1920 - self.body2_surface.get_width()) // 2
        screen.blit(self.body2_surface, (body2_x, 500))

        screen.blit(self.cup, (50, 1080 - self.cup.get_height() - 50))
        screen.blit(self.car, (1920 - self.car.get_width() - 50, 1080 - self.car.get_height() - 50))
        flag_x = (1920 - self.flag.get_width()) // 2
        screen.blit(self.flag, (flag_x, 1080 - self.flag.get_height() - 50))

        blink_surface = self.body3_surface if self._blink_visible else self.off_body3_surface
        blink_rect = blink_surface.get_rect(center=(1920 // 2, 1080 - 100))
        screen.blit(blink_surface, blink_rect)

    def reset(self) -> None:
        self._next_state = None
        self._blink_visible = True
        self._blink_elapsed = 0.0

