import pygame
from typing import Optional
from user_interface.scene import Scene
from user_interface.states import State

class MainMenuScene(Scene):
    def __init__(self):
        super().__init__()
        # Load fonts
        self.title_font = pygame.font.Font("./assets/joystix_monospace.ttf", 144)
        self.body_font = pygame.font.Font("./assets/joystix_monospace.ttf", 72)
        self.start_font = pygame.font.Font("./assets/joystix_monospace.ttf", 36)

        # Pre-render static text surfaces
        self.title_surface = self.title_font.render("EVORACER", True, (0, 0, 0))
        self.title_shadow_surface = self.title_font.render("EVORACER", True, (65, 26, 64))
        self.body1_surface = self.body_font.render("TRAIN THE BEST CAR!", True, (0, 0, 0))
        self.body2_surface = self.body_font.render("WIN A PRIZE!", True, (0, 0, 0))
        self.body3_surface = self.start_font.render("[PRESS ANY BUTTON TO START]", True, (0, 0, 0))
        # An empty surface to represent the blink-off state
        self.off_body3_surface = self.start_font.render("", True, (0, 0, 0))

        # Load and scale images
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

        # Next state is set when a user interaction occurs
        self._next_state: Optional[State] = None

        # Variables for blinking text logic
        self._blink_visible = True
        self._blink_elapsed = 0.0

    def handle_events(self, events):
        # Transition to instructions scene on any mouse click
        for ev in events:
            if ev.type == pygame.MOUSEBUTTONDOWN:
                self._next_state = State.INSTRUCTIONS

    def update(self, dt: float) -> Optional[State]:
        # Update blinking timer: toggle visibility every second
        self._blink_elapsed += dt
        if self._blink_elapsed >= 1.0:
            self._blink_visible = not self._blink_visible
            self._blink_elapsed -= 1.0

        return self._next_state

    def draw(self, screen):
        # Use the provided screen; do not reinitialize it here.
        screen_rect = screen.get_rect()

        # Draw the background image
        screen.blit(self.background, (0, 0))

        # Draw the title with a shadow offset
        title_shadow_x = screen_rect.centerx - self.title_shadow_surface.get_width() // 2 + 10
        title_y = 150
        screen.blit(self.title_shadow_surface, (title_shadow_x, title_y))
        title_x = screen_rect.centerx - self.title_surface.get_width() // 2
        screen.blit(self.title_surface, (title_x, title_y))

        # Draw the body messages
        body1_x = screen_rect.centerx - self.body1_surface.get_width() // 2
        screen.blit(self.body1_surface, (body1_x, 400))
        body2_x = screen_rect.centerx - self.body2_surface.get_width() // 2
        screen.blit(self.body2_surface, (body2_x, 500))

        # Draw additional images
        screen.blit(self.cup, (50, 700))
        screen.blit(self.car, (1450, 725))
        flag_x = screen_rect.centerx - self.flag.get_width() // 2
        screen.blit(self.flag, (flag_x, 700))

        # Draw blinking text that alternates every second
        blink_surface = self.body3_surface if self._blink_visible else self.off_body3_surface
        blink_rect = blink_surface.get_rect(center=(screen_rect.centerx, 700))
        screen.blit(blink_surface, blink_rect)

    def reset(self):
        # Reset the scene state including blink timer and the next state trigger
        self._next_state = None
        self._blink_visible = True
        self._blink_elapsed = 0.0
