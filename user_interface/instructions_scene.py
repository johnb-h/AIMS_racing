import pygame
from typing import Optional
from user_interface.scene import Scene
from user_interface.states import State
from user_interface.constants import MENU_FPS

class InstructionsScene(Scene):
    """Scene that displays game instructions on a 1920x1080 canvas."""
    def __init__(self) -> None:
        super().__init__()
        self.title_font = pygame.font.Font("./assets/joystix_monospace.ttf", 144)
        self.body_font = pygame.font.Font("./assets/joystix_monospace.ttf", 50)
        self.start_font = pygame.font.Font("./assets/joystix_monospace.ttf", 36)
        
        self.background = pygame.image.load("assets/Background2_1920_1080.png").convert()

        self.heading = self.title_font.render("INSTRUCTIONS", True, (0, 0, 0))
        self.heading_shadow = self.title_font.render("INSTRUCTIONS", True, (65, 26, 64))
        
        self.blink_surface = self.start_font.render("[PRESS ANY BUTTON TO START]", True, (0, 0, 0))
        self.blink_off_surface = self.start_font.render("", True, (0, 0, 0))
        
        self.instructions = [
            "1. Look at the different cars racing.",
            "2. Choose the best-performing car.",
            "3. Watch them evolve and finish the track!"
        ]
        self.instructions_surfaces = [
            self.body_font.render(line, True, (0, 0, 0)) for line in self.instructions
        ]
        self._next_state: Optional[State] = None
        self._blink_visible = True
        self._blink_elapsed = 0.0

        self._update_accumulator = 0.0
        self._update_interval = 1.0 / MENU_FPS 

    def handle_events(self, events) -> None:
        for ev in events:
            if ev.type == pygame.MOUSEBUTTONDOWN:
                self._next_state = State.GAME

    def update(self, dt: float) -> Optional[State]:
        self._update_accumulator += dt
        if self._update_accumulator < self._update_interval:
            # Not enough time has elapsed to perform another update.
            return None

        # Subtract the interval (or use a while loop for robustness if dt is large).
        self._update_accumulator -= self._update_interval

        # Now, update blinking or any other time-dependent logic.
        self._blink_elapsed += self._update_interval
        if self._blink_elapsed >= 1.0:
            self._blink_visible = not self._blink_visible
            self._blink_elapsed -= 1.0

        return self._next_state


    def draw(self, screen: pygame.Surface) -> None:
        # Draw the full background (1920x1080); the entire image will be visible.
        screen.blit(self.background, (0, 0))
        
        # Draw heading with a shadow.
        heading_x = (1920 - self.heading.get_width()) // 2
        heading_y = 100
        screen.blit(self.heading_shadow, (heading_x + 10, heading_y))
        screen.blit(self.heading, (heading_x, heading_y))
        
        # Draw instruction text lines.
        for i, line_surface in enumerate(self.instructions_surfaces):
            text_x = (1920 - line_surface.get_width()) // 2
            text_y = 400 + i * 100
            screen.blit(line_surface, (text_x, text_y))
        
        # Draw blinking text near the bottom.
        blink = self.blink_surface if self._blink_visible else self.blink_off_surface
        blink_rect = blink.get_rect(center=(1920 // 2, 1080 - 100))
        screen.blit(blink, blink_rect)

    def reset(self) -> None:
        self._next_state = None
        self._blink_visible = True
        self._blink_elapsed = 0.0


