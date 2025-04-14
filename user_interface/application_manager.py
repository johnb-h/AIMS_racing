import pygame
from typing import Optional
from user_interface.constants import TARGET_FPS
from user_interface.game_scene import GameScene
from user_interface.instructions_scene import InstructionsScene
from user_interface.main_menu_scene import MainMenuScene
from user_interface.name_entry_scene import NameEntryScene
from user_interface.high_scores_scene import HighScoresScene
from user_interface.states import State

class ApplicationManager:
    """
    Manages the overall application using a fixed virtual resolution (base_resolution)
    and scales it to the actual window while preserving aspect ratio.
    
    Here, the base resolution is set to 1920x1080 so that your full background images
    (which are also 1920x1080) are always visible.
    """
    def __init__(self, window_width: int = 1200, window_height: int = 800) -> None:
        # For this design, fix the base (virtual) resolution to 1920x1080.
        self._base_resolution = (1920, 1080)
        self._window_width = window_width
        self._window_height = window_height

        pygame.init()
        pygame.display.set_caption("Evolving Cars")
        self._set_screen(window_width, window_height)

        # Create the virtual surface where all drawing will occur.
        self._virtual_surface = pygame.Surface(self._base_resolution)

        # Initialize scenes.
        self._state: State = State.MAIN_MENU
        self._scenes = {
            State.MAIN_MENU: MainMenuScene(),
            State.INSTRUCTIONS: InstructionsScene(),
            State.GAME: GameScene(),
            State.NAME_ENTRY: NameEntryScene(),
            State.HIGH_SCORES: HighScoresScene(),
        }
        self._scene = self._scenes[self._state]

    def _set_screen(self, window_width: int, window_height: int) -> None:
        """
        Updates the window’s display on resize.
        """
        self._screen = pygame.display.set_mode((window_width, window_height), pygame.RESIZABLE)
        self._screen_rect = self._screen.get_rect()
        pygame.display.update()

    def run_game_loop(self) -> None:
        """
        Main game loop: events are processed and the current scene is drawn onto a 
        1920x1080 virtual surface which is then uniformly scaled to the window.
        """
        clock = pygame.time.Clock()
        running = True

        while running:
            dt = clock.tick(TARGET_FPS) / 1000.0  # Delta time (seconds)
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.VIDEORESIZE:
                    self._set_screen(event.w, event.h)

            self._scene.handle_events(events)
            next_scene = self._scene.update(dt)
            if next_scene is not None:
                self._change_scene(next_scene)

            # Clear the virtual surface.
            self._virtual_surface.fill((0, 0, 0))
            self._scene.draw(self._virtual_surface)

            # --- Scale the virtual surface to the window while preserving aspect ratio ---
            # Use the smaller ratio to avoid stretching.
            scale_factor = min(
                self._screen_rect.width / self._base_resolution[0],
                self._screen_rect.height / self._base_resolution[1]
            )
            new_width = int(self._base_resolution[0] * scale_factor)
            new_height = int(self._base_resolution[1] * scale_factor)
            scaled_surface = pygame.transform.smoothscale(self._virtual_surface, (new_width, new_height))

            # Center the scaled surface within the actual window.
            offset_x = (self._screen_rect.width - new_width) // 2
            offset_y = (self._screen_rect.height - new_height) // 2

            self._screen.fill((0, 0, 0))  # Fill background (black or choose your color)
            self._screen.blit(scaled_surface, (offset_x, offset_y))
            pygame.display.flip()

        pygame.quit()

    def _change_scene(self, state: State) -> None:
        """
        Changes the current scene to a new scene and resets its state.
        """
        self._state = state
        self._scene = self._scenes[state]
        self._scene.reset()

