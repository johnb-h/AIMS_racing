import pygame
from typing import Optional
from user_interface.constants import TARGET_FPS, MENU_FPS
from user_interface.game_scene import GameScene
from user_interface.instructions_scene import InstructionsScene
from user_interface.main_menu_scene import MainMenuScene
from user_interface.name_entry_scene import NameEntryScene
from user_interface.high_scores_scene import HighScoresScene
from user_interface.states import State

class ApplicationManager:
    """
    Manages the overall application with a fixed virtual resolution (1920x1080),
    scales it to the window, and uses lazy scene loading for faster startup.
    """
    def __init__(self, window_width: Optional[int] = None, window_height: Optional[int] = None) -> None:
        # Fixed virtual resolution.
        self._base_resolution = (1920, 1080)
        
        # Get screen info if dimensions not specified
        if window_width is None or window_height is None:
            screen_info = pygame.display.Info()
            window_width = window_width or screen_info.current_w
            window_height = window_height or screen_info.current_h

        self._window_width = window_width
        self._window_height = window_height

        # Basic Pygame initialization.
        pygame.init()
        pygame.display.set_caption("Evolving Cars")
        self._set_screen(window_width, window_height)

        # Create the virtual surface for drawing.
        self._virtual_surface = pygame.Surface(self._base_resolution)

        # Use lazy loading: map states to their scene classes,
        # and only instantiate the current scene.
        self._scene_map = {
            State.MAIN_MENU: MainMenuScene,
            State.INSTRUCTIONS: InstructionsScene,
            State.GAME: GameScene,
            State.NAME_ENTRY: NameEntryScene,
            State.HIGH_SCORES: HighScoresScene,
        }
        self._scenes = {}  # Cache for instantiated scenes.

        # Start immediately with the Main Menu.
        self._state: State = State.MAIN_MENU
        self._scene = self._get_scene(self._state)

    def _set_screen(self, window_width: int, window_height: int) -> None:
        """
        Sets and updates the window's display. Called on initialization and resize.
        """
        self._screen = pygame.display.set_mode((window_width, window_height), pygame.RESIZABLE)
        self._screen_rect = self._screen.get_rect()
        pygame.display.update()

    def _get_scene(self, state: State):
        """
        Lazily instantiate and return the scene for the given state.
        """
        if state not in self._scenes:
            self._scenes[state] = self._scene_map[state]()
        return self._scenes[state]

    def run_game_loop(self) -> None:
        """
        The main loop: processes events, updates the current scene based on a dynamic FPS
        and scales the virtual surface to the display.
        """
        clock = pygame.time.Clock()
        running = True

        while running:
            # Use MENU_FPS for menu scenes; else TARGET_FPS.
            current_fps = MENU_FPS if self._state in (State.MAIN_MENU, State.INSTRUCTIONS) else TARGET_FPS
            dt = clock.tick(current_fps) / 1000.0
            events = pygame.event.get()

            for event in events:
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.VIDEORESIZE:
                    self._set_screen(event.w, event.h)

            # Process the current scene.
            self._scene.handle_events(events)
            next_scene = self._scene.update(dt)
            if next_scene is not None:
                self._state = next_scene
                self._scene = self._get_scene(self._state)
                self._scene.reset()

            # Render the current scene onto the virtual surface.
            self._virtual_surface.fill((0, 0, 0))
            self._scene.draw(self._virtual_surface)

            # Scale the virtual surface to the actual window.
            scale_factor = min(
                self._screen_rect.width / self._base_resolution[0],
                self._screen_rect.height / self._base_resolution[1]
            )
            new_width = int(self._base_resolution[0] * scale_factor)
            new_height = int(self._base_resolution[1] * scale_factor)
            scaled_surface = pygame.transform.smoothscale(self._virtual_surface, (new_width, new_height))

            # Center the scaled surface.
            offset_x = (self._screen_rect.width - new_width) // 2
            offset_y = (self._screen_rect.height - new_height) // 2
            self._screen.fill((0, 0, 0))
            self._screen.blit(scaled_surface, (offset_x, offset_y))
            pygame.display.flip()

        pygame.quit()

    def _change_scene(self, state: State) -> None:
        """
        Allows for changing the scene externally if needed.
        """
        self._state = state
        self._scene = self._get_scene(state)
        self._scene.reset()

