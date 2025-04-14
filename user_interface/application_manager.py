from typing import Dict, Type

import pygame

from user_interface.constants import TARGET_FPS
from user_interface.game_scene import GameScene
from user_interface.instructions_scene import InstructionsScene
from user_interface.main_menu_scene import MainMenuScene
from user_interface.name_entry_scene import NameEntryScene
from user_interface.scene import Scene
from user_interface.states import State


class ApplicationManager:
    def __init__(self, window_width=None, window_height=None):
        # Initialize pygame first
        pygame.init()
        pygame.display.set_caption("Evolving Cars")

        # Get screen info if dimensions not specified
        if window_width is None or window_height is None:
            screen_info = pygame.display.Info()
            window_width = window_width or screen_info.current_w
            window_height = window_height or screen_info.current_h

        self._window_width = window_width
        self._window_height = window_height

        # Create scenes before setting up the screen
        self._state = State.MAIN_MENU
        self._scenes = {
            State.MAIN_MENU: MainMenuScene(),
            State.INSTRUCTIONS: InstructionsScene(),
            State.GAME: GameScene(),
            State.NAME_ENTRY: NameEntryScene(),
        }
        self._scene = self._scenes[self._state]

        # Now set up the screen, which will resize all scenes
        self._set_screen(window_width, window_height)

    def _set_screen(self, window_width, window_height):
        self._screen = pygame.display.set_mode(
            (window_width, window_height), pygame.RESIZABLE
        )
        self._screen_rect = self._screen.get_rect()
        # Resize the current scene if it supports resizing
        for scene in self._scenes.values():
            if hasattr(scene, "resize"):
                scene.resize(window_width, window_height)
        pygame.display.update()

    def run_game_loop(self):
        clock = pygame.time.Clock()
        running = True

        while running:
            dt = clock.tick(TARGET_FPS) / 1000.0  # Delta time in seconds

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

            self._scene.draw(self._screen)

            pygame.display.flip()

        pygame.quit()

    def _change_scene(self, state):
        self._state = state
        self._scene = self._scenes[state]
        self._scene.reset()
