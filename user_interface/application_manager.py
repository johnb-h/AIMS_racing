import pygame
import json
import concurrent.futures
from typing import Optional

from user_interface.game_scene import GameScene
from user_interface.instructions_scene import InstructionsScene
from user_interface.main_menu_scene import MainMenuScene
from user_interface.name_entry_scene import NameEntryScene
from user_interface.high_scores_scene import HighScoresScene
from user_interface.states import State
from user_interface.sound_player import SoundPlayer
from hardware_interface.mqtt_communication import MQTTClient
from hardware_interface.communication_protocol import RaceCar

from user_interface.constants import (
    GAME_RENDER_FPS,
    MENU_FPS,
    CROSSFADE_DURATION_MS,
    CROSSFADE_STEPS,
    MIN_SCENE_DURATION,
)


class ApplicationManager:
    """
    Manages the overall application with a fixed virtual resolution (1920x1080),
    scales it to the window, and uses lazy scene loading for faster startup,
    plus fade transitions between scenes.
    """
    def __init__(self, window_width: Optional[int] = None, window_height: Optional[int] = None) -> None:
        # Initialize MQTT Instance
        with open("configs/mqtt_config.json", "r", encoding="utf-8") as f:
            mqtt_config = json.load(f)

        self._base_resolution = (1920, 1080)

        pygame.init()
        pygame.display.set_caption("Evolving Cars")

        # Prepare fade surface
        self._fade_surf = pygame.Surface(self._base_resolution)
        self._fade_surf.fill((0, 0, 0))

        if window_width is None or window_height is None:
            screen_info = pygame.display.Info()
            window_width = window_width or screen_info.current_w
            window_height = window_height or screen_info.current_h

        self._window_width = window_width
        self._window_height = window_height
        self._set_screen(self._window_width, self._window_height)

        self._virtual_surface = pygame.Surface(self._base_resolution)

        # Create a shared data dictionary to pass information between scenes.
        self.shared_data = {}

        # Initialize MQTT
        self.mqtt_client = MQTTClient(mqtt_config=mqtt_config)
        self.mqtt_client.connect()

        # Sound player
        self._sound_player = SoundPlayer()

        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

        self._scene_map = {
            State.MAIN_MENU:  MainMenuScene(self.shared_data, self.mqtt_client, self._sound_player),
            State.INSTRUCTIONS: InstructionsScene(self.shared_data, self.mqtt_client, self._sound_player),
            State.GAME: GameScene(self.shared_data, self.mqtt_client, self._sound_player, executor=self._executor),
            State.NAME_ENTRY: NameEntryScene(self.shared_data, self.mqtt_client, self._sound_player),
            State.HIGH_SCORES: HighScoresScene(self.shared_data, self.mqtt_client, self._sound_player),
        }
        self._scenes = {}

        # Start immediately with the Main Menu.
        self._state: State = State.MAIN_MENU
        self._scene = self._get_scene(self._state)
        self._scene.reset()

        self._scene_start_ticks = pygame.time.get_ticks()

    def _set_screen(self, window_width: int, window_height: int) -> None:
        self._screen = pygame.display.set_mode((window_width, window_height), pygame.RESIZABLE)
        self._screen_rect = self._screen.get_rect()
        pygame.display.update()

    def _get_scene(self, state: State):
        if state not in self._scenes:
            # Instantiate scene with shared data, mqtt client, sound player
            self._scenes[state] = self._scene_map[state]
        return self._scenes[state]

    def run_game_loop(self) -> None:
        running = True
        # (re)connect MQTT
        self.mqtt_client.connect()
        self.mqtt_client.start_loop()
        self.mqtt_client.subscribe(RaceCar.topic)

        clock = pygame.time.Clock()
        render_acc = 0.0

        while running:
            # 1) Run loop up to a high cap (so update/inputs stay snappy)
            dt = clock.tick(240) / 1000.0

            # 2) Gather events, track if we need an immediate render on resize
            events = pygame.event.get()
            force_render = False
            for event in events:
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.VIDEORESIZE:
                    self._set_screen(event.w, event.h)
                    force_render = True

            # 3) Handle MQTT, input & scene logic at full rate
            elapsed_s = (pygame.time.get_ticks() - self._scene_start_ticks) / 1000.0
            if elapsed_s >= MIN_SCENE_DURATION:
                self._scene.handle_mqtt()
                self._scene.handle_events(events)
            next_scene = self._scene.update(dt)
            if next_scene is not None:
                self._change_scene(next_scene)
                continue

            # 4) Always draw into the *virtual* surface
            self._virtual_surface.fill((0, 0, 0))
            self._scene.draw(self._virtual_surface)

            # 5) Scale & center
            scale = min(
                self._screen_rect.width  / self._base_resolution[0],
                self._screen_rect.height / self._base_resolution[1],
            )
            new_w = int(self._base_resolution[0] * scale)
            new_h = int(self._base_resolution[1] * scale)
            scaled = pygame.transform.smoothscale(self._virtual_surface, (new_w, new_h))
            offset_x = (self._screen_rect.width  - new_w) // 2
            offset_y = (self._screen_rect.height - new_h) // 2
            self._screen.fill((0,0,0))
            self._screen.blit(scaled, (offset_x, offset_y))

            # 6) Update menu music if we’re in one of the menu scenes
            if self._state in (
                State.MAIN_MENU,
                State.INSTRUCTIONS,
                State.NAME_ENTRY,
                State.HIGH_SCORES,
            ):
                self._sound_player.update()

            # 7) Throttle the *actual* flip() to MENU_FPS in menus, GAME_RENDER_FPS in‐game
            render_acc += dt
            target_fps = GAME_RENDER_FPS if self._state == State.GAME else MENU_FPS
            if force_render or render_acc >= 1.0/target_fps:
                pygame.display.flip()
                render_acc %= 1.0/target_fps

        pygame.quit()

    def _change_scene(self, new_state: State) -> None:
        self._crossfade_transition(new_state)

        # ← right after the fade: if we’re entering the game, stop any menu music
        if new_state == State.GAME:
            self._sound_player.stop_menu_music()

        self._scene_start_ticks = pygame.time.get_ticks()

    def _render_current_frame(self) -> None:
        """
        Draw the current scene once (used inside fade loops).
        """
        # Draw on virtual surface
        self._virtual_surface.fill((0, 0, 0))
        self._scene.draw(self._virtual_surface)

        # Scale to screen
        scale_factor = min(
            self._screen_rect.width / self._base_resolution[0],
            self._screen_rect.height / self._base_resolution[1]
        )
        new_w = int(self._base_resolution[0] * scale_factor)
        new_h = int(self._base_resolution[1] * scale_factor)
        scaled = pygame.transform.smoothscale(self._virtual_surface, (new_w, new_h))

        # Center on screen
        offset_x = (self._screen_rect.width - new_w) // 2
        offset_y = (self._screen_rect.height - new_h) // 2
        self._screen.fill((0, 0, 0))
        self._screen.blit(scaled, (offset_x, offset_y))

    def _crossfade_transition(self, new_state: State) -> None:
        """
        Fade out the current scene, switch to `new_state`, then fade in,
        hiding any blink-text during the crossfade.
        """
        # 0) Hide blink-text on the outgoing scene
        setattr(self._scene, "_hide_blink", True)

        # 1) Snapshot the old scene into a virtual surface
        old_virtual = pygame.Surface(self._base_resolution, pygame.SRCALPHA).convert_alpha()
        self._scene.draw(old_virtual)
        
        # 2) Prepare the new scene
        self._state = new_state
        self._scene = self._get_scene(new_state)
        self._scene.reset()
        # 2a) Hide blink-text on the incoming scene
        setattr(self._scene, "_hide_blink", True)

        # 3) Snapshot the new scene’s first frame
        new_virtual = pygame.Surface(self._base_resolution, pygame.SRCALPHA).convert_alpha()
        self._scene.draw(new_virtual)
        
        # 4) Prepare a combined surface
        combined = pygame.Surface(self._base_resolution, pygame.SRCALPHA).convert_alpha()
        
        duration_ms = CROSSFADE_DURATION_MS
        steps      = CROSSFADE_STEPS
        delay      = duration_ms // steps

        # 5) Crossfade loop
        for i in range(steps + 1):
            alpha = int(255 * (i / steps))
            old_virtual.set_alpha(255 - alpha)
            new_virtual.set_alpha(alpha)
            
            # Composite both onto `combined`
            combined.fill((0, 0, 0, 0))
            combined.blit(old_virtual, (0, 0))
            combined.blit(new_virtual, (0, 0))
            
            # Now scale & center as usual
            scale_factor = min(
                self._screen_rect.width  / self._base_resolution[0],
                self._screen_rect.height / self._base_resolution[1]
            )
            w = int(self._base_resolution[0] * scale_factor)
            h = int(self._base_resolution[1] * scale_factor)
            scaled = pygame.transform.smoothscale(combined, (w, h))
            
            offset_x = (self._screen_rect.width  - w) // 2
            offset_y = (self._screen_rect.height - h) // 2
            
            self._screen.fill((0, 0, 0))
            self._screen.blit(scaled, (offset_x, offset_y))
            pygame.display.flip()
            pygame.time.delay(delay)

        # 6) After fade, restore blink-text on all scenes
        for scene in self._scenes.values():
            setattr(scene, "_hide_blink", False)

