import pygame
import json
from typing import Optional
from user_interface.constants import TARGET_FPS, MENU_FPS
from user_interface.game_scene import GameScene
# from user_interface.dummy_game_scene import DummyGameScene
from user_interface.instructions_scene import InstructionsScene
from user_interface.main_menu_scene import MainMenuScene
from user_interface.name_entry_scene import NameEntryScene
from user_interface.high_scores_scene import HighScoresScene
from user_interface.states import State
from hardware_interface.mqtt_communication import MQTTClient
from hardware_interface.communication_protocol import RaceCar

class ApplicationManager:
    """
    Manages the overall application with a fixed virtual resolution (1920x1080),
    scales it to the window, and uses lazy scene loading for faster startup.
    """
    def __init__(self, window_width: Optional[int] = None, window_height: Optional[int] = None) -> None:
        # Initialize MQTT Instance
        with open("configs/mqtt_config.json", "r", encoding="utf-8") as f:
            mqtt_config = json.load(f)
            f.close()
        self.mqtt_client = MQTTClient(mqtt_config=mqtt_config)
        self.mqtt_client.connect()

        self._base_resolution = (1920, 1080)

        pygame.init()
        pygame.display.set_caption("Evolving Cars")

        # Sound
        self._general_race_noise = pygame.mixer.Sound("./assets/general_race_noise.wav")
        self._cheer = pygame.mixer.Sound("./assets/cheers.wav")
        self._general_race_noise.play()

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

        # Lazy scene loading: map states to scene classes and pass shared_data.
        self._scene_map = {
            State.MAIN_MENU: MainMenuScene,
            State.INSTRUCTIONS: InstructionsScene,
            State.GAME: GameScene,
            # State.GAME: DummyGameScene,
            State.NAME_ENTRY: NameEntryScene,
            State.HIGH_SCORES: HighScoresScene,
        }
        self._scenes = {}

        # Start immediately with the Main Menu.
        self._state: State = State.MAIN_MENU
        self._scene = self._get_scene(self._state)

    def _set_screen(self, window_width: int, window_height: int) -> None:
        self._screen = pygame.display.set_mode((window_width, window_height), pygame.RESIZABLE)
        self._screen_rect = self._screen.get_rect()
        pygame.display.update()

    def _get_scene(self, state: State):
        if state not in self._scenes:
            if state == State.GAME:
                self._general_race_noise.stop()
            elif state == state.NAME_ENTRY or state == state.HIGH_SCORES:
                self._cheer.play()
                self._cheer.fadeout(5)
            else:
                self._general_race_noise.play()
            self._scenes[state] = self._scene_map[state](self.shared_data, self.mqtt_client)
        return self._scenes[state]

    def run_game_loop(self) -> None:
        clock = pygame.time.Clock()
        running = True
        self.mqtt_client.connect()
        self.mqtt_client.start_loop()
        self.mqtt_client.subscribe(RaceCar.topic)
        while running:
            # MQTT Handle
            if not self.mqtt_client.queue_empty():
                topic, msg = self.mqtt_client.pop_queue()
                if RaceCar.topic in topic:
                    race_car = RaceCar()
                    race_car.deserialise(msg)
                    car_index = race_car.id
                    if self._state == State.GAME:
                        if car_index < len(self._scene.cars):
                            self._scene.cars[car_index].selected = not self._scene.cars[
                                car_index
                            ].selected

            # Use MENU_FPS for menu scenes; else TARGET_FPS.
            current_fps = MENU_FPS if self._state in (State.MAIN_MENU, State.INSTRUCTIONS) else TARGET_FPS
            dt = clock.tick(current_fps) / 1000.0
            events = pygame.event.get()

            for event in events:
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.VIDEORESIZE:
                    self._set_screen(event.w, event.h)

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
        self._state = state
        self._scene = self._get_scene(state)
        self._scene.reset()


