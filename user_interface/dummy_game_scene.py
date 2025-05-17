import pygame
from user_interface.scene import Scene
from user_interface.states import State
from hardware_interface.mqtt_communication import MQTTClient
from user_interface.sound_player import SoundPlayer

class DummyGameScene(Scene):
    def __init__(
        self, shared_data: dict,
        mqtt_client: MQTTClient,
        sound_player: SoundPlayer,
    ) -> None:
        super().__init__(shared_data, mqtt_client, sound_player)
        self.font = pygame.font.Font(None, 36)
        self.input_text = ""
        self.simulated_score = 0.0

    def handle_events(self, events):
        for ev in events:
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    self.reset()
                    self._next_state = State.MAIN_MENU
                elif ev.key == pygame.K_RETURN:
                    self.set_next_state()
                elif ev.key == pygame.K_BACKSPACE:
                    self.input_text = self.input_text[:-1]
                else:
                    self.input_text += ev.unicode

    def update(self, dt):
        return self._next_state

    def draw(self, screen):
        screen.fill((128, 0, 128))
        prompt = self.font.render("Enter your score:", True, (255, 255, 255))
        score_surface = self.font.render(self.input_text, True, (255, 255, 255))
        screen.blit(prompt, (50, 50))
        screen.blit(score_surface, (50, 100))

    def reset(self):
        self._next_state = None
        self.input_text = ""
        self.simulated_score = 0.0

    def handle_mqtt(self) -> None:
        """Handle incoming MQTT messages."""
        if not self._mqtt_client.queue_empty():
            topic, msg = self._mqtt_client.pop_queue()
            self.set_next_state()

    def set_next_state(self) -> None:
        try:
            self.simulated_score = float(self.input_text)
        except ValueError:
            self.simulated_score = 100
        self.shared_data["score"] = self.simulated_score
        self.shared_data["round"] = 3
        self._next_state = State.NAME_ENTRY
