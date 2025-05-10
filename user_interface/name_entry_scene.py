import pygame
import os
import json
from typing import Optional
from user_interface.scene import Scene
from user_interface.states import State
from hardware_interface.mqtt_communication import MQTTClient
from user_interface.sound_player import SoundPlayer

HIGH_SCORES_FILE = "high_scores.json"

def load_high_scores():
    if os.path.exists(HIGH_SCORES_FILE):
        with open(HIGH_SCORES_FILE, "r") as f:
            try:
                scores = json.load(f)
            except json.JSONDecodeError:
                scores = []
    else:
        scores = []
    return scores

class NameEntryScene(Scene):
    def __init__(
        self, shared_data: dict,
        mqtt_client: MQTTClient,
        sound_player: SoundPlayer,
    ) -> None:
        super().__init__(shared_data, mqtt_client, sound_player)
        # Load fonts.
        self.title_font = pygame.font.Font("./assets/joystix_monospace.ttf", 110)
        self.body_font = pygame.font.Font("./assets/joystix_monospace.ttf", 72)
        self.prompt_font = pygame.font.Font("./assets/joystix_monospace.ttf", 36)

        # Load shared background.
        self.background = pygame.image.load("assets/Background1_1920_1080.png").convert()
        
        # Get the score from shared data.
        self.score = self.shared_data.get("score", 0.0)
        # Load previously saved scores.
        self.high_scores = load_high_scores()
        # Compute ranking: number of scores strictly better (i.e. with lower value) plus one.
        self.ranking = sum(1 for entry in self.high_scores if entry["score"] < self.score) + 1

        # Pre-render score and ranking texts.
        score_text = f"YOUR TIME: {self.score:.2f}s"
        rank_text = f"YOUR RANK: {self.ranking}"
        self.score_surface = self.title_font.render(score_text, True, (0, 0, 0))
        self.rank_surface = self.body_font.render(rank_text, True, (0, 0, 0))
        
        # The prompt text to ask the player to type in their name.
        self.prompt_text = "ENTER YOUR NAME: "
        self.prompt_surface = self.prompt_font.render(self.prompt_text, True, (0, 0, 0))

        # User input string.
        self.player_name = ""
        self._next_state = None

    def handle_events(self, events) -> None:
        for ev in events:
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_RETURN:
                    if self.player_name.strip() != "":
                        new_entry = {"name": self.player_name.strip(), "score": self.score}
                        # Append the new entry to the complete list (saving all scores)
                        self.high_scores.append(new_entry)
                        with open(HIGH_SCORES_FILE, "w") as f:
                            json.dump(self.high_scores, f, indent=4)
                        # Also store the recent entry to shared_data for the high scores screen.
                        self.shared_data["recent_entry"] = new_entry
                    self.set_next_state()
                elif ev.key == pygame.K_BACKSPACE:
                    self.player_name = self.player_name[:-1]
                else:
                    self.player_name += ev.unicode


    def handle_mqtt(self) -> None:
        """Handle incoming MQTT messages."""
        if not self._mqtt_client.queue_empty():
            topic, msg = self._mqtt_client.pop_queue()
            self.set_next_state()

    def set_next_state(self) -> Optional[State]:
        self._sound_player.play_menu_click()
        self._next_state = State.HIGH_SCORES

    def update(self, dt: float) -> Optional[State]:
        # No blinking timers; simply return the next state when set.
        return self._next_state

    def draw(self, screen: pygame.Surface) -> None:
        screen.blit(self.background, (0, 0))
        # Draw the score at the top center.
        score_x = (1920 - self.score_surface.get_width()) // 2
        score_y = 350
        screen.blit(self.score_surface, (score_x, score_y))
        # Draw the ranking beneath the score.
        rank_x = (1920 - self.rank_surface.get_width()) // 2
        rank_y = score_y + self.score_surface.get_height() + 20
        screen.blit(self.rank_surface, (rank_x, rank_y))
        # Draw the static prompt and the player's current input.
        full_prompt = self.prompt_text + self.player_name
        prompt_surface = self.prompt_font.render(full_prompt, True, (0, 0, 0))
        prompt_x = (1920 - prompt_surface.get_width()) // 2
        prompt_y = rank_y + self.rank_surface.get_height() + 50
        screen.blit(prompt_surface, (prompt_x, prompt_y))

    def reset(self) -> None:
        # Reload data when the scene is reactivated.
        self._next_state = None
        self.player_name = ""
        self.score = self.shared_data.get("score", 0.0)
        self.high_scores = load_high_scores()
        self.ranking = sum(1 for entry in self.high_scores if entry["score"] < self.score) + 1
        
        score_text = f"YOUR SCORE: {self.score:.2f}"
        rank_text = f"YOUR RANK: {self.ranking}"
        self.score_surface = self.title_font.render(score_text, True, (0, 0, 0))
        self.rank_surface = self.body_font.render(rank_text, True, (0, 0, 0))

