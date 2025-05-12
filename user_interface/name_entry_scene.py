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

        self.score = -1
        self.rounds = -1
        self.ranking = -1
        self.high_scores = load_high_scores()

        # Prompt text
        self.prompt_text = "ENTER YOUR NAME: "
        self.prompt_surface = self.prompt_font.render(self.prompt_text, True, (0, 0, 0))

        # User input string.
        self.player_name = ""
        self._next_state = None

    def handle_events(self, events) -> None:
        for ev in events:
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    self._sound_player.play_menu_click()
                    self._next_state = State.MAIN_MENU
                if ev.key == pygame.K_RETURN:
                    # Only submit if name length >= 2
                    if 2 <= len(self.player_name) <= 12:
                        name = self.player_name.strip()
                        if name:
                            self._save_score_entry(name)
                        self.set_next_state()
                    # else: ignore submit
                elif ev.key == pygame.K_BACKSPACE:
                    # Remove last character
                    self.player_name = self.player_name[:-1]
                else:
                    # Only allow alphabetic chars, up to max length 12
                    char = ev.unicode
                    if char.isalpha() and len(self.player_name) < 12:
                        self._sound_player.play_key_press_sound()
                        self.player_name += char
                    # else: ignore invalid character

    def handle_mqtt(self) -> None:
        """Handle incoming MQTT messages."""
        if not self._mqtt_client.queue_empty():
            topic, msg = self._mqtt_client.pop_queue()

    def _save_score_entry(self, name: str) -> None:
        """Merge the new score in, drop any duplicate name with a worse time, and write out."""
        new_entry = {"name": name, "rounds": self.rounds, "score": self.score}

        found = False
        for entry in self.high_scores:
            if entry["name"] == name:
                found = True
                # replace only if strictly better: fewer rounds, or same rounds + faster
                if (new_entry["rounds"], new_entry["score"]) < (entry["rounds"], entry["score"]):
                    entry["rounds"] = new_entry["rounds"]
                    entry["score"]  = new_entry["score"]
                break

        if not found:
            self.high_scores.append(new_entry)

        self.high_scores.sort(key=lambda e: (e["rounds"], e["score"]))

        with open(HIGH_SCORES_FILE, "w") as f:
            json.dump(self.high_scores, f, indent=4)

        survivor = next(e for e in self.high_scores if e["name"] == name)
        self.shared_data["recent_entry"] = survivor

    def set_next_state(self) -> Optional[State]:
        self._sound_player.play_menu_click()
        self._next_state = State.HIGH_SCORES

    def update(self, dt: float) -> Optional[State]:
        return self._next_state

    def draw(self, screen: pygame.Surface) -> None:
        self.rounds = self.shared_data["rounds"]
        self.score = self.shared_data["score"]
        def is_better(e):
            return (e["rounds"], e["score"]) < (self.rounds, self.score)
        self.ranking = sum(1 for entry in self.high_scores if is_better(entry)) + 1

        rounds_text = f"NUM ROUNDS: {self.rounds}"
        score_text = f"TIME: {self.score:.2f}"
        rank_text = f"YOUR RANK: {self.ranking}"
        self.rounds_surface = self.title_font.render(rounds_text, True, (0, 0, 0))
        self.score_surface = self.title_font.render(score_text, True, (0, 0, 0))
        self.rank_surface = self.body_font.render(rank_text, True, (0, 0, 0))

        screen.blit(self.background, (0, 0))
        # Draw the rounds at the top center.
        score_x = (1920 - self.rounds_surface.get_width()) // 2
        score_y = 200
        screen.blit(self.rounds_surface, (score_x, score_y))
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
        self.score = 100
        self.rounds = 100
        self.high_scores = load_high_scores()
        self.ranking = sum(1 for entry in self.high_scores if (entry["rounds"], entry["score"]) < (self.rounds, self.score)) + 1

        rounds_text = f"NUM ROUNDS: {self.rounds}"
        score_text = f"TIME: {self.score:.2f}"
        rank_text = f"YOUR RANK: {self.ranking}"
        self.rounds_surface = self.title_font.render(rounds_text, True, (0, 0, 0))
        self.score_surface = self.title_font.render(score_text, True, (0, 0, 0))
        self.rank_surface = self.body_font.render(rank_text, True, (0, 0, 0))

