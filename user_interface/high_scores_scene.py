import pygame
import os
import json
from typing import Optional
from user_interface.scene import Scene
from user_interface.states import State
from hardware_interface.mqtt_communication import MQTTClient

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

class HighScoresScene(Scene):
    def __init__(
        self, shared_data: dict,
        mqtt_client: MQTTClient,
    ) -> None:
        super().__init__(shared_data)
        # Load fonts.
        self.title_font = pygame.font.Font("./assets/joystix_monospace.ttf", 144)
        self.body_font = pygame.font.Font("./assets/joystix_monospace.ttf", 72)
        self.line_font = pygame.font.Font("./assets/joystix_monospace.ttf", 36)
        
        # Load the shared background.
        self.background = pygame.image.load("assets/Background1_1920_1080.png").convert()
        
        # Pre-render the title with shadow.
        self.title_surface = self.title_font.render("HIGH SCORES", True, (0, 0, 0))
        self.title_shadow_surface = self.title_font.render("HIGH SCORES", True, (65, 26, 64))
        
        # Load and scale the trophy image.
        self.trophy = pygame.image.load("assets/Prize.png").convert_alpha()
        self.trophy = pygame.transform.scale(self.trophy, (200, 200))
        
        self.high_scores = load_high_scores()
        self.scores_surfaces = []
        self.update_scores_surfaces()
        self._next_state = None

    def update_scores_surfaces(self):
        self.high_scores = load_high_scores()
        # Sort scores in ascending order (lower scores are better).
        sorted_scores = sorted(self.high_scores, key=lambda x: x["score"])
        self.scores_surfaces = []
        # Render the top 10.
        top10 = sorted_scores[:10]
        for idx, entry in enumerate(top10, start=1):
            line = f"{idx}. {entry['name']} - {entry['score']:.2f}"
            surface = self.line_font.render(line, True, (0, 0, 0))
            self.scores_surfaces.append(surface)
            
        # Check if there is a recent entry and if it is not in the top 10.
        recent_entry = self.shared_data.get("recent_entry")
        if recent_entry and recent_entry not in top10:
            # Append a "..." separator.
            dots_surface = self.line_font.render("...", True, (0, 0, 0))
            self.scores_surfaces.append(dots_surface)
            # Find the ranking of the recent entry.
            try:
                rank = sorted_scores.index(recent_entry) + 1
            except ValueError:
                rank = "?"
            recent_line = f"{rank}. {recent_entry['name']} - {recent_entry['score']:.2f}"
            recent_surface = self.line_font.render(recent_line, True, (0, 0, 0))
            self.scores_surfaces.append(recent_surface)

    def handle_events(self, events) -> None:
        for ev in events:
            # Advance on any mouse click OR any key press
            if ev.type == pygame.MOUSEBUTTONDOWN or ev.type == pygame.KEYDOWN:
                self._next_state = State.MAIN_MENU

    def update(self, dt: float) -> Optional[State]:
        return self._next_state

    def draw(self, screen: pygame.Surface) -> None:
        screen.blit(self.background, (0, 0))
        # Draw the trophy in the top right.
        trophy_x = 1920 - self.trophy.get_width() - 50
        trophy_y = 50
        screen.blit(self.trophy, (trophy_x, trophy_y))
        # Draw the title with shadow.
        title_shadow_x = (1920 - self.title_shadow_surface.get_width()) // 2 + 10
        title_y = 150
        screen.blit(self.title_shadow_surface, (title_shadow_x, title_y))
        title_x = (1920 - self.title_surface.get_width()) // 2
        screen.blit(self.title_surface, (title_x, title_y))
        # Draw the high scores list.
        start_y = title_y + self.title_surface.get_height() + 50
        for i, surface in enumerate(self.scores_surfaces):
            x = (1920 - surface.get_width()) // 2
            y = start_y + i * (surface.get_height() + 10)
            screen.blit(surface, (x, y))

    def reset(self) -> None:
        self._next_state = None
        self.update_scores_surfaces()

