import pygame
from user_interface.scene import Scene
from user_interface.states import State

class DummyGameScene(Scene):
    def __init__(self, shared_data: dict):
        super().__init__(shared_data)
        self.font = pygame.font.Font(None, 36)
        self.input_text = ""
        self.simulated_score = 0.0

    def handle_events(self, events):
        for ev in events:
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_RETURN:
                    try:
                        self.simulated_score = float(self.input_text)
                    except ValueError:
                        self.simulated_score = 0.0
                    # Save the game score as a float in the shared data.
                    self.shared_data["score"] = self.simulated_score
                    self._next_state = State.NAME_ENTRY
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

