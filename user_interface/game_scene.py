import pygame

from scene import Scene
from states import State

# Game Scene
class GameScene(Scene):
    def __init__(self):
        super().__init__()
        self.player_pos = [400, 300]
        self.player_speed = 5

    def handle_events(self, events):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            self.player_pos[0] -= self.player_speed
        if keys[pygame.K_RIGHT]:
            self.player_pos[0] += self.player_speed
        if keys[pygame.K_UP]:
            self.player_pos[1] -= self.player_speed
        if keys[pygame.K_DOWN]:
            self.player_pos[1] += self.player_speed

        for ev in events:
            if ev.type == pygame.MOUSEBUTTONDOWN:
                self._next_state = State.NAME_ENTRY

    def update(self):
        return self._next_state

    def draw(self, screen):
        screen.fill((0, 128, 0))  # Green background
        pygame.draw.rect(screen, (255, 0, 0), (*self.player_pos, 50, 50))

    def reset(self):
        print("Resetting Game Scene")
        self._next_state = None
        self.player_pos = [400, 300]

