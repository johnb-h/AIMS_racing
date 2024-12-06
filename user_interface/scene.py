import pygame

from abc import ABC, abstractmethod

# Base Scene Class
class Scene(ABC):
    def __init__(self):
        self._next_state = None

    @abstractmethod
    def handle_events(self, events):
        """Handle input events like keyboard or mouse."""
        pass

    @abstractmethod
    def update(self):
        """Update game state."""
        pass

    @abstractmethod
    def draw(self, screen):
        """Draw the scene to the screen."""
        pass

    @abstractmethod
    def reset(self):
        pass
