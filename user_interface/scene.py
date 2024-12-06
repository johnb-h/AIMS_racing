from abc import ABC, abstractmethod

from states import State
from typing import Optional

# Base Scene Class
class Scene(ABC):
    def __init__(self):
        self._next_state = None

    @abstractmethod
    def handle_events(self, events):
        """Handle input events like keyboard or mouse."""
        pass

    @abstractmethod
    def update(self, dt) -> Optional[State]:
        """Update game state."""
        pass

    @abstractmethod
    def draw(self, screen):
        """Draw the scene to the screen."""
        pass

    @abstractmethod
    def reset(self):
        pass
