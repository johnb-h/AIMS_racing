from abc import ABC, abstractmethod
from typing import Optional
from user_interface.states import State
from hardware_interface.mqtt_communication import MQTTClient
from user_interface.sound_player import SoundPlayer

class Scene(ABC):
    def __init__(
        self, 
        shared_data: dict,
        mqtt_client: MQTTClient,
        sound_player: SoundPlayer,
    ) -> None:
        self.shared_data = shared_data
        self._mqtt_client = mqtt_client
        self._sound_player = sound_player
        self._next_state: Optional[State] = None

    @abstractmethod
    def handle_events(self, events) -> None:
        """Process input events."""
        pass

    @abstractmethod
    def update(self, dt: float) -> Optional[State]:
        """Update the scene logic."""
        pass

    @abstractmethod
    def draw(self, screen) -> None:
        """Draw the scene content on the given surface."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset the scene state to its initial conditions."""
        pass

    @abstractmethod
    def handle_mqtt(self) -> None:
        """Reset the scene state to its initial conditions."""
        pass
