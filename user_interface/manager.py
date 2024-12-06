from states import State
from main_menu_scene import MainMenuScene
from instructions_scene import InstructionsScene
from game_scene import GameScene
from name_entry_scene import NameEntryScene

# Scene Manager Class
class Manager:
    def __init__(self, start_scene):
        self._state = None
        self._scene = None
        self._scenes = None
        self._create_states()
        self.change_scene(start_scene)

    def change_scene(self, state):
        """Switch to a different scene."""
        self._state = state
        self._scene = self._scenes[state]
        self._scene.reset()

    def handle_events(self, events):
        self._scene.handle_events(events)

    def update(self, dt):
        next_scene = self._scene.update(dt)
        if next_scene is not None:
            self.change_scene(next_scene)

    def draw(self, screen):
        self._scene.draw(screen)

    def _create_states(self):
        self._scenes = {
            State.MAIN_MENU: MainMenuScene(),
            State.INSTRUCTIONS: InstructionsScene(),
            State.GAME: GameScene(),
            State.NAME_ENTRY: NameEntryScene()
        }
