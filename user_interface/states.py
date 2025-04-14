from enum import Enum

class State(Enum):
    MAIN_MENU = "main_menu"
    INSTRUCTIONS = "instructions"
    GAME = "game"
    NAME_ENTRY = "name_entry"
    HIGH_SCORES = "high_scores"
