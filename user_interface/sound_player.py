import pygame
import random
import os

from user_interface.constants import (
    ASSETS_DIR,
    STARTING_MENU_MUSIC_FILE,
    SECONDARY_MENU_MUSIC_FILE,
    MENU_MUSIC_DIR,
    MENU_MUSIC_CROSSFADE,
    MENU_CLICK_SOUND_FILE,
    GAME_START_SOUND_FILE,
    GAME_BACKGROUND_MUSIC_FILE,
    RACE_BEEP_1_SOUND_FILE,
    RACE_BEEP_2_SOUND_FILE,
)

class SoundPlayer:
    """
    Plays all your sounds, and for menu music does an overlapping X-second
    crossfade between tracks.
    """

    def __init__(self):
        self._initial_playlist = [
            STARTING_MENU_MUSIC_FILE, SECONDARY_MENU_MUSIC_FILE
        ]

        self._all_menu_music_files = [
            f for f in os.listdir(MENU_MUSIC_DIR)
            if f.lower().endswith(".wav") and f != STARTING_MENU_MUSIC_FILE
        ]

        # How long we crossfade (ms)
        self._starting_sound = True
        self._crossfade_ms = int(MENU_MUSIC_CROSSFADE * 1000)
        self._menu_music_channel = None
        self._start_ticks = None
        self._track_length_ms = None

        self._menu_click_sound = self._load_sound(MENU_CLICK_SOUND_FILE)
        self._game_start_sound = self._load_sound(GAME_START_SOUND_FILE)
        self._game_background_music = self._load_sound(GAME_BACKGROUND_MUSIC_FILE)
        self._race_beep_1_sound = self._load_sound(RACE_BEEP_1_SOUND_FILE)
        self._race_beep_2_sound = self._load_sound(RACE_BEEP_2_SOUND_FILE)

    def _load_sound(self, file_path) -> pygame.mixer.Sound:
        menu_click_file_path = os.path.join(ASSETS_DIR, file_path)
        return pygame.mixer.Sound(menu_click_file_path)


    def _load_next_menu_sound(self) -> pygame.mixer.Sound:
        """
        If we still have tracks in the fixed playlist, pop and play the next one.
        Otherwise, pick randomly (excluding whatever just played).
        """
        # 1) If there’s an explicit “first two” playlist, use that
        if self._initial_playlist:
            filename = self._initial_playlist.pop(0)
        else:
            # 2) Otherwise, pick a random .wav (but not the one just used)
            candidates = self._all_menu_music_files.copy()
            if hasattr(self, "_last_file") and self._last_file in candidates and len(candidates) > 1:
                candidates.remove(self._last_file)
            filename = random.choice(candidates)

        # remember for next‐time exclusion
        self._last_file = filename

        # load and return
        path = os.path.join(MENU_MUSIC_DIR, filename)
        return pygame.mixer.Sound(path)

    def update(self) -> None:
        now = pygame.time.get_ticks()

        # 1) No track yet → start the first one
        if self._menu_music_channel is None:
            sound = self._load_next_menu_sound()
            self._menu_music_channel = sound.play(fade_ms=self._crossfade_ms)
            self._start_ticks = now
            self._track_length_ms = sound.get_length() * 1000
            return

        # 2) Time left before end?
        elapsed = now - (self._start_ticks or 0)
        time_left = (self._track_length_ms or 0) - elapsed

        if time_left <= self._crossfade_ms:
            # begin crossfade
            old_channel = self._menu_music_channel

            # load & play next
            next_sound = self._load_next_menu_sound()
            next_sound.set_volume(0.5)
            if self._starting_sound:
                self._starting_sound = False
                next_channel = next_sound.play()
            else:
                next_channel = next_sound.play(fade_ms=self._crossfade_ms)

            # fade out the old
            old_channel.fadeout(self._crossfade_ms)

            # swap pointers so next frame refers to the new track
            self._menu_music_channel = next_channel
            self._start_ticks = now
            self._track_length_ms = next_sound.get_length() * 1000

    def stop_menu_music(self) -> None:
        """Fade out whatever menu music is still playing."""
        if self._menu_music_channel:
            self._menu_music_channel.fadeout(self._crossfade_ms)
            self._menu_music_channel = None
            self._start_ticks = None
            self._track_length_ms = None

    def play_menu_click(self) -> None:
        self._menu_click_sound.play()

    def play_game_start(self) -> None:
        self._game_start_sound.play()

    def play_game_background_music(self) -> None:
        self._game_background_music.set_volume(0.3)
        self._game_background_music.play()

    def stop_game_background_music(self) -> None:
        self._game_background_music.stop()

    def play_race_beep_1(self) -> None:
        self._race_beep_1_sound.play()

    def play_race_beep_2(self) -> None:
        self._race_beep_2_sound.play()

