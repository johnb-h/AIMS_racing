def pixels_to_meters(pixels):
    # Irrespective of resolution, we assume the track is a fixed length across
    return pixels * WINDOW_WIDTH_IN_M / RESOLUTION[0]


RESOLUTION = (1920, 1080)
WINDOW_WIDTH_IN_M = 400  # In meters
WINDOW_HEIGHT_IN_M = pixels_to_meters(RESOLUTION[1])
SCREEN_CENTER_METERS = (WINDOW_WIDTH_IN_M / 2, WINDOW_HEIGHT_IN_M / 2)

# Constants
CONST_SPEED = 10  # In m/s
CONST_STEER = -0.05  # In radians

WHEELBASE = 2.5
CAR_WIDTH = 4
CAR_LENGTH = 12

MAX_STEERING_RADIUS = 400

N_CARS = 5

OFF_TRACK_MULTIPLIER = 0.2

GAME_RENDER_FPS = 120
SIMULATION_FPS = 60

SPRITE_SCALE_MULTIPLIER = 0.02
MENU_FPS = 20

CROSSFADE_DURATION_MS = 200
CROSSFADE_STEPS = 30

MIN_SCENE_DURATION = 1.0

GAME_START_WAIT_TIME = 2.0

# Sounds

STARTING_MENU_MUSIC_FILE = "race_start.wav"
SECONDARY_MENU_MUSIC_FILE = "main_menu.wav"
MENU_MUSIC_DIR = "./assets/menu_music"
ASSETS_DIR = "./assets"
MENU_CLICK_SOUND_FILE = "menu_click.wav"
GAME_START_SOUND_FILE = "game_start.wav"
GAME_BACKGROUND_MUSIC_FILE = "snowboarding.wav"
RACE_BEEP_1_SOUND_FILE = "race_beep_1.wav"
RACE_BEEP_2_SOUND_FILE = "race_beep_2.wav"

MENU_MUSIC_CROSSFADE = 3
