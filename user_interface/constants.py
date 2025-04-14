def pixels_to_meters(pixels):
    # Irrespective of resolution, we assume the track is a fixed length across
    return pixels * WINDOW_WIDTH_IN_M / RESOLUTION[0]


RESOLUTION = (1920, 1080)
WINDOW_WIDTH_IN_M = 400  # In meters
WINDOW_HEIGHT_IN_M = pixels_to_meters(RESOLUTION[1])
SCREEN_CENTER_METERS = (WINDOW_WIDTH_IN_M / 2, WINDOW_HEIGHT_IN_M / 2)

# Constants
CONST_SPEED = 50  # In m/s
CONST_STEER = -0.05  # In radians

WHEELBASE = 2.5
CAR_WIDTH = 4
CAR_LENGTH = 12

MAX_STEERING_RADIUS = 400

N_CARS = 5

OFF_TRACK_MULTIPLIER = 0.2

TARGET_FPS = 120
MENU_FPS = 20
