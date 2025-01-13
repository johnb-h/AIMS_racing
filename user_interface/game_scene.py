import pygame
import matplotlib
from typing import Optional, List, Tuple

from scene import Scene
from states import State
from race_track import RaceTrack
from car import Car
from constants import (
    WINDOW_WIDTH_IN_M,
    WINDOW_HEIGHT_IN_M,
    CONST_STEER,
    N_CARS
)

# Game Scene
class GameScene(Scene):
    def __init__(self, car_colours: Optional[list] = None):
        super().__init__()
        self._track = RaceTrack(
            width_extent=0.9 * WINDOW_WIDTH_IN_M,
            length_extent=0.9 * WINDOW_HEIGHT_IN_M,
            track_width=50,
        )
        self._cars = None

        if car_colours is None:
            car_colours = matplotlib.cm.get_cmap("tab10")(range(N_CARS))
            car_colours = [(int(r * 255), int(g * 255), int(b * 255)) for r, g, b, _ in car_colours]
        
        self._init_cars(car_colours=car_colours) 

    def handle_events(self, events):
        for ev in events:
            if ev.type == pygame.MOUSEBUTTONDOWN:
                self._next_state = State.NAME_ENTRY

    def update(self, dt):
        for car in self._cars:
            car.is_on_track = self._track.is_on_track(car.pixel_position)
            car.update(dt)
        return self._next_state

    def draw(self, screen):
        self._track.draw(screen)
        for car in self._cars:
            car.draw(screen)

    def reset(self):
        self._next_state = None

    def _init_cars(self, car_colours: List[Tuple[int, int, int]]):
        self._cars = self._track.spawn_cars_on_starting_line(
            car_colours, [lambda _: CONST_STEER for _ in range(N_CARS)]
        )
