import math
import time
from dataclasses import dataclass
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, Future
from typing import List, Optional, Tuple

import matplotlib
import numpy as np
import pygame
import pygame.freetype
pygame.freetype.init()
from hardware_interface.mqtt_communication import MQTTClient
from hardware_interface import RaceCar, LedCtrl, LedMode
from user_interface.constants import RESOLUTION
from user_interface.sound_player import SoundPlayer

from evolution.evolution import CarEvolution
from user_interface.constants import (
    GAME_RENDER_FPS,
    SIMULATION_FPS,
    WINDOW_HEIGHT_IN_M,
    WINDOW_WIDTH_IN_M,
    SPRITE_SCALE_MULTIPLIER,
    GAME_START_WAIT_TIME,
)
from user_interface.scene import Scene
from user_interface.states import State

BACKGROUND_COLOR = (0, 102, 16)
TRACK_COLOR = (50, 50, 50)


@dataclass
class Car:
    positions: List[Tuple[float, float]]
    selected: bool = False
    color: Tuple[int, int, int] = (255, 255, 255)
    last_position: Tuple[float, float] = (0, 0)


class GameScene(Scene):
    def __init__(
        self,
        shared_data: dict,
        mqtt_client: MQTTClient,
        sound_player: SoundPlayer,
        num_visualised_cars: int = 10,
        car_colours: Optional[List[Tuple[int, int, int]]] = None,
        show_instructions: bool = False,
        executor: concurrent.futures.ThreadPoolExecutor | None = None
    ):
        super().__init__(shared_data, mqtt_client, sound_player)
        self._executor = executor
        self._load_future: Optional[Future] = None
        self._sim_time_acc = 0.0
        self._sim_step_acc = 0.0
        self.num_visualised_cars = num_visualised_cars
        self.car_colours = car_colours
        self.show_instructions = show_instructions
        if self.car_colours is None:
            self.car_colours = matplotlib.cm.get_cmap("tab10")(
                range(self.num_visualised_cars)
            )
            self.car_colours = [
                (255, 255, 255),  # White
                (0, 0, 255),      # Blue
                (255, 255, 0),    # Yellow
                (0, 255, 0),      # Green
                (255, 0, 0)       # Red
            ]
            self.car_colours += self.car_colours
        self._crashed_sound_played = set()

        self.crashed_time = None
        self.waiting_for_start = True
        self.start_wait_time = pygame.time.get_ticks()
        self.countdown = False
        self.countdown_time = pygame.time.get_ticks()
        self.countdown_text = ""
        self.font    = pygame.font.Font("./assets/joystix_monospace.ttf", 36)
        self.font_ft  = pygame.freetype.Font("./assets/joystix_monospace.ttf", 36)
        self.font_go_ft = pygame.freetype.Font("./assets/joystix_monospace.ttf", 100)

        # Load sprites & backgrounds
        # Load the F1 car sprite
        self.f1_car_sprite = pygame.image.load("assets/cleaned_f1.tiff").convert_alpha()
        self.f1_car_bar_sprite = pygame.image.load(
            "assets/cleaned_f1_bar.tiff"
        ).convert_alpha()
        self.index_to_sprite = {
            i: (
                self.f1_car_sprite
                if i >= len(self.car_colours) // 2
                else self.f1_car_bar_sprite
            )
            for i, _ in enumerate(self.car_colours)
        }

        self.track_background = pygame.image.load(
            "assets/Background5_1920_1080.png"
        ).convert_alpha()
        self.background = pygame.image.load(
            "assets/Background3_1920_1080.png"
        ).convert_alpha()

        # Build everything
        self._init_display()
        self._init_track()
        self._init_evolution()
        self._init_state()
        self._init_visualization_cache()

        self._prev_selected: list[bool] = []

        self.auto_transition_delay = 2000
        self._finish_transition_start = None

    def _init_display(self):
        """Initialize display settings based on the *virtual* 1920×1080 surface."""
        self.width, self.height = RESOLUTION
        # then compute track_scale off of those:
        self.track_scale = min(
            self.width / (WINDOW_WIDTH_IN_M * 1.1),
            self.height / (WINDOW_HEIGHT_IN_M * 1.1),
        )

    def _init_track(self):
        """Initialize track with dimensions relative to window size."""
        # Track setup with relative dimensions
        self.track_center = (self.width // 2, self.height // 2)

        # Scale track dimensions based on window size while maintaining aspect ratio
        track_height = (
            WINDOW_HEIGHT_IN_M * self.track_scale
        )
        track_outer_width = (
            WINDOW_WIDTH_IN_M * self.track_scale
        )
        track_inner_width = track_outer_width * 0.5  # Inner track is 50% of outer width
        track_inner_height = track_height * 0.5  # Inner track is 50% of outer height

        self.track_outer = self._generate_oval(
            self.track_center[0], self.track_center[1], track_outer_width, track_height
        )
        self.track_inner = self._generate_oval(
            self.track_center[0],
            self.track_center[1],
            track_inner_width,
            track_inner_height,
        )

        # Finish line setup - align vertically between inner and outer track
        outer_top = self.track_center[1] - track_height / 2
        inner_top = self.track_center[1] - track_inner_height / 2
        finish_line_center = (outer_top + inner_top) / 2
        track_gap = abs(outer_top - inner_top)
        self.finish_line = [
            (self.track_center[0], finish_line_center - track_gap / 2),
            (self.track_center[0], finish_line_center + track_gap / 2),
        ]

        # Store track dimensions for evolution
        self.track_outer_width = track_outer_width
        self.track_outer_height = track_height
        self.track_inner_width = track_inner_width
        self.track_inner_height = track_inner_height

    def _init_evolution(self):
        """Initialize evolution with track dimensions."""
        outer_top = self.track_center[1] - self.track_outer_height / 2
        inner_top = self.track_center[1] - self.track_inner_height / 2
        finish_line_center = (outer_top + inner_top) / 2

        self.evolution = CarEvolution(
            track_center=self.track_center,
            start_position=(self.track_center[0], finish_line_center),
            track_outer_width=self.track_outer_width,
            track_outer_height=self.track_outer_height,
            track_inner_width=self.track_inner_width,
            track_inner_height=self.track_inner_height,
            track_outer=self.track_outer,
            track_inner=self.track_inner,
            num_visualize=self.num_visualised_cars,
        )

    def resize(self, new_width: int, new_height: int):
        """Handle window resize events."""
        self.width = new_width
        self.height = new_height
        self._init_track()
        self._init_evolution()
        self._init_visualization_cache()
        self.generate_new_population()

    def _generate_oval(
        self, cx: float, cy: float, width: float, height: float
    ) -> List[Tuple[float, float]]:
        """Generate points for an oval track."""
        return [
            (cx + width / 2 * np.cos(angle), cy + height / 2 * np.sin(angle))
            for angle in np.linspace(0, 2 * np.pi, 100)
        ]

    def _angle_between(self, p1, p2, p3):
        """Angle between vectors (in radians) formed by p1→p2 and p2→p3."""
        v1 = (p1[0] - p2[0], p1[1] - p2[1])
        v2 = (p3[0] - p2[0], p3[1] - p2[1])
        dot = v1[0]*v2[0] + v1[1]*v2[1]
        mag1 = math.hypot(*v1)
        mag2 = math.hypot(*v2)
        if mag1 == 0 or mag2 == 0:
            return 0
        cos_angle = max(-1, min(1, dot / (mag1 * mag2)))
        return math.acos(cos_angle)

    def _draw_track_outline_checkered_line(self, surface, points, base_length=12, width=10, min_length=10, max_length=20, curve_threshold=0.9775):
        red = (255, 0, 0)
        white = (255, 255, 255)

        n = len(points)

        for i in range(n):
            start = points[i]
            end = points[(i + 1) % n]
            next_pt = points[(i + 2) % n]

            # Calculate curvature
            curve_angle = self._angle_between(start, end, next_pt)
            curvature = curve_angle / math.pi  # Normalize to 0–1

            dx = end[0] - start[0]
            dy = end[1] - start[1]
            distance = math.hypot(dx, dy)
            angle = math.atan2(dy, dx)

            if curvature < curve_threshold:
                # Curved section → red-white curb
                segment_length = max(min_length, min(max_length, base_length * (1.0 - curvature + 0.1)))
                segments = max(1, int(distance // segment_length))

                for s in range(segments):
                    t = s / segments
                    x = start[0] + dx * t
                    y = start[1] + dy * t

                    color = red if s % 2 == 0 else white

                    rect = pygame.Surface((segment_length, width), pygame.SRCALPHA)
                    rect.fill(color)

                    rotated = pygame.transform.rotate(rect, -math.degrees(angle))
                    offset_x = rotated.get_width() / 2
                    offset_y = rotated.get_height() / 2
                    surface.blit(rotated, (x - offset_x, y - offset_y))
            else:
                # Straight section → solid white line
                pygame.draw.line(surface, white, start, end, width)

    def _create_track_surface(self) -> pygame.Surface:
        """Create the static track surface with boundaries and finish line."""
        # Create a surface for the track using custom image
        # Load and prepare texture
        track_background = self.track_background
        # Final surface to return
        surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        # 1. Create a mask surface with per-pixel alpha
        track_mask = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        # 2. Fill the outer polygon (opaque)
        pygame.draw.polygon(track_mask, (255, 255, 255, 255), self.track_outer)
        # 3. Cut out the inner polygon (transparent)
        pygame.draw.polygon(track_mask, (0, 0, 0, 0), self.track_inner)
        # 4. Create texture surface with tiling
        texture_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        tex_w, tex_h = track_background.get_size()
        for x in range(0, self.width, tex_w):
            for y in range(0, self.height, tex_h):
                texture_surface.blit(track_background, (x, y))
        # 5. Apply mask
        texture_surface.blit(track_mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
        # 6. Blit the masked texture onto the main surface
        surface.blit(texture_surface, (0, 0))

        # Draw track boundaries
        self._draw_track_outline_checkered_line(surface, self.track_outer)
        self._draw_track_outline_checkered_line(surface, self.track_inner)

        # Draw finish line
        x_start = self.finish_line[0][0]
        y_start = self.finish_line[0][1]
        x_end = self.finish_line[1][0]
        y_end = self.finish_line[1][1]

        total_height = y_end - y_start
        stripe_width = 8
        stripe_height = 8
        total_width = stripe_width * 3  # Number of stripes in the finish line

        for row in range(int(total_height // stripe_height)):
            for col in range(3):  # Number of stripes
                color = (255, 255, 255) if (row + col) % 2 == 0 else (0, 0, 0)
                rect = pygame.Rect(
                    x_start + col * stripe_width,
                    y_start + row * stripe_height,
                    stripe_width,
                    stripe_height,
                )
                pygame.draw.rect(surface, color, rect)

        return surface

    def _check_finish_line_crossing(
        self, prev_pos: Tuple[float, float], current_pos: Tuple[float, float]
    ) -> bool:
        """Check if a line segment crosses the finish line."""

        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

        if self.current_step > 50 and prev_pos[0] < current_pos[0]:
            return ccw(prev_pos, self.finish_line[0], self.finish_line[1]) != ccw(
                current_pos, self.finish_line[0], self.finish_line[1]
            ) and ccw(prev_pos, current_pos, self.finish_line[0]) != ccw(
                prev_pos, current_pos, self.finish_line[1]
            )
        return False

    def generate_new_population(self):
        """Generate and initialize a new population of cars."""

        # ─── reset the countdown/start flags for this generation ───
        self.start_wait_time   = pygame.time.get_ticks()
        self.countdown         = True
        self.countdown_time    = self.start_wait_time
        self.countdown_text    = ""

        # (re-)play background music if you like
        self._sound_player.play_game_background_music()

        # now rebuild the cars as before
        self.cars = [
            Car(positions=traj, color=color)
            for traj, color in zip(self.evolution.ask(), self.car_colours)
        ]
        self.current_step    = 0
        self.crashed_time    = None
        self.cars_driving    = True
        self.finish_line_crossed = False
        self.finish_time     = None
        self._finish_transition_start = None
        self._crashed_sound_played.clear()
        self._prev_selected  = [car.selected for car in self.cars]


    def _update_mean_trajectory(self):
        """Update the cached mean trajectory visualization."""
        if self.generation == self.last_generation:
            return

        mean_traj = self.evolution.get_mean_trajectory()
        if mean_traj and len(mean_traj) > 1:
            self.mean_trajectory_surface = pygame.Surface(
                (self.width, self.height), pygame.SRCALPHA
            )
            points = mean_traj[::1]
            if len(points) >= 2:
                pygame.draw.lines(
                    self.mean_trajectory_surface,
                    (255, 215, 0, 180),
                    False,
                    points,
                    2,
                )
        self.last_generation = self.generation

    def _check_for_finish(self, up_to_step: int) -> bool:
        """Check if any car has crossed the finish line up to the given step."""
        if not self.finish_line_crossed:
            for car in self.cars:
                if len(car.positions) > 1:
                    positions = car.positions[: up_to_step + 1]
                    if len(positions) >= 2:
                        if self._check_finish_line_crossing(
                            positions[-2], positions[-1]
                        ):
                            self.finish_line_crossed = True
                            self.finish_time = up_to_step
                            self.cars_driving = False
                            return True
        return False

    def _draw_cars(self, screen):
        """Draw all car trajectories and current positions."""
        trajectory_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)

        # First check for finish line crossing
        self._check_for_finish(self.current_step)

        all_cars_crashed = True

        # Then draw cars and handle crashes
        for i, car in enumerate(self.cars):

            car_sprite = self.index_to_sprite[i]

            if len(car.positions) <= 1:
                continue

            # Get car status
            crashed = (
                self.evolution.displayed_crashed[i]
                if hasattr(self.evolution, "displayed_crashed")
                else False
            )
            crash_step = (
                self.evolution.displayed_crash_steps[i]
                if hasattr(self.evolution, "displayed_crash_steps")
                else len(car.positions)
            )
            all_cars_crashed = all_cars_crashed and crash_step < self.current_step

            # Determine how much of trajectory to draw
            max_step = self.current_step + 1
            max_step = min(max_step, crash_step)
            if self.finish_line_crossed:
                max_step = min(max_step, self.finish_time)

            car.last_position = car.positions[max_step - 1]
            positions = car.positions[:max_step]

            # Draw trajectory
            if len(positions) >= 2:
                # Create a translucent color by adding alpha to the car's color
                translucent_color = (*car.color, 64)  # 64 is 25% opacity
                pygame.draw.lines(trajectory_surface, translucent_color, False, positions, 2)

            # Draw current position marker
            if positions:
                current_pos = positions[-1]
                if (
                    0 <= current_pos[0] <= self.width
                    and 0 <= current_pos[1] <= self.height
                ):
                    if len(positions) > 1:
                        prev_pos = positions[-2]
                        dx, dy = (
                            current_pos[0] - prev_pos[0],
                            current_pos[1] - prev_pos[1],
                        )
                        angle = np.arctan2(dy, dx)

                        scaled_car = pygame.transform.scale_by(
                            car_sprite,
                            self.track_scale * SPRITE_SCALE_MULTIPLIER,
                        )

                        # Create a colored version of the car
                        colored_car = scaled_car.copy()
                        colored_car.fill(car.color, special_flags=pygame.BLEND_RGBA_MULT)

                        # Calculate the rear axle position (pivot point for rotation)
                        rear_axle_offset = pygame.Vector2(
                            scaled_car.get_width() / 2, 0  # Use the actual scaled width
                        )
                        rear_axle_offset = rear_axle_offset.rotate_rad(angle)
                        rear_axle_position = current_pos - rear_axle_offset

                        # Rotate the car surface based on the direction angle
                        # Add 90 degrees to correct the initial orientation
                        rotated_car = pygame.transform.rotate(
                            colored_car,
                            -math.degrees(angle) + 90
                        )

                        # Get the rectangle of the rotated car for correct positioning
                        rotated_car_rect = rotated_car.get_rect()
                        rotated_car_rect.center = (
                            int(rear_axle_position.x),
                            int(rear_axle_position.y),
                        )

                        # Draw the rotated car onto the screen
                        screen.blit(rotated_car, rotated_car_rect.topleft)

                    # Draw crash marker
                    if crashed and max_step >= crash_step:
                        pass

            # Draw selection highlight
            if car.selected:
                pygame.draw.rect(trajectory_surface, (0, 255, 0), rotated_car_rect, width=2)

            if crashed and max_step == crash_step and i not in self._crashed_sound_played:
                self._sound_player.play_car_crash_1()
                self._crashed_sound_played.add(i)

        if all_cars_crashed:
            self._sound_player.stop_car_sounds()
            self.cars_driving = False
            if self.crashed_time is None:
                self.crashed_time = self.current_step

        screen.blit(trajectory_surface, (0, 0))

    def _draw_ui(self, screen):
        """Draw UI elements including timer and instructions."""
        # Draw generation counter
        gen_text = self.font.render(f"Round: {self.generation}", True, (255, 255, 255))
        screen.blit(gen_text, (20, 20))

        # Draw timer
        if self.finish_line_crossed and self.finish_time is not None:
            elapsed_time = self.finish_time
        elif self.crashed_time is not None:
            elapsed_time = self.crashed_time
        else:
            elapsed_time = self.current_step

        elapsed_time = elapsed_time / GAME_RENDER_FPS
        timer_text = self.font.render(f"Time: {elapsed_time:.2f}s", True, (255, 255, 255))
        timer_rect = timer_text.get_rect(topright=(self.width - 20, 20))
        screen.blit(timer_text, timer_rect)

        # Draw instructions
        instructions = (
            "Racing..."
            # Press SPACE to end simulation"
            if self.cars_driving
            else "Which cars performed best?"
            # Press SPACE for next generation"
        )

        # Draw Finishd
        finished_text = "Finished!"

        # # Draw restart instruction
        # restart_text = self.font.render("(r) restart", True, (255, 255, 255))
        # screen.blit(restart_text, (20, self.height - 120))

        # Draw exit instruction
        exit_text = self.font.render("(esc) exit", True, (255, 255, 255))
        screen.blit(exit_text, (20, self.height - 140))


        # Countdown from 5 to GO
        if self.finish_line_crossed:
            txt = finished_text
            _, rect = self.font_ft.render(txt, (255,255,255), size=36)
            center = (self.width//2 - rect.width//2, self.height//2 - rect.height//2)
            self._draw_text_with_outline(screen, self.font_ft, txt, center, size=36)
        elif self.countdown:
            if self.countdown:
                txt = self.countdown_text
                _, rect = self.font_go_ft.render(txt, (255,255,255), size=72)
                center = (self.width//2 - rect.width//2, self.height//2 - rect.height//2)
                self._draw_text_with_outline(screen, self.font_go_ft, txt, center, size=72)
            pygame.display.flip()
            if pygame.time.get_ticks() - self.countdown_time > 1000:
                self.countdown_time = pygame.time.get_ticks()
                if self.countdown_text == "":
                    self._mqtt_client.publish_message(topic=LedCtrl.topic,
                                                     message=LedCtrl(mode=LedMode.RACE_START).serialise())
                    self._sound_player.play_race_beep_1()
                    self.countdown_text = "5"
                elif self.countdown_text == "5":
                    self._sound_player.play_car_sounds()
                    self._sound_player.play_race_beep_1()
                    self.countdown_text = "4"
                elif self.countdown_text == "4":
                    self._sound_player.play_race_beep_1()
                    self.countdown_text = "3"
                elif self.countdown_text == "3":
                    self._sound_player.play_race_beep_1()
                    self.countdown_text = "2"
                elif self.countdown_text == "2":
                    self._sound_player.play_race_beep_1()
                    self.countdown_text = "1"
                elif self.countdown_text == "1":
                    self._sound_player.play_race_beep_2()
                    self.countdown_text = "GO!"
                elif self.countdown_text == "GO!":
                    self.countdown = False
        else:
            if not self.cars_driving:
                # Draw next instruction
                next_text = self.font.render("(space) next round", True, (255, 255, 255))
                screen.blit(next_text, (20, self.height - 80))
                # draw the “Press SPACE…” line with the same tiny outline
            
            txt = instructions
            _, rect = self.font_ft.render(txt, (255,255,255), size=36)
            center = (self.width//2 - rect.width//2, self.height//2 - rect.height//2)
            self._draw_text_with_outline(screen, self.font_ft, txt, center, size=36)

    def handle_events(self, events):
        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()
                for car in self.cars:
                    if not car.positions:
                        continue
                    pos = car.last_position
                    if (
                        np.sqrt(
                            (pos[0] - mouse_pos[0]) ** 2 + (pos[1] - mouse_pos[1]) ** 2
                        )
                        < 15
                    ):
                        car.selected = not car.selected

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self._next_state = State.MAIN_MENU
                elif event.key == pygame.K_SPACE:
                    self._handle_space_key()
                elif event.key == pygame.K_m:
                    self.show_mean = not self.show_mean
                elif pygame.K_0 <= event.key <= pygame.K_9:
                    # TODO: Add kill switch or param for max wait
                    car_index = event.key - pygame.K_0
                    if car_index < len(self.cars):
                        self.cars[car_index].selected = not self.cars[
                            car_index
                        ].selected

    def _handle_space_key(self):
        if self.countdown:
            return
        if self.cars_driving:
            max_steps = max(len(car.positions) for car in self.cars)
            if hasattr(self.evolution, "displayed_crash_steps"):
                max_steps = max(self.evolution.displayed_crash_steps)

            while self.current_step < max_steps - 1:
                self.current_step += 1
                if self._check_for_finish(self.current_step):
                    break

            if not self.finish_line_crossed:
                self.cars_driving = False
        else:
            self._mqtt_client.publish_message(message=LedCtrl(mode=LedMode.ALL_OFF).serialise(), topic=LedCtrl.topic)
            selected = [i for i, car in enumerate(self.cars) if car.selected]
            if selected:
                self.evolution.tell(selected)
                self.generation += 1
                self.generate_new_population()

    def handle_mqtt(self) -> None:
        """Handle incoming MQTT messages."""
        if not self._mqtt_client.queue_empty():
            topic, msg = self._mqtt_client.pop_queue()
            if RaceCar.topic in topic:
                race_car = RaceCar()
                race_car.deserialise(msg)
                car_index = race_car.id
                if car_index < len(self.cars):
                    self.cars[car_index].selected = not self.cars[
                        car_index
                    ].selected

    def update(self, dt):
        # 1) Pre‐countdown wait
        if self.waiting_for_start:
            now = pygame.time.get_ticks()
            if now - self.start_wait_time >= GAME_START_WAIT_TIME * 1000:
                self.waiting_for_start = False
                self.countdown = True
                self.countdown_time = now
                self.countdown_text = ""
            else:
                # even during the initial pause we still want to catch selects
                self._check_selection_sounds()
                return self._next_state

        # 2) If we're in the countdown, skip simulation stepping
        if self.countdown:
            # allow selection sounds during countdown too
            self._check_selection_sounds()
            return self._next_state

        # 3) Normal simulation stepping
        self._sim_step_acc += dt * SIMULATION_FPS
        steps = int(self._sim_step_acc)
        self._sim_step_acc -= steps
        if steps:
            self.current_step += steps
            self._check_for_finish(self.current_step)

        self._check_selection_sounds()

        if self.finish_line_crossed and not self.cars_driving:
            now = pygame.time.get_ticks()
            if self._finish_transition_start is None:
                self._finish_transition_start = now
            elif now - self._finish_transition_start >= self.auto_transition_delay:
                # time’s up: push score into shared_data and go to NameEntryScene
                score = 0 if self.finish_time is None else self.finish_time
                score /= GAME_RENDER_FPS
                self.shared_data["score"] = score
                self.shared_data["rounds"] = self.generation
                # clean up any music/LEDs
                self._sound_player.stop_game_background_music()
                self._mqtt_client.publish_message(
                    message=LedCtrl(mode=LedMode.INIT).serialise(),
                    topic=LedCtrl.topic
                )
                self._next_state = State.NAME_ENTRY

        if self._next_state == State.MAIN_MENU:
            self._sound_player.stop_game_background_music()
            self._sound_player.stop_car_sounds()

        # 5) Return next state as before
        return self._next_state

    def draw(self, screen):
        # draw background & track
        screen.blit(self.background, (0, 0))
        screen.blit(self.track_surface, (0, 0))

        # optional mean trajectory
        if self.show_mean:
            self._update_mean_trajectory()
            if self.mean_trajectory_surface:
                screen.blit(self.mean_trajectory_surface, (0, 0))

        # draw cars only after wait & countdown are done
        if not self.countdown and not self.waiting_for_start:
            self._draw_cars(screen)

        # draw UI (countdown & instructions) only after the initial wait
        if not self.waiting_for_start:
            self._draw_ui(screen)

    def reset(self):
        self._next_state = None
        self._init_state()
        self._init_evolution()
        self._init_visualization_cache()
        self.generation = 1
        self._sound_player.play_game_background_music()
        if self._executor is not None:
            self._load_future = self._executor.submit(self.generate_new_population)

    def _init_state(self):
        """Initialize state variables."""
        self._next_state = None
        self.cars: List[Car] = []
        self.current_step = 0
        self.cars_driving = True
        self.generation = 1
        self.show_mean = False
        self.finish_line_crossed = False
        self.finish_time = None

    def _init_visualization_cache(self):
        """Initialize visualization cache."""
        self.mean_trajectory_surface = None
        self.last_generation = -1
        self.track_surface = self._create_track_surface()

    def _draw_text_with_outline(
        self,
        surf: pygame.Surface,
        font: pygame.freetype.Font,
        text: str,
        pos: tuple[int,int],
        fg: tuple[int,int,int]=(255,255,255),
        outline_color: tuple[int,int,int]=(0,0,0),
        outline_width: int=2,
        size: int=0,
    ) -> None:
        # render both fg and outline versions
        text_surf, _    = font.render(text, fg, size=size)          # :contentReference[oaicite:0]{index=0}
        outline_surf, _ = font.render(text, outline_color, size=size)
        w, h = text_surf.get_size()
        # build a tiny Surface that can hold the outline + text
        o_surf = pygame.Surface((w + 2*outline_width, h + 2*outline_width), pygame.SRCALPHA)
        # blit outline eight times around
        for dx in (-outline_width, 0, outline_width):
            for dy in (-outline_width, 0, outline_width):
                if dx==0 and dy==0: continue
                o_surf.blit(outline_surf, (dx + outline_width, dy + outline_width))
        # blit the white text on top
        o_surf.blit(text_surf, (outline_width, outline_width))
        # finally, blit to your target surface, offset so it's centered at pos
        surf.blit(o_surf, (pos[0] - outline_width, pos[1] - outline_width))

    def _check_selection_sounds(self):
        """
        Play a button‐note whenever a car flips from unselected → selected.
        This runs once per frame, after any events (keyboard, mouse or MQTT).
        """
        for i, car in enumerate(self.cars):
            if car.selected and not self._prev_selected[i]:
                # fire the note for car index i
                self._sound_player.play_button_sound(i)
            # update history
            self._prev_selected[i] = car.selected
