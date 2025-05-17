import pygame
import os
import json
from typing import Optional
from user_interface.scene import Scene
from user_interface.states import State
from hardware_interface.mqtt_communication import MQTTClient
from user_interface.sound_player import SoundPlayer

HIGH_SCORES_FILE = "high_scores.json"


def load_high_scores():
    if os.path.exists(HIGH_SCORES_FILE):
        with open(HIGH_SCORES_FILE, "r") as f:
            try:
                scores = json.load(f)
            except json.JSONDecodeError:
                scores = []
    else:
        scores = []
    return scores


class HighScoresScene(Scene):
    def __init__(
        self, shared_data: dict,
        mqtt_client: MQTTClient,
        sound_player: SoundPlayer,
    ) -> None:
        super().__init__(shared_data, mqtt_client, sound_player)
        # Load fonts.
        self.title_font = pygame.font.Font("./assets/joystix_monospace.ttf", 144)
        self.header_font = pygame.font.Font("./assets/joystix_monospace.ttf", 72)
        self.line_font = pygame.font.Font("./assets/joystix_monospace.ttf", 40)
        
        # Load the shared background.
        self.background = pygame.image.load("assets/Background1_1920_1080.png").convert()
        
        # Pre-render the title with shadow.
        self.title_surface = self.title_font.render("HIGH SCORES", True, (0, 0, 0))
        self.title_shadow_surface = self.title_font.render("HIGH SCORES", True, (65, 26, 64))
        
        # Load and scale the trophy image.
        self.trophy = pygame.image.load("assets/Prize.png").convert_alpha()
        self.trophy = pygame.transform.scale(self.trophy, (200, 200))
        
        self.high_scores = load_high_scores()
        # Prepare surfaces for header and rows
        self.header_surfaces = []  # type: list[pygame.Surface]
        self.row_surfaces = []     # type: list[tuple[pygame.Surface, pygame.Surface, pygame.Surface, pygame.Surface]]
        self.col_x_positions = []  # type: list[int]
        
        self.update_scores_surfaces()
        self._next_state = None
        self._mqtt_client = mqtt_client

    def update_scores_surfaces(self):
        """
        Load high scores, prepare header and row surfaces, and compute column positions.
        Takes into account the recent_entry if it's outside the top 10 when determining column widths.
        """
        # Reload and sort scores
        self.high_scores = load_high_scores()
        sorted_scores = sorted(self.high_scores, key=lambda x: (x["rounds"], x["score"]))
        top10 = sorted_scores[:10]

        # Prepare header surfaces
        rank_header = self.header_font.render("Rank", True, (0, 0, 0))
        name_header = self.header_font.render("Name", True, (0, 0, 0))
        rounds_header = self.header_font.render("Rounds", True, (0, 0, 0))
        time_header = self.header_font.render("Time", True, (0, 0, 0))

        # Prepare top-10 row surfaces
        rank_surfs = [self.line_font.render(f"{i+1}.", True, (0, 0, 0)) for i in range(len(top10))]
        name_surfs = [self.line_font.render(entry['name'], True, (0, 0, 0)) for entry in top10]
        rounds_surfs = [self.line_font.render(str(entry['rounds']), True, (0, 0, 0)) for entry in top10]
        time_surfs = [self.line_font.render(f"{entry['score']:.2f}s", True, (0, 0, 0)) for entry in top10]

        # Check if there's a recent entry beyond the top 10
        recent = self.shared_data.get("recent_entry")
        extra_rank_surf = extra_name_surf = extra_rounds_surf = extra_time_surf = None
        if recent and recent not in top10:
            # Create surfaces for the extra entry
            position = self._find_rank(recent)  # you may implement a helper to find absolute rank
            extra_rank_surf = self.line_font.render(f"{position}.", True, (0, 0, 0))
            extra_name_surf = self.line_font.render(recent['name'], True, (0, 0, 0))
            extra_rounds_surf = self.line_font.render(str(recent['rounds']), True, (0, 0, 0))
            extra_time_surf = self.line_font.render(f"{recent['score']:.2f}s", True, (0, 0, 0))

        # Collect widths for each column
        widths_rank = [rank_header.get_width()] + [surf.get_width() for surf in rank_surfs]
        widths_name = [name_header.get_width()] + [surf.get_width() for surf in name_surfs]
        widths_rounds = [rounds_header.get_width()] + [surf.get_width() for surf in rounds_surfs]
        widths_time = [time_header.get_width()] + [surf.get_width() for surf in time_surfs]
        if extra_rank_surf:
            widths_rank.append(extra_rank_surf.get_width())
            widths_name.append(extra_name_surf.get_width())
            widths_rounds.append(extra_rounds_surf.get_width())
            widths_time.append(extra_time_surf.get_width())

        # Determine max widths
        max_rank_w = max(widths_rank)
        max_name_w = max(widths_name)
        max_rounds_w = max(widths_rounds)
        max_time_w = max(widths_time)
        spacing = 50
        total_w = max_rank_w + spacing + max_name_w + spacing + max_rounds_w + spacing + max_time_w

        # Compute column x-positions
        start_x = (1920 - total_w) // 2
        x_rank = start_x
        x_name = x_rank + max_rank_w + spacing
        x_rounds = x_name + max_name_w + spacing
        x_time = x_rounds + max_rounds_w + spacing

        # Store header positions
        self.header_surfaces = [
            (rank_header, x_rank),
            (name_header, x_name),
            (rounds_header, x_rounds),
            (time_header, x_time),
        ]

        # Store row surfaces for top-10
        self.row_surfaces = [
            (rank_s, x_rank, name_s, x_name, rounds_s, x_rounds, time_s, x_time)
            for rank_s, name_s, rounds_s, time_s in zip(rank_surfs, name_surfs, rounds_surfs, time_surfs)
        ]

        # If extra entry exists, store it separately for drawing in `draw`
        self.extra_entry_surfaces = None
        if extra_rank_surf:
            self.extra_entry_surfaces = (
                extra_rank_surf, x_rank,
                extra_name_surf, x_name,
                extra_rounds_surf, x_rounds,
                extra_time_surf, x_time,
            )

    def _find_rank(self, entry):
        """Return 1-based rank of the given entry in the full high score list."""
        sorted_all = sorted(self.high_scores, key=lambda x: (x['rounds'], x['score']))
        for idx, e in enumerate(sorted_all, start=1):
            if e['name'] == entry['name'] and e['rounds'] == entry['rounds'] and abs(e['score'] - entry['score']) < 1e-6:
                return idx
        return len(sorted_all)

    def handle_events(self, events) -> None:
        for ev in events:
            # Advance on any mouse click OR any key press
            if ev.type == pygame.MOUSEBUTTONDOWN or ev.type == pygame.KEYDOWN:
                self.set_next_state()

    def handle_mqtt(self) -> None:
        """Handle incoming MQTT messages."""
        if not self._mqtt_client.queue_empty():
            topic, msg = self._mqtt_client.pop_queue()
            self.set_next_state()

    def set_next_state(self) -> Optional[State]:
        self._sound_player.play_menu_click()
        self._next_state = State.MAIN_MENU

    def update(self, dt: float) -> Optional[State]:
        return self._next_state

    def draw(self, screen) -> None:
        # Draw background
        screen.blit(self.background, (0, 0))

        # Draw the trophy in the top right.
        trophy_x = 1920 - self.trophy.get_width() - 50
        trophy_y = 50
        screen.blit(self.trophy, (trophy_x, trophy_y))

        # Draw the title with shadow.
        title_shadow_x = (1920 - self.title_shadow_surface.get_width()) // 2 + 10
        title_y = 50
        screen.blit(self.title_shadow_surface, (title_shadow_x, title_y))
        title_x = (1920 - self.title_surface.get_width()) // 2
        screen.blit(self.title_surface, (title_x, title_y))

        # Calculate starting y for the grid (below title)
        start_y = title_y + self.title_surface.get_height()
        row_height = self.line_font.get_height() + 10

        # Draw headers
        for header_surf, x in self.header_surfaces:
            screen.blit(header_surf, (x, start_y))

        # Prepare column x-positions for later use
        x_rank   = self.header_surfaces[0][1]
        x_name   = self.header_surfaces[1][1]
        x_rounds = self.header_surfaces[2][1]
        x_time   = self.header_surfaces[3][1]

        # Where the first data row starts
        row_start_y = start_y + 30

        # 1) Draw only the Top-10 rows
        for idx, (rank_s, x_ra, name_s, x_n, rounds_s, x_ro, time_s, x_t) in enumerate(self.row_surfaces, start=1):
            if idx > 10:
                break
            y = row_start_y + idx * row_height
            screen.blit(rank_s,   (x_ra, y))
            screen.blit(name_s,   (x_n,  y))
            screen.blit(rounds_s, (x_ro, y))
            screen.blit(time_s,   (x_t,  y))

        # 2) If there's a recent_entry, find its overall rank
        recent = self.shared_data.get("recent_entry")
        if recent:
            all_scores = load_high_scores()
            sorted_scores = sorted(all_scores, key=lambda e: (e["rounds"], e["score"]))
            # locate the matching entry
            user_rank = next(
                (i + 1 for i, e in enumerate(sorted_scores)
                 if e["name"] == recent["name"]
                 and e["rounds"] == recent["rounds"]
                 and abs(e["score"] - recent["score"]) < 1e-6),
                None
            )

            if user_rank and user_rank > 10:
                # Determine the Y position for the extra row(s)
                if user_rank == 11:
                    # 2a) Exactly 11th → draw row 11 immediately
                    extra_y = row_start_y + 11 * row_height
                else:
                    # 2b) >11th → draw ellipsis at row 11, then entry at row 12
                    ellipsis = self.line_font.render("...", True, (0, 0, 0))
                    ellipsis_x = (1920 - ellipsis.get_width()) // 2
                    ellipsis_y = row_start_y + 11 * row_height
                    screen.blit(ellipsis, (ellipsis_x, ellipsis_y))
                    extra_y = ellipsis_y + row_height

                # Render the recent_entry row
                rank_s   = self.line_font.render(f"{user_rank}.", True, (0, 0, 0))
                name_s   = self.line_font.render(recent["name"], True, (0, 0, 0))
                rounds_s = self.line_font.render(str(recent["rounds"]), True, (0, 0, 0))
                time_s   = self.line_font.render(f"{recent['score']:.2f}s", True, (0, 0, 0))

                screen.blit(rank_s,   (x_rank,   extra_y))
                screen.blit(name_s,   (x_name,   extra_y))
                screen.blit(rounds_s, (x_rounds, extra_y))
                screen.blit(time_s,   (x_time,   extra_y))

    def reset(self) -> None:
        self._next_state = None
        self.update_scores_surfaces()

