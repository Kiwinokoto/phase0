from __future__ import annotations

import argparse
import secrets
from dataclasses import dataclass, replace

import numpy as np
import pygame

from .config import PlanetConfig
from .planet import Planet

LayerName = str
OverlayMode = str
WeatherOverlayMode = str
ProjectionMode = str
Color = tuple[int, int, int]

LIFE_LAYER_NAMES: tuple[LayerName, ...] = ("dead_matter", "biomass", "diversity", "dominant_life", "biotic_pressure", "migration_pressure", "isolation_pressure")
ATMOSPHERE_LAYER_NAMES: tuple[LayerName, ...] = ("clouds", "rain")
LIFE_OVERLAY_MODES: tuple[OverlayMode, ...] = ("off", "biomass", "dominant")
WEATHER_OVERLAY_MODES: tuple[WeatherOverlayMode, ...] = ("off", "clouds", "rain", "all")
PROJECTION_MODES: tuple[ProjectionMode, ...] = ("2d", "3d")


@dataclass(frozen=True)
class SetupField:
    key: str
    label: str
    step: float
    minimum: float
    maximum: float
    decimals: int = 2
    low_color: Color = (52, 64, 88)
    high_color: Color = (120, 185, 150)

    def format_value(self, value: object) -> str:
        if self.decimals == 0:
            return str(int(round(float(value))))
        return f"{float(value):.{self.decimals}f}"


PRIMARY_SETUP_FIELDS: tuple[SetupField, ...] = (
    SetupField("sea_level", "Sea level", 0.02, 0.34, 0.68, 2, (82, 132, 76), (52, 132, 205)),
    SetupField("continent_scale", "Continent scale", 1.0, 2.0, 9.0, 0, (70, 58, 48), (210, 180, 104)),
    SetupField("volcanic_activity_fraction", "Volcanism", 0.01, 0.01, 0.24, 2, (62, 42, 76), (255, 116, 45)),
    SetupField("equator_temperature_c", "Equator temp", 1.0, 8.0, 48.0, 0, (60, 86, 170), (210, 72, 46)),
    SetupField("pole_temperature_c", "Pole temp", 1.0, -45.0, 8.0, 0, (35, 70, 155), (215, 230, 238)),
)

DETAIL_SETUP_FIELDS: tuple[SetupField, ...] = (
    SetupField("detail_octaves", "Detail octaves", 1.0, 1.0, 7.0, 0, (60, 66, 80), (205, 212, 198)),
    SetupField("detail_gain", "Detail gain", 0.03, 0.35, 0.72, 2, (72, 60, 50), (220, 205, 132)),
)

PLANET_SETUP_FIELDS: tuple[SetupField, ...] = PRIMARY_SETUP_FIELDS + DETAIL_SETUP_FIELDS


@dataclass(frozen=True)
class LayerLegend:
    title: str
    description: tuple[str, ...]
    colors: tuple[Color, ...] = ()
    labels: tuple[str, ...] = ()
    categories: tuple[tuple[str, Color], ...] = ()


@dataclass(frozen=True)
class GeologicalIntroStage:
    title: str
    description: tuple[str, ...]


INTRO_DURATION_FRAMES = 1560
EXTINCT_CRIMSON: Color = (205, 46, 62)
EXTINCT_DARK: Color = (82, 28, 38)


LAYER_LEGENDS: dict[LayerName, LayerLegend] = {
    "biome": LayerLegend(
        title="Biome",
        description=("Synthetic view: water, temp,", "humidity, altitude and coasts."),
        categories=(
            ("deep ocean", (25, 55, 125)),
            ("shallow sea", (42, 105, 165)),
            ("coast", (205, 190, 120)),
            ("wet land", (47, 115, 68)),
            ("dry land", (178, 151, 86)),
            ("cold/high", (230, 235, 238)),
        ),
    ),
    "elevation": LayerLegend(
        title="Elevation",
        description=("Terrain height before sea level", "classification."),
        colors=((15, 28, 64), (245, 245, 230)),
        labels=("low", "high"),
    ),
    "temperature": LayerLegend(
        title="Temperature",
        description=("Dynamic climate: latitude,", "altitude, ocean and season."),
        colors=((35, 70, 155), (230, 225, 170), (180, 45, 35)),
        labels=("cold", "mild", "hot"),
    ),
    "water": LayerLegend(
        title="Water",
        description=("Surface water / ocean depth.", "Land is low, oceans are high."),
        colors=((25, 25, 35), (40, 130, 220)),
        labels=("dry", "water"),
    ),
    "humidity": LayerLegend(
        title="Humidity",
        description=("Ecological moisture on land.", "Includes simple rain shadow."),
        colors=((120, 95, 45), (45, 165, 95)),
        labels=("arid", "humid"),
    ),
    "light": LayerLegend(
        title="Light",
        description=("Solar input by latitude and", "season."),
        colors=((20, 18, 45), (255, 235, 140)),
        labels=("dark", "bright"),
    ),
    "clouds": LayerLegend(
        title="Clouds",
        description=("Procedural atmospheric veil.", "Opacity drifts with season/tick."),
        colors=((18, 22, 32), (120, 135, 155), (235, 240, 245)),
        labels=("clear", "haze", "cloud"),
    ),
    "rain": LayerLegend(
        title="Rain / storms",
        description=("Stylized rain over cloud cover.", "Storm flashes favor wet volcanic zones."),
        colors=((12, 16, 26), (82, 112, 155), (225, 238, 255)),
        labels=("dry", "rain", "storm"),
    ),
    "volcanism": LayerLegend(
        title="Volcanism",
        description=("Base activity plus temporary", "volcanic pulses."),
        colors=((18, 16, 24), (92, 46, 72), (255, 116, 45)),
        labels=("quiet", "active", "pulse"),
    ),
    "minerals": LayerLegend(
        title="Minerals",
        description=("Static geological resource.", "Higher near mountains/volcanism."),
        colors=((24, 22, 22), (115, 102, 82), (230, 220, 185)),
        labels=("poor", "rocky", "rich"),
    ),
    "nutrients": LayerLegend(
        title="Nutrients",
        description=("Dynamic soluble resources.", "Diffuse, recharge and wash out."),
        colors=((28, 24, 18), (105, 115, 52), (210, 190, 78)),
        labels=("depleted", "usable", "rich"),
    ),
    "chemical_energy": LayerLegend(
        title="Chemical energy",
        description=("Non-light energy source for", "future chemosynthesis."),
        colors=((16, 18, 34), (80, 55, 150), (245, 175, 80)),
        labels=("none", "source", "surge"),
    ),
    "toxicity": LayerLegend(
        title="Toxicity",
        description=("Abiotic hostility from heat,", "volcanism and chemistry."),
        colors=((18, 22, 20), (90, 70, 130), (210, 55, 85)),
        labels=("safe", "stress", "hostile"),
    ),
    "fertility": LayerLegend(
        title="Fertility",
        description=("Composite proto-life potential,", "penalized by toxicity."),
        colors=((28, 22, 18), (65, 120, 68), (165, 220, 92)),
        labels=("hostile", "viable", "promising"),
    ),
    "dead_matter": LayerLegend(
        title="Dead matter",
        description=("Dead/recycled biomass.", "Auto-scaled so faint debris shows."),
        colors=((20, 16, 12), (115, 76, 38), (225, 170, 80)),
        labels=("none", "debris", "rich"),
    ),
    "biomass": LayerLegend(
        title="Biomass",
        description=("Total living proto-life", "population density."),
        colors=((10, 14, 12), (35, 130, 65), (180, 245, 120)),
        labels=("empty", "alive", "dense"),
    ),
    "diversity": LayerLegend(
        title="Diversity",
        description=("Local number of coexisting", "lineages, normalized."),
        colors=((16, 14, 26), (60, 105, 160), (230, 210, 110)),
        labels=("single/none", "mixed", "diverse"),
    ),
    "dominant_life": LayerLegend(
        title="Dominant life",
        description=("Color = dominant lineage.", "Brightness = biomass density."),
        colors=((8, 10, 12), (95, 160, 120), (240, 240, 180)),
        labels=("barren", "lineage", "dense"),
    ),
    "biotic_pressure": LayerLegend(
        title="Biotic pressure",
        description=("Life-on-life consumption pressure.", "Phase 5 ecological interaction field."),
        colors=((10, 12, 18), (110, 54, 85), (245, 118, 70)),
        labels=("quiet", "pressure", "intense"),
    ),
    "migration_pressure": LayerLegend(
        title="Migration / colonization",
        description=("Active drift toward better niches.", "Bright zones are recent frontiers."),
        colors=((8, 10, 18), (42, 118, 150), (120, 240, 210)),
        labels=("quiet", "moving", "frontier"),
    ),
    "isolation_pressure": LayerLegend(
        title="Isolation pressure",
        description=("Frontier/island colonies likely", "to branch into descendants."),
        colors=((12, 10, 18), (94, 70, 150), (255, 218, 92)),
        labels=("connected", "edge", "isolated"),
    ),
}


class PlanetViewer:
    """Small Pygame viewer for Phase 5 richer proto-ecology maps."""

    layers: tuple[LayerName, ...] = (
        "biome",
        "elevation",
        "temperature",
        "water",
        "humidity",
        "light",
        "clouds",
        "rain",
        "volcanism",
        "minerals",
        "nutrients",
        "chemical_energy",
        "toxicity",
        "fertility",
        "dead_matter",
        "biomass",
        "diversity",
        "dominant_life",
        "biotic_pressure",
        "migration_pressure",
        "isolation_pressure",
    )

    def __init__(self, planet: Planet, scale: int = 4, start_fullscreen: bool = True) -> None:
        pygame.init()
        self.planet = planet
        self.scale = max(1, int(scale))
        self.layer_index = 0
        self.paused = False
        self.in_setup_screen = True
        self.intro_active = False
        self.intro_frame = 0
        self.intro_duration_frames = INTRO_DURATION_FRAMES
        self.skip_intro = True
        self.fullscreen = bool(start_fullscreen)
        self.life_overlay_mode: OverlayMode = "biomass"
        self.weather_overlay_mode: WeatherOverlayMode = "clouds"
        self.projection_mode: ProjectionMode = "2d"
        self.speed = planet.config.initial_speed
        self.selected_cell: tuple[int, int] | None = None  # stored as (x, y) map coordinates
        self.selected_species_id: int | None = None
        self.selected_radius = 5
        self.species_row_rects: list[tuple[pygame.Rect, int]] = []
        self.genealogy_button_rect = pygame.Rect(0, 0, 0, 0)
        self.genealogy_modal_species_id: int | None = None
        self.genealogy_modal_rect = pygame.Rect(0, 0, 0, 0)
        self.genealogy_modal_close_rect = pygame.Rect(0, 0, 0, 0)
        self.genealogy_modal_row_rects: list[tuple[pygame.Rect, int]] = []
        self.event_log_button_rect = pygame.Rect(0, 0, 0, 0)
        self.event_log_modal_open = False
        self.event_log_modal_rect = pygame.Rect(0, 0, 0, 0)
        self.event_log_modal_close_rect = pygame.Rect(0, 0, 0, 0)
        self.event_log_filter_button_rects: list[tuple[pygame.Rect, str]] = []
        self.event_filter_show_volcanism = False
        self.event_birth_filter_mode = "early"  # early/all/hidden
        self.setup_control_rects: list[tuple[pygame.Rect, str, str]] = []
        self.setup_slider_rects: list[tuple[pygame.Rect, str]] = []
        self.active_setup_slider: tuple[str, pygame.Rect] | None = None
        self.section_header_rects: list[tuple[pygame.Rect, str]] = []
        self.collapsed_sections: set[str] = {"simulation", "planet", "life", "legend"}

        self.font = pygame.font.SysFont("monospace", 16)
        self.layer_font = pygame.font.SysFont("monospace", 20, bold=True)
        self.small_font = pygame.font.SysFont("monospace", 12)
        self.tiny_font = pygame.font.SysFont("monospace", 11)

        height, width = self.planet.shape
        self.base_map_size = (width * self.scale, height * self.scale)
        self.side_panel_width = 410
        self.windowed_size = (self.base_map_size[0] + self.side_panel_width, self.base_map_size[1])
        if self.fullscreen:
            self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        else:
            self.screen = pygame.display.set_mode(self.windowed_size, pygame.RESIZABLE)
        pygame.display.set_caption("Artificial Life Sandbox — Phase 5")
        self.map_rect = pygame.Rect(0, 0, *self.base_map_size)
        self.panel_rect = pygame.Rect(self.base_map_size[0], 0, self.side_panel_width, self.base_map_size[1])
        self.fullscreen_button_rect = pygame.Rect(0, 0, 0, 0)
        self.life_overlay_button_rect = pygame.Rect(0, 0, 0, 0)
        self.weather_overlay_button_rect = pygame.Rect(0, 0, 0, 0)
        self._update_layout()

        self.clock = pygame.time.Clock()
        self.cached_layer: LayerName | None = None
        self.cached_overlay_mode: OverlayMode | None = None
        self.cached_weather_overlay_mode: WeatherOverlayMode | None = None
        self.cached_projection_mode: ProjectionMode | None = None
        self.cached_surface: pygame.Surface | None = None
        self.cached_surface_size: tuple[int, int] | None = None

    @property
    def current_layer(self) -> LayerName:
        return self.layers[self.layer_index]

    def run(self) -> None:
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    running = self._handle_key(event.key)
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self._handle_mouse_click(event.pos)
                elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    self.active_setup_slider = None
                elif event.type == pygame.MOUSEMOTION and self.active_setup_slider is not None:
                    field_key, rect = self.active_setup_slider
                    self._set_setup_field_from_slider(field_key, event.pos[0], rect)
                elif event.type == pygame.VIDEORESIZE and not self.fullscreen:
                    self.screen = pygame.display.set_mode(event.size, pygame.RESIZABLE)
                    self.windowed_size = self.screen.get_size()
                    self._update_layout()
                    self._invalidate_cache()

            if self.intro_active:
                self.intro_frame += 1
                if self.intro_frame >= self.intro_duration_frames:
                    self._start_simulation()
                self._invalidate_cache()
            elif not self.in_setup_screen and not self.paused:
                self.planet.step(self.speed)
                self._invalidate_cache()

            self._draw()
            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()

    def _handle_mouse_click(self, pos: tuple[int, int]) -> None:
        if self.event_log_modal_open:
            self._handle_event_log_modal_click(pos)
            return

        if self.genealogy_modal_species_id is not None:
            self._handle_genealogy_modal_click(pos)
            return

        if self.fullscreen_button_rect.collidepoint(pos):
            self._toggle_fullscreen()
            return

        if self.projection_button_rect.collidepoint(pos):
            self._cycle_projection_mode()
            return

        if self.in_setup_screen:
            for rect, field_key in self.setup_slider_rects:
                if rect.collidepoint(pos):
                    self.active_setup_slider = (field_key, rect)
                    self._set_setup_field_from_slider(field_key, pos[0], rect)
                    return
            for rect, action, key in self.setup_control_rects:
                if rect.collidepoint(pos):
                    self._handle_setup_action(action, key)
                    return
            return

        if self.life_overlay_button_rect.collidepoint(pos):
            self._cycle_life_overlay()
            return

        if self.weather_overlay_button_rect.collidepoint(pos):
            self._cycle_weather_overlay()
            return

        if self.genealogy_button_rect.collidepoint(pos) and self.selected_species_id is not None:
            self.genealogy_modal_species_id = self.selected_species_id
            return

        if self.event_log_button_rect.collidepoint(pos):
            self.event_log_modal_open = True
            return

        for rect, section_key in self.section_header_rects:
            if rect.collidepoint(pos):
                self._toggle_section(section_key)
                return

        for rect, species_id in self.species_row_rects:
            if rect.collidepoint(pos):
                self.selected_species_id = species_id
                return

        if self.map_rect.collidepoint(pos):
            cell = self._screen_pos_to_cell(pos)
            if cell is not None:
                self.selected_cell = cell

    def _handle_genealogy_modal_click(self, pos: tuple[int, int]) -> None:
        if self.genealogy_modal_close_rect.collidepoint(pos):
            self.genealogy_modal_species_id = None
            return

        for rect, species_id in self.genealogy_modal_row_rects:
            if rect.collidepoint(pos):
                self.selected_species_id = species_id
                self.genealogy_modal_species_id = species_id
                return

        # Click outside the modal to close it.
        if not self.genealogy_modal_rect.collidepoint(pos):
            self.genealogy_modal_species_id = None

    def _handle_event_log_modal_click(self, pos: tuple[int, int]) -> None:
        if self.event_log_modal_close_rect.collidepoint(pos):
            self.event_log_modal_open = False
            return

        for rect, action in self.event_log_filter_button_rects:
            if rect.collidepoint(pos):
                if action == "toggle_volcanism":
                    self.event_filter_show_volcanism = not self.event_filter_show_volcanism
                elif action == "cycle_births":
                    modes = ("early", "hidden", "all")
                    current = modes.index(self.event_birth_filter_mode) if self.event_birth_filter_mode in modes else 0
                    self.event_birth_filter_mode = modes[(current + 1) % len(modes)]
                return

        # Click outside the modal to close it.
        if not self.event_log_modal_rect.collidepoint(pos):
            self.event_log_modal_open = False

    def _handle_key(self, key: int) -> bool:
        if self.event_log_modal_open and key == pygame.K_ESCAPE:
            self.event_log_modal_open = False
            return True
        if self.genealogy_modal_species_id is not None and key == pygame.K_ESCAPE:
            self.genealogy_modal_species_id = None
            return True
        if key in (pygame.K_ESCAPE, pygame.K_q):
            return False
        if self.in_setup_screen:
            if key in (pygame.K_f, pygame.K_F11):
                self._toggle_fullscreen()
            elif key == pygame.K_g:
                self._cycle_projection_mode()
            elif key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
                if self.intro_active:
                    self._start_simulation()
                else:
                    self._begin_start_flow()
            elif key == pygame.K_r and not self.intro_active:
                self._randomize_setup_seed()
            elif key == pygame.K_s:
                self._save_screenshot()
            return True
        if key == pygame.K_SPACE:
            self.paused = not self.paused
        elif key in (pygame.K_TAB, pygame.K_RIGHT):
            self.layer_index = (self.layer_index + 1) % len(self.layers)
            self._invalidate_cache()
        elif key == pygame.K_LEFT:
            self.layer_index = (self.layer_index - 1) % len(self.layers)
            self._invalidate_cache()
        elif key in (pygame.K_UP, pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
            self.speed = min(self.speed * 2, 4096)
        elif key in (pygame.K_DOWN, pygame.K_MINUS, pygame.K_UNDERSCORE, pygame.K_KP_MINUS):
            self.speed = max(1, self.speed // 2)
        elif key in (pygame.K_f, pygame.K_F11):
            self._toggle_fullscreen()
        elif key == pygame.K_g:
            self._cycle_projection_mode()
        elif key == pygame.K_o:
            self._cycle_life_overlay()
        elif key == pygame.K_w:
            self._cycle_weather_overlay()
        elif key == pygame.K_r:
            self.planet = self.planet.regenerate(seed=random_seed())
            self.selected_cell = None
            self.selected_species_id = None
            self._invalidate_cache()
        elif key == pygame.K_s:
            self._save_screenshot()
        return True

    def _toggle_fullscreen(self) -> None:
        self.fullscreen = not self.fullscreen
        if self.fullscreen:
            self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        else:
            self.screen = pygame.display.set_mode(self.windowed_size, pygame.RESIZABLE)
        self._update_layout()
        self._invalidate_cache()

    def _cycle_life_overlay(self) -> None:
        index = LIFE_OVERLAY_MODES.index(self.life_overlay_mode)
        self.life_overlay_mode = LIFE_OVERLAY_MODES[(index + 1) % len(LIFE_OVERLAY_MODES)]
        self._invalidate_cache()

    def _cycle_weather_overlay(self) -> None:
        index = WEATHER_OVERLAY_MODES.index(self.weather_overlay_mode)
        self.weather_overlay_mode = WEATHER_OVERLAY_MODES[(index + 1) % len(WEATHER_OVERLAY_MODES)]
        self._invalidate_cache()

    def _cycle_projection_mode(self) -> None:
        index = PROJECTION_MODES.index(self.projection_mode)
        self.projection_mode = PROJECTION_MODES[(index + 1) % len(PROJECTION_MODES)]
        self._invalidate_cache()

    def _update_layout(self) -> None:
        screen_w, screen_h = self.screen.get_size()
        panel_w = min(self.side_panel_width, max(300, screen_w // 3))
        map_area_w = max(1, screen_w - panel_w)
        map_area_h = max(1, screen_h)

        original_w, original_h = self.planet.config.width, self.planet.config.height
        aspect = original_w / original_h
        map_w = min(map_area_w, int(map_area_h * aspect))
        map_h = min(map_area_h, int(map_area_w / aspect))
        map_w = max(1, map_w)
        map_h = max(1, map_h)
        map_left = max(0, (map_area_w - map_w) // 2)
        map_top = max(0, (map_area_h - map_h) // 2)

        self.map_rect = pygame.Rect(map_left, map_top, map_w, map_h)
        self.panel_rect = pygame.Rect(map_area_w, 0, panel_w, screen_h)

    def _invalidate_cache(self) -> None:
        self.cached_layer = None
        self.cached_overlay_mode = None
        self.cached_weather_overlay_mode = None
        self.cached_projection_mode = None
        self.cached_surface = None
        self.cached_surface_size = None

    def _draw(self) -> None:
        self.screen.fill((16, 18, 24))
        self.screen.blit(self._get_map_surface(), self.map_rect.topleft)
        if not self.in_setup_screen:
            self._draw_selected_species_distribution()
            self._draw_selection_marker()
        self._draw_panel()
        if self.genealogy_modal_species_id is not None:
            self._draw_genealogy_modal()
        if self.event_log_modal_open:
            self._draw_event_log_modal()

    def _get_map_surface(self) -> pygame.Surface:
        layer = "biome" if self.in_setup_screen else self.current_layer
        overlay_mode = "off" if self.in_setup_screen else self.life_overlay_mode
        weather_overlay_mode = "off" if self.in_setup_screen else self.weather_overlay_mode
        if (
            self.cached_layer == layer
            and self.cached_overlay_mode == overlay_mode
            and self.cached_weather_overlay_mode == weather_overlay_mode
            and self.cached_projection_mode == self.projection_mode
            and self.cached_surface is not None
            and self.cached_surface_size == self.map_rect.size
        ):
            return self.cached_surface

        if self.intro_active:
            progress = self._intro_progress()
            rgb = render_geological_intro_layer(self.planet, progress)
        else:
            rgb = render_layer(self.planet, layer, overlay_mode=overlay_mode, weather_overlay_mode=weather_overlay_mode)
            if self.projection_mode == "3d":
                rgb = render_globe_texture(
                    rgb,
                    self.map_rect.size,
                    rotation=self._globe_rotation(),
                    star_rotation=self._starfield_rotation(),
                    seed=self.planet.config.seed,
                )
        surface = pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))
        if surface.get_size() != self.map_rect.size:
            surface = pygame.transform.scale(surface, self.map_rect.size)

        self.cached_layer = layer
        self.cached_overlay_mode = overlay_mode
        self.cached_weather_overlay_mode = weather_overlay_mode
        self.cached_projection_mode = self.projection_mode
        self.cached_surface = surface
        self.cached_surface_size = self.map_rect.size
        return surface

    def _draw_panel(self) -> None:
        panel = self.panel_rect
        pygame.draw.rect(self.screen, (18, 20, 28), panel)
        pygame.draw.line(self.screen, (70, 76, 96), panel.topleft, panel.bottomleft, 1)

        self.species_row_rects = []
        self.genealogy_modal_row_rects = []
        self.genealogy_button_rect = pygame.Rect(0, 0, 0, 0)
        self.genealogy_modal_close_rect = pygame.Rect(0, 0, 0, 0)
        self.event_log_button_rect = pygame.Rect(0, 0, 0, 0)
        self.event_log_modal_close_rect = pygame.Rect(0, 0, 0, 0)
        self.event_log_filter_button_rects = []
        self.weather_overlay_button_rect = pygame.Rect(0, 0, 0, 0)
        self.projection_button_rect = pygame.Rect(0, 0, 0, 0)
        self.setup_control_rects = []
        self.setup_slider_rects = []
        self.section_header_rects = []

        x = panel.left + 18
        y = 18
        content_w = panel.width - 36

        if self.in_setup_screen:
            self._draw_setup_panel(x, y, content_w)
            return

        self._draw_text("Phase 5 — Richer Ecology", x, y, self.font, (235, 238, 245))
        y += 25

        y = self._draw_active_layer_header(x, y)
        y += 9

        y = self._draw_settings_row(x, y)
        y += 8

        y = self._draw_compact_controls(x, y, content_w)
        y += 10

        legend_height = 36 if self._is_collapsed("legend") else 178
        legend_y = max(y + 250, panel.bottom - legend_height)
        section_limit = legend_y - 12

        for draw_section in (
            self._draw_simulation_summary,
            self._draw_event_log,
            self._draw_planet_summary,
            self._draw_life_summary,
            self._draw_selected_zone,
            self._draw_selected_lineage,
        ):
            if y > section_limit - 34:
                self._draw_text("Collapse sections above to reveal more details.", x, y, self.tiny_font, (190, 170, 130))
                y += 16
                break
            y = draw_section(x, y, content_w)
            y += 12

        self._draw_current_layer_legend(x, legend_y, content_w)


    def _draw_setup_panel(self, x: int, y: int, width: int) -> None:
        if self.intro_active:
            self._draw_intro_panel(x, y, width)
            return

        self._draw_text("Artificial Life Sandbox", x, y, self.font, (235, 238, 245))
        y += 24
        self._draw_text("Planet setup — preview before simulation", x, y, self.tiny_font, (165, 174, 196))
        y += 24

        button_h = 28
        gap = 8
        half_w = max(120, (width - gap) // 2)
        fullscreen_label = "Window" if self.fullscreen else "Fullscreen"
        self.fullscreen_button_rect = pygame.Rect(x, y, half_w, button_h)
        self._draw_button(self.fullscreen_button_rect, fullscreen_label)
        self.projection_button_rect = pygame.Rect(x + half_w + gap, y, half_w, button_h)
        self._draw_button(self.projection_button_rect, f"View: {self.projection_mode.upper()}")
        y += button_h + 12

        y = self._draw_section_title("Generated planet", x, y, width)
        y = self._draw_key_value_grid(
            (
                ("seed", str(self.planet.config.seed)),
                ("size", f"{self.planet.config.width}x{self.planet.config.height}"),
                ("land", f"{100.0 * self.planet.land.mean():.1f}%"),
                ("ocean", f"{100.0 * (1.0 - self.planet.land.mean()):.1f}%"),
                ("temp avg", f"{self.planet.temperature_c.mean():.1f} C"),
                ("temp range", f"{self.planet.temperature_c.min():.1f}/{self.planet.temperature_c.max():.1f}"),
                ("humidity", f"{self.planet.humidity.mean():.2f}"),
                ("fertility", f"{self.planet.fertility.mean():.2f}"),
                ("nutrients", f"{self.planet.nutrients.mean():.2f}"),
                ("volcanism", f"{self.planet.volcanism.mean():.2f}"),
                ("year", f"{self.planet.config.seasonal_period_ticks} ticks"),
                ("cloud/rain", self._weather_mean_label()),
                ("bio land", f"{self._land_biomass_share():.1f}%"),
                ("bio ocean", self._ocean_biomass_share_label()),
                ("migr", f"{self.planet.migration_pressure.mean():.3f}"),
                ("isolate", f"{self.planet.isolation_pressure.mean():.3f}"),
            ),
            x,
            y,
            width,
        )
        y += 12

        y = self._draw_section_title("Seed", x, y, width)
        y = self._draw_seed_setup_row(x, y, width)
        y += 8

        y = self._draw_section_title("Planet parameters", x, y, width)
        for field in PRIMARY_SETUP_FIELDS:
            if y > self.panel_rect.bottom - 72:
                self._draw_text("Panel too short: resize or use fullscreen.", x, y, self.tiny_font, (190, 170, 130))
                break
            y = self._draw_setup_field_row(field, x, y, width)

        if y <= self.panel_rect.bottom - 126:
            y += 8
            self._draw_text("Terrain detail — deterministic", x, y, self.tiny_font, (150, 160, 184))
            y += 16
            for field in DETAIL_SETUP_FIELDS:
                if y > self.panel_rect.bottom - 72:
                    self._draw_text("Panel too short: resize or use fullscreen.", x, y, self.tiny_font, (190, 170, 130))
                    break
                y = self._draw_setup_field_row(field, x, y, width)

        hint_y = max(y + 12, self.panel_rect.bottom - 124)
        for line in ("Changes regenerate the preview immediately.", "Enter/Space starts.  R chooses a random seed."):
            if hint_y + 14 < self.panel_rect.bottom - 74:
                self._draw_text(line, x, hint_y, self.tiny_font, (150, 160, 184))
                hint_y += 14

        checkbox_rect = pygame.Rect(x, self.panel_rect.bottom - 70, width, 22)
        self.setup_control_rects.append((checkbox_rect, "toggle_skip_intro", ""))
        self._draw_checkbox_row(checkbox_rect, "Skip formation intro", self.skip_intro)

        start_rect = pygame.Rect(x, self.panel_rect.bottom - 42, width, 30)
        self.setup_control_rects.append((start_rect, "start", ""))
        self._draw_button(start_rect, "Start simulation")

    def _draw_intro_panel(self, x: int, y: int, width: int) -> None:
        progress = self._intro_progress()
        stage = geological_intro_stage(progress)

        self._draw_text("Artificial Life Sandbox", x, y, self.font, (235, 238, 245))
        y += 24
        self._draw_text("Geological prelude — visual only", x, y, self.tiny_font, (165, 174, 196))
        y += 24

        button_h = 28
        fullscreen_label = "Window" if self.fullscreen else "Fullscreen"
        self.fullscreen_button_rect = pygame.Rect(x, y, min(170, width), button_h)
        self._draw_button(self.fullscreen_button_rect, fullscreen_label)
        y += button_h + 14

        y = self._draw_section_title("Formation stage", x, y, width)
        self._draw_text(stage.title, x, y, self.small_font, (235, 238, 245))
        y += 18
        for line in stage.description:
            self._draw_text(line, x, y, self.tiny_font, (170, 180, 204))
            y += 14
        y += 8

        bar = pygame.Rect(x, y, width, 16)
        pygame.draw.rect(self.screen, (20, 24, 34), bar, border_radius=8)
        pygame.draw.rect(self.screen, (54, 64, 86), bar, 1, border_radius=8)
        filled = pygame.Rect(bar.left, bar.top, int(bar.width * progress), bar.height)
        if filled.width > 0:
            self._draw_gradient_bar(filled, ((80, 42, 42), (100, 140, 190), (148, 215, 150)))
        pygame.draw.rect(self.screen, (84, 96, 126), bar, 1, border_radius=8)
        pct = self.tiny_font.render(f"{int(progress * 100):3d}%", True, (230, 234, 244))
        self.screen.blit(pct, (bar.centerx - pct.get_width() // 2, bar.top + 1))
        y += 28

        y = self._draw_section_title("Planet outcome", x, y, width)
        y = self._draw_key_value_grid(
            (
                ("seed", str(self.planet.config.seed)),
                ("land", f"{100.0 * self.planet.land.mean():.1f}%"),
                ("ocean", f"{100.0 * (1.0 - self.planet.land.mean()):.1f}%"),
                ("volcanism", f"{self.planet.volcanism.mean():.2f}"),
                ("fertility", f"{self.planet.fertility.mean():.2f}"),
                ("temp avg", f"{self.planet.temperature_c.mean():.1f} C"),
            ),
            x,
            y,
            width,
        )

        note_y = max(y + 18, self.panel_rect.bottom - 92)
        for line in ("The prelude does not alter the generated planet.", "Enter/Space skips to simulation."):
            if note_y + 14 < self.panel_rect.bottom - 48:
                self._draw_text(line, x, note_y, self.tiny_font, (150, 160, 184))
                note_y += 14

        skip_rect = pygame.Rect(x, self.panel_rect.bottom - 42, width, 30)
        self.setup_control_rects.append((skip_rect, "skip_intro_now", ""))
        self._draw_button(skip_rect, "Skip intro / start now")

    def _draw_seed_setup_row(self, x: int, y: int, width: int) -> int:
        random_w = 88
        gap = 8
        value_rect = pygame.Rect(x, y, max(80, width - random_w - gap), 24)
        pygame.draw.rect(self.screen, (24, 29, 40), value_rect, border_radius=6)
        pygame.draw.rect(self.screen, (54, 64, 86), value_rect, 1, border_radius=6)
        self._draw_text(str(self.planet.config.seed), value_rect.left + 8, value_rect.top + 4, self.tiny_font, (224, 230, 242))

        random_rect = pygame.Rect(value_rect.right + gap, y, random_w, 24)
        self.setup_control_rects.append((random_rect, "random_seed", ""))
        self._draw_button(random_rect, "Random")
        return y + 30

    def _draw_setup_field_row(self, field: SetupField, x: int, y: int, width: int) -> int:
        row_h = 50
        self._draw_text(field.label, x, y + 5, self.tiny_font, (184, 193, 214))

        plus_w = 30
        value_w = 70
        gap = 5
        plus_rect = pygame.Rect(x + width - plus_w, y + 1, plus_w, 23)
        minus_rect = pygame.Rect(plus_rect.left - gap - plus_w, y + 1, plus_w, 23)
        value_rect = pygame.Rect(minus_rect.left - gap - value_w, y + 1, value_w, 23)

        value = getattr(self.planet.config, field.key)
        pygame.draw.rect(self.screen, (24, 29, 40), value_rect, border_radius=6)
        pygame.draw.rect(self.screen, (54, 64, 86), value_rect, 1, border_radius=6)
        value_text = field.format_value(value)
        rendered = self.tiny_font.render(value_text, True, (224, 230, 242))
        self.screen.blit(rendered, (value_rect.centerx - rendered.get_width() // 2, value_rect.top + 5))

        self.setup_control_rects.append((minus_rect, "field_delta", f"{field.key}:-1"))
        self.setup_control_rects.append((plus_rect, "field_delta", f"{field.key}:1"))
        self._draw_button(minus_rect, "-")
        self._draw_button(plus_rect, "+")

        slider_y = y + 29
        slider_rect = pygame.Rect(x, slider_y, width, 12)
        self.setup_slider_rects.append((slider_rect, field.key))
        self._draw_setup_slider(field, slider_rect, value)

        min_text = field.format_value(field.minimum)
        max_text = field.format_value(field.maximum)
        self._draw_text(min_text, slider_rect.left, slider_rect.bottom + 1, self.tiny_font, (118, 128, 152))
        max_rendered = self.tiny_font.render(max_text, True, (118, 128, 152))
        self.screen.blit(max_rendered, (slider_rect.right - max_rendered.get_width(), slider_rect.bottom + 1))
        return y + row_h

    def _draw_setup_slider(self, field: SetupField, rect: pygame.Rect, value: object) -> None:
        fraction = self._setup_field_fraction(field, value)
        bg = rect.inflate(0, 2)
        pygame.draw.rect(self.screen, (20, 24, 34), bg, border_radius=6)
        pygame.draw.rect(self.screen, (48, 58, 78), bg, 1, border_radius=6)
        self._draw_gradient_bar(rect, (field.low_color, field.high_color))
        overlay = pygame.Surface(rect.size, pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 92))
        hidden_w = int(rect.width * (1.0 - fraction))
        if hidden_w > 0:
            self.screen.blit(overlay, (rect.right - hidden_w, rect.top), pygame.Rect(rect.width - hidden_w, 0, hidden_w, rect.height))
        knob_x = rect.left + int(round(fraction * rect.width))
        pygame.draw.line(self.screen, (245, 248, 238), (knob_x, rect.top - 3), (knob_x, rect.bottom + 3), 2)

    def _setup_field_fraction(self, field: SetupField, value: object) -> float:
        span = max(1e-9, float(field.maximum) - float(field.minimum))
        return float(np.clip((float(value) - float(field.minimum)) / span, 0.0, 1.0))

    def _set_setup_field_from_slider(self, field_key: str, mouse_x: int, rect: pygame.Rect) -> None:
        field = next((item for item in PLANET_SETUP_FIELDS if item.key == field_key), None)
        if field is None:
            return
        fraction = float(np.clip((mouse_x - rect.left) / max(1, rect.width), 0.0, 1.0))
        raw = field.minimum + fraction * (field.maximum - field.minimum)
        if field.decimals == 0:
            value = int(round(raw))
        else:
            value = round(float(raw), field.decimals)
        current = getattr(self.planet.config, field.key)
        if current == value:
            return
        self._set_setup_config(**{field.key: value})

    def _handle_setup_action(self, action: str, key: str) -> None:
        if action == "start":
            self._begin_start_flow()
        elif action == "skip_intro_now":
            self._start_simulation()
        elif action == "toggle_skip_intro":
            self.skip_intro = not self.skip_intro
        elif action == "random_seed":
            self._randomize_setup_seed()
        elif action == "seed_delta":
            self._set_setup_config(seed=max(0, int(self.planet.config.seed) + int(key)))
        elif action == "field_delta":
            field_key, direction_text = key.split(":", 1)
            self._adjust_setup_field(field_key, int(direction_text))

    def _begin_start_flow(self) -> None:
        if self.skip_intro:
            self._start_simulation()
        else:
            self._start_geological_intro()

    def _start_geological_intro(self) -> None:
        self.intro_active = True
        self.intro_frame = 0
        self.paused = True
        self.selected_cell = None
        self.selected_species_id = None
        self.active_setup_slider = None
        self._invalidate_cache()

    def _intro_progress(self) -> float:
        if not self.intro_active:
            return 0.0
        return float(np.clip(self.intro_frame / max(1, self.intro_duration_frames), 0.0, 1.0))

    def _start_simulation(self) -> None:
        self.intro_active = False
        self.intro_frame = 0
        self.in_setup_screen = False
        self.paused = False
        self.layer_index = 0
        self.life_overlay_mode = "biomass"
        self.selected_cell = None
        self.selected_species_id = None
        self._invalidate_cache()

    def _randomize_setup_seed(self) -> None:
        self._set_setup_config(seed=random_seed())

    def _adjust_setup_field(self, field_key: str, direction: int) -> None:
        field = next((item for item in PLANET_SETUP_FIELDS if item.key == field_key), None)
        if field is None:
            return
        current = float(getattr(self.planet.config, field.key))
        value = float(np.clip(current + direction * field.step, field.minimum, field.maximum))
        if field.decimals == 0:
            value = int(round(value))
        else:
            value = round(value, field.decimals)
        self._set_setup_config(**{field.key: value})

    def _set_setup_config(self, **changes: object) -> None:
        try:
            new_config = replace(self.planet.config, **changes)
            new_config.validate()
        except Exception as exc:
            print(f"Ignored invalid setup config change {changes}: {exc}")
            return
        self.planet = Planet.generate(new_config)
        self.speed = self.planet.config.initial_speed
        self.selected_cell = None
        self.selected_species_id = None
        self._update_layout()
        self._invalidate_cache()

    def _draw_active_layer_header(self, x: int, y: int) -> int:
        legend = LAYER_LEGENDS[self.current_layer]
        max_w = self.panel_rect.width - 36
        rect = pygame.Rect(x, y, max_w, 35)
        pygame.draw.rect(self.screen, (34, 42, 58), rect, border_radius=8)
        pygame.draw.rect(self.screen, (92, 112, 150), rect, 1, border_radius=8)
        self._draw_text(f"Layer: {legend.title}", x + 12, y + 5, self.layer_font, (235, 246, 255))
        raw = self.tiny_font.render(self.current_layer, True, (160, 174, 200))
        self.screen.blit(raw, (rect.right - raw.get_width() - 10, y + 18))
        return y + rect.height

    def _draw_compact_controls(self, x: int, y: int, width: int) -> int:
        line1 = "click map: inspect zone   |   space: pause"
        line2 = "tab/←/→: layer   ↑/↓: speed   r: new seed   s: shot"
        self._draw_text(line1, x, y, self.tiny_font, (176, 184, 204))
        y += 13
        self._draw_text(line2, x, y, self.tiny_font, (176, 184, 204))
        return y + 14

    def _draw_section_title(self, title: str, x: int, y: int, width: int, key: str | None = None) -> int:
        rect = pygame.Rect(x, y, width, 25)
        hovered = rect.collidepoint(pygame.mouse.get_pos())
        fill = (42, 49, 68) if hovered and key is not None else (30, 36, 50)
        border = (80, 94, 126) if hovered and key is not None else (56, 66, 90)

        # Dark spacer/backplate: this makes each section read as a separate block
        # without needing a full layout engine. The content below keeps the same
        # background, but the title creates a clear visual anchor and gap.
        pygame.draw.rect(self.screen, (14, 17, 24), rect.inflate(2, 6), border_radius=8)
        pygame.draw.rect(self.screen, fill, rect, border_radius=7)
        pygame.draw.rect(self.screen, border, rect, 1, border_radius=7)

        text_x = x + 10
        if key is not None:
            self.section_header_rects.append((rect, key))
            marker = "+" if self._is_collapsed(key) else "-"
            self._draw_text(marker, x + 10, y + 4, self.small_font, (178, 196, 226))
            text_x = x + 28
        self._draw_text(title, text_x, y + 4, self.small_font, (236, 241, 252))
        return y + 33

    def _draw_key_value_grid(
        self,
        pairs: tuple[tuple[str, str], ...],
        x: int,
        y: int,
        width: int,
        *,
        columns: int = 2,
        row_h: int = 15,
    ) -> int:
        columns = max(1, columns)
        col_w = max(1, width // columns)
        for index, (key, value) in enumerate(pairs):
            col = index % columns
            row = index // columns
            sx = x + col * col_w
            sy = y + row * row_h
            self._draw_text(f"{key}:", sx, sy, self.tiny_font, (145, 154, 178))
            value_x = sx + min(82, col_w // 2)
            max_chars = max(6, (col_w - (value_x - sx)) // 7)
            self._draw_text(self._clip_text(value, max_chars), value_x, sy, self.tiny_font, (214, 221, 238))
        rows = (len(pairs) + columns - 1) // columns
        return y + rows * row_h

    def _clip_text(self, text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        return text[: max(0, max_chars - 1)] + "…"

    def _toggle_section(self, section_key: str) -> None:
        if section_key in self.collapsed_sections:
            self.collapsed_sections.remove(section_key)
        else:
            self.collapsed_sections.add(section_key)

    def _is_collapsed(self, section_key: str) -> bool:
        return section_key in self.collapsed_sections

    def _draw_simulation_summary(self, x: int, y: int, width: int) -> int:
        y = self._draw_section_title("Simulation", x, y, width, key="simulation")
        if self._is_collapsed("simulation"):
            return y
        return self._draw_key_value_grid(
            (
                ("seed", str(self.planet.config.seed)),
                ("tick", str(self.planet.tick)),
                ("year/day", self._season_position_label()),
                ("season", self._season_label()),
                ("speed", f"x{self.speed}"),
                ("state", "paused" if self.paused else "running"),
                ("mode", "fullscreen" if self.fullscreen else "window"),
                ("view", self.projection_mode),
                ("life", self.life_overlay_mode),
                ("weather", self.weather_overlay_mode),
            ),
            x,
            y,
            width,
        )

    def _draw_event_log(self, x: int, y: int, width: int) -> int:
        y = self._draw_section_title("Event log", x, y, width, key="events")
        if self._is_collapsed("events"):
            return y

        events = self._filtered_events_newest(limit=5)
        if not events:
            self._draw_text("No visible events with current filters.", x, y, self.tiny_font, (145, 154, 178))
            y += 18
        else:
            for event in events:
                text = f"t{event.tick}: {event.message}"
                self._draw_text(self._clip_text(text, 52), x, y, self.tiny_font, self._event_kind_color(event.kind))
                y += 14

        y += 5
        self.event_log_button_rect = pygame.Rect(x, y, width, 24)
        self._draw_button(self.event_log_button_rect, "Open event summary")
        return y + 30

    def _event_kind_color(self, kind: str) -> Color:
        if kind == "birth":
            return (142, 222, 158)
        if kind == "branch":
            # Descendant/speciation events should pop visually in logs.
            return (255, 218, 92)
        if kind == "extinction":
            return EXTINCT_CRIMSON
        if kind == "volcanism":
            return (238, 174, 92)
        return (202, 210, 230)


    def _draw_life_summary(self, x: int, y: int, width: int) -> int:
        y = self._draw_section_title("Life summary", x, y, width, key="life")
        if self._is_collapsed("life"):
            return y
        y = self._draw_key_value_grid(
            (
                ("living", f"{self.planet.living_species_count}"),
                ("extinct", f"{self.planet.extinction_count}"),
                ("biomass", f"{self.planet.total_biomass:.1f}"),
                ("dead", f"{self.planet.total_dead_matter:.1f}"),
                ("lineages", f"{len(self.planet.species)}/{self.planet.config.max_species}"),
                ("max bio", f"{self.planet.biomass.max():.3f}"),
                ("biotic", f"{self.planet.biotic_pressure.mean():.3f}"),
            ),
            x,
            y,
            width,
        )
        y += 4
        return self._draw_top_species(x, y, width)

    def _draw_planet_summary(self, x: int, y: int, width: int) -> int:
        y = self._draw_section_title("Planet averages", x, y, width, key="planet")
        if self._is_collapsed("planet"):
            return y
        return self._draw_key_value_grid(
            (
                ("land", f"{100.0 * self.planet.land.mean():.1f}%"),
                ("ocean", f"{100.0 * (1.0 - self.planet.land.mean()):.1f}%"),
                ("temp", f"{self.planet.temperature_c.mean():.1f} C"),
                ("range", f"{self.planet.temperature_c.min():.1f}/{self.planet.temperature_c.max():.1f}"),
                ("humid", f"{self.planet.humidity.mean():.2f}"),
                ("light", f"{self.planet.light.mean():.2f}"),
                ("nutr", f"{self.planet.nutrients.mean():.2f}"),
                ("chem", f"{self.planet.chemical_energy.mean():.2f}"),
                ("tox", f"{self.planet.toxicity.mean():.2f}"),
                ("fert", f"{self.planet.fertility.mean():.2f}"),
                ("cloud/rain", self._weather_mean_label()),
                ("bio land", f"{self._land_biomass_share():.1f}%"),
                ("bio ocean", self._ocean_biomass_share_label()),
                ("migr", f"{self.planet.migration_pressure.mean():.3f}"),
                ("isolate", f"{self.planet.isolation_pressure.mean():.3f}"),
            ),
            x,
            y,
            width,
        )

    def _draw_selected_zone(self, x: int, y: int, width: int) -> int:
        y = self._draw_section_title("Selected zone", x, y, width, key="zone")
        if self._is_collapsed("zone"):
            return y

        if self.selected_cell is None:
            self._draw_text("Click anywhere on the map to inspect local ecology.", x, y, self.tiny_font, (178, 186, 206))
            y += 14
            self._draw_text("This will show local biomass and top lineages.", x, y, self.tiny_font, (145, 154, 178))
            return y + 17

        cell_x, cell_y = self.selected_cell
        cell_x = int(np.clip(cell_x, 0, self.planet.config.width - 1))
        cell_y = int(np.clip(cell_y, 0, self.planet.config.height - 1))
        self.selected_cell = (cell_x, cell_y)

        y = self._draw_key_value_grid(
            (
                ("cell", f"x{cell_x} y{cell_y}"),
                ("radius", str(self.selected_radius)),
                ("biomass", f"{self.planet.biomass[cell_y, cell_x]:.3f}"),
                ("diversity", f"{self.planet.diversity[cell_y, cell_x]:.2f}"),
                ("fert", f"{self.planet.fertility[cell_y, cell_x]:.2f}"),
                ("tox", f"{self.planet.toxicity[cell_y, cell_x]:.2f}"),
                ("dead", f"{self.planet.dead_matter[cell_y, cell_x]:.3f}"),
                ("biotic", f"{self.planet.biotic_pressure[cell_y, cell_x]:.3f}"),
                ("migr", f"{self.planet.migration_pressure[cell_y, cell_x]:.3f}"),
                ("isolate", f"{self.planet.isolation_pressure[cell_y, cell_x]:.3f}"),
            ),
            x,
            y,
            width,
        )
        y += 4

        local_top = self.planet.top_species_near(cell_x, cell_y, radius=self.selected_radius, limit=4)
        if not local_top:
            self._draw_text("Local lineages: none", x, y, self.tiny_font, (145, 154, 178))
            return y + 16

        self._draw_text("Local top lineages  (click to inspect)", x, y, self.tiny_font, (220, 226, 240))
        y += 14
        for species, local_total, _global_total in local_top:
            label = f"{species.name}  {local_total:.2f}  {self.planet.species_strategy_label(species)}"
            y = self._draw_species_row(species, label, x, y, width)
        return y

    def _draw_selected_lineage(self, x: int, y: int, width: int) -> int:
        y = self._draw_section_title("Selected lineage / habitat", x, y, width, key="lineage")
        if self._is_collapsed("lineage"):
            return y

        species = self.planet.species_by_id(self.selected_species_id)
        if species is None:
            if self.selected_species_id is not None:
                self.selected_species_id = None
            self._draw_text("Click a local/global lineage row to open its card.", x, y, self.tiny_font, (178, 186, 206))
            y += 14
            self._draw_text("The selected lineage is highlighted on the map.", x, y, self.tiny_font, (145, 154, 178))
            return y + 17

        summary = self.planet.lineage_habitat_summary(species.id)
        strategy = self.planet.species_strategy_label(species)
        total_pop = summary.total_population
        status = "extinct" if species.is_extinct else "living"
        age_end = species.extinct_tick if species.extinct_tick is not None else self.planet.tick
        age = max(0, age_end - species.created_tick)
        parent = self.planet.species_by_id(species.parent_id)
        parent_name = "seed" if parent is None else parent.name
        children = self.planet.descendant_count(species.id)

        # Small color/name header. Extinct selected lineages stay visible so
        # the observer does not lose context when a watched branch disappears.
        header_rect = pygame.Rect(x, y, width, 22)
        header_fill = EXTINCT_DARK if species.is_extinct else (23, 28, 38)
        header_border = EXTINCT_CRIMSON if species.is_extinct else (58, 68, 90)
        title_color = EXTINCT_CRIMSON if species.is_extinct else (232, 238, 250)
        pygame.draw.rect(self.screen, header_fill, header_rect, border_radius=5)
        pygame.draw.rect(self.screen, header_border, header_rect, 1, border_radius=5)
        swatch = pygame.Rect(x + 7, y + 5, 12, 12)
        pygame.draw.rect(self.screen, species.color, swatch)
        pygame.draw.rect(self.screen, (112, 120, 142), swatch, 1)
        title = f"{species.name} — {strategy}"
        self._draw_text(self._clip_text(title, 44), x + 26, y + 3, self.small_font, title_color)
        y += 28
        if species.is_extinct:
            extinct_text = "EXTINCT" if species.extinct_tick is None else f"EXTINCT at tick {species.extinct_tick}"
            self._draw_text(extinct_text, x, y, self.tiny_font, EXTINCT_CRIMSON)
            y += 15

        y = self._draw_key_value_grid(
            (
                ("status", status),
                ("age", str(age)),
                ("parent", parent_name),
                ("children", str(children)),
                ("biomass", f"{total_pop:.2f}"),
                ("peak", f"{species.population_peak:.2f}"),
                ("cells", str(summary.occupied_cells)),
                ("habitat", summary.main_habitat),
            ),
            x,
            y,
            width,
        )
        y += 4

        if summary.strongest_cell is not None:
            sx, sy = summary.strongest_cell
            self._draw_text(f"strongest: x{sx} y{sy}", x, y, self.tiny_font, (165, 174, 196))
            y += 14

        self._draw_text("Habitat summary", x, y, self.tiny_font, (220, 226, 240))
        y += 14
        y = self._draw_key_value_grid(
            (
                ("temp", f"{summary.mean_temperature_c:.1f} C"),
                ("water", f"{summary.mean_water_access:.2f}"),
                ("fert", f"{summary.mean_fertility:.2f}"),
                ("tox", f"{summary.mean_toxicity:.2f}"),
                ("nutr", f"{summary.mean_nutrients:.2f}"),
                ("chem", f"{summary.mean_chemical_energy:.2f}"),
                ("dead", f"{summary.mean_dead_matter:.3f}"),
                ("biotic", f"{summary.mean_biotic_pressure:.3f}"),
                ("migr", f"{summary.mean_migration_pressure:.3f}"),
                ("isolate", f"{summary.mean_isolation_pressure:.3f}"),
                ("light", f"{summary.mean_light:.2f}"),
            ),
            x,
            y,
            width,
        )
        y += 4

        self._draw_text("Traits", x, y, self.tiny_font, (220, 226, 240))
        y += 14
        traits = species.traits
        y = self._draw_key_value_grid(
            (
                ("photo", f"{traits.photosynthesis:.2f}"),
                ("chemo", f"{traits.chemosynthesis:.2f}"),
                ("detrit", f"{traits.organic_absorption:.2f}"),
                ("living", f"{traits.living_consumption:.2f}"),
                ("def", f"{traits.defense:.2f}"),
                ("storage", f"{traits.storage:.2f}"),
                ("disp", f"{traits.dispersal:.2f}"),
                ("tox tol", f"{traits.toxicity_tolerance:.2f}"),
                ("mut", f"{traits.mutation_rate:.3f}"),
                ("temp opt", f"{traits.temperature_optimum_c:.1f}"),
                ("temp tol", f"{traits.temperature_tolerance_c:.1f}"),
            ),
            x,
            y,
            width,
        )
        y += 8
        self.genealogy_button_rect = pygame.Rect(x, y, width, 26)
        self._draw_button(self.genealogy_button_rect, "Open genealogy tree")
        return y + 32

    def _draw_species_row(self, species, label: str, x: int, y: int, width: int) -> int:
        row_h = 16
        rect = pygame.Rect(x, y - 1, width, row_h)
        self.species_row_rects.append((rect, species.id))
        hovered = rect.collidepoint(pygame.mouse.get_pos())
        selected = self.selected_species_id == species.id
        if selected or hovered:
            fill = (45, 56, 76) if selected else (30, 36, 50)
            pygame.draw.rect(self.screen, fill, rect, border_radius=4)
            border = (135, 176, 150) if selected else (72, 84, 108)
            pygame.draw.rect(self.screen, border, rect, 1, border_radius=4)

        color_rect = pygame.Rect(x + 3, y + 2, 10, 10)
        pygame.draw.rect(self.screen, species.color, color_rect)
        pygame.draw.rect(self.screen, (90, 96, 116), color_rect, 1)
        if species.is_extinct:
            text_color = EXTINCT_CRIMSON
        else:
            text_color = (234, 242, 235) if selected else (190, 199, 218)
        self._draw_text(self._clip_text(label, 49), x + 18, y, self.tiny_font, text_color)
        return y + row_h

    def _draw_event_log_modal(self) -> None:
        all_events = list(reversed(self.planet.event_log))
        events = self._filtered_events_newest(limit=None)
        screen_w, screen_h = self.screen.get_size()
        overlay = pygame.Surface((screen_w, screen_h), pygame.SRCALPHA)
        overlay.fill((3, 5, 10, 168))
        self.screen.blit(overlay, (0, 0))

        modal_w = min(940, screen_w - 90)
        modal_h = min(720, screen_h - 86)
        modal_w = max(560, modal_w)
        modal_h = max(460, modal_h)
        modal = pygame.Rect((screen_w - modal_w) // 2, (screen_h - modal_h) // 2, modal_w, modal_h)
        self.event_log_modal_rect = modal
        self.event_log_filter_button_rects = []

        pygame.draw.rect(self.screen, (18, 22, 32), modal, border_radius=12)
        pygame.draw.rect(self.screen, (84, 96, 126), modal, 1, border_radius=12)
        pygame.draw.rect(self.screen, (42, 49, 68), modal.inflate(-2, -2), 1, border_radius=11)

        x = modal.left + 22
        y = modal.top + 18
        width = modal.width - 44
        close_rect = pygame.Rect(modal.right - 40, modal.top + 14, 24, 24)
        self.event_log_modal_close_rect = close_rect
        self._draw_button(close_rect, "×")

        self._draw_text("World event summary", x, y, self.font, (238, 243, 252))
        y += 24
        self._draw_text(
            "Births/extinctions can be filtered; branch descendants are always highlighted in yellow.",
            x,
            y,
            self.tiny_font,
            (165, 176, 200),
        )
        y += 22

        counts = self._event_counts()
        y = self._draw_key_value_grid(
            (
                ("events", str(len(self.planet.event_log))),
                ("visible", str(len(events))),
                ("births", str(counts.get("birth", 0))),
                ("branches", str(counts.get("branch", 0))),
                ("extinct", str(counts.get("extinction", 0))),
                ("volcanic", str(counts.get("volcanism", 0))),
                ("current tick", str(self.planet.tick)),
                ("birth cutoff", f"t{self._birth_event_cutoff_tick()}"),
            ),
            x,
            y,
            width,
            columns=4,
        )
        y += 14

        button_gap = 10
        button_h = 25
        half = (width - button_gap) // 2
        volcano_label = "Volcanoes: shown" if self.event_filter_show_volcanism else "Volcanoes: hidden"
        if self.event_birth_filter_mode == "all":
            birth_label = "Births: all"
        elif self.event_birth_filter_mode == "hidden":
            birth_label = "Births: hidden"
        else:
            birth_label = "Births: early only"

        volcano_rect = pygame.Rect(x, y, half, button_h)
        birth_rect = pygame.Rect(x + half + button_gap, y, half, button_h)
        self.event_log_filter_button_rects.append((volcano_rect, "toggle_volcanism"))
        self.event_log_filter_button_rects.append((birth_rect, "cycle_births"))
        self._draw_button(volcano_rect, volcano_label)
        self._draw_button(birth_rect, birth_label)
        y += button_h + 12

        row_top = y
        row_bottom = modal.bottom - 44
        if not all_events:
            self._draw_text("No major event yet. Let the world run.", x, row_top, self.tiny_font, (150, 160, 184))
        elif not events:
            self._draw_text("All current events are hidden by filters.", x, row_top, self.tiny_font, (150, 160, 184))
        else:
            visible_rows = max(1, (row_bottom - row_top) // 18)
            for event in events[:visible_rows]:
                color = self._event_kind_color(event.kind)
                loc = "" if event.location is None else f"  x{event.location[0]} y{event.location[1]}"
                marker = "★ DESCENDANT " if event.kind == "branch" else ""
                text = f"{marker}t{event.tick:<5} {event.kind:<10} {event.message}{loc}"
                self._draw_text(self._clip_text(text, max(24, width // 7)), x, y, self.tiny_font, color)
                y += 18
            remaining = len(events) - visible_rows
            if remaining > 0 and y <= row_bottom - 14:
                self._draw_text(f"… {remaining} older visible events hidden", x, y, self.tiny_font, (150, 160, 184))

        self._draw_text(
            "Use Births: all to audit roots. Descendant/speciation events stay visible and yellow.",
            x,
            modal.bottom - 28,
            self.tiny_font,
            (155, 166, 190),
        )

    def _birth_event_cutoff_tick(self) -> int:
        return max(1200, int(self.planet.config.seasonal_period_ticks))

    def _event_visible_by_filter(self, event) -> bool:
        if event.kind == "branch":
            return True
        if event.kind == "extinction":
            return True
        if event.kind == "volcanism":
            return self.event_filter_show_volcanism
        if event.kind == "birth":
            if self.event_birth_filter_mode == "all":
                return True
            if self.event_birth_filter_mode == "hidden":
                return False
            return int(event.tick) <= self._birth_event_cutoff_tick()
        return True

    def _filtered_events_newest(self, limit: int | None = None) -> list:
        events = [event for event in reversed(self.planet.event_log) if self._event_visible_by_filter(event)]
        if limit is None:
            return events
        return events[: max(0, int(limit))]

    def _event_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for event in self.planet.event_log:
            counts[event.kind] = counts.get(event.kind, 0) + 1
        return counts

    def _draw_genealogy_modal(self) -> None:
        species = self.planet.species_by_id(self.genealogy_modal_species_id)
        if species is None:
            self.genealogy_modal_species_id = None
            return

        self.genealogy_modal_row_rects = []
        screen_w, screen_h = self.screen.get_size()
        overlay = pygame.Surface((screen_w, screen_h), pygame.SRCALPHA)
        overlay.fill((3, 5, 10, 168))
        self.screen.blit(overlay, (0, 0))

        modal_w = min(920, screen_w - 90)
        modal_h = min(660, screen_h - 86)
        modal_w = max(520, modal_w)
        modal_h = max(420, modal_h)
        modal = pygame.Rect((screen_w - modal_w) // 2, (screen_h - modal_h) // 2, modal_w, modal_h)
        self.genealogy_modal_rect = modal

        modal_border = EXTINCT_CRIMSON if species.is_extinct else species.color
        pygame.draw.rect(self.screen, (18, 22, 32), modal, border_radius=12)
        pygame.draw.rect(self.screen, modal_border, modal, 1, border_radius=12)
        pygame.draw.rect(self.screen, (74, 84, 112), modal.inflate(-2, -2), 1, border_radius=11)

        x = modal.left + 22
        y = modal.top + 18
        width = modal.width - 44
        close_rect = pygame.Rect(modal.right - 40, modal.top + 14, 24, 24)
        self.genealogy_modal_close_rect = close_rect
        self._draw_button(close_rect, "×")

        pygame.draw.rect(self.screen, species.color, pygame.Rect(x, y + 4, 14, 14))
        pygame.draw.rect(self.screen, (110, 118, 140), pygame.Rect(x, y + 4, 14, 14), 1)
        self._draw_text("Genealogy tree", x + 22, y, self.font, (238, 243, 252))
        status = "extinct" if species.is_extinct else "living"
        self._draw_text(
            self._clip_text(f"{species.name} — {self.planet.species_strategy_label(species)} — {status}", 86),
            x + 22,
            y + 21,
            self.tiny_font,
            EXTINCT_CRIMSON if species.is_extinct else (178, 188, 210),
        )
        y += 54

        ancestors = self.planet.lineage_ancestors(species.id, include_self=True)
        descendants = self.planet.lineage_descendants(species.id)
        direct_children = self.planet.lineage_children(species.id)

        summary_pairs = (
            ("ancestors", str(max(0, len(ancestors) - 1))),
            ("children", str(len(direct_children))),
            ("desc", str(len(descendants))),
            ("created", f"t{species.created_tick}"),
            ("peak", f"{species.population_peak:.1f}"),
            ("current", f"{self.planet.species_total_population(species.id):.1f}"),
        )
        y = self._draw_key_value_grid(summary_pairs, x, y, width, columns=3)
        y += 16

        col_gap = 18
        left_w = max(210, int(width * 0.35))
        right_w = width - left_w - col_gap
        left_x = x
        right_x = x + left_w + col_gap
        content_bottom = modal.bottom - 48

        self._draw_text("Ancestral line", left_x, y, self.small_font, (230, 236, 248))
        self._draw_text("Descendants", right_x, y, self.small_font, (230, 236, 248))
        y0 = y + 22

        ay = y0
        if not ancestors:
            self._draw_text("No ancestry data.", left_x, ay, self.tiny_font, (150, 160, 184))
        else:
            for index, ancestor in enumerate(ancestors):
                if ay > content_bottom - 18:
                    self._draw_text("…", left_x, ay, self.tiny_font, (150, 160, 184))
                    break
                connector = "└─" if index == len(ancestors) - 1 else "├─"
                depth = max(0, index)
                ay = self._draw_genealogy_row(
                    ancestor,
                    f"{connector} {ancestor.name}",
                    left_x,
                    ay,
                    left_w,
                    depth=min(depth, 5),
                    selected=ancestor.id == species.id,
                )

        dy = y0
        if not descendants:
            self._draw_text("No descendants yet.", right_x, dy, self.tiny_font, (150, 160, 184))
            dy += 16
        else:
            max_rows = max(8, (content_bottom - dy) // 18)
            visible = descendants[:max_rows]
            for depth, child in visible:
                prefix = "└─" if depth == 1 else "↳"
                label = f"{prefix} {child.name}"
                dy = self._draw_genealogy_row(
                    child,
                    label,
                    right_x,
                    dy,
                    right_w,
                    depth=min(depth - 1, 7),
                    selected=child.id == self.selected_species_id,
                )
            remaining = len(descendants) - len(visible)
            if remaining > 0 and dy <= content_bottom - 14:
                self._draw_text(f"… {remaining} more descendants", right_x, dy, self.tiny_font, (150, 160, 184))
                dy += 16

        footer_y = modal.bottom - 32
        self._draw_text(
            "Click a lineage row to inspect it. Esc or outside click closes this tree.",
            x,
            footer_y,
            self.tiny_font,
            (155, 166, 190),
        )

    def _draw_genealogy_row(
        self,
        species,
        label: str,
        x: int,
        y: int,
        width: int,
        *,
        depth: int = 0,
        selected: bool = False,
    ) -> int:
        row_h = 18
        indent = min(86, max(0, depth) * 14)
        rect = pygame.Rect(x, y, width, row_h)
        self.genealogy_modal_row_rects.append((rect, species.id))
        hovered = rect.collidepoint(pygame.mouse.get_pos())
        if selected or hovered:
            fill = (45, 56, 76) if selected else (30, 36, 50)
            border = (135, 176, 150) if selected else (72, 84, 108)
            pygame.draw.rect(self.screen, fill, rect, border_radius=5)
            pygame.draw.rect(self.screen, border, rect, 1, border_radius=5)

        swatch = pygame.Rect(x + 5 + indent, y + 4, 10, 10)
        pygame.draw.rect(self.screen, species.color, swatch)
        pygame.draw.rect(self.screen, (92, 100, 122), swatch, 1)
        total = self.planet.species_total_population(species.id)
        extinct = "† " if species.is_extinct else ""
        text = f"{extinct}{label}  t{species.created_tick}  {total:.1f}"
        if species.is_extinct:
            color = EXTINCT_CRIMSON
        else:
            color = (235, 242, 236) if selected else (190, 200, 220)
        max_chars = max(12, (width - indent - 22) // 7)
        self._draw_text(self._clip_text(text, max_chars), x + 20 + indent, y + 2, self.tiny_font, color)
        return y + row_h

    def _draw_selected_species_distribution(self) -> None:
        species = self.planet.species_by_id(self.selected_species_id)
        if species is None:
            return
        index = self.planet.species_index_by_id(species.id)
        if index is None:
            return
        pop = np.clip(self.planet.populations[index], 0.0, 1.0)
        has_visible_distribution = float(pop.max()) > 0.0

        if has_visible_distribution:
            h, w = pop.shape
            alpha = np.clip(28 + 190 * np.sqrt(pop / max(float(pop.max()), 0.025)), 0, 218).astype(np.uint8)
            alpha[pop <= 0.004] = 0
            if int(alpha.max()) > 0:
                if self.projection_mode == "3d":
                    overlay = render_globe_scalar_overlay(
                        pop,
                        species.color,
                        self.map_rect.size,
                        rotation=self._globe_rotation(),
                    )
                    self.screen.blit(overlay, self.map_rect.topleft)
                else:
                    rgb = np.zeros((h, w, 3), dtype=np.uint8)
                    color = np.array(species.color, dtype=np.uint8)
                    rgb[:, :] = color
                    surface = pygame.Surface((w, h), pygame.SRCALPHA)
                    pixels = pygame.surfarray.pixels3d(surface)
                    pixels[:] = np.transpose(rgb, (1, 0, 2))
                    del pixels
                    pixels_alpha = pygame.surfarray.pixels_alpha(surface)
                    pixels_alpha[:] = np.transpose(alpha, (1, 0))
                    del pixels_alpha
                    scaled = pygame.transform.scale(surface, self.map_rect.size)
                    self.screen.blit(scaled, self.map_rect.topleft)

        # Small readable label on the map itself. Keep it visible even after
        # extinction so the current observer state never looks like it vanished.
        suffix = " (extinct)" if species.is_extinct else ""
        label = f"selected: {species.name}{suffix}"
        text_color = EXTINCT_CRIMSON if species.is_extinct else (245, 248, 240)
        border_color = EXTINCT_CRIMSON if species.is_extinct else species.color
        text = self.small_font.render(label, True, text_color)
        bg = pygame.Rect(self.map_rect.left + 12, self.map_rect.top + 12, text.get_width() + 14, text.get_height() + 8)
        pygame.draw.rect(self.screen, (16, 20, 26), bg, border_radius=6)
        pygame.draw.rect(self.screen, border_color, bg, 1, border_radius=6)
        self.screen.blit(text, (bg.left + 7, bg.top + 4))


    def _globe_rotation(self) -> float:
        period = max(1, int(self.planet.config.seasonal_period_ticks))
        seed_offset = (int(self.planet.config.seed) % 1000) / 1000.0 * 2.0 * np.pi
        return seed_offset + (float(self.planet.tick) / max(1.0, period * 0.72)) * 2.0 * np.pi

    def _starfield_rotation(self) -> float:
        """Rotate the distant sky once per local year.

        The globe itself rotates a bit faster than this; the star field is a
        slow background reference tied to the seeded seasonal period.
        """
        period = max(1, int(self.planet.config.seasonal_period_ticks))
        seed_offset = ((int(self.planet.config.seed) >> 8) % 1000) / 1000.0 * 2.0 * np.pi
        return seed_offset + (float(self.planet.tick) / float(period)) * 2.0 * np.pi

    def _globe_center_radius(self) -> tuple[tuple[int, int], int]:
        radius = max(8, int(min(self.map_rect.width, self.map_rect.height) * 0.46))
        center = (self.map_rect.centerx, self.map_rect.centery)
        return center, radius

    def _cell_to_screen_pos(self, cell_x: int, cell_y: int) -> tuple[int, int] | None:
        if self.projection_mode != "3d":
            width = self.planet.config.width
            height = self.planet.config.height
            px = self.map_rect.left + int((cell_x / width) * self.map_rect.width)
            py = self.map_rect.top + int((cell_y / height) * self.map_rect.height)
            return (px, py)

        width = self.planet.config.width
        height = self.planet.config.height
        u = (float(cell_x) + 0.5) / max(1, width)
        v = (float(cell_y) + 0.5) / max(1, height)
        lon = 2.0 * np.pi * u - self._globe_rotation()
        lon = (lon + np.pi) % (2.0 * np.pi) - np.pi
        lat = (0.5 - v) * np.pi
        screen_x = np.sin(lon)
        z = np.cos(lon) * np.cos(lat)
        if z <= 0.0:
            return None
        screen_y = -np.sin(lat)
        center, radius = self._globe_center_radius()
        return (center[0] + int(screen_x * np.cos(lat) * radius), center[1] + int(screen_y * radius))
    def _screen_pos_to_cell(self, pos: tuple[int, int]) -> tuple[int, int] | None:
        if not self.map_rect.collidepoint(pos):
            return None

        if self.projection_mode == "3d":
            center, radius = self._globe_center_radius()
            nx = (pos[0] - center[0]) / max(1, radius)
            ny = (pos[1] - center[1]) / max(1, radius)
            r2 = nx * nx + ny * ny
            if r2 > 1.0:
                return None
            nz = float(np.sqrt(max(0.0, 1.0 - r2)))
            lon = np.arctan2(nx, nz) + self._globe_rotation()
            lat = np.arcsin(float(np.clip(-ny, -1.0, 1.0)))
            u = (lon / (2.0 * np.pi)) % 1.0
            v = 0.5 - (lat / np.pi)
            cell_x = int(np.clip(u * self.planet.config.width, 0, self.planet.config.width - 1))
            cell_y = int(np.clip(v * self.planet.config.height, 0, self.planet.config.height - 1))
            return (cell_x, cell_y)

        rel_x = (pos[0] - self.map_rect.left) / max(1, self.map_rect.width)
        rel_y = (pos[1] - self.map_rect.top) / max(1, self.map_rect.height)
        cell_x = int(np.clip(rel_x * self.planet.config.width, 0, self.planet.config.width - 1))
        cell_y = int(np.clip(rel_y * self.planet.config.height, 0, self.planet.config.height - 1))
        return (cell_x, cell_y)

    def _draw_selection_marker(self) -> None:
        if self.selected_cell is None:
            return
        cell_x, cell_y = self.selected_cell
        width = self.planet.config.width
        center = self._cell_to_screen_pos(cell_x, cell_y)
        if center is None:
            return
        if self.projection_mode == "3d":
            _globe_center, globe_radius = self._globe_center_radius()
            radius_px = max(7, int(self.selected_radius * (globe_radius / max(1, width))))
        else:
            radius_px = max(8, int(self.selected_radius * (self.map_rect.width / width)))
        pygame.draw.circle(self.screen, (245, 245, 235), center, radius_px, 1)
        pygame.draw.line(self.screen, (245, 245, 235), (center[0] - radius_px - 3, center[1]), (center[0] + radius_px + 3, center[1]), 1)
        pygame.draw.line(self.screen, (245, 245, 235), (center[0], center[1] - radius_px - 3), (center[0], center[1] + radius_px + 3), 1)

    def _draw_top_species(self, x: int, y: int, width: int) -> int:
        top = self.planet.top_species(limit=3)
        if not top:
            self._draw_text("Global top lineages: none yet", x, y, self.tiny_font, (145, 154, 178))
            return y + 16

        self._draw_text("Global top lineages", x, y, self.tiny_font, (220, 226, 240))
        y += 14
        for species, total in top:
            status = "†" if species.is_extinct else ""
            label = f"{species.name}{status}  {total:.1f}  {self.planet.species_strategy_label(species)}"
            y = self._draw_species_row(species, label, x, y, width)
        return y

    def _draw_settings_row(self, x: int, y: int) -> int:
        button_h = 28
        gap = 8
        available_w = self.panel_rect.width - 36
        button_w = max(126, min(168, (available_w - gap) // 2))

        fullscreen_label = "Window" if self.fullscreen else "Fullscreen"
        self.fullscreen_button_rect = pygame.Rect(x, y, button_w, button_h)
        self._draw_button(self.fullscreen_button_rect, fullscreen_label)

        projection_label = f"View: {self.projection_mode.upper()}"
        self.projection_button_rect = pygame.Rect(x + button_w + gap, y, button_w, button_h)
        self._draw_button(self.projection_button_rect, projection_label)

        y += button_h + 7
        overlay_label = f"Life: {self.life_overlay_mode}"
        self.life_overlay_button_rect = pygame.Rect(x, y, button_w, button_h)
        self._draw_button(self.life_overlay_button_rect, overlay_label)

        weather_label = f"Weather: {self.weather_overlay_mode}"
        self.weather_overlay_button_rect = pygame.Rect(x + button_w + gap, y, button_w, button_h)
        self._draw_button(self.weather_overlay_button_rect, weather_label)
        return y + button_h

    def _draw_checkbox_row(self, rect: pygame.Rect, label: str, checked: bool) -> None:
        hovered = rect.collidepoint(pygame.mouse.get_pos())
        fill = (30, 36, 50) if hovered else (22, 26, 36)
        pygame.draw.rect(self.screen, fill, rect, border_radius=6)
        box = pygame.Rect(rect.left + 5, rect.top + 3, 16, 16)
        pygame.draw.rect(self.screen, (24, 29, 40), box, border_radius=3)
        pygame.draw.rect(self.screen, (96, 112, 145), box, 1, border_radius=3)
        if checked:
            pygame.draw.line(self.screen, (152, 224, 170), (box.left + 3, box.centery), (box.left + 7, box.bottom - 4), 2)
            pygame.draw.line(self.screen, (152, 224, 170), (box.left + 7, box.bottom - 4), (box.right - 3, box.top + 4), 2)
        self._draw_text(label, rect.left + 28, rect.top + 4, self.tiny_font, (210, 218, 236))

    def _draw_button(self, rect: pygame.Rect, label: str) -> None:
        mouse_pos = pygame.mouse.get_pos()
        hovered = rect.collidepoint(mouse_pos)
        fill = (48, 58, 82) if hovered else (38, 44, 62)
        border = (132, 146, 178) if hovered else (86, 96, 122)
        pygame.draw.rect(self.screen, fill, rect, border_radius=7)
        pygame.draw.rect(self.screen, border, rect, 1, border_radius=7)
        text = self.small_font.render(label, True, (230, 234, 244))
        text_pos = (
            rect.centerx - text.get_width() // 2,
            rect.centery - text.get_height() // 2,
        )
        self.screen.blit(text, text_pos)

    def _draw_current_layer_legend(self, x: int, y: int, width: int | None = None) -> int:
        legend = LAYER_LEGENDS[self.current_layer]
        max_bar_w = width if width is not None else max(140, self.panel_rect.width - 44)

        y = self._draw_section_title("Layer legend", x, y, max_bar_w, key="legend")
        if self._is_collapsed("legend"):
            return y
        self._draw_text(legend.title, x, y, self.small_font, (220, 226, 240))
        y += 16
        for line in legend.description:
            self._draw_text(line, x, y, self.tiny_font, (178, 186, 206))
            y += 13

        if self._should_apply_life_overlay(self.current_layer, self.life_overlay_mode):
            y += 2
            self._draw_text(
                f"+ life overlay: {self.life_overlay_mode}",
                x,
                y,
                self.tiny_font,
                (132, 214, 154),
            )
            y += 13
        if self._should_apply_weather_overlay(self.current_layer, self.weather_overlay_mode):
            y += 2
            self._draw_text(
                f"+ weather overlay: {self.weather_overlay_mode}",
                x,
                y,
                self.tiny_font,
                (170, 198, 232),
            )
            y += 13

        y += 5
        if legend.categories:
            columns = 2
            col_w = max_bar_w // columns
            for index, (label, color) in enumerate(legend.categories):
                row = index // columns
                col = index % columns
                sx = x + col * col_w
                sy = y + row * 18
                pygame.draw.rect(self.screen, color, pygame.Rect(sx, sy + 2, 12, 12))
                pygame.draw.rect(self.screen, (90, 96, 116), pygame.Rect(sx, sy + 2, 12, 12), 1)
                self._draw_text(label, sx + 18, sy, self.tiny_font, (185, 192, 210))
            y += ((len(legend.categories) + columns - 1) // columns) * 18
        elif legend.colors:
            bar = pygame.Rect(x, y, max_bar_w, 14)
            self._draw_gradient_bar(bar, legend.colors)
            pygame.draw.rect(self.screen, (90, 96, 116), bar, 1)
            y += 18
            if legend.labels:
                if len(legend.labels) == 2:
                    self._draw_text(legend.labels[0], bar.left, y, self.tiny_font, (185, 192, 210))
                    right_text = self.tiny_font.render(legend.labels[1], True, (185, 192, 210))
                    self.screen.blit(right_text, (bar.right - right_text.get_width(), y))
                elif len(legend.labels) == 3:
                    self._draw_text(legend.labels[0], bar.left, y, self.tiny_font, (185, 192, 210))
                    mid_text = self.tiny_font.render(legend.labels[1], True, (185, 192, 210))
                    self.screen.blit(mid_text, (bar.centerx - mid_text.get_width() // 2, y))
                    right_text = self.tiny_font.render(legend.labels[2], True, (185, 192, 210))
                    self.screen.blit(right_text, (bar.right - right_text.get_width(), y))
                y += 16
        return y

    def _draw_gradient_bar(self, rect: pygame.Rect, colors: tuple[Color, ...]) -> None:
        if len(colors) == 2:
            low, high = colors
            for dx in range(rect.width):
                t = dx / max(1, rect.width - 1)
                color = _lerp_color(low, high, t)
                pygame.draw.line(self.screen, color, (rect.left + dx, rect.top), (rect.left + dx, rect.bottom - 1))
        elif len(colors) == 3:
            low, mid, high = colors
            for dx in range(rect.width):
                t = dx / max(1, rect.width - 1)
                if t <= 0.5:
                    color = _lerp_color(low, mid, t / 0.5)
                else:
                    color = _lerp_color(mid, high, (t - 0.5) / 0.5)
                pygame.draw.line(self.screen, color, (rect.left + dx, rect.top), (rect.left + dx, rect.bottom - 1))
        else:
            pygame.draw.rect(self.screen, colors[0] if colors else (90, 96, 116), rect)

    def _draw_text(self, line: str, x: int, y: int, font: pygame.font.Font, color: Color) -> None:
        text = font.render(line, True, color)
        self.screen.blit(text, (x, y))

    def _should_apply_life_overlay(self, layer: LayerName, mode: OverlayMode) -> bool:
        return should_apply_life_overlay(layer, mode)

    def _should_apply_weather_overlay(self, layer: LayerName, mode: WeatherOverlayMode) -> bool:
        return should_apply_weather_overlay(layer, mode)

    def _season_position_label(self) -> str:
        year, day, period = season_position(self.planet)
        return f"Y{year} d{day}/{period}"

    def _season_label(self) -> str:
        return season_label(self.planet)

    def _weather_mean_label(self) -> str:
        cloud_alpha, rain_alpha, _lightning, _cloud_rgb = _atmosphere_visual_fields(self.planet)
        return f"{float(cloud_alpha.mean()):.2f}/{float(rain_alpha.mean()):.2f}"

    def _land_biomass_share(self) -> float:
        total = float(self.planet.biomass.sum())
        if total <= 1e-9:
            return 0.0
        land_total = float(self.planet.biomass[self.planet.land].sum())
        return 100.0 * land_total / total

    def _ocean_biomass_share_label(self) -> str:
        if self.planet.total_biomass <= 0.0:
            return "0.0%"
        return f"{100.0 - self._land_biomass_share():.1f}%"

    def _save_screenshot(self) -> None:
        suffixes: list[str] = []
        if self.life_overlay_mode != "off":
            suffixes.append(f"life_{self.life_overlay_mode}")
        if self.weather_overlay_mode != "off":
            suffixes.append(f"weather_{self.weather_overlay_mode}")
        overlay_suffix = "" if not suffixes else "_" + "_".join(suffixes)
        filename = f"planet_seed_{self.planet.config.seed}_{self.current_layer}{overlay_suffix}.png"
        pygame.image.save(self.screen, filename)
        print(f"Saved screenshot: {filename}")


def _lerp_color(low: Color, high: Color, t: float) -> Color:
    return tuple(int(a * (1.0 - t) + b * t) for a, b in zip(low, high))  # type: ignore[return-value]



def _smoothstep(edge0: float, edge1: float, x: float) -> float:
    if edge0 == edge1:
        return 1.0 if x >= edge1 else 0.0
    t = float(np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


def season_position(planet: Planet) -> tuple[int, int, int]:
    period = max(10, int(planet.config.seasonal_period_ticks))
    year = int(planet.tick // period) + 1
    day = int(planet.tick % period) + 1
    return year, day, period


def season_label(planet: Planet) -> str:
    _year, day, period = season_position(planet)
    phase = day / max(1, period)
    # The model has opposite hemispheric seasons. This label is a compact
    # observer hint, not a physical calendar.
    if phase < 0.125 or phase >= 0.875:
        return "equinox / transition"
    if phase < 0.375:
        return "north cooling / south warming"
    if phase < 0.625:
        return "solstice-like peak"
    return "north warming / south cooling"


def geological_intro_stage(progress: float) -> GeologicalIntroStage:
    """Return the narrative stage for the optional formation intro."""
    p = float(np.clip(progress, 0.0, 1.0))
    if p < 0.14:
        return GeologicalIntroStage(
            "Void / stellar nursery",
            ("A quiet field of dust and light", "begins to gather in the dark."),
        )
    if p < 0.30:
        return GeologicalIntroStage(
            "Cloud collapse",
            ("The cloud folds inward, bright", "filaments falling toward a core."),
        )
    if p < 0.48:
        return GeologicalIntroStage(
            "Explosive volcanic world",
            ("The young surface burns, cracks", "and throws fire into the sky."),
        )
    if p < 0.62:
        return GeologicalIntroStage(
            "Smoke and primordial clouds",
            ("Ash, steam and a thick atmosphere", "hide the cooling crust below."),
        )
    if p < 0.76:
        return GeologicalIntroStage(
            "Condensation / long rains",
            ("Clouds collapse into rain; basins", "begin to collect the first seas."),
        )
    if p < 0.90:
        return GeologicalIntroStage(
            "Oceans and continents emerge",
            ("Blue water finds the low places", "while highlands harden into land."),
        )
    return GeologicalIntroStage(
        "Young stable planet",
        ("The previewed world is ready", "for proto-ecology to begin."),
    )


def render_geological_intro_layer(planet: Planet, progress: float) -> np.ndarray:
    """Render a deterministic, visual-only formation prelude frame.

    This is intentionally cinematic, not a geophysical model. It uses the final
    generated planet fields as targets, then tells a deterministic formation
    story: void, cloud collapse, magma, smoke/clouds, rain, and final reveal.
    The function must never mutate simulation state.
    """
    p = float(np.clip(progress, 0.0, 1.0))
    elevation = np.clip(planet.elevation, 0.0, 1.0).astype(np.float32)
    volcanism = np.clip(planet.volcanism, 0.0, 1.0).astype(np.float32)
    water = np.clip(planet.water, 0.0, 1.0).astype(np.float32)
    final_biome = _render_biome(planet).astype(np.float32)

    h, w = planet.shape
    yy = np.linspace(-1.0, 1.0, h, dtype=np.float32)[:, None]
    xx = np.linspace(-1.0, 1.0, w, dtype=np.float32)[None, :]
    radius = np.sqrt((xx * 0.92) ** 2 + (yy * 1.22) ** 2)
    angle = np.arctan2(yy, xx + 1e-6)
    wave = 0.5 + 0.5 * np.sin((xx * 19.0 + yy * 13.0 + p * 10.0) * np.pi)
    spiral = 0.5 + 0.5 * np.sin(8.5 * angle + 17.0 * radius - p * 18.0)
    fine = 0.5 + 0.5 * np.sin((xx * 73.0 - yy * 41.0 + p * 26.0) * np.pi)

    volcanic_setting = float(np.clip(planet.config.volcanic_activity_fraction / 0.24, 0.0, 1.0))
    heat_setting = float(np.clip((planet.config.equator_temperature_c - 8.0) / 40.0, 0.0, 1.0))
    sea_setting = float(np.clip((planet.config.sea_level - 0.34) / 0.34, 0.0, 1.0))

    # 1) Deep void + magical accretion cloud. The core narrows as p advances.
    collapse = _smoothstep(0.04, 0.30, p)
    cloud_radius = 1.18 - 0.70 * collapse
    cloud = np.exp(-(radius ** 2) / max(0.06, cloud_radius ** 2))
    filaments = np.clip(0.50 * cloud + 0.32 * cloud * spiral + 0.22 * fine * cloud, 0.0, 1.0)
    star_sparkles = (fine > 0.985).astype(np.float32) * (1.0 - 0.55 * collapse)
    void_rgb = np.zeros((h, w, 3), dtype=np.float32)
    void_rgb[:, :] = np.array((2, 3, 9), dtype=np.float32)
    nebula_color = np.array((96, 72, 190), dtype=np.float32) * (1.0 - collapse) + np.array((255, 154, 74), dtype=np.float32) * collapse
    core_color = np.array((255, 220, 150), dtype=np.float32)
    void_rgb += filaments[..., None] * nebula_color * (0.35 + 0.75 * collapse)
    void_rgb += np.clip(cloud * collapse * 1.4, 0.0, 1.0)[..., None] * core_color * 0.38
    void_rgb += star_sparkles[..., None] * np.array((190, 210, 255), dtype=np.float32)

    # 2) Magma / volcanic phase. Final elevation and volcanism steer where fire
    # and hardening crust become visible.
    magma_heat = np.clip(
        0.46 + 0.30 * elevation + 0.56 * volcanism + 0.18 * heat_setting + 0.16 * wave,
        0.0,
        1.0,
    )
    magma = _three_color_gradient(magma_heat, (32, 6, 5), (190, 45, 18), (255, 218, 96)).astype(np.float32)
    crack_pattern = np.clip((volcanism * 1.7 + wave * 0.45 + fine * 0.22) - 0.62, 0.0, 1.0)
    magma += crack_pattern[..., None] * np.array((255, 170, 42), dtype=np.float32) * (0.55 + 0.45 * volcanic_setting)
    crust_mask = np.clip((elevation - (0.62 - 0.08 * _smoothstep(0.32, 0.56, p))) * 3.0, 0.0, 1.0)[..., None]
    magma = magma * (1.0 - crust_mask * 0.62) + np.array((58, 48, 44), dtype=np.float32) * (crust_mask * 0.62)

    # 3) Smoke / ash / cloud deck, visually tied to volcanism and water setting.
    smoke_noise = np.clip(0.42 * wave + 0.40 * spiral + 0.28 * fine + 0.38 * volcanism, 0.0, 1.0)
    smoke_alpha = np.clip((smoke_noise - 0.28) * (1.00 + 0.55 * volcanic_setting), 0.0, 0.86)
    smoke_color = np.array((72, 70, 82), dtype=np.float32)
    smoky = magma * (1.0 - smoke_alpha[..., None]) + smoke_color * smoke_alpha[..., None]
    cloud_alpha = np.clip((0.55 * wave + 0.55 * fine + 0.25 * water + 0.22 * sea_setting) - 0.38, 0.0, 0.82)
    cloud_color = np.array((184, 192, 202), dtype=np.float32)
    cloudy = smoky * (1.0 - cloud_alpha[..., None] * 0.72) + cloud_color * (cloud_alpha[..., None] * 0.72)

    # 4) Rain / condensation: dark curtains and diagonal streaks over a muted
    # crust, with oceans slowly filling low places.
    ocean_birth = np.clip(water * (0.08 + 0.92 * _smoothstep(0.58, 0.82, p + 0.04 * sea_setting)), 0.0, 1.0)
    muted_land = _three_color_gradient(elevation, (52, 46, 43), (110, 98, 76), (184, 178, 152)).astype(np.float32)
    muted_water = _gradient(ocean_birth, (30, 42, 58), (42, 115, 178)).astype(np.float32)
    rain_world = muted_land * (1.0 - ocean_birth[..., None]) + muted_water * ocean_birth[..., None]
    rain_lines = (0.5 + 0.5 * np.sin((xx * 38.0 + yy * 108.0 + p * 130.0) * np.pi)).astype(np.float32)
    rain_mask = np.clip((rain_lines - 0.78) * 4.2, 0.0, 1.0) * (0.35 + 0.65 * sea_setting)
    rain_world = rain_world * (1.0 - rain_mask[..., None] * 0.35) + np.array((170, 190, 215), dtype=np.float32) * rain_mask[..., None] * 0.42
    rain_world = rain_world * 0.84 + np.array((18, 24, 34), dtype=np.float32) * 0.16

    # 5) Young planet reveal with residual volcanic glow and fading cloud bands.
    young = final_biome.copy()
    hot_spots = np.clip(volcanism * (0.55 + 0.62 * volcanic_setting), 0.0, 1.0)[..., None]
    young = young * (1.0 - hot_spots * 0.32) + np.array((255, 112, 44), dtype=np.float32) * (hot_spots * 0.32)
    residual_cloud = np.clip((cloud_alpha - 0.18) * (1.0 - _smoothstep(0.78, 1.0, p)), 0.0, 0.62)
    young = young * (1.0 - residual_cloud[..., None] * 0.36) + np.array((210, 216, 222), dtype=np.float32) * (residual_cloud[..., None] * 0.36)

    if p < 0.08:
        # Full black fade-in into space.
        t = _smoothstep(0.00, 0.08, p)
        rgb = void_rgb * t
    elif p < 0.30:
        rgb = void_rgb
    elif p < 0.48:
        t = _smoothstep(0.30, 0.48, p)
        rgb = void_rgb * (1.0 - t) + magma * t
    elif p < 0.62:
        t = _smoothstep(0.48, 0.62, p)
        rgb = magma * (1.0 - t) + cloudy * t
    elif p < 0.76:
        t = _smoothstep(0.62, 0.76, p)
        rgb = cloudy * (1.0 - t) + rain_world * t
    elif p < 0.90:
        t = _smoothstep(0.76, 0.90, p)
        rgb = rain_world * (1.0 - t) + young * t
    else:
        t = _smoothstep(0.90, 1.00, p)
        rgb = young * (1.0 - t) + final_biome * t

    # A cinematic vignette keeps early phases cosmic and focuses the collapse.
    vignette = np.clip(1.0 - 0.48 * radius * (1.0 - _smoothstep(0.72, 1.0, p)), 0.54, 1.0)
    rgb *= vignette[..., None]
    return np.clip(rgb, 0, 255).astype(np.uint8)




def render_star_background(target_size: tuple[int, int], *, seed: int, rotation: float = 0.0) -> np.ndarray:
    """Render a deterministic slow-rotating star field for the 3D projection.

    This is deliberately visual-only: stars are a sky reference, not a stellar
    mechanics simulation. The caller drives ``rotation``; the viewer uses one
    full sky turn per local year.
    """
    target_w, target_h = max(1, int(target_size[0])), max(1, int(target_size[1]))
    out = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    # Slight vertical gradient keeps space from feeling perfectly flat.
    y_grad = np.linspace(0.0, 1.0, target_h, dtype=np.float32)[:, None]
    base = np.array((5, 7, 14), dtype=np.float32)
    glow = np.array((10, 12, 24), dtype=np.float32)
    out[:, :] = np.clip(base + glow * (0.28 + 0.35 * (1.0 - y_grad)), 0, 255).astype(np.uint8)[:, None, :]

    rng = np.random.default_rng((int(seed) ^ 0xA571FE) & 0xFFFFFFFF)
    area = target_w * target_h
    star_count = int(np.clip(area / 3200, 140, 900))
    cx = (target_w - 1) / 2.0
    cy = (target_h - 1) / 2.0
    max_radius = float(np.hypot(target_w, target_h)) * 0.72

    # Generate in polar space around the screen center so a yearly rotation is
    # perceptible without requiring real 3D camera math.
    radii = np.sqrt(rng.random(star_count)) * max_radius
    angles = rng.random(star_count) * 2.0 * np.pi + float(rotation)
    xs = np.rint(cx + radii * np.cos(angles)).astype(np.int32)
    ys = np.rint(cy + radii * np.sin(angles)).astype(np.int32)
    brightness = rng.uniform(0.38, 1.0, star_count)
    sizes = rng.choice(np.array([1, 1, 1, 1, 2, 2, 3]), size=star_count, p=[0.34, 0.24, 0.16, 0.10, 0.08, 0.06, 0.02])
    palettes = np.array(
        [
            (210, 220, 255),
            (245, 248, 255),
            (255, 236, 195),
            (190, 210, 255),
        ],
        dtype=np.float32,
    )
    colors = palettes[rng.integers(0, len(palettes), size=star_count)]

    for x, y, b, size, color in zip(xs, ys, brightness, sizes, colors):
        if x < -2 or y < -2 or x >= target_w + 2 or y >= target_h + 2:
            continue
        main = np.clip(color * (0.42 + 0.58 * b), 0, 255).astype(np.uint8)
        if 0 <= x < target_w and 0 <= y < target_h:
            out[y, x] = np.maximum(out[y, x], main)
        if size >= 2:
            halo = np.clip(color * (0.16 + 0.18 * b), 0, 255).astype(np.uint8)
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                xx = x + dx
                yy = y + dy
                if 0 <= xx < target_w and 0 <= yy < target_h:
                    out[yy, xx] = np.maximum(out[yy, xx], halo)
        if size >= 3:
            halo2 = np.clip(color * (0.08 + 0.08 * b), 0, 255).astype(np.uint8)
            for dx, dy in ((-2, 0), (2, 0), (0, -2), (0, 2), (-1, -1), (1, -1), (-1, 1), (1, 1)):
                xx = x + dx
                yy = y + dy
                if 0 <= xx < target_w and 0 <= yy < target_h:
                    out[yy, xx] = np.maximum(out[yy, xx], halo2)

    return out

def render_globe_texture(
    texture_rgb: np.ndarray,
    target_size: tuple[int, int],
    *,
    rotation: float = 0.0,
    star_rotation: float | None = None,
    seed: int = 0,
) -> np.ndarray:
    """Project an equirectangular layer texture onto a rotating orthographic globe."""
    target_w, target_h = max(1, int(target_size[0])), max(1, int(target_size[1]))
    tex_h, tex_w = texture_rgb.shape[:2]
    if star_rotation is None:
        out = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        out[:, :] = np.array((8, 10, 18), dtype=np.uint8)
    else:
        out = render_star_background((target_w, target_h), seed=seed, rotation=star_rotation)

    radius = max(2.0, min(target_w, target_h) * 0.46)
    cx = (target_w - 1) / 2.0
    cy = (target_h - 1) / 2.0
    yy, xx = np.indices((target_h, target_w), dtype=np.float32)
    nx = (xx - cx) / radius
    ny = (yy - cy) / radius
    r2 = nx * nx + ny * ny
    mask = r2 <= 1.0
    if not np.any(mask):
        return out

    nz = np.sqrt(np.clip(1.0 - r2, 0.0, 1.0))
    lon = np.arctan2(nx, nz) + float(rotation)
    lat = np.arcsin(np.clip(-ny, -1.0, 1.0))
    u = (lon / (2.0 * np.pi)) % 1.0
    v = np.clip(0.5 - lat / np.pi, 0.0, 0.999999)
    src_x = np.clip((u * tex_w).astype(np.int32), 0, tex_w - 1)
    src_y = np.clip((v * tex_h).astype(np.int32), 0, tex_h - 1)

    sampled = texture_rgb[src_y, src_x].astype(np.float32)
    light = np.clip(0.38 + 0.62 * (0.72 * nz + 0.20 * nx - 0.10 * ny), 0.20, 1.08)
    limb = np.clip((1.0 - r2) * 7.0, 0.0, 1.0)
    shaded = sampled * light[..., None]
    shaded = shaded * (0.78 + 0.22 * limb[..., None])
    out[mask] = np.clip(shaded[mask], 0, 255).astype(np.uint8)

    # Soft atmospheric rim around the visible disk.
    rim = (r2 > 0.965) & (r2 <= 1.04)
    if np.any(rim):
        rim_alpha = np.clip((1.04 - r2[rim]) / 0.075, 0.0, 1.0)[:, None]
        rim_color = np.array((72, 104, 155), dtype=np.float32)
        current = out[rim].astype(np.float32)
        out[rim] = np.clip(current * (1.0 - 0.45 * rim_alpha) + rim_color * (0.45 * rim_alpha), 0, 255).astype(np.uint8)
    return out


def render_globe_scalar_overlay(
    field: np.ndarray,
    color: Color,
    target_size: tuple[int, int],
    *,
    rotation: float = 0.0,
) -> pygame.Surface:
    """Project one scalar field to a transparent globe overlay surface."""
    target_w, target_h = max(1, int(target_size[0])), max(1, int(target_size[1]))
    h, w = field.shape
    rgba = np.zeros((target_h, target_w, 4), dtype=np.uint8)

    radius = max(2.0, min(target_w, target_h) * 0.46)
    cx = (target_w - 1) / 2.0
    cy = (target_h - 1) / 2.0
    yy, xx = np.indices((target_h, target_w), dtype=np.float32)
    nx = (xx - cx) / radius
    ny = (yy - cy) / radius
    r2 = nx * nx + ny * ny
    mask = r2 <= 1.0
    if np.any(mask):
        nz = np.sqrt(np.clip(1.0 - r2, 0.0, 1.0))
        lon = np.arctan2(nx, nz) + float(rotation)
        lat = np.arcsin(np.clip(-ny, -1.0, 1.0))
        u = (lon / (2.0 * np.pi)) % 1.0
        v = np.clip(0.5 - lat / np.pi, 0.0, 0.999999)
        src_x = np.clip((u * w).astype(np.int32), 0, w - 1)
        src_y = np.clip((v * h).astype(np.int32), 0, h - 1)
        sampled = np.clip(field[src_y, src_x], 0.0, 1.0)
        alpha = np.clip(34 + 198 * np.sqrt(sampled / max(float(field.max()), 0.025)), 0, 226).astype(np.uint8)
        alpha[sampled <= 0.004] = 0
        rgba[..., :3] = np.array(color, dtype=np.uint8)
        rgba[..., 3] = np.where(mask, alpha, 0).astype(np.uint8)

    surface = pygame.Surface((target_w, target_h), pygame.SRCALPHA)
    pixels = pygame.surfarray.pixels3d(surface)
    pixels[:] = np.transpose(rgba[..., :3], (1, 0, 2))
    del pixels
    pixels_alpha = pygame.surfarray.pixels_alpha(surface)
    pixels_alpha[:] = np.transpose(rgba[..., 3], (1, 0))
    del pixels_alpha
    return surface

def render_layer(
    planet: Planet,
    layer: LayerName,
    overlay_mode: OverlayMode = "off",
    weather_overlay_mode: WeatherOverlayMode = "off",
) -> np.ndarray:
    base = _render_base_layer(planet, layer)
    if should_apply_weather_overlay(layer, weather_overlay_mode):
        base = apply_weather_overlay(planet, base, weather_overlay_mode)
    if should_apply_life_overlay(layer, overlay_mode):
        return apply_life_overlay(planet, base, overlay_mode)
    return base


def should_apply_life_overlay(layer: LayerName, overlay_mode: OverlayMode) -> bool:
    return overlay_mode != "off" and layer not in LIFE_LAYER_NAMES and layer not in ATMOSPHERE_LAYER_NAMES


def should_apply_weather_overlay(layer: LayerName, weather_overlay_mode: WeatherOverlayMode) -> bool:
    return weather_overlay_mode != "off" and layer == "biome"


def apply_life_overlay(planet: Planet, base_rgb: np.ndarray, overlay_mode: OverlayMode) -> np.ndarray:
    """Blend abstract life information onto any non-life layer.

    This keeps the pretty biome/abiotic maps readable while making expanding life
    visible without switching to a dedicated life layer.
    """
    biomass = np.clip(planet.biomass, 0.0, 1.0)
    mask = biomass > 0.012
    if not np.any(mask):
        return base_rgb

    rgb = base_rgb.astype(np.float32).copy()
    alpha = np.clip(0.12 + 0.62 * np.sqrt(biomass), 0.0, 0.68)[..., None]
    land_mask = planet.land.astype(bool)

    if overlay_mode == "dominant" and planet.species:
        overlay = np.zeros_like(rgb)
        for index, species in enumerate(planet.species):
            species_mask = mask & (planet.dominant_species_index == index)
            if np.any(species_mask):
                overlay[species_mask] = np.array(species.color, dtype=np.float32)
        # Barren/no-dominant cells stay untouched.
        valid = mask & (overlay.sum(axis=2) > 0.0)
        rgb[valid] = rgb[valid] * (1.0 - alpha[valid]) + overlay[valid] * alpha[valid]
        return np.clip(rgb, 0, 255).astype(np.uint8)

    if overlay_mode == "biomass":
        # Ocean and land biomass deliberately use slightly different tones so
        # terrestrial colonization is visible on the biome map.
        ocean_glow = np.array((70, 245, 130), dtype=np.float32)
        land_glow = np.array((178, 238, 92), dtype=np.float32)
        ocean_mask = mask & ~land_mask
        terrestrial_mask = mask & land_mask
        if np.any(ocean_mask):
            rgb[ocean_mask] = rgb[ocean_mask] * (1.0 - alpha[ocean_mask]) + ocean_glow * alpha[ocean_mask]
        if np.any(terrestrial_mask):
            land_alpha = np.clip(alpha[..., 0] * 1.12 + 0.05, 0.0, 0.78)[..., None]
            rgb[terrestrial_mask] = (
                rgb[terrestrial_mask] * (1.0 - land_alpha[terrestrial_mask])
                + land_glow * land_alpha[terrestrial_mask]
            )
        return np.clip(rgb, 0, 255).astype(np.uint8)

    return base_rgb


def apply_weather_overlay(planet: Planet, base_rgb: np.ndarray, weather_overlay_mode: WeatherOverlayMode) -> np.ndarray:
    """Blend visual-only clouds/rain on top of the biome map.

    Weather remains a consequence/visualization of existing fields; it does not
    feed back into humidity, nutrients or fertility yet.
    """
    if weather_overlay_mode == "off":
        return base_rgb

    cloud_alpha, rain_alpha, lightning, cloud_rgb = _atmosphere_visual_fields(planet)
    rgb = base_rgb.astype(np.float32).copy()
    land_boost = np.where(planet.land, 1.18, 0.88).astype(np.float32)

    def blend_clouds(current: np.ndarray, *, strength: float) -> np.ndarray:
        shaped = np.power(np.clip(cloud_alpha, 0.0, 1.0), 1.35)
        alpha = np.clip((0.015 + strength * shaped) * land_boost, 0.0, strength + 0.05)[..., None]
        mixed = current * (1.0 - alpha) + cloud_rgb * alpha
        # Keep clouds patchy: clear cells stay clear instead of applying a full-map filter.
        shadow = np.clip(shaped * 0.045 * land_boost, 0.0, 0.075)[..., None]
        return mixed * (1.0 - shadow)

    def blend_rain(current: np.ndarray, *, strength: float) -> np.ndarray:
        h, w = planet.shape
        yy = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
        xx = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :]
        tick = float(planet.tick)
        streaks = 0.5 + 0.5 * np.sin((xx * 62.0 + yy * 150.0 + tick * 0.37) * np.pi)
        streaks2 = 0.5 + 0.5 * np.sin((xx * 27.0 + yy * 96.0 + tick * 0.19) * np.pi)
        streak_mask = np.clip((0.72 * streaks + 0.28 * streaks2 - 0.72) * 5.0, 0.0, 1.0)
        visible_rain = np.clip(rain_alpha * (0.25 + 0.75 * streak_mask) * land_boost, 0.0, 1.0)
        rain_color = np.array((188, 211, 236), dtype=np.float32)
        mixed = current * (1.0 - visible_rain[..., None] * strength) + rain_color * (visible_rain[..., None] * strength)
        mixed *= 1.0 - np.clip(rain_alpha * 0.055 * land_boost, 0.0, 0.09)[..., None]
        if float(lightning.max()) > 0.0:
            flash = np.clip(lightning[..., None] * 0.58, 0.0, 0.58)
            mixed = mixed * (1.0 - flash) + np.array((235, 244, 255), dtype=np.float32) * flash
        return mixed

    if weather_overlay_mode == "clouds":
        return np.clip(blend_clouds(rgb, strength=0.30), 0, 255).astype(np.uint8)

    if weather_overlay_mode == "rain":
        rgb = blend_clouds(rgb, strength=0.14)
        rgb = blend_rain(rgb, strength=0.40)
        return np.clip(rgb, 0, 255).astype(np.uint8)

    if weather_overlay_mode == "all":
        rgb = blend_clouds(rgb, strength=0.24)
        rgb = blend_rain(rgb, strength=0.34)
        return np.clip(rgb, 0, 255).astype(np.uint8)

    return base_rgb

def _render_base_layer(planet: Planet, layer: LayerName) -> np.ndarray:
    if layer == "biome":
        return _render_biome(planet)
    if layer == "elevation":
        return _gradient(planet.elevation, (15, 28, 64), (245, 245, 230))
    if layer == "temperature":
        temp_norm = np.clip((planet.temperature_c + 25.0) / 65.0, 0.0, 1.0)
        return _three_color_gradient(temp_norm, (35, 70, 155), (230, 225, 170), (180, 45, 35))
    if layer == "water":
        return _gradient(planet.water, (25, 25, 35), (40, 130, 220))
    if layer == "humidity":
        return _gradient(planet.humidity, (120, 95, 45), (45, 165, 95))
    if layer == "light":
        return _gradient(planet.light, (20, 18, 45), (255, 235, 140))
    if layer == "clouds":
        return _render_clouds(planet)
    if layer == "rain":
        return _render_rain(planet)
    if layer == "volcanism":
        return _three_color_gradient(planet.volcanism, (18, 16, 24), (92, 46, 72), (255, 116, 45))
    if layer == "minerals":
        return _three_color_gradient(planet.minerals, (24, 22, 22), (115, 102, 82), (230, 220, 185))
    if layer == "nutrients":
        return _three_color_gradient(planet.nutrients, (28, 24, 18), (105, 115, 52), (210, 190, 78))
    if layer == "chemical_energy":
        return _three_color_gradient(planet.chemical_energy, (16, 18, 34), (80, 55, 150), (245, 175, 80))
    if layer == "toxicity":
        return _three_color_gradient(planet.toxicity, (18, 22, 20), (90, 70, 130), (210, 55, 85))
    if layer == "fertility":
        return _three_color_gradient(planet.fertility, (28, 22, 18), (65, 120, 68), (165, 220, 92))
    if layer == "dead_matter":
        return _render_dead_matter(planet)
    if layer == "biomass":
        return _three_color_gradient(planet.biomass, (10, 14, 12), (35, 130, 65), (180, 245, 120))
    if layer == "diversity":
        return _three_color_gradient(planet.diversity, (16, 14, 26), (60, 105, 160), (230, 210, 110))
    if layer == "dominant_life":
        return _render_dominant_life(planet)
    if layer == "biotic_pressure":
        return _render_biotic_pressure(planet)
    if layer == "migration_pressure":
        return _render_migration_pressure(planet)
    if layer == "isolation_pressure":
        return _render_isolation_pressure(planet)
    raise ValueError(f"Unknown layer: {layer}")


def _atmosphere_visual_fields(planet: Planet) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return procedural cloud, rain, lightning and cloud-color fields.

    This is visual-only weather. It is deterministic from the current tick,
    planet fields and setup values, so it drifts while the simulation runs but
    never changes ecology/resource dynamics.
    """
    h, w = planet.shape
    yy = np.linspace(-1.0, 1.0, h, dtype=np.float32)[:, None]
    xx = np.linspace(-1.0, 1.0, w, dtype=np.float32)[None, :]
    tick = float(planet.tick)
    period = float(max(10, planet.config.seasonal_period_ticks))
    phase = 2.0 * np.pi * ((tick % period) / period)

    # Smooth moving bands. A few sine fields are cheaper than storing a full
    # weather simulation and still read well as slow atmospheric drift.
    drift = tick / period
    band_a = 0.5 + 0.5 * np.sin((xx * 7.0 + yy * 2.1 + drift * 2.6) * np.pi)
    band_b = 0.5 + 0.5 * np.sin((xx * -4.4 + yy * 8.0 + drift * -1.7) * np.pi)
    fine = 0.5 + 0.5 * np.sin((xx * 37.0 + yy * 19.0 + drift * 10.5) * np.pi)
    swirl = 0.5 + 0.5 * np.sin((xx * 11.0 - yy * 5.0 + 0.85 * np.sin(phase)) * np.pi)
    pattern = np.clip(0.38 * band_a + 0.28 * band_b + 0.22 * swirl + 0.12 * fine, 0.0, 1.0)

    water_access = np.where(planet.land, planet.humidity, 0.56 + 0.44 * planet.water).astype(np.float32)
    warmth = np.clip((planet.temperature_c + 12.0) / 46.0, 0.0, 1.0).astype(np.float32)
    seasonal_moisture = (0.5 + 0.5 * np.sin(phase - yy * np.pi)).astype(np.float32)
    # Slow storm envelope: rain belts wax/wane instead of staying uniformly
    # active. The spatial term keeps different regions out of phase.
    storm_cycle = 0.5 + 0.5 * np.sin((phase * 1.55) + xx * 3.4 - yy * 2.2 + tick * 0.0021)
    storm_cycle = (0.30 + 0.70 * storm_cycle).astype(np.float32)
    sea_setting = float(np.clip((planet.config.sea_level - 0.34) / 0.34, 0.0, 1.0))
    volcanic_setting = float(np.clip(planet.config.volcanic_activity_fraction / 0.24, 0.0, 1.0))

    source = (
        0.42 * water_access
        + 0.20 * planet.humidity
        + 0.13 * planet.water
        + 0.11 * seasonal_moisture
        + 0.10 * planet.fertility
        + 0.08 * sea_setting
        + 0.06 * planet.volcanism * volcanic_setting
        + 0.07 * planet.land.astype(np.float32) * planet.humidity
    )
    raw_clouds = source + 0.52 * (pattern - 0.50)
    # Keep clouds patchy enough to read as clouds, not as a full-map filter.
    cloud_alpha = np.clip((raw_clouds - 0.43) * 2.65, 0.0, 1.0).astype(np.float32)
    cloud_alpha = np.power(cloud_alpha, 1.18).astype(np.float32)
    cloud_alpha = _soften_visual_field(cloud_alpha, passes=1)

    storm_potential = cloud_alpha * (0.42 + 0.58 * water_access) * (0.30 + 0.70 * warmth)
    rain_alpha = np.clip((storm_potential - 0.28) * 2.55, 0.0, 1.0).astype(np.float32)
    rain_alpha *= (0.48 + 0.52 * sea_setting) * storm_cycle
    rain_alpha = _soften_visual_field(rain_alpha, passes=1)

    # Deterministic sparse storm flashes. They appear only when rain and
    # volcanism/storm potential are high enough, and change with tick.
    flash_wave = 0.5 + 0.5 * np.sin((xx * 91.0 + yy * 47.0 + tick * 0.137) * np.pi)
    storm_core = np.clip(rain_alpha * (0.35 + 0.65 * planet.volcanism), 0.0, 1.0)
    lightning = np.where((flash_wave > 0.992) & (storm_core > 0.28), storm_core, 0.0).astype(np.float32)
    lightning = _soften_visual_field(lightning, passes=1)

    coldness = np.clip((8.0 - planet.temperature_c) / 34.0, 0.0, 1.0)[..., None]
    heat = np.clip((planet.temperature_c - 18.0) / 36.0, 0.0, 1.0)[..., None]
    smoky = np.clip(planet.volcanism * (0.35 + 0.65 * volcanic_setting), 0.0, 1.0)[..., None]
    base = np.array((182, 190, 202), dtype=np.float32)
    cold_color = np.array((214, 226, 238), dtype=np.float32)
    warm_color = np.array((205, 182, 160), dtype=np.float32)
    smoke_color = np.array((116, 105, 116), dtype=np.float32)
    cloud_rgb = base * (1.0 - 0.34 * coldness) + cold_color * (0.34 * coldness)
    cloud_rgb = cloud_rgb * (1.0 - 0.22 * heat) + warm_color * (0.22 * heat)
    cloud_rgb = cloud_rgb * (1.0 - 0.42 * smoky) + smoke_color * (0.42 * smoky)
    return cloud_alpha, rain_alpha.astype(np.float32), lightning, cloud_rgb.astype(np.float32)


def _soften_visual_field(field: np.ndarray, *, passes: int = 1) -> np.ndarray:
    result = field.astype(np.float32)
    for _ in range(max(0, passes)):
        result = (
            result * 0.52
            + 0.12 * np.roll(result, 1, axis=0)
            + 0.12 * np.roll(result, -1, axis=0)
            + 0.12 * np.roll(result, 1, axis=1)
            + 0.12 * np.roll(result, -1, axis=1)
        )
    return np.clip(result, 0.0, 1.0).astype(np.float32)


def _render_clouds(planet: Planet) -> np.ndarray:
    cloud_alpha, _rain_alpha, _lightning, cloud_rgb = _atmosphere_visual_fields(planet)
    biome = _render_biome(planet).astype(np.float32)
    sky_tint = np.array((14, 19, 30), dtype=np.float32)
    base = biome * 0.58 + sky_tint * 0.42
    land_boost = np.where(planet.land, 1.25, 1.0).astype(np.float32)
    alpha = np.clip((0.12 + 0.78 * cloud_alpha) * land_boost, 0.0, 0.86)[..., None]
    rgb = base * (1.0 - alpha) + cloud_rgb * alpha
    # Clear areas remain visibly planetary instead of black.
    rgb = rgb * (0.82 + 0.18 * cloud_alpha[..., None])
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _render_rain(planet: Planet) -> np.ndarray:
    cloud_alpha, rain_alpha, lightning, cloud_rgb = _atmosphere_visual_fields(planet)
    biome = _render_biome(planet).astype(np.float32)
    storm_base = biome * 0.44 + np.array((9, 14, 25), dtype=np.float32) * 0.56
    cloud_mix = np.clip(0.08 + 0.55 * cloud_alpha, 0.0, 0.62)[..., None]
    rgb = storm_base * (1.0 - cloud_mix) + cloud_rgb * cloud_mix

    h, w = planet.shape
    yy = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    xx = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :]
    tick = float(planet.tick)
    # Diagonal curtains, stylized rather than physical.
    streaks = 0.5 + 0.5 * np.sin((xx * 58.0 + yy * 142.0 + tick * 0.34) * np.pi)
    streaks2 = 0.5 + 0.5 * np.sin((xx * 24.0 + yy * 88.0 + tick * 0.21) * np.pi)
    streak_mask = np.clip((0.72 * streaks + 0.28 * streaks2 - 0.73) * 4.7, 0.0, 1.0)
    land_boost = np.where(planet.land, 1.35, 1.0).astype(np.float32)
    visible_rain = np.clip(rain_alpha * (0.32 + 0.68 * streak_mask) * land_boost, 0.0, 1.0)
    rain_color = np.array((170, 195, 226), dtype=np.float32)
    rgb = rgb * (1.0 - visible_rain[..., None] * 0.46) + rain_color * (visible_rain[..., None] * 0.46)
    rgb *= (1.0 - rain_alpha[..., None] * 0.18)

    if float(lightning.max()) > 0.0:
        flash = np.clip(lightning[..., None] * 0.92, 0.0, 0.92)
        rgb = rgb * (1.0 - flash) + np.array((235, 244, 255), dtype=np.float32) * flash
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _render_biotic_pressure(planet: Planet) -> np.ndarray:
    field = np.clip(planet.biotic_pressure, 0.0, 1.0)
    if float(field.max()) <= 1e-7:
        return _three_color_gradient(field, (10, 12, 18), (110, 54, 85), (245, 118, 70))
    nonzero = field[field > 1e-7]
    scale = float(np.quantile(nonzero, 0.985)) if nonzero.size else float(field.max())
    scale = max(scale, 0.025)
    visible = np.clip(field / scale, 0.0, 1.0)
    return _three_color_gradient(visible, (10, 12, 18), (110, 54, 85), (245, 118, 70))


def _render_migration_pressure(planet: Planet) -> np.ndarray:
    field = np.clip(planet.migration_pressure, 0.0, 1.0)
    if float(field.max()) <= 1e-7:
        return _three_color_gradient(field, (8, 10, 18), (42, 118, 150), (120, 240, 210))
    nonzero = field[field > 1e-7]
    scale = float(np.quantile(nonzero, 0.985)) if nonzero.size else float(field.max())
    scale = max(scale, 0.018)
    visible = np.clip(field / scale, 0.0, 1.0)
    return _three_color_gradient(visible, (8, 10, 18), (42, 118, 150), (120, 240, 210))


def _render_isolation_pressure(planet: Planet) -> np.ndarray:
    field = np.clip(planet.isolation_pressure, 0.0, 1.0)
    if float(field.max()) <= 1e-7:
        return _three_color_gradient(field, (12, 10, 18), (94, 70, 150), (255, 218, 92))
    nonzero = field[field > 1e-7]
    scale = float(np.quantile(nonzero, 0.985)) if nonzero.size else float(field.max())
    scale = max(scale, 0.020)
    visible = np.clip(field / scale, 0.0, 1.0)
    return _three_color_gradient(visible, (12, 10, 18), (94, 70, 150), (255, 218, 92))

def _render_dead_matter(planet: Planet) -> np.ndarray:
    # Dead matter is usually a thinner field than living biomass. The simulation
    # keeps raw values in [0, 1], but this view stretches current non-zero debris
    # so crashes/turnover are readable instead of almost black.
    field = np.clip(planet.dead_matter, 0.0, 1.0)
    if float(field.max()) <= 1e-7:
        return _three_color_gradient(field, (20, 16, 12), (115, 76, 38), (225, 170, 80))
    nonzero = field[field > 1e-7]
    scale = float(np.quantile(nonzero, 0.98)) if nonzero.size else float(field.max())
    scale = max(scale, 0.025)
    visible = np.clip(field / scale, 0.0, 1.0)
    return _three_color_gradient(visible, (20, 16, 12), (115, 76, 38), (225, 170, 80))


def _render_dominant_life(planet: Planet) -> np.ndarray:
    h, w = planet.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    if not planet.species:
        return rgb

    # Dark biome background helps barren areas stay readable.
    background = (_render_biome(planet).astype(np.float32) * 0.18).astype(np.uint8)
    rgb[:] = background
    biomass = np.clip(planet.biomass, 0.0, 1.0)
    brightness = (0.25 + 0.75 * biomass)[..., None]
    for index, species in enumerate(planet.species):
        mask = planet.dominant_species_index == index
        if not np.any(mask):
            continue
        color = np.array(species.color, dtype=np.float32)
        rgb[mask] = np.clip(color * brightness[mask], 0, 255).astype(np.uint8)
    return rgb


def _render_biome(planet: Planet) -> np.ndarray:
    h, w = planet.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    # Ocean depth.
    deep = planet.water > 0.45
    shallow = (planet.water > 0.0) & ~deep
    rgb[deep] = (25, 55, 125)
    rgb[shallow] = (42, 105, 165)

    land = planet.land
    cold = planet.temperature_c < 0
    dry = planet.humidity < 0.28
    wet = planet.humidity > 0.62
    high = planet.elevation > 0.78

    rgb[land] = (96, 132, 62)        # temperate land
    rgb[land & dry] = (178, 151, 86) # desert / dry steppe
    rgb[land & wet] = (47, 115, 68)  # wet forest-ish
    rgb[land & cold] = (185, 195, 200)
    rgb[land & high] = (135, 122, 105)
    rgb[land & high & cold] = (230, 235, 238)

    # Coastline highlight.
    coast = planet.land & (
        np.roll(~planet.land, 1, axis=0)
        | np.roll(~planet.land, -1, axis=0)
        | np.roll(~planet.land, 1, axis=1)
        | np.roll(~planet.land, -1, axis=1)
    )
    rgb[coast] = (205, 190, 120)
    return rgb


def _gradient(values: np.ndarray, low: Color, high: Color) -> np.ndarray:
    v = np.clip(values, 0.0, 1.0)[..., None]
    a = np.array(low, dtype=np.float32)
    b = np.array(high, dtype=np.float32)
    return (a * (1.0 - v) + b * v).astype(np.uint8)


def _three_color_gradient(values: np.ndarray, low: Color, mid: Color, high: Color) -> np.ndarray:
    v = np.clip(values, 0.0, 1.0)
    rgb = np.zeros((*v.shape, 3), dtype=np.float32)
    low_arr = np.array(low, dtype=np.float32)
    mid_arr = np.array(mid, dtype=np.float32)
    high_arr = np.array(high, dtype=np.float32)

    lower = v <= 0.5
    upper = ~lower
    t_low = (v[lower] / 0.5)[..., None]
    t_high = ((v[upper] - 0.5) / 0.5)[..., None]
    rgb[lower] = low_arr * (1.0 - t_low) + mid_arr * t_low
    rgb[upper] = mid_arr * (1.0 - t_high) + high_arr * t_high
    return rgb.astype(np.uint8)


def random_seed() -> int:
    """Return a positive 32-bit seed suitable for NumPy RNGs."""
    return secrets.randbits(32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Artificial Life Sandbox — Phase 5")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="World seed. Omit for a fresh random planet each run.",
    )
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--sea-level", type=float, default=0.50)
    parser.add_argument(
        "--windowed",
        action="store_true",
        help="Start in a resizable window instead of fullscreen.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PlanetConfig(
        seed=args.seed if args.seed is not None else random_seed(),
        width=args.width,
        height=args.height,
        sea_level=args.sea_level,
    )
    viewer = PlanetViewer(Planet.generate(config), scale=args.scale, start_fullscreen=not args.windowed)
    viewer.run()


if __name__ == "__main__":
    main()
