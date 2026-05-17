from __future__ import annotations

import argparse
import secrets
from dataclasses import dataclass

import numpy as np
import pygame

from .config import PlanetConfig
from .planet import Planet

LayerName = str
OverlayMode = str
Color = tuple[int, int, int]

LIFE_LAYER_NAMES: tuple[LayerName, ...] = ("dead_matter", "biomass", "diversity", "dominant_life")
LIFE_OVERLAY_MODES: tuple[OverlayMode, ...] = ("off", "biomass", "dominant")


@dataclass(frozen=True)
class LayerLegend:
    title: str
    description: tuple[str, ...]
    colors: tuple[Color, ...] = ()
    labels: tuple[str, ...] = ()
    categories: tuple[tuple[str, Color], ...] = ()


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
}


class PlanetViewer:
    """Small Pygame viewer for Phase 4 constrained proto-ecology maps."""

    layers: tuple[LayerName, ...] = (
        "biome",
        "elevation",
        "temperature",
        "water",
        "humidity",
        "light",
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
    )

    def __init__(self, planet: Planet, scale: int = 4, start_fullscreen: bool = True) -> None:
        pygame.init()
        self.planet = planet
        self.scale = max(1, int(scale))
        self.layer_index = 0
        self.paused = False
        self.fullscreen = bool(start_fullscreen)
        self.life_overlay_mode: OverlayMode = "biomass"
        self.speed = planet.config.initial_speed
        self.selected_cell: tuple[int, int] | None = None  # stored as (x, y) map coordinates
        self.selected_radius = 5

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
        pygame.display.set_caption("Artificial Life Sandbox — Phase 4")
        self.map_rect = pygame.Rect(0, 0, *self.base_map_size)
        self.panel_rect = pygame.Rect(self.base_map_size[0], 0, self.side_panel_width, self.base_map_size[1])
        self.fullscreen_button_rect = pygame.Rect(0, 0, 0, 0)
        self.life_overlay_button_rect = pygame.Rect(0, 0, 0, 0)
        self._update_layout()

        self.clock = pygame.time.Clock()
        self.cached_layer: LayerName | None = None
        self.cached_overlay_mode: OverlayMode | None = None
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
                elif event.type == pygame.VIDEORESIZE and not self.fullscreen:
                    self.screen = pygame.display.set_mode(event.size, pygame.RESIZABLE)
                    self.windowed_size = self.screen.get_size()
                    self._update_layout()
                    self._invalidate_cache()

            if not self.paused:
                self.planet.step(self.speed)
                self._invalidate_cache()

            self._draw()
            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()

    def _handle_mouse_click(self, pos: tuple[int, int]) -> None:
        if self.fullscreen_button_rect.collidepoint(pos):
            self._toggle_fullscreen()
        elif self.life_overlay_button_rect.collidepoint(pos):
            self._cycle_life_overlay()
        elif self.map_rect.collidepoint(pos):
            cell = self._screen_pos_to_cell(pos)
            if cell is not None:
                self.selected_cell = cell

    def _handle_key(self, key: int) -> bool:
        if key in (pygame.K_ESCAPE, pygame.K_q):
            return False
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
        elif key == pygame.K_o:
            self._cycle_life_overlay()
        elif key == pygame.K_r:
            self.planet = self.planet.regenerate(seed=random_seed())
            self.selected_cell = None
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
        self.cached_surface = None
        self.cached_surface_size = None

    def _draw(self) -> None:
        self.screen.fill((16, 18, 24))
        self.screen.blit(self._get_map_surface(), self.map_rect.topleft)
        self._draw_selection_marker()
        self._draw_panel()

    def _get_map_surface(self) -> pygame.Surface:
        if (
            self.cached_layer == self.current_layer
            and self.cached_overlay_mode == self.life_overlay_mode
            and self.cached_surface is not None
            and self.cached_surface_size == self.map_rect.size
        ):
            return self.cached_surface

        rgb = render_layer(self.planet, self.current_layer, overlay_mode=self.life_overlay_mode)
        surface = pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))
        if surface.get_size() != self.map_rect.size:
            surface = pygame.transform.scale(surface, self.map_rect.size)

        self.cached_layer = self.current_layer
        self.cached_overlay_mode = self.life_overlay_mode
        self.cached_surface = surface
        self.cached_surface_size = self.map_rect.size
        return surface

    def _draw_panel(self) -> None:
        panel = self.panel_rect
        pygame.draw.rect(self.screen, (18, 20, 28), panel)
        pygame.draw.line(self.screen, (70, 76, 96), panel.topleft, panel.bottomleft, 1)

        x = panel.left + 18
        y = 18
        content_w = panel.width - 36

        self._draw_text("Phase 4 — Constrained Ecology", x, y, self.font, (235, 238, 245))
        y += 25

        y = self._draw_active_layer_header(x, y)
        y += 9

        y = self._draw_settings_row(x, y)
        y += 8

        y = self._draw_compact_controls(x, y, content_w)
        y += 10

        y = self._draw_simulation_summary(x, y, content_w)
        y += 9

        y = self._draw_selected_zone(x, y, content_w)
        y += 9

        y = self._draw_life_summary(x, y, content_w)
        y += 9

        # On shorter windows the inspector and life sections matter more than
        # global planet averages, so keep the lower sections conditional.
        if y < panel.bottom - 220:
            y = self._draw_planet_summary(x, y, content_w)
            y += 9

        if y < panel.bottom - 120:
            y = self._draw_current_layer_legend(x, y, content_w)

        note_y = max(y + 12, panel.bottom - 44)
        for line in (
            "Resource-limited proto-ecology.",
            "No hard-coded plants/animals.",
        ):
            if note_y + 14 < panel.bottom:
                self._draw_text(line, x, note_y, self.tiny_font, (165, 172, 190))
                note_y += 14


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

    def _draw_section_title(self, title: str, x: int, y: int, width: int) -> int:
        rect = pygame.Rect(x, y, width, 22)
        pygame.draw.rect(self.screen, (28, 33, 46), rect, border_radius=6)
        pygame.draw.rect(self.screen, (52, 62, 84), rect, 1, border_radius=6)
        self._draw_text(title, x + 9, y + 3, self.small_font, (232, 238, 250))
        return y + 28

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

    def _draw_simulation_summary(self, x: int, y: int, width: int) -> int:
        y = self._draw_section_title("Simulation", x, y, width)
        return self._draw_key_value_grid(
            (
                ("seed", str(self.planet.config.seed)),
                ("tick", str(self.planet.tick)),
                ("speed", f"x{self.speed}"),
                ("state", "paused" if self.paused else "running"),
                ("mode", "fullscreen" if self.fullscreen else "window"),
                ("life", self.life_overlay_mode),
            ),
            x,
            y,
            width,
        )

    def _draw_life_summary(self, x: int, y: int, width: int) -> int:
        y = self._draw_section_title("Life summary", x, y, width)
        y = self._draw_key_value_grid(
            (
                ("living", f"{self.planet.living_species_count}"),
                ("extinct", f"{self.planet.extinction_count}"),
                ("biomass", f"{self.planet.total_biomass:.1f}"),
                ("dead", f"{self.planet.total_dead_matter:.1f}"),
                ("lineages", f"{len(self.planet.species)}/{self.planet.config.max_species}"),
                ("max bio", f"{self.planet.biomass.max():.3f}"),
            ),
            x,
            y,
            width,
        )
        y += 4
        return self._draw_top_species(x, y, width)

    def _draw_planet_summary(self, x: int, y: int, width: int) -> int:
        y = self._draw_section_title("Planet averages", x, y, width)
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
            ),
            x,
            y,
            width,
        )

    def _draw_selected_zone(self, x: int, y: int, width: int) -> int:
        y = self._draw_section_title("Selected zone", x, y, width)

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

        self._draw_text("Local top lineages", x, y, self.tiny_font, (220, 226, 240))
        y += 14
        for species, local_total, _global_total in local_top:
            color_rect = pygame.Rect(x, y + 2, 10, 10)
            pygame.draw.rect(self.screen, species.color, color_rect)
            pygame.draw.rect(self.screen, (90, 96, 116), color_rect, 1)
            label = f"{species.name}  {local_total:.2f}  {self.planet.species_strategy_label(species)}"
            self._draw_text(self._clip_text(label, 48), x + 16, y, self.tiny_font, (190, 199, 218))
            y += 14
        return y

    def _screen_pos_to_cell(self, pos: tuple[int, int]) -> tuple[int, int] | None:
        if not self.map_rect.collidepoint(pos):
            return None
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
        height = self.planet.config.height
        px = self.map_rect.left + int((cell_x / width) * self.map_rect.width)
        py = self.map_rect.top + int((cell_y / height) * self.map_rect.height)
        cell_w = max(2, int(np.ceil(self.map_rect.width / width)))
        cell_h = max(2, int(np.ceil(self.map_rect.height / height)))
        radius_px = max(8, int(self.selected_radius * (self.map_rect.width / width)))
        center = (px + cell_w // 2, py + cell_h // 2)
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
            color_rect = pygame.Rect(x, y + 2, 10, 10)
            pygame.draw.rect(self.screen, species.color, color_rect)
            pygame.draw.rect(self.screen, (90, 96, 116), color_rect, 1)
            status = "†" if species.is_extinct else ""
            label = f"{species.name}{status}  {total:.1f}  {self.planet.species_strategy_label(species)}"
            self._draw_text(self._clip_text(label, 48), x + 16, y, self.tiny_font, (190, 199, 218))
            y += 14
        return y

    def _draw_settings_row(self, x: int, y: int) -> int:
        button_h = 28
        gap = 8
        available_w = self.panel_rect.width - 36
        button_w = max(126, min(168, (available_w - gap) // 2))

        fullscreen_label = "Window" if self.fullscreen else "Fullscreen"
        self.fullscreen_button_rect = pygame.Rect(x, y, button_w, button_h)
        self._draw_button(self.fullscreen_button_rect, fullscreen_label)

        overlay_label = f"Life: {self.life_overlay_mode}"
        self.life_overlay_button_rect = pygame.Rect(x + button_w + gap, y, button_w, button_h)
        self._draw_button(self.life_overlay_button_rect, overlay_label)
        return y + button_h

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

        y = self._draw_section_title("Layer legend", x, y, max_bar_w)
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

    def _save_screenshot(self) -> None:
        overlay_suffix = "" if self.life_overlay_mode == "off" else f"_{self.life_overlay_mode}_overlay"
        filename = f"planet_seed_{self.planet.config.seed}_{self.current_layer}{overlay_suffix}.png"
        pygame.image.save(self.screen, filename)
        print(f"Saved screenshot: {filename}")


def _lerp_color(low: Color, high: Color, t: float) -> Color:
    return tuple(int(a * (1.0 - t) + b * t) for a, b in zip(low, high))  # type: ignore[return-value]


def render_layer(planet: Planet, layer: LayerName, overlay_mode: OverlayMode = "off") -> np.ndarray:
    base = _render_base_layer(planet, layer)
    if should_apply_life_overlay(layer, overlay_mode):
        return apply_life_overlay(planet, base, overlay_mode)
    return base


def should_apply_life_overlay(layer: LayerName, overlay_mode: OverlayMode) -> bool:
    return overlay_mode != "off" and layer not in LIFE_LAYER_NAMES


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
        glow = np.array((70, 245, 120), dtype=np.float32)
        rgb[mask] = rgb[mask] * (1.0 - alpha[mask]) + glow * alpha[mask]
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
    raise ValueError(f"Unknown layer: {layer}")

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
    parser = argparse.ArgumentParser(description="Artificial Life Sandbox — Phase 4")
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
