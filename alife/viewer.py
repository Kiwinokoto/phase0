from __future__ import annotations

import argparse
import secrets
from dataclasses import dataclass

import numpy as np
import pygame

from .config import PlanetConfig
from .planet import Planet

LayerName = str
Color = tuple[int, int, int]


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
        description=("Composite future proto-life", "potential, penalized by toxicity."),
        colors=((28, 22, 18), (65, 120, 68), (165, 220, 92)),
        labels=("hostile", "viable", "promising"),
    ),
}


class PlanetViewer:
    """Small Pygame viewer for Phase 2 dynamic abiotic maps."""

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
    )

    def __init__(self, planet: Planet, scale: int = 4) -> None:
        pygame.init()
        self.planet = planet
        self.scale = max(1, int(scale))
        self.layer_index = 0
        self.paused = False
        self.fullscreen = False
        self.speed = planet.config.initial_speed
        self.font = pygame.font.SysFont("monospace", 16)
        self.small_font = pygame.font.SysFont("monospace", 12)
        self.tiny_font = pygame.font.SysFont("monospace", 11)

        height, width = self.planet.shape
        self.base_map_size = (width * self.scale, height * self.scale)
        self.side_panel_width = 380
        self.windowed_size = (self.base_map_size[0] + self.side_panel_width, self.base_map_size[1])
        self.screen = pygame.display.set_mode(self.windowed_size)
        pygame.display.set_caption("Artificial Life Sandbox — Phase 2")
        self.map_rect = pygame.Rect(0, 0, *self.base_map_size)
        self.panel_rect = pygame.Rect(self.base_map_size[0], 0, self.side_panel_width, self.base_map_size[1])
        self.fullscreen_button_rect = pygame.Rect(0, 0, 0, 0)
        self._update_layout()

        self.clock = pygame.time.Clock()
        self.cached_layer: LayerName | None = None
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
        elif key == pygame.K_r:
            self.planet = self.planet.regenerate(seed=random_seed())
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
        self.cached_surface = None
        self.cached_surface_size = None

    def _draw(self) -> None:
        self.screen.fill((16, 18, 24))
        self.screen.blit(self._get_map_surface(), self.map_rect.topleft)
        self._draw_panel()

    def _get_map_surface(self) -> pygame.Surface:
        if (
            self.cached_layer == self.current_layer
            and self.cached_surface is not None
            and self.cached_surface_size == self.map_rect.size
        ):
            return self.cached_surface

        rgb = render_layer(self.planet, self.current_layer)
        surface = pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))
        if surface.get_size() != self.map_rect.size:
            surface = pygame.transform.scale(surface, self.map_rect.size)

        self.cached_layer = self.current_layer
        self.cached_surface = surface
        self.cached_surface_size = self.map_rect.size
        return surface

    def _draw_panel(self) -> None:
        panel = self.panel_rect
        pygame.draw.rect(self.screen, (22, 24, 32), panel)
        pygame.draw.line(self.screen, (70, 76, 96), panel.topleft, panel.bottomleft, 1)

        x = panel.left + 18
        y = 18
        self._draw_text("Phase 2 — Dynamic Abiotic Planet", x, y, self.font, (235, 238, 245))
        y += 30

        y = self._draw_settings_row(x, y)
        y += 10

        for line in (
            f"seed: {self.planet.config.seed}",
            f"tick: {self.planet.tick}",
            f"speed: x{self.speed}",
            f"status: {'paused' if self.paused else 'running'}",
            f"mode: {'fullscreen' if self.fullscreen else 'window'}",
            f"layer: {self.current_layer}",
        ):
            self._draw_text(line, x, y, self.small_font, (195, 202, 220))
            y += 15

        y += 8
        self._draw_text("Controls:", x, y, self.small_font, (235, 238, 245))
        y += 16
        for line in (
            "space pause  |  button fullscreen",
            "tab/←/→ layer | ↑/↓ speed",
            "r new seed   |  s screenshot",
            "q/esc quit",
        ):
            self._draw_text(line, x, y, self.tiny_font, (185, 192, 210))
            y += 14

        y += 8
        self._draw_text("Planet stats:", x, y, self.small_font, (235, 238, 245))
        y += 16
        for line in (
            f"land/ocean: {100.0 * self.planet.land.mean():.1f}% / {100.0 * (1.0 - self.planet.land.mean()):.1f}%",
            f"temp avg: {self.planet.temperature_c.mean():.1f} °C",
            f"temp min/max: {self.planet.temperature_c.min():.1f}/{self.planet.temperature_c.max():.1f} °C",
            f"humidity/light: {self.planet.humidity.mean():.2f} / {self.planet.light.mean():.2f}",
            f"nutrients/chem: {self.planet.nutrients.mean():.2f} / {self.planet.chemical_energy.mean():.2f}",
            f"tox/fertility: {self.planet.toxicity.mean():.2f} / {self.planet.fertility.mean():.2f}",
        ):
            self._draw_text(line, x, y, self.tiny_font, (185, 192, 210))
            y += 14

        y += 12
        y = self._draw_current_layer_legend(x, y)

        # Keep the phase note pinned near the bottom when there is enough space.
        note_y = max(y + 12, panel.bottom - 68)
        for line in (
            "Phase 2 still has no life yet.",
            "Abiotic fields now change over time.",
            "Phase 3 will add proto-replicators.",
        ):
            if note_y + 14 < panel.bottom:
                self._draw_text(line, x, note_y, self.tiny_font, (165, 172, 190))
                note_y += 14

    def _draw_settings_row(self, x: int, y: int) -> int:
        button_w = min(180, self.panel_rect.width - 36)
        button_h = 28
        label = "Window mode" if self.fullscreen else "Fullscreen"
        self.fullscreen_button_rect = pygame.Rect(x, y, button_w, button_h)
        self._draw_button(self.fullscreen_button_rect, label)
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

    def _draw_current_layer_legend(self, x: int, y: int) -> int:
        legend = LAYER_LEGENDS[self.current_layer]
        max_bar_w = max(140, self.panel_rect.width - 44)

        self._draw_text("Layer legend:", x, y, self.small_font, (235, 238, 245))
        y += 17
        self._draw_text(legend.title, x, y, self.small_font, (220, 226, 240))
        y += 16
        for line in legend.description:
            self._draw_text(line, x, y, self.tiny_font, (185, 192, 210))
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

    def _save_screenshot(self) -> None:
        filename = f"planet_seed_{self.planet.config.seed}_{self.current_layer}.png"
        pygame.image.save(self.screen, filename)
        print(f"Saved screenshot: {filename}")


def _lerp_color(low: Color, high: Color, t: float) -> Color:
    return tuple(int(a * (1.0 - t) + b * t) for a, b in zip(low, high))  # type: ignore[return-value]


def render_layer(planet: Planet, layer: LayerName) -> np.ndarray:
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
    raise ValueError(f"Unknown layer: {layer}")


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
    parser = argparse.ArgumentParser(description="Artificial Life Sandbox — Phase 2")
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PlanetConfig(
        seed=args.seed if args.seed is not None else random_seed(),
        width=args.width,
        height=args.height,
        sea_level=args.sea_level,
    )
    viewer = PlanetViewer(Planet.generate(config), scale=args.scale)
    viewer.run()


if __name__ == "__main__":
    main()
