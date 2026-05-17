from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .config import PlanetConfig
from .noise import fractal_noise_2d


@dataclass
class Planet:
    """Generated 2D planet fields for Phase 2.

    Field shapes are always (height, width). Values are normalized to [0, 1]
    except temperature_c.

    Phase 2 still has no life. It turns the abiotic planet into a dynamic
    system: seasons, diffusion, recharge/decay, and volcanic pulses continually
    reshape the future niches where proto-life will appear in Phase 3.
    """

    config: PlanetConfig
    rng: np.random.Generator
    elevation: np.ndarray
    land: np.ndarray
    water: np.ndarray
    humidity: np.ndarray
    base_light: np.ndarray
    light: np.ndarray
    base_temperature_c: np.ndarray
    temperature_c: np.ndarray
    base_volcanism: np.ndarray
    volcanic_pulses: np.ndarray
    volcanism: np.ndarray
    minerals: np.ndarray
    nutrient_source: np.ndarray
    nutrients: np.ndarray
    chemical_energy: np.ndarray
    toxicity: np.ndarray
    fertility: np.ndarray
    tick: int = 0

    @classmethod
    def generate(cls, config: PlanetConfig) -> "Planet":
        config.validate()
        rng = np.random.default_rng(config.seed)

        elevation = _generate_elevation(config, rng)
        land = elevation >= config.sea_level
        water = np.clip((config.sea_level - elevation) / config.sea_level, 0.0, 1.0)

        base_light = _generate_light(config.height, config.width)
        base_temperature_c = _generate_temperature(config, elevation, water)
        humidity = _generate_humidity(config, rng, elevation, land, water, base_temperature_c)

        base_volcanism = _generate_volcanism(config, rng, elevation, land, water)
        volcanic_pulses = np.zeros_like(base_volcanism, dtype=np.float32)
        volcanism = base_volcanism.copy()
        minerals = _generate_minerals(config, rng, elevation, land, volcanism)
        nutrient_source = _generate_nutrients(config, rng, elevation, land, water, humidity, minerals)
        nutrients = nutrient_source.copy()
        chemical_energy = _generate_chemical_energy(config, volcanism, water)
        toxicity = _generate_toxicity(config, volcanism, base_temperature_c, water)
        fertility = _generate_fertility(
            config=config,
            land=land,
            water=water,
            humidity=humidity,
            light=base_light,
            temperature_c=base_temperature_c,
            nutrients=nutrients,
            chemical_energy=chemical_energy,
            toxicity=toxicity,
        )

        return cls(
            config=config,
            rng=rng,
            elevation=elevation,
            land=land,
            water=water,
            humidity=humidity,
            base_light=base_light,
            light=base_light.copy(),
            base_temperature_c=base_temperature_c,
            temperature_c=base_temperature_c.copy(),
            base_volcanism=base_volcanism,
            volcanic_pulses=volcanic_pulses,
            volcanism=volcanism,
            minerals=minerals,
            nutrient_source=nutrient_source,
            nutrients=nutrients,
            chemical_energy=chemical_energy,
            toxicity=toxicity,
            fertility=fertility,
        )

    @property
    def shape(self) -> tuple[int, int]:
        return self.elevation.shape

    def step(self, steps: int = 1) -> None:
        """Advance abiotic dynamics by `steps` simulation ticks.

        Fast-forward is handled as a macro-step, not thousands of tiny loops.
        This keeps the viewer responsive while preserving the direction of the
        processes: diffusion smooths, sources recharge, unstable fields decay,
        and volcanic pulses inject temporary energy/toxicity hotspots.
        """
        steps = int(steps)
        if steps < 1:
            return

        self.tick += steps
        self._update_seasonal_climate()
        self._update_volcanism(steps)
        self._update_resources(steps)
        self._update_fertility()

    def regenerate(self, seed: int | None = None) -> "Planet":
        """Return a new planet with a new or explicit seed."""
        new_seed = self.config.seed + 1 if seed is None else seed
        return Planet.generate(replace(self.config, seed=new_seed))

    def _update_seasonal_climate(self) -> None:
        period = max(10, self.config.seasonal_period_ticks)
        phase = 2.0 * np.pi * (self.tick % period) / period
        season = np.sin(phase)

        height, width = self.shape
        lat = np.linspace(-1.0, 1.0, height, dtype=np.float32)[:, None]
        lat_full = np.repeat(lat, width, axis=1)

        # Opposite hemispheres warm/brighten at opposite times. Oceans dampen
        # the seasonal temperature swing so coasts and seas are more stable.
        hemisphere_signal = -lat_full * season
        land_weight = self.land.astype(np.float32)
        ocean_damping = 0.42 + 0.58 * land_weight
        seasonal_temp = (
            self.config.seasonal_temperature_swing_c
            * hemisphere_signal
            * ocean_damping
        )
        self.temperature_c = (self.base_temperature_c + seasonal_temp).astype(np.float32)

        light_multiplier = 1.0 + self.config.seasonal_light_swing * hemisphere_signal
        self.light = np.clip(self.base_light * light_multiplier, 0.03, 1.0).astype(np.float32)

    def _update_volcanism(self, steps: int) -> None:
        self.volcanic_pulses *= _decay_factor(self.config.volcanic_pulse_decay_rate, steps)

        expected_pulses = self.config.volcanic_pulse_rate * steps
        pulse_count = int(self.rng.poisson(expected_pulses))
        if pulse_count > 0:
            # Cap very large fast-forward bursts to keep interaction smooth.
            pulse_count = min(pulse_count, 18)
            weights = (self.base_volcanism + 0.02).astype(np.float64)
            weights /= weights.sum()
            flat_indices = self.rng.choice(weights.size, size=pulse_count, replace=True, p=weights.ravel())
            height, width = self.shape
            for flat_index in np.atleast_1d(flat_indices):
                y, x = divmod(int(flat_index), width)
                self._add_volcanic_pulse(y, x)

        self.volcanic_pulses = np.clip(self.volcanic_pulses, 0.0, 1.0).astype(np.float32)
        self.volcanism = np.clip(self.base_volcanism + self.volcanic_pulses, 0.0, 1.0).astype(np.float32)

    def _add_volcanic_pulse(self, y0: int, x0: int) -> None:
        height, width = self.shape
        y = np.arange(height, dtype=np.float32)[:, None]
        x = np.arange(width, dtype=np.float32)[None, :]
        dx = np.abs(x - float(x0))
        dx = np.minimum(dx, float(width) - dx)  # wrap horizontally
        dy = y - float(y0)
        radius = self.config.volcanic_pulse_radius
        bump = np.exp(-(dx * dx + dy * dy) / (2.0 * radius * radius))
        self.volcanic_pulses += (self.config.volcanic_pulse_strength * bump).astype(np.float32)

    def _update_resources(self, steps: int) -> None:
        dynamic_nutrient_source = np.clip(
            self.nutrient_source
            + 0.10 * _blur4(self.minerals * self.humidity * self.land.astype(np.float32), passes=2)
            + 0.08 * _blur4(self.volcanism, passes=1),
            0.0,
            1.0,
        )
        self.nutrients = _relax_to_source(
            self.nutrients,
            dynamic_nutrient_source,
            self.config.nutrient_recharge_rate,
            steps,
        )
        self.nutrients = _diffuse(self.nutrients, self.config.nutrient_diffusion_rate, steps)
        self.nutrients *= _decay_factor(self.config.nutrient_leaching_rate, steps)
        self.nutrients = np.clip(self.nutrients, 0.0, 1.0).astype(np.float32)

        chemical_source = _generate_chemical_energy(self.config, self.volcanism, self.water)
        self.chemical_energy = _relax_to_source(
            self.chemical_energy,
            chemical_source,
            self.config.chemical_energy_recharge_rate,
            steps,
        )
        self.chemical_energy = _diffuse(
            self.chemical_energy,
            self.config.chemical_energy_diffusion_rate,
            steps,
        )
        self.chemical_energy *= _decay_factor(self.config.chemical_energy_decay_rate, steps)
        self.chemical_energy = np.clip(self.chemical_energy, 0.0, 1.0).astype(np.float32)

        toxicity_source = _generate_toxicity(self.config, self.volcanism, self.temperature_c, self.water)
        self.toxicity = _relax_to_source(
            self.toxicity,
            toxicity_source,
            self.config.toxicity_recharge_rate,
            steps,
        )
        self.toxicity = _diffuse(self.toxicity, self.config.toxicity_diffusion_rate, steps)
        self.toxicity *= _decay_factor(self.config.toxicity_decay_rate, steps)
        self.toxicity = np.clip(self.toxicity, 0.0, 1.0).astype(np.float32)

    def _update_fertility(self) -> None:
        self.fertility = _generate_fertility(
            config=self.config,
            land=self.land,
            water=self.water,
            humidity=self.humidity,
            light=self.light,
            temperature_c=self.temperature_c,
            nutrients=self.nutrients,
            chemical_energy=self.chemical_energy,
            toxicity=self.toxicity,
        )


def _generate_elevation(config: PlanetConfig, rng: np.random.Generator) -> np.ndarray:
    terrain = fractal_noise_2d(
        rng,
        width=config.width,
        height=config.height,
        base_grid=config.continent_scale,
        octaves=config.detail_octaves,
        gain=config.detail_gain,
    )

    # Make poles slightly more oceanic/low and keep mid-latitudes varied.
    lat = np.linspace(-1.0, 1.0, config.height)[:, None]
    polar_drop = 0.12 * np.abs(lat) ** 1.7
    equatorial_lift = 0.04 * (1.0 - np.abs(lat))
    elevation = terrain - polar_drop + equatorial_lift

    return _normalize(elevation).astype(np.float32)


def _generate_light(height: int, width: int) -> np.ndarray:
    lat = np.linspace(-1.0, 1.0, height)[:, None]
    # 1.0 at equator, low but non-zero near poles.
    light_col = np.clip(np.cos(np.abs(lat) * np.pi / 2.0), 0.05, 1.0)
    return np.repeat(light_col, width, axis=1).astype(np.float32)


def _generate_temperature(
    config: PlanetConfig,
    elevation: np.ndarray,
    water: np.ndarray,
) -> np.ndarray:
    lat = np.linspace(-1.0, 1.0, config.height)[:, None]
    lat_warmth = np.clip(np.cos(np.abs(lat) * np.pi / 2.0), 0.0, 1.0)
    base_temp = config.pole_temperature_c + (
        config.equator_temperature_c - config.pole_temperature_c
    ) * lat_warmth

    land_height = np.clip((elevation - config.sea_level) / (1.0 - config.sea_level), 0.0, 1.0)
    temp = base_temp - config.altitude_cooling_c * land_height

    mild_ocean_temp = 12.0
    ocean_weight = water * config.ocean_moderation
    temp = temp * (1.0 - ocean_weight) + mild_ocean_temp * ocean_weight
    return temp.astype(np.float32)


def _generate_humidity(
    config: PlanetConfig,
    rng: np.random.Generator,
    elevation: np.ndarray,
    land: np.ndarray,
    water: np.ndarray,
    temperature_c: np.ndarray,
) -> np.ndarray:
    """Generate ecological humidity, not just surface water.

    Oceans are wet, but inland rain depends on crude wind transport,
    temperature, altitude and rain shadow effects. This is a toy climate model,
    not physical meteorology.
    """
    height, width = elevation.shape
    climate_noise = fractal_noise_2d(
        rng,
        width=config.width,
        height=config.height,
        base_grid=7,
        octaves=4,
        gain=0.55,
    )
    normalized_temp = np.clip((temperature_c + 20.0) / 55.0, 0.0, 1.0)
    land_height = np.clip((elevation - config.sea_level) / (1.0 - config.sea_level), 0.0, 1.0)

    humidity = np.zeros((height, width), dtype=np.float32)

    for y in range(height):
        lat = (y / max(height - 1, 1)) * 2.0 - 1.0
        west_to_east = abs(lat) < 0.35 or abs(lat) > 0.72
        x_iter = range(width) if west_to_east else range(width - 1, -1, -1)
        air_moisture = 0.12 + 0.18 * float(climate_noise[y].mean())
        previous_height = 0.0

        for x in x_iter:
            if water[y, x] > 0.0:
                air_moisture = min(1.0, air_moisture + 0.07 + 0.08 * float(normalized_temp[y, x]))
                humidity[y, x] = 0.80 + 0.20 * water[y, x]
                previous_height = 0.0
                continue

            slope_lift = max(0.0, float(land_height[y, x] - previous_height))
            rainfall = air_moisture * (0.04 + 0.25 * slope_lift + 0.06 * float(climate_noise[y, x]))
            warmth_loss = 0.015 + 0.025 * float(normalized_temp[y, x])
            altitude_loss = 0.06 * float(land_height[y, x])
            humidity[y, x] = rainfall + 0.15 * float(climate_noise[y, x])
            air_moisture = max(0.0, air_moisture - rainfall - warmth_loss - altitude_loss)
            previous_height = float(land_height[y, x])

    humidity = _blur4(humidity, passes=2)
    humidity = 0.70 * humidity + 0.20 * climate_noise + 0.10 * normalized_temp
    humidity -= 0.20 * land_height
    humidity[water > 0.0] = np.maximum(humidity[water > 0.0], 0.75 + 0.20 * water[water > 0.0])
    return np.clip(humidity, 0.0, 1.0).astype(np.float32)


def _generate_volcanism(
    config: PlanetConfig,
    rng: np.random.Generator,
    elevation: np.ndarray,
    land: np.ndarray,
    water: np.ndarray,
) -> np.ndarray:
    tectonic_noise = fractal_noise_2d(
        rng,
        width=config.width,
        height=config.height,
        base_grid=4,
        octaves=5,
        gain=0.62,
    )
    local_relief = _local_relief(elevation)
    high_land = np.clip((elevation - 0.62) / 0.28, 0.0, 1.0) * land
    ocean_vents = np.clip((water - 0.25) / 0.55, 0.0, 1.0) * (1.0 - land)

    raw = 0.46 * tectonic_noise + 0.34 * local_relief + 0.22 * high_land + 0.18 * ocean_vents
    threshold = np.quantile(raw, 1.0 - config.volcanic_activity_fraction)
    volcanism = np.clip((raw - threshold) / max(float(raw.max() - threshold), 1e-9), 0.0, 1.0)
    volcanism = _blur4(volcanism, passes=1)
    return np.clip(volcanism, 0.0, 1.0).astype(np.float32)


def _generate_minerals(
    config: PlanetConfig,
    rng: np.random.Generator,
    elevation: np.ndarray,
    land: np.ndarray,
    volcanism: np.ndarray,
) -> np.ndarray:
    mineral_noise = fractal_noise_2d(
        rng,
        width=config.width,
        height=config.height,
        base_grid=8,
        octaves=4,
        gain=0.50,
    )
    land_height = np.clip((elevation - config.sea_level) / (1.0 - config.sea_level), 0.0, 1.0)
    mountains = np.clip((elevation - 0.68) / 0.25, 0.0, 1.0) * land
    minerals = 0.34 * mineral_noise + 0.32 * land_height + 0.24 * mountains + 0.42 * volcanism
    minerals = _blur4(minerals, passes=1)
    return np.clip(minerals, 0.0, 1.0).astype(np.float32)


def _generate_nutrients(
    config: PlanetConfig,
    rng: np.random.Generator,
    elevation: np.ndarray,
    land: np.ndarray,
    water: np.ndarray,
    humidity: np.ndarray,
    minerals: np.ndarray,
) -> np.ndarray:
    nutrient_noise = fractal_noise_2d(
        rng,
        width=config.width,
        height=config.height,
        base_grid=9,
        octaves=3,
        gain=0.48,
    )
    coast = _coast_mask(land).astype(np.float32)
    shallow_water = ((water > 0.0) & (water < 0.45)).astype(np.float32)
    lowland = np.clip(1.0 - (elevation - config.sea_level) / max(1.0 - config.sea_level, 1e-9), 0.0, 1.0)

    eroded_minerals = _blur4(minerals * (0.35 + 0.65 * humidity) * land, passes=4)
    coastal_nutrients = _blur4(coast + shallow_water, passes=3)

    nutrients = (
        0.30 * nutrient_noise
        + 0.34 * eroded_minerals
        + 0.34 * coastal_nutrients
        + 0.20 * humidity * lowland
    )
    nutrients = nutrients * (0.35 + 0.65 * land.astype(np.float32)) + 0.22 * shallow_water
    return np.clip(nutrients, 0.0, 1.0).astype(np.float32)


def _generate_chemical_energy(
    config: PlanetConfig,
    volcanism: np.ndarray,
    water: np.ndarray,
) -> np.ndarray:
    hydrothermal_bonus = volcanism * np.clip(water * 1.5, 0.0, 1.0)
    energy = 0.75 * volcanism + 0.35 * hydrothermal_bonus
    energy = _blur4(energy, passes=2)
    return np.clip(energy, 0.0, 1.0).astype(np.float32)


def _generate_toxicity(
    config: PlanetConfig,
    volcanism: np.ndarray,
    temperature_c: np.ndarray,
    water: np.ndarray,
) -> np.ndarray:
    heat_stress = np.clip((temperature_c - 38.0) / 22.0, 0.0, 1.0)
    cold_stress = np.clip((-temperature_c - 18.0) / 25.0, 0.0, 1.0)
    toxicity = 0.58 * volcanism + 0.20 * heat_stress + 0.12 * cold_stress
    toxicity *= 1.0 - 0.18 * water
    toxicity = _blur4(toxicity, passes=1)
    return np.clip(toxicity, 0.0, 1.0).astype(np.float32)


def _generate_fertility(
    config: PlanetConfig,
    land: np.ndarray,
    water: np.ndarray,
    humidity: np.ndarray,
    light: np.ndarray,
    temperature_c: np.ndarray,
    nutrients: np.ndarray,
    chemical_energy: np.ndarray,
    toxicity: np.ndarray,
) -> np.ndarray:
    temp_suitability = np.exp(-((temperature_c - config.life_friendly_temperature_c) / 22.0) ** 2)
    shallow_water = ((water > 0.0) & (water < 0.45)).astype(np.float32)
    water_access = np.where(land, humidity, 0.45 + 0.55 * shallow_water)
    energy_access = np.maximum(0.65 * light, 0.95 * chemical_energy)
    habitat_bonus = np.where(land, 0.85, 0.25 + 0.65 * shallow_water)

    fertility = (
        config.fertility_nutrient_weight * nutrients
        + config.fertility_energy_weight * energy_access
        + config.fertility_water_weight * water_access
        + config.fertility_temperature_weight * temp_suitability
    )
    fertility *= habitat_bonus
    fertility *= 1.0 - config.toxicity_fertility_penalty * toxicity
    return np.clip(fertility, 0.0, 1.0).astype(np.float32)


def _diffuse(values: np.ndarray, rate: float, steps: int) -> np.ndarray:
    # One stable macro diffusion pass. The exponential scaling means x1 and
    # x256 have the same qualitative direction without doing 256 Python loops.
    amount = float(np.clip(1.0 - np.exp(-rate * steps), 0.0, 0.38))
    neighbor_avg = (
        np.roll(values, 1, axis=0)
        + np.roll(values, -1, axis=0)
        + np.roll(values, 1, axis=1)
        + np.roll(values, -1, axis=1)
    ) / 4.0
    return values * (1.0 - amount) + neighbor_avg * amount


def _relax_to_source(current: np.ndarray, source: np.ndarray, rate: float, steps: int) -> np.ndarray:
    amount = float(np.clip(1.0 - np.exp(-rate * steps), 0.0, 1.0))
    return current * (1.0 - amount) + source * amount


def _decay_factor(rate: float, steps: int) -> float:
    return float(np.exp(-max(0.0, rate) * max(0, steps)))


def _coast_mask(land: np.ndarray) -> np.ndarray:
    return land & _neighbor_mask(~land)


def _neighbor_mask(mask: np.ndarray) -> np.ndarray:
    return (
        np.roll(mask, 1, axis=0)
        | np.roll(mask, -1, axis=0)
        | np.roll(mask, 1, axis=1)
        | np.roll(mask, -1, axis=1)
    )


def _local_relief(values: np.ndarray) -> np.ndarray:
    relief = np.maximum.reduce(
        [
            np.abs(values - np.roll(values, 1, axis=0)),
            np.abs(values - np.roll(values, -1, axis=0)),
            np.abs(values - np.roll(values, 1, axis=1)),
            np.abs(values - np.roll(values, -1, axis=1)),
        ]
    )
    return _normalize(relief)


def _blur4(values: np.ndarray, passes: int = 1) -> np.ndarray:
    out = values.astype(np.float32, copy=True)
    for _ in range(max(0, passes)):
        out = (
            out
            + np.roll(out, 1, axis=0)
            + np.roll(out, -1, axis=0)
            + np.roll(out, 1, axis=1)
            + np.roll(out, -1, axis=1)
        ) / 5.0
    return out


def _normalize(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32, copy=False)
    span = float(values.max() - values.min())
    if span <= 1e-9:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - values.min()) / span).astype(np.float32)
