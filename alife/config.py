from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PlanetConfig:
    """Configuration for a generated 2D planet.

    Phase 2 keeps the world deliberately abiotic: no life yet. The planet is
    now dynamic, so resources, chemical energy, toxicity, light, temperature,
    and fertility evolve over time.
    """

    width: int = 256
    height: int = 128
    seed: int = 42

    # Terrain
    sea_level: float = 0.50
    continent_scale: int = 5
    detail_octaves: int = 5
    detail_gain: float = 0.52

    # Climate
    equator_temperature_c: float = 32.0
    pole_temperature_c: float = -18.0
    altitude_cooling_c: float = 30.0
    ocean_moderation: float = 0.35
    seasonal_period_ticks: int = 2400
    seasonal_temperature_swing_c: float = 8.0
    seasonal_light_swing: float = 0.28

    # Static abiotic resource generation
    volcanic_activity_fraction: float = 0.08
    life_friendly_temperature_c: float = 22.0
    toxicity_fertility_penalty: float = 0.75
    fertility_nutrient_weight: float = 0.34
    fertility_energy_weight: float = 0.24
    fertility_water_weight: float = 0.22
    fertility_temperature_weight: float = 0.20

    # Phase 2 dynamics. Rates are per tick; step() scales them for fast-forward.
    nutrient_diffusion_rate: float = 0.020
    nutrient_recharge_rate: float = 0.0025
    nutrient_leaching_rate: float = 0.0008

    chemical_energy_diffusion_rate: float = 0.010
    chemical_energy_recharge_rate: float = 0.010
    chemical_energy_decay_rate: float = 0.0035

    toxicity_diffusion_rate: float = 0.012
    toxicity_recharge_rate: float = 0.006
    toxicity_decay_rate: float = 0.004

    volcanic_pulse_rate: float = 0.010
    volcanic_pulse_decay_rate: float = 0.010
    volcanic_pulse_radius: float = 4.0
    volcanic_pulse_strength: float = 0.55

    # Visual/simulation clock
    initial_speed: int = 1

    def validate(self) -> None:
        if self.width < 32 or self.height < 16:
            raise ValueError("Planet map is too small; use at least 32x16.")
        if not 0.0 < self.sea_level < 1.0:
            raise ValueError("sea_level must be between 0 and 1.")
        if self.detail_octaves < 1:
            raise ValueError("detail_octaves must be >= 1.")
        if not 0.0 < self.volcanic_activity_fraction < 0.5:
            raise ValueError("volcanic_activity_fraction must be between 0 and 0.5.")
        if self.seasonal_period_ticks < 10:
            raise ValueError("seasonal_period_ticks must be >= 10.")
        if self.volcanic_pulse_radius <= 0:
            raise ValueError("volcanic_pulse_radius must be positive.")
        if self.initial_speed < 1:
            raise ValueError("initial_speed must be >= 1.")
