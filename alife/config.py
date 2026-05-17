from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PlanetConfig:
    """Configuration for a generated 2D planet.

    Phase 3 adds the first abstract life layer on top of the dynamic abiotic
    planet: rare abiogenesis events, proto-lineages, population growth,
    extinction, and light mutation-driven branching.
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

    # Phase 2 abiotic dynamics. Rates are per tick; step() scales them for fast-forward.
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


    # Phase 3 proto-life. Populations are stored per lineage and per map cell;
    # these are still abstract population fields, not individual organisms.
    max_species: int = 64
    abiogenesis_rate: float = 0.0060
    abiogenesis_fertility_threshold: float = 0.50
    initial_seed_population: float = 0.18

    life_time_scale: float = 0.045
    life_stress_rate: float = 0.09
    life_crowding_rate: float = 0.30
    life_resource_consumption_rate: float = 0.003
    life_dispersal_rate: float = 0.018

    dead_matter_diffusion_rate: float = 0.012
    dead_matter_decay_rate: float = 0.003
    dead_matter_recycling_rate: float = 0.55

    speciation_rate: float = 0.000075
    mutation_strength: float = 0.09
    extinction_population_threshold: float = 0.010

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
        if self.max_species < 1:
            raise ValueError("max_species must be >= 1.")
        if self.abiogenesis_rate < 0.0:
            raise ValueError("abiogenesis_rate must be >= 0.")
        if not 0.0 <= self.abiogenesis_fertility_threshold <= 1.0:
            raise ValueError("abiogenesis_fertility_threshold must be between 0 and 1.")
        if self.initial_speed < 1:
            raise ValueError("initial_speed must be >= 1.")
