from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from .config import PlanetConfig
from .noise import fractal_noise_2d
from .life import (
    LifeSpecies,
    infer_strategy_label,
    make_species_name,
    mutate_traits,
    seed_traits_from_environment,
    species_color,
)




@dataclass(frozen=True)
class LineageHabitatSummary:
    """Computed observation data for one lineage.

    This is viewer/support data only: it does not affect simulation dynamics.
    Values are population-weighted over currently occupied cells.
    """

    species_id: int
    total_population: float
    occupied_cells: int
    strongest_cell: tuple[int, int] | None
    main_habitat: str
    mean_temperature_c: float
    mean_water_access: float
    mean_fertility: float
    mean_toxicity: float
    mean_nutrients: float
    mean_chemical_energy: float
    mean_dead_matter: float
    mean_light: float
    land_share: float

@dataclass
class Planet:
    """Generated 2D planet fields for Phase 4.

    Field shapes are always (height, width). Values are normalized to [0, 1]
    except temperature_c. Phase 4 keeps proto-life abstract, but adds stronger
    ecological pressure: local carrying capacity, maintenance consumption,
    starvation/turnover, dead matter, and more meaningful extinction dynamics.
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
    dead_matter: np.ndarray
    populations: np.ndarray
    biomass: np.ndarray
    diversity: np.ndarray
    dominant_species_index: np.ndarray
    species: list[LifeSpecies] = field(default_factory=list)
    next_species_id: int = 1
    extinction_count: int = 0
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
        dead_matter = np.zeros_like(fertility, dtype=np.float32)
        populations = np.zeros((config.max_species, config.height, config.width), dtype=np.float32)
        biomass = np.zeros_like(fertility, dtype=np.float32)
        diversity = np.zeros_like(fertility, dtype=np.float32)
        dominant_species_index = np.full_like(fertility, -1, dtype=np.int16)

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
            dead_matter=dead_matter,
            populations=populations,
            biomass=biomass,
            diversity=diversity,
            dominant_species_index=dominant_species_index,
        )

    @property
    def shape(self) -> tuple[int, int]:
        return self.elevation.shape

    @property
    def living_species_count(self) -> int:
        return sum(1 for species in self.species if not species.is_extinct)

    @property
    def total_biomass(self) -> float:
        return float(self.biomass.sum())

    @property
    def total_dead_matter(self) -> float:
        return float(self.dead_matter.sum())

    def top_species(self, limit: int = 5) -> list[tuple[LifeSpecies, float]]:
        totals: list[tuple[LifeSpecies, float]] = []
        for index, species in enumerate(self.species):
            total = float(self.populations[index].sum())
            if total > 0.0 or not species.is_extinct:
                totals.append((species, total))
        totals.sort(key=lambda item: item[1], reverse=True)
        return totals[: max(0, limit)]

    def species_index_by_id(self, species_id: int) -> int | None:
        """Return the internal array index for a public lineage id."""
        for index, species in enumerate(self.species):
            if species.id == species_id:
                return index
        return None

    def species_by_id(self, species_id: int | None) -> LifeSpecies | None:
        """Return a lineage by id, or None when it no longer belongs to this planet."""
        if species_id is None:
            return None
        index = self.species_index_by_id(int(species_id))
        return None if index is None else self.species[index]

    def species_total_population(self, species_id: int) -> float:
        index = self.species_index_by_id(species_id)
        if index is None:
            return 0.0
        return float(self.populations[index].sum())

    def descendant_count(self, species_id: int) -> int:
        """Return the number of direct and indirect descendant lineages."""
        descendants = 0
        frontier = {int(species_id)}
        seen: set[int] = set()
        while frontier:
            parent_id = frontier.pop()
            if parent_id in seen:
                continue
            seen.add(parent_id)
            children = {species.id for species in self.species if species.parent_id == parent_id}
            descendants += len(children)
            frontier.update(children)
        return descendants

    def lineage_habitat_summary(self, species_id: int, threshold: float = 0.005) -> LineageHabitatSummary:
        """Compute a compact habitat card for a lineage.

        The summary is based on current population distribution. It intentionally
        remains observational: it should help the user understand why a lineage
        is succeeding without feeding back into the simulation.
        """
        index = self.species_index_by_id(species_id)
        if index is None:
            return _empty_habitat_summary(int(species_id))

        pop = np.clip(self.populations[index], 0.0, None).astype(np.float64)
        total = float(pop.sum())
        occupied = pop > max(0.0, float(threshold))
        occupied_cells = int(occupied.sum())
        if total <= 1e-12 or occupied_cells == 0:
            return _empty_habitat_summary(int(species_id), total_population=total)

        strongest_flat = int(np.argmax(pop))
        strongest_y, strongest_x = divmod(strongest_flat, self.config.width)

        def weighted_mean(values: np.ndarray) -> float:
            return float((values.astype(np.float64) * pop).sum() / total)

        water_access = self._water_access()
        land_share = float((self.land.astype(np.float64) * pop).sum() / total)
        mean_temperature = weighted_mean(self.temperature_c)
        mean_water_access = weighted_mean(water_access)
        mean_fertility = weighted_mean(self.fertility)
        mean_toxicity = weighted_mean(self.toxicity)
        mean_nutrients = weighted_mean(self.nutrients)
        mean_chemical_energy = weighted_mean(self.chemical_energy)
        mean_dead_matter = weighted_mean(self.dead_matter)
        mean_light = weighted_mean(self.light)
        main_habitat = _infer_habitat_label(
            land_share=land_share,
            mean_temperature_c=mean_temperature,
            mean_water_access=mean_water_access,
            mean_fertility=mean_fertility,
        )

        return LineageHabitatSummary(
            species_id=int(species_id),
            total_population=total,
            occupied_cells=occupied_cells,
            strongest_cell=(int(strongest_x), int(strongest_y)),
            main_habitat=main_habitat,
            mean_temperature_c=mean_temperature,
            mean_water_access=mean_water_access,
            mean_fertility=mean_fertility,
            mean_toxicity=mean_toxicity,
            mean_nutrients=mean_nutrients,
            mean_chemical_energy=mean_chemical_energy,
            mean_dead_matter=mean_dead_matter,
            mean_light=mean_light,
            land_share=land_share,
        )

    def step(self, steps: int = 1) -> None:
        """Advance abiotic and proto-life dynamics by `steps` simulation ticks.

        Abiotic fields still use macro-steps. Life uses a few bounded internal
        substeps so population growth remains stable during fast-forward.
        """
        steps = int(steps)
        if steps < 1:
            return

        self.tick += steps
        self._update_seasonal_climate()
        self._update_volcanism(steps)
        self._update_resources(steps)
        self._update_fertility()
        self._maybe_seed_abiogenesis(steps)
        self._update_life(steps)
        self._maybe_branch_lineages(steps)
        self._update_biomass_maps()

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
        # Dead biomass slowly returns soluble nutrients. Extra recycling is also
        # handled during life substeps, but this keeps paused-low-biomass fields
        # chemically active between major population changes.
        if float(self.dead_matter.max()) > 0.0:
            recycled = self.dead_matter * (1.0 - _decay_factor(self.config.dead_matter_decay_rate * 0.35, steps))
            self.dead_matter -= recycled
            self.nutrients += (self.config.dead_matter_recycling_rate * 0.55 * recycled).astype(np.float32)
            self.dead_matter = np.clip(self.dead_matter, 0.0, 1.0).astype(np.float32)

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

    def _water_access(self) -> np.ndarray:
        shallow_water = np.clip(self.water * 1.45, 0.0, 1.0)
        return np.where(self.land, self.humidity, 0.45 + 0.55 * shallow_water).astype(np.float32)

    def _maybe_seed_abiogenesis(self, steps: int) -> None:
        if len(self.species) >= self.config.max_species:
            return

        viable = np.clip(self.fertility - self.config.abiogenesis_fertility_threshold, 0.0, 1.0)
        if float(viable.max()) <= 0.0:
            return

        expected = self.config.abiogenesis_rate * max(1, steps)
        event_count = int(self.rng.poisson(expected))
        if event_count <= 0:
            return

        event_count = min(event_count, 3, self.config.max_species - len(self.species))
        weights = (viable ** 2) * (0.25 + self.nutrients + 0.65 * self.chemical_energy) * (1.0 - 0.65 * self.toxicity)
        weights = np.clip(weights, 0.0, None).astype(np.float64)
        total_weight = float(weights.sum())
        if total_weight <= 0.0:
            return
        weights /= total_weight

        height, width = self.shape
        flat_indices = self.rng.choice(weights.size, size=event_count, replace=True, p=weights.ravel())
        for flat_index in np.atleast_1d(flat_indices):
            if len(self.species) >= self.config.max_species:
                break
            y, x = divmod(int(flat_index), width)
            self._create_seed_species(y, x)

    def _create_seed_species(self, y: int, x: int) -> None:
        species_id = self.next_species_id
        self.next_species_id += 1
        water_access = float(self._water_access()[y, x])
        traits = seed_traits_from_environment(
            self.rng,
            temperature_c=float(self.temperature_c[y, x]),
            water_access=water_access,
            light=float(self.light[y, x]),
            chemical_energy=float(self.chemical_energy[y, x]),
            dead_matter=float(self.dead_matter[y, x]),
            toxicity=float(self.toxicity[y, x]),
        )
        species = LifeSpecies(
            id=species_id,
            parent_id=None,
            name=make_species_name(self.rng, species_id),
            color=species_color(traits),
            traits=traits,
            created_tick=self.tick,
        )
        index = len(self.species)
        self.species.append(species)
        self._add_population_blob(index, y, x, self.config.initial_seed_population, radius=2.3)

    def _add_population_blob(self, index: int, y0: int, x0: int, amount: float, radius: float) -> None:
        height, width = self.shape
        y = np.arange(height, dtype=np.float32)[:, None]
        x = np.arange(width, dtype=np.float32)[None, :]
        dx = np.abs(x - float(x0))
        dx = np.minimum(dx, float(width) - dx)
        dy = y - float(y0)
        bump = np.exp(-(dx * dx + dy * dy) / (2.0 * radius * radius))
        self.populations[index] += (amount * bump).astype(np.float32)
        self.populations[index] = np.clip(self.populations[index], 0.0, 1.0).astype(np.float32)

    def _update_life(self, steps: int) -> None:
        if not self.species:
            return

        # Keep large fast-forward steps stable without doing one Python loop per tick.
        chunks = max(1, min(48, int(np.ceil(steps / 32))))
        dt = max(1.0, float(steps) / float(chunks))
        for _ in range(chunks):
            self._update_dead_matter(dt)
            water_access = self._water_access()
            biomass_before = np.clip(self.populations[: len(self.species)].sum(axis=0), 0.0, 1.0)

            for index, species in enumerate(self.species):
                if species.is_extinct:
                    continue
                pop = self.populations[index]
                if float(pop.sum()) <= 0.0:
                    self._mark_extinct_if_needed(index)
                    continue

                traits = species.traits
                temp_fit = np.exp(-((self.temperature_c - traits.temperature_optimum_c) / traits.temperature_tolerance_c) ** 2)
                water_fit = np.clip(
                    1.0 - np.abs(water_access - traits.water_preference) / traits.water_tolerance,
                    0.0,
                    1.0,
                )
                tox_over = np.maximum(0.0, self.toxicity - traits.toxicity_tolerance)
                tox_fit = np.clip(1.0 - tox_over / max(1.0 - traits.toxicity_tolerance, 0.05), 0.0, 1.0)
                habitat_fit = temp_fit * water_fit * tox_fit

                photo_energy = traits.photosynthesis * self.light * self.nutrients * water_fit
                chemo_energy = traits.chemosynthesis * self.chemical_energy * (0.25 + 0.75 * self.minerals)
                organic_energy = traits.organic_absorption * self.dead_matter
                energy_gain = photo_energy + chemo_energy + organic_energy

                # Phase 4: life is no longer only "growth minus generic crowding".
                # Each cell has a rough carrying capacity derived from current
                # fertility and usable energy. Dense mats in poor cells now crash,
                # which produces visible dead matter and frees nutrients again.
                local_capacity = np.clip(
                    0.02 + 0.72 * self.fertility + 0.26 * energy_gain * habitat_fit,
                    0.015,
                    1.0,
                )
                over_capacity = np.maximum(0.0, biomass_before - local_capacity)

                growth = traits.reproduction_rate * energy_gain * habitat_fit
                stress = self.config.life_stress_rate * (1.0 - habitat_fit)
                starvation = np.maximum(0.0, traits.metabolism_cost - 0.55 * energy_gain * habitat_fit)
                crowding = self.config.life_crowding_rate * over_capacity
                net = growth - traits.metabolism_cost - stress - starvation - crowding

                multiplier = np.exp(np.clip(net * dt * self.config.life_time_scale, -0.95, 0.42))
                updated = np.clip(pop * multiplier, 0.0, 1.0).astype(np.float32)

                turnover_pressure = np.clip(
                    0.35 + 2.50 * stress + 2.10 * starvation + 1.40 * crowding,
                    0.0,
                    7.0,
                )
                turnover_fraction = 1.0 - np.exp(-self.config.life_turnover_rate * dt * turnover_pressure)
                turnover_deaths = updated * turnover_fraction
                updated = np.maximum(updated - turnover_deaths, 0.0).astype(np.float32)

                if traits.dispersal > 0.0:
                    updated = _diffuse(
                        updated,
                        self.config.life_dispersal_rate * traits.dispersal,
                        max(1, int(round(dt))),
                    ).astype(np.float32)

                deaths = np.maximum(pop - updated, 0.0)
                growth_amount = np.maximum(updated - pop, 0.0)
                self.dead_matter += (0.42 * deaths + 0.35 * turnover_deaths).astype(np.float32)

                growth_use = self.config.life_resource_consumption_rate * growth_amount
                maintenance_use = self.config.life_maintenance_consumption_rate * dt * updated
                resource_use = growth_use + maintenance_use
                self.nutrients -= resource_use * (0.62 * traits.photosynthesis + 0.32 * traits.organic_absorption + 0.08)
                self.chemical_energy -= resource_use * (0.90 * traits.chemosynthesis)
                self.dead_matter -= resource_use * (0.60 * traits.organic_absorption)

                updated[updated < 1e-5] = 0.0
                self.populations[index] = np.clip(updated, 0.0, 1.0).astype(np.float32)
                species.population_peak = max(species.population_peak, float(self.populations[index].sum()))
                self._mark_extinct_if_needed(index)

            # During fast-forward, abiotic recharge must continue while life consumes
            # resources; otherwise one macro-step lets populations strip the map
            # before the next environmental recharge.
            recharge_steps = max(1, int(round(dt)))
            self.nutrients = _relax_to_source(
                self.nutrients,
                self.nutrient_source,
                self.config.nutrient_recharge_rate * 0.75,
                recharge_steps,
            )
            self.chemical_energy = _relax_to_source(
                self.chemical_energy,
                _generate_chemical_energy(self.config, self.volcanism, self.water),
                self.config.chemical_energy_recharge_rate * 0.55,
                recharge_steps,
            )
            self.nutrients = np.clip(self.nutrients, 0.0, 1.0).astype(np.float32)
            self.chemical_energy = np.clip(self.chemical_energy, 0.0, 1.0).astype(np.float32)
            self.dead_matter = np.clip(self.dead_matter, 0.0, 1.0).astype(np.float32)

    def _update_dead_matter(self, dt: float) -> None:
        if float(self.dead_matter.max()) <= 0.0:
            return
        steps = max(1, int(round(dt)))
        decay_amount = self.dead_matter * (1.0 - _decay_factor(self.config.dead_matter_decay_rate, steps))
        self.dead_matter -= decay_amount
        self.nutrients += (self.config.dead_matter_recycling_rate * decay_amount).astype(np.float32)
        self.dead_matter = _diffuse(self.dead_matter, self.config.dead_matter_diffusion_rate, steps)
        self.dead_matter = np.clip(self.dead_matter, 0.0, 1.0).astype(np.float32)

    def _mark_extinct_if_needed(self, index: int) -> None:
        species = self.species[index]
        if species.is_extinct:
            return
        total = float(self.populations[index].sum())
        if total <= self.config.extinction_population_threshold and self.tick > species.created_tick + 80:
            species.extinct_tick = self.tick
            self.populations[index].fill(0.0)
            self.extinction_count += 1

    def _maybe_branch_lineages(self, steps: int) -> None:
        if len(self.species) >= self.config.max_species or not self.species:
            return

        living_indices = [i for i, species in enumerate(self.species) if not species.is_extinct]
        if not living_indices:
            return

        totals = np.array([float(self.populations[i].sum()) for i in living_indices], dtype=np.float64)
        mutation_rates = np.array([self.species[i].traits.mutation_rate for i in living_indices], dtype=np.float64)
        weights = totals * mutation_rates
        total_weight = float(weights.sum())
        if total_weight <= 2.0:
            return

        expected = self.config.speciation_rate * max(1, steps) * min(12.0, total_weight / 28.0)
        event_count = int(self.rng.poisson(expected))
        if event_count <= 0:
            return
        event_count = min(event_count, 2, self.config.max_species - len(self.species))
        weights /= total_weight

        for _ in range(event_count):
            if len(self.species) >= self.config.max_species:
                break
            parent_index = int(self.rng.choice(living_indices, p=weights))
            parent_pop = self.populations[parent_index]
            if float(parent_pop.max()) <= 0.02:
                continue
            flat_index = int(np.argmax(parent_pop))
            y, x = divmod(flat_index, self.config.width)
            self._create_mutant_species(parent_index, y, x)

    def _create_mutant_species(self, parent_index: int, y: int, x: int) -> None:
        parent = self.species[parent_index]
        species_id = self.next_species_id
        self.next_species_id += 1
        traits = mutate_traits(self.rng, parent.traits, self.config.mutation_strength)
        species = LifeSpecies(
            id=species_id,
            parent_id=parent.id,
            name=make_species_name(self.rng, species_id),
            color=species_color(traits),
            traits=traits,
            created_tick=self.tick,
        )
        index = len(self.species)
        self.species.append(species)
        seed_amount = float(np.clip(self.populations[parent_index, y, x] * 0.35, 0.025, 0.12))
        self._add_population_blob(index, y, x, seed_amount, radius=2.0)
        self.populations[parent_index, y, x] *= 0.92

    def _update_biomass_maps(self) -> None:
        if not self.species:
            self.biomass.fill(0.0)
            self.diversity.fill(0.0)
            self.dominant_species_index.fill(-1)
            return

        active = self.populations[: len(self.species)]
        self.biomass = np.clip(active.sum(axis=0), 0.0, 1.0).astype(np.float32)
        self.diversity = np.clip((active > 0.01).sum(axis=0) / max(1, min(8, len(self.species))), 0.0, 1.0).astype(np.float32)
        dominant = np.argmax(active, axis=0).astype(np.int16)
        dominant[self.biomass <= 0.01] = -1
        self.dominant_species_index = dominant


    def top_species_near(
        self,
        x: int,
        y: int,
        *,
        radius: int = 5,
        limit: int = 5,
    ) -> list[tuple[LifeSpecies, float, float]]:
        """Return top lineages in a local map zone.

        Coordinates are map-style (x, y). The zone wraps horizontally and clips
        vertically, matching the planet projection. Returned totals are
        (species, local_total, global_total), sorted by local_total descending.
        """
        if not self.species or limit <= 0:
            return []

        height, width = self.shape
        cell_x = int(np.clip(x, 0, width - 1))
        cell_y = int(np.clip(y, 0, height - 1))
        r = max(0, int(radius))
        y0 = max(0, cell_y - r)
        y1 = min(height, cell_y + r + 1)
        x_indices = np.arange(cell_x - r, cell_x + r + 1) % width

        active = self.populations[: len(self.species)]
        region = active[:, y0:y1, :][:, :, x_indices]
        local_totals = region.sum(axis=(1, 2))
        global_totals = active.sum(axis=(1, 2))

        rows: list[tuple[LifeSpecies, float, float]] = []
        for index, species in enumerate(self.species):
            local_total = float(local_totals[index])
            if local_total > 0.005:
                rows.append((species, local_total, float(global_totals[index])))
        rows.sort(key=lambda item: item[1], reverse=True)
        return rows[:limit]

    def species_strategy_label(self, species: LifeSpecies) -> str:
        return infer_strategy_label(species.traits)




def _empty_habitat_summary(species_id: int, total_population: float = 0.0) -> LineageHabitatSummary:
    return LineageHabitatSummary(
        species_id=int(species_id),
        total_population=float(total_population),
        occupied_cells=0,
        strongest_cell=None,
        main_habitat="none",
        mean_temperature_c=0.0,
        mean_water_access=0.0,
        mean_fertility=0.0,
        mean_toxicity=0.0,
        mean_nutrients=0.0,
        mean_chemical_energy=0.0,
        mean_dead_matter=0.0,
        mean_light=0.0,
        land_share=0.0,
    )


def _infer_habitat_label(
    *,
    land_share: float,
    mean_temperature_c: float,
    mean_water_access: float,
    mean_fertility: float,
) -> str:
    if land_share < 0.25:
        if mean_temperature_c < 2.0:
            return "cold ocean"
        if mean_fertility >= 0.55:
            return "fertile water"
        return "open water"
    if land_share < 0.75:
        return "coastal mix"
    if mean_temperature_c < 0.0:
        return "cold land"
    if mean_water_access < 0.28:
        return "dry land"
    if mean_water_access > 0.64:
        return "wet land"
    return "temperate land"


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
