from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LifeTraits:
    """Heritable parameters for an abstract proto-lineage.

    These traits deliberately describe verbs/capabilities, not fixed categories
    like plant, animal, herbivore, or predator. Phase 5 adds life-on-life
    ecological pressure, still in this same verb-based model: a lineage may
    extract energy from living biomass, resist being consumed, or buffer bad
    periods with storage.
    """

    photosynthesis: float
    chemosynthesis: float
    organic_absorption: float
    living_consumption: float
    defense: float
    storage: float
    temperature_optimum_c: float
    temperature_tolerance_c: float
    water_preference: float
    water_tolerance: float
    toxicity_tolerance: float
    reproduction_rate: float
    metabolism_cost: float
    dispersal: float
    mutation_rate: float

    def clipped(self) -> "LifeTraits":
        return LifeTraits(
            photosynthesis=_clip01(self.photosynthesis),
            chemosynthesis=_clip01(self.chemosynthesis),
            organic_absorption=_clip01(self.organic_absorption),
            living_consumption=_clip01(self.living_consumption),
            defense=_clip01(self.defense),
            storage=_clip01(self.storage),
            temperature_optimum_c=float(np.clip(self.temperature_optimum_c, -18.0, 48.0)),
            temperature_tolerance_c=float(np.clip(self.temperature_tolerance_c, 6.0, 34.0)),
            water_preference=_clip01(self.water_preference),
            water_tolerance=float(np.clip(self.water_tolerance, 0.08, 0.85)),
            toxicity_tolerance=_clip01(self.toxicity_tolerance),
            reproduction_rate=float(np.clip(self.reproduction_rate, 0.10, 1.80)),
            metabolism_cost=float(np.clip(self.metabolism_cost, 0.02, 0.60)),
            dispersal=_clip01(self.dispersal),
            mutation_rate=float(np.clip(self.mutation_rate, 0.002, 0.080)),
        )


@dataclass
class LifeSpecies:
    """A lineage tracked by the simulation.

    The population grid for a species is stored in Planet.populations at the
    same index as this object in Planet.species.
    """

    id: int
    parent_id: int | None
    name: str
    color: tuple[int, int, int]
    traits: LifeTraits
    created_tick: int
    population_peak: float = 0.0
    extinct_tick: int | None = None

    @property
    def is_extinct(self) -> bool:
        return self.extinct_tick is not None


_SYLLABLES = (
    "ka", "zu", "mi", "ra", "lo", "ve", "an", "tu", "shi", "or", "ne", "fi",
    "xa", "po", "ul", "dra", "ki", "sa", "elu", "mor", "ith", "qua", "ren", "vo",
)


def make_species_name(rng: np.random.Generator, species_id: int) -> str:
    parts = rng.choice(_SYLLABLES, size=int(rng.integers(2, 4)), replace=True)
    return "".join(str(part) for part in parts).capitalize() + f"-{species_id:03d}"


def seed_traits_from_environment(
    rng: np.random.Generator,
    *,
    temperature_c: float,
    water_access: float,
    light: float,
    chemical_energy: float,
    dead_matter: float,
    toxicity: float,
) -> LifeTraits:
    """Create a plausible first genotype for a local abiogenesis event."""
    photo_bias = max(0.0, light) + 0.08 * float(rng.random())
    chemo_bias = max(0.0, chemical_energy) + 0.08 * float(rng.random())
    organic_bias = max(0.0, dead_matter) + 0.04 * float(rng.random())
    total = photo_bias + chemo_bias + organic_bias + 1e-6

    # First life is usually primitive and low-efficiency, but biased toward the
    # local energy source that made the niche fertile. Living-biomass extraction
    # is intentionally low at abiogenesis; it becomes interesting through
    # mutation once other biomass exists.
    photosynthesis = 0.22 + 0.68 * photo_bias / total
    chemosynthesis = 0.15 + 0.65 * chemo_bias / total
    organic_absorption = 0.05 + 0.38 * organic_bias / total
    living_consumption = 0.02 + 0.08 * float(rng.random())

    return LifeTraits(
        photosynthesis=float(np.clip(photosynthesis + rng.normal(0.0, 0.05), 0.0, 1.0)),
        chemosynthesis=float(np.clip(chemosynthesis + rng.normal(0.0, 0.05), 0.0, 1.0)),
        organic_absorption=float(np.clip(organic_absorption + rng.normal(0.0, 0.04), 0.0, 1.0)),
        living_consumption=float(np.clip(living_consumption + rng.normal(0.0, 0.025), 0.0, 0.18)),
        defense=float(np.clip(0.06 + toxicity * 0.10 + rng.random() * 0.14, 0.0, 0.32)),
        storage=float(np.clip(0.08 + rng.random() * 0.18, 0.0, 0.34)),
        temperature_optimum_c=float(temperature_c + rng.normal(0.0, 4.0)),
        temperature_tolerance_c=float(np.clip(13.0 + rng.normal(0.0, 3.0), 7.0, 26.0)),
        water_preference=float(np.clip(water_access + rng.normal(0.0, 0.08), 0.0, 1.0)),
        water_tolerance=float(np.clip(0.25 + rng.random() * 0.30, 0.10, 0.75)),
        toxicity_tolerance=float(np.clip(toxicity + 0.18 + rng.normal(0.0, 0.08), 0.0, 1.0)),
        reproduction_rate=float(np.clip(0.92 + rng.normal(0.0, 0.18), 0.30, 1.50)),
        metabolism_cost=float(np.clip(0.065 + rng.normal(0.0, 0.025), 0.025, 0.18)),
        dispersal=float(np.clip(0.08 + rng.random() * 0.18, 0.0, 0.40)),
        mutation_rate=float(np.clip(0.012 + rng.random() * 0.018, 0.003, 0.060)),
    ).clipped()


def mutate_traits(
    rng: np.random.Generator,
    parent: LifeTraits,
    strength: float,
) -> LifeTraits:
    """Return a nearby genotype, preserving parent similarity."""
    # A small chance of a bigger trophic/defensive innovation makes Phase 5 more
    # watchable without hard-coding consumers. Most mutations remain subtle.
    trophic_strength = strength * (1.65 if rng.random() < 0.12 else 0.95)
    return LifeTraits(
        photosynthesis=float(parent.photosynthesis + rng.normal(0.0, strength)),
        chemosynthesis=float(parent.chemosynthesis + rng.normal(0.0, strength)),
        organic_absorption=float(parent.organic_absorption + rng.normal(0.0, strength * 0.85)),
        living_consumption=float(parent.living_consumption + rng.normal(0.0, trophic_strength * 0.95)),
        defense=float(parent.defense + rng.normal(0.0, trophic_strength * 0.80)),
        storage=float(parent.storage + rng.normal(0.0, strength * 0.75)),
        temperature_optimum_c=float(parent.temperature_optimum_c + rng.normal(0.0, 5.0 * strength / 0.10)),
        temperature_tolerance_c=float(parent.temperature_tolerance_c + rng.normal(0.0, 2.5 * strength / 0.10)),
        water_preference=float(parent.water_preference + rng.normal(0.0, strength)),
        water_tolerance=float(parent.water_tolerance + rng.normal(0.0, strength * 0.75)),
        toxicity_tolerance=float(parent.toxicity_tolerance + rng.normal(0.0, strength * 0.75)),
        reproduction_rate=float(parent.reproduction_rate + rng.normal(0.0, strength * 1.4)),
        metabolism_cost=float(parent.metabolism_cost + rng.normal(0.0, strength * 0.45)),
        dispersal=float(parent.dispersal + rng.normal(0.0, strength * 0.80)),
        mutation_rate=float(parent.mutation_rate + rng.normal(0.0, strength * 0.08)),
    ).clipped()


def species_color(traits: LifeTraits) -> tuple[int, int, int]:
    """Stable visible color derived from ecological strategy traits."""
    photo = traits.photosynthesis
    chemo = traits.chemosynthesis
    detritus = traits.organic_absorption
    living = traits.living_consumption
    defense = traits.defense
    mobility = traits.dispersal
    r = int(np.clip(50 + 85 * detritus + 125 * living + 95 * chemo + 20 * defense, 35, 255))
    g = int(np.clip(60 + 155 * photo + 40 * detritus + 30 * defense, 45, 255))
    b = int(np.clip(70 + 115 * chemo + 58 * mobility + 42 * defense, 50, 255))
    return (r, g, b)


def infer_strategy_label(traits: LifeTraits) -> str:
    """Human-facing interpretation; not a simulation class."""
    values = {
        "photo": traits.photosynthesis,
        "chemo": traits.chemosynthesis,
        "detritus": traits.organic_absorption,
        "living": traits.living_consumption,
    }
    primary = max(values, key=values.get)
    if traits.defense >= 0.62 and traits.dispersal < 0.22:
        return "defensive mat"
    if primary == "living" and traits.dispersal >= 0.25:
        return "mobile consumer"
    if primary == "living":
        return "biomass consumer"
    if primary == "photo" and traits.dispersal < 0.18:
        return "producer-like"
    if primary == "chemo":
        return "chemo-producer"
    if primary == "detritus":
        return "detritivore-like"
    if traits.storage >= 0.62:
        return "storage specialist"
    if traits.dispersal >= 0.28:
        return "mobile protoform"
    return "mixed protoform"


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))
