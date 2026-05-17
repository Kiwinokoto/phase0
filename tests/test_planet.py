import numpy as np

from alife import Planet, PlanetConfig


NORMALIZED_FIELD_NAMES = [
    "elevation",
    "water",
    "humidity",
    "light",
    "base_light",
    "base_volcanism",
    "volcanic_pulses",
    "volcanism",
    "minerals",
    "nutrient_source",
    "nutrients",
    "chemical_energy",
    "toxicity",
    "fertility",
    "dead_matter",
    "biomass",
    "diversity",
]

DYNAMIC_FIELD_NAMES = [
    "light",
    "temperature_c",
    "volcanism",
    "nutrients",
    "chemical_energy",
    "toxicity",
    "fertility",
    "dead_matter",
    "biomass",
    "diversity",
]

STATIC_FIELD_NAMES = [
    "elevation",
    "land",
    "water",
    "humidity",
    "base_light",
    "base_temperature_c",
    "base_volcanism",
    "minerals",
    "nutrient_source",
]


def test_planet_shapes_are_consistent():
    config = PlanetConfig(width=96, height=48, seed=1)
    planet = Planet.generate(config)

    for name in NORMALIZED_FIELD_NAMES:
        assert getattr(planet, name).shape == (48, 96)
    assert planet.land.shape == (48, 96)
    assert planet.temperature_c.shape == (48, 96)
    assert planet.base_temperature_c.shape == (48, 96)
    assert planet.populations.shape == (config.max_species, 48, 96)
    assert planet.dominant_species_index.shape == (48, 96)


def test_normalized_fields_stay_in_expected_range():
    planet = Planet.generate(PlanetConfig(width=96, height=48, seed=2))

    for name in NORMALIZED_FIELD_NAMES:
        field = getattr(planet, name)
        assert float(field.min()) >= 0.0, name
        assert float(field.max()) <= 1.0, name


def test_equator_is_brighter_than_poles_at_initial_tick():
    planet = Planet.generate(PlanetConfig(width=96, height=48, seed=3))

    equator_light = planet.light[planet.config.height // 2].mean()
    north_pole_light = planet.light[0].mean()
    south_pole_light = planet.light[-1].mean()

    assert equator_light > north_pole_light
    assert equator_light > south_pole_light


def test_equator_is_generally_warmer_than_poles_at_initial_tick():
    planet = Planet.generate(PlanetConfig(width=96, height=48, seed=4))

    equator_temp = planet.temperature_c[planet.config.height // 2].mean()
    pole_temp = np.concatenate([planet.temperature_c[:4].ravel(), planet.temperature_c[-4:].ravel()]).mean()

    assert equator_temp > pole_temp


def test_generation_is_deterministic_for_same_seed():
    config = PlanetConfig(width=96, height=48, seed=123)
    p1 = Planet.generate(config)
    p2 = Planet.generate(config)

    for name in NORMALIZED_FIELD_NAMES:
        np.testing.assert_allclose(getattr(p1, name), getattr(p2, name))
    np.testing.assert_allclose(p1.temperature_c, p2.temperature_c)
    np.testing.assert_allclose(p1.populations, p2.populations)


def test_regenerate_changes_seed():
    p1 = Planet.generate(PlanetConfig(width=96, height=48, seed=123))
    p2 = p1.regenerate()

    assert p2.config.seed == 124
    assert not np.allclose(p1.elevation, p2.elevation)


def test_phase2_resource_layers_are_not_empty():
    planet = Planet.generate(PlanetConfig(width=128, height=64, seed=99))

    assert float(planet.volcanism.max()) > 0.05
    assert float(planet.minerals.mean()) > 0.05
    assert float(planet.nutrients.mean()) > 0.05
    assert float(planet.chemical_energy.max()) > 0.05
    assert float(planet.fertility.max()) > 0.05


def test_chemical_energy_is_higher_near_volcanism():
    planet = Planet.generate(PlanetConfig(width=128, height=64, seed=101))
    hot = planet.volcanism >= np.quantile(planet.volcanism, 0.90)
    calm = planet.volcanism <= np.quantile(planet.volcanism, 0.40)

    assert planet.chemical_energy[hot].mean() > planet.chemical_energy[calm].mean()


def test_toxicity_penalizes_fertility_on_average():
    planet = Planet.generate(PlanetConfig(width=128, height=64, seed=202))
    toxic = planet.toxicity >= np.quantile(planet.toxicity, 0.90)
    mild = planet.toxicity <= np.quantile(planet.toxicity, 0.40)

    assert planet.fertility[toxic].mean() < planet.fertility[mild].mean() + 0.15


def test_humidity_is_not_just_the_water_map():
    planet = Planet.generate(PlanetConfig(width=128, height=64, seed=303))
    corr = np.corrcoef(planet.water.ravel(), planet.humidity.ravel())[0, 1]

    assert corr < 0.95


def test_step_advances_tick_and_changes_dynamic_fields():
    planet = Planet.generate(PlanetConfig(width=128, height=64, seed=404))
    before = {name: getattr(planet, name).copy() for name in DYNAMIC_FIELD_NAMES}

    planet.step(600)

    assert planet.tick == 600
    changed = [not np.allclose(before[name], getattr(planet, name)) for name in DYNAMIC_FIELD_NAMES]
    assert any(changed)
    assert not np.allclose(before["fertility"], planet.fertility)


def test_step_keeps_static_fields_stable():
    planet = Planet.generate(PlanetConfig(width=128, height=64, seed=505))
    before = {name: getattr(planet, name).copy() for name in STATIC_FIELD_NAMES}

    planet.step(800)

    for name in STATIC_FIELD_NAMES:
        np.testing.assert_allclose(before[name], getattr(planet, name), err_msg=name)


def test_dynamic_fields_remain_in_range_after_many_steps():
    planet = Planet.generate(PlanetConfig(width=128, height=64, seed=606))
    for _ in range(8):
        planet.step(500)

    for name in NORMALIZED_FIELD_NAMES:
        field = getattr(planet, name)
        assert float(field.min()) >= 0.0, name
        assert float(field.max()) <= 1.0, name


def test_same_seed_evolves_deterministically_for_same_steps():
    config = PlanetConfig(width=96, height=48, seed=707)
    p1 = Planet.generate(config)
    p2 = Planet.generate(config)

    for steps in (1, 13, 200, 777):
        p1.step(steps)
        p2.step(steps)

    assert p1.tick == p2.tick
    for name in DYNAMIC_FIELD_NAMES:
        np.testing.assert_allclose(getattr(p1, name), getattr(p2, name), err_msg=name)
    np.testing.assert_allclose(p1.populations, p2.populations)
    assert [s.name for s in p1.species] == [s.name for s in p2.species]


def test_phase3_starts_without_life_but_has_life_capacity():
    config = PlanetConfig(width=96, height=48, seed=808)
    planet = Planet.generate(config)

    assert planet.species == []
    assert planet.living_species_count == 0
    assert planet.total_biomass == 0.0
    assert planet.populations.shape == (config.max_species, 48, 96)


def test_abiogenesis_creates_proto_lineages_in_fertile_world():
    config = PlanetConfig(
        width=96,
        height=48,
        seed=909,
        abiogenesis_rate=0.20,
        abiogenesis_fertility_threshold=0.25,
    )
    planet = Planet.generate(config)
    planet.step(500)

    assert len(planet.species) > 0
    assert planet.living_species_count > 0
    assert planet.total_biomass > 0.0
    assert float(planet.biomass.max()) > 0.0


def test_life_fields_remain_bounded_after_many_steps():
    config = PlanetConfig(
        width=96,
        height=48,
        seed=1001,
        abiogenesis_rate=0.08,
        abiogenesis_fertility_threshold=0.30,
    )
    planet = Planet.generate(config)
    for _ in range(6):
        planet.step(700)

    assert len(planet.species) <= config.max_species
    assert float(planet.populations.min()) >= 0.0
    assert float(planet.populations.max()) <= 1.0
    assert float(planet.biomass.min()) >= 0.0
    assert float(planet.biomass.max()) <= 1.0
    assert float(planet.diversity.min()) >= 0.0
    assert float(planet.diversity.max()) <= 1.0


def test_life_evolution_is_deterministic_for_same_seed():
    config = PlanetConfig(
        width=96,
        height=48,
        seed=1112,
        abiogenesis_rate=0.08,
        abiogenesis_fertility_threshold=0.30,
    )
    p1 = Planet.generate(config)
    p2 = Planet.generate(config)

    for steps in (100, 300, 900, 1500):
        p1.step(steps)
        p2.step(steps)

    assert len(p1.species) == len(p2.species)
    assert [s.name for s in p1.species] == [s.name for s in p2.species]
    np.testing.assert_allclose(p1.populations, p2.populations)
    np.testing.assert_allclose(p1.biomass, p2.biomass)
