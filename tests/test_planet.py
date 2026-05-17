import inspect

import numpy as np
import pygame

from alife import Planet, PlanetConfig
from alife.planet import SimulationEvent
from alife.viewer import DETAIL_SETUP_FIELDS, PLANET_SETUP_FIELDS, PRIMARY_SETUP_FIELDS, PlanetViewer, geological_intro_stage, list_world_presets, load_world_preset, planet_config_from_preset, planet_config_to_preset, decode_world_thumbnail, random_world_name, read_world_preset_metadata, render_geological_intro_layer, render_globe_texture, render_layer, render_star_background, save_world_preset, season_label, season_position, should_apply_life_overlay, should_apply_weather_overlay, specimen_keywords


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
    "biotic_pressure",
    "migration_pressure",
    "isolation_pressure",
    "morphology_index",
    "climate_stress",
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
    "biotic_pressure",
    "migration_pressure",
    "isolation_pressure",
    "morphology_index",
    "climate_stress",
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




def test_static_maps_are_horizontally_seam_compatible():
    planet = Planet.generate(PlanetConfig(width=160, height=80, seed=7171))

    for name in ("elevation", "humidity", "minerals", "nutrient_source"):
        field = getattr(planet, name).astype(np.float32)
        seam_delta = float(np.mean(np.abs(field[:, 0] - field[:, -1])))
        typical_delta = float(np.mean(np.abs(field[:, 1:] - field[:, :-1])))
        assert seam_delta <= max(0.035, typical_delta * 3.5), name


def test_globe_render_has_no_obvious_vertical_texture_join():
    planet = Planet.generate(PlanetConfig(width=160, height=80, seed=8282))
    texture = render_layer(planet, "biome")
    globe = render_globe_texture(texture, (420, 300), rotation=0.0, seed=planet.config.seed)

    h, w = globe.shape[:2]
    cx = w // 2
    center_band = globe[h // 5 : h - h // 5, cx - 1 : cx + 2].astype(np.float32)
    neighbor_band = globe[h // 5 : h - h // 5, cx + 5 : cx + 8].astype(np.float32)
    # This guards against the previous artificial meridian where the two
    # rectangular-map edges looked pasted together as a straight center line.
    assert float(np.mean(np.abs(center_band - neighbor_band))) < 55.0

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


def test_life_overlay_changes_biome_render_when_biomass_exists():
    config = PlanetConfig(
        width=64,
        height=32,
        seed=1213,
        abiogenesis_rate=0.20,
        abiogenesis_fertility_threshold=0.25,
    )
    planet = Planet.generate(config)
    planet.step(600)

    plain = render_layer(planet, "biome", overlay_mode="off")
    overlay = render_layer(planet, "biome", overlay_mode="biomass")

    assert should_apply_life_overlay("biome", "biomass")
    assert not should_apply_life_overlay("biomass", "biomass")
    assert plain.shape == overlay.shape == (32, 64, 3)
    assert not np.array_equal(plain, overlay)


def test_life_overlay_does_not_modify_dedicated_life_layers():
    planet = Planet.generate(PlanetConfig(width=64, height=32, seed=1314))
    planet.step(600)

    plain = render_layer(planet, "biomass", overlay_mode="off")
    overlay = render_layer(planet, "biomass", overlay_mode="dominant")

    np.testing.assert_array_equal(plain, overlay)


def test_phase4_dead_matter_accumulates_after_life_turnover():
    config = PlanetConfig(
        width=96,
        height=48,
        seed=1415,
        abiogenesis_rate=0.12,
        abiogenesis_fertility_threshold=0.28,
    )
    planet = Planet.generate(config)
    for _ in range(8):
        planet.step(600)

    assert planet.total_biomass > 0.0
    assert planet.total_dead_matter > 0.0
    assert float(planet.dead_matter.max()) > 0.0


def test_phase4_hostile_pressure_can_reduce_existing_biomass():
    config = PlanetConfig(
        width=64,
        height=32,
        seed=1516,
        abiogenesis_rate=0.25,
        abiogenesis_fertility_threshold=0.20,
    )
    planet = Planet.generate(config)
    planet.step(700)
    before = planet.total_biomass

    # Directly simulate a severe environmental shock. This uses the model state,
    # not a public gameplay event yet; future phases can expose it as disasters.
    planet.toxicity.fill(1.0)
    planet.temperature_c.fill(75.0)
    planet._update_life(350)
    planet._update_biomass_maps()

    assert before > 0.0
    assert planet.total_biomass < before
    assert planet.total_dead_matter > 0.0


def test_dead_matter_render_becomes_visible_for_small_nonzero_values():
    planet = Planet.generate(PlanetConfig(width=32, height=16, seed=1617))
    planet.dead_matter[4:8, 10:14] = 0.015

    rgb = render_layer(planet, "dead_matter")

    assert rgb.shape == (16, 32, 3)
    assert rgb[5, 11].mean() > rgb[0, 0].mean()


def test_top_species_near_returns_local_lineages_sorted():
    config = PlanetConfig(
        width=64,
        height=32,
        seed=1718,
        abiogenesis_rate=0.25,
        abiogenesis_fertility_threshold=0.20,
    )
    planet = Planet.generate(config)
    planet.step(900)
    assert planet.species

    y, x = np.unravel_index(np.argmax(planet.biomass), planet.biomass.shape)
    local = planet.top_species_near(int(x), int(y), radius=5, limit=4)

    assert local
    assert len(local) <= 4
    assert all(local[i][1] >= local[i + 1][1] for i in range(len(local) - 1))
    assert all(local_total > 0.0 for _species, local_total, _global_total in local)
    assert all(global_total >= local_total for _species, local_total, global_total in local)


def test_lineage_lookup_and_descendant_count_are_available():
    config = PlanetConfig(
        width=64,
        height=32,
        seed=1819,
        abiogenesis_rate=0.30,
        abiogenesis_fertility_threshold=0.18,
        speciation_rate=0.0020,
    )
    planet = Planet.generate(config)
    for _ in range(6):
        planet.step(600)
    assert planet.species

    first = planet.species[0]
    assert planet.species_by_id(first.id) is first
    assert planet.species_index_by_id(first.id) == 0
    assert planet.species_total_population(first.id) >= 0.0
    assert planet.descendant_count(first.id) >= 0
    assert planet.species_by_id(999999) is None
    assert planet.species_index_by_id(999999) is None


def test_lineage_genealogy_helpers_return_ordered_family_rows():
    config = PlanetConfig(
        width=64,
        height=32,
        seed=1820,
        abiogenesis_rate=0.35,
        abiogenesis_fertility_threshold=0.18,
        speciation_rate=0.0030,
    )
    planet = Planet.generate(config)
    for _ in range(8):
        planet.step(600)
    assert planet.species

    # Prefer a lineage that actually has a parent when available.
    selected = next((species for species in planet.species if species.parent_id is not None), planet.species[0])
    ancestors = planet.lineage_ancestors(selected.id, include_self=True)
    children = planet.lineage_children(selected.id)
    descendants = planet.lineage_descendants(selected.id)

    assert ancestors
    assert ancestors[-1].id == selected.id
    assert all(child.parent_id == selected.id for child in children)
    assert all(depth >= 1 for depth, _species in descendants)
    assert planet.descendant_count(selected.id) == len(descendants)
    if selected.parent_id is not None:
        assert any(ancestor.id == selected.parent_id for ancestor in ancestors)


def test_lineage_habitat_summary_describes_current_distribution():
    config = PlanetConfig(
        width=64,
        height=32,
        seed=1920,
        abiogenesis_rate=0.30,
        abiogenesis_fertility_threshold=0.18,
    )
    planet = Planet.generate(config)
    planet.step(900)
    assert planet.species

    species, _total = planet.top_species(limit=1)[0]
    summary = planet.lineage_habitat_summary(species.id)

    assert summary.species_id == species.id
    assert summary.total_population > 0.0
    assert summary.occupied_cells > 0
    assert summary.strongest_cell is not None
    sx, sy = summary.strongest_cell
    assert 0 <= sx < planet.config.width
    assert 0 <= sy < planet.config.height
    assert 0.0 <= summary.mean_water_access <= 1.0
    assert 0.0 <= summary.mean_fertility <= 1.0
    assert 0.0 <= summary.mean_toxicity <= 1.0
    assert summary.main_habitat != "none"


def test_habitat_summary_for_missing_lineage_is_empty():
    planet = Planet.generate(PlanetConfig(width=64, height=32, seed=2021))
    summary = planet.lineage_habitat_summary(123456)

    assert summary.species_id == 123456
    assert summary.total_population == 0.0
    assert summary.occupied_cells == 0
    assert summary.strongest_cell is None
    assert summary.main_habitat == "none"


def test_setup_fields_target_valid_planet_config_values():
    config = PlanetConfig()
    keys = {field.key for field in PLANET_SETUP_FIELDS}

    assert "sea_level" in keys
    assert "volcanic_activity_fraction" in keys
    assert "equator_temperature_c" in keys

    for field in PLANET_SETUP_FIELDS:
        value = getattr(config, field.key)
        assert field.minimum <= float(value) <= field.maximum
        assert field.format_value(value)
        assert len(field.low_color) == 3
        assert len(field.high_color) == 3
        assert all(0 <= channel <= 255 for channel in (*field.low_color, *field.high_color))



def test_setup_detail_fields_are_kept_in_deterministic_bottom_group():
    primary_keys = [field.key for field in PRIMARY_SETUP_FIELDS]
    detail_keys = [field.key for field in DETAIL_SETUP_FIELDS]
    all_keys = [field.key for field in PLANET_SETUP_FIELDS]

    assert detail_keys == ["detail_octaves", "detail_gain"]
    assert "detail_octaves" not in primary_keys
    assert "detail_gain" not in primary_keys
    assert all_keys[-2:] == detail_keys


def test_major_events_are_logged_for_observer_panel():
    config = PlanetConfig(
        width=64,
        height=32,
        seed=2122,
        abiogenesis_rate=0.35,
        abiogenesis_fertility_threshold=0.18,
        speciation_rate=0.0025,
        volcanic_pulse_rate=0.20,
    )
    planet = Planet.generate(config)
    for _ in range(5):
        planet.step(500)

    events = planet.recent_events(limit=12)
    assert events
    assert all(event.tick <= planet.tick for event in events)
    assert any(event.kind in {"birth", "branch", "volcanism"} for event in events)
    assert len(planet.recent_events(limit=3)) <= 3


def test_geological_intro_stage_progression_is_ordered():
    assert geological_intro_stage(0.0).title.startswith("Void")
    assert geological_intro_stage(0.20).title == "Cloud collapse"
    assert geological_intro_stage(0.40).title.startswith("Explosive volcanic")
    assert geological_intro_stage(0.56).title.startswith("Smoke")
    assert geological_intro_stage(0.70).title.startswith("Condensation")
    assert geological_intro_stage(0.84).title.startswith("Oceans")
    assert geological_intro_stage(1.0).title == "Young stable planet"


def test_geological_intro_layer_is_visual_only_and_converges_to_planet_shape():
    planet = Planet.generate(PlanetConfig(width=64, height=32, seed=2324))
    before_tick = planet.tick
    before_biomass = planet.total_biomass

    early = render_geological_intro_layer(planet, 0.0)
    late = render_geological_intro_layer(planet, 1.0)
    biome = render_layer(planet, "biome", overlay_mode="off")

    assert early.shape == (32, 64, 3)
    assert late.shape == (32, 64, 3)
    assert early.dtype == np.uint8
    assert late.dtype == np.uint8
    assert not np.array_equal(early, late)
    assert np.mean(np.abs(late.astype(float) - biome.astype(float))) < 16.0
    assert planet.tick == before_tick
    assert planet.total_biomass == before_biomass


def test_phase5_traits_include_consumption_defense_and_storage():
    config = PlanetConfig(
        width=64,
        height=32,
        seed=2425,
        abiogenesis_rate=0.30,
        abiogenesis_fertility_threshold=0.18,
    )
    planet = Planet.generate(config)
    planet.step(900)
    assert planet.species

    traits = planet.species[0].traits
    assert 0.0 <= traits.living_consumption <= 1.0
    assert 0.0 <= traits.defense <= 1.0
    assert 0.0 <= traits.storage <= 1.0
    assert planet.species_strategy_label(planet.species[0])


def test_phase5_biotic_pressure_layer_appears_when_life_interacts():
    config = PlanetConfig(
        width=64,
        height=32,
        seed=2526,
        abiogenesis_rate=0.35,
        abiogenesis_fertility_threshold=0.18,
        speciation_rate=0.0025,
    )
    planet = Planet.generate(config)
    for _ in range(8):
        planet.step(500)

    assert planet.total_biomass > 0.0
    assert float(planet.biotic_pressure.min()) >= 0.0
    assert float(planet.biotic_pressure.max()) <= 1.0
    assert float(planet.biotic_pressure.max()) > 0.0

    rgb = render_layer(planet, "biotic_pressure")
    assert rgb.shape == (32, 64, 3)
    assert rgb.dtype == np.uint8


def test_phase5_consumer_advantage_depends_on_available_living_biomass():
    config = PlanetConfig(width=48, height=24, seed=2627, abiogenesis_rate=0.0)
    planet = Planet.generate(config)

    from alife.life import LifeSpecies, LifeTraits, species_color

    prey_traits = LifeTraits(
        photosynthesis=0.8, chemosynthesis=0.1, organic_absorption=0.1,
        living_consumption=0.0, defense=0.05, storage=0.1,
        size=0.22, structure=0.16, surface_area=0.62, armor=0.04,
        speed=0.05, longevity=0.18, fragility=0.36, complexity=0.08,
        temperature_optimum_c=20.0, temperature_tolerance_c=30.0,
        water_preference=0.5, water_tolerance=0.8, toxicity_tolerance=1.0,
        reproduction_rate=0.9, metabolism_cost=0.04, dispersal=0.0, mutation_rate=0.01,
    )
    consumer_traits = LifeTraits(
        photosynthesis=0.0, chemosynthesis=0.0, organic_absorption=0.0,
        living_consumption=1.0, defense=0.1, storage=0.1,
        size=0.30, structure=0.12, surface_area=0.28, armor=0.05,
        speed=0.42, longevity=0.18, fragility=0.38, complexity=0.20,
        temperature_optimum_c=20.0, temperature_tolerance_c=30.0,
        water_preference=0.5, water_tolerance=0.8, toxicity_tolerance=1.0,
        reproduction_rate=1.1, metabolism_cost=0.04, dispersal=0.0, mutation_rate=0.01,
    )
    planet.species.extend([
        LifeSpecies(1, None, "Prey-001", species_color(prey_traits), prey_traits, planet.tick),
        LifeSpecies(2, None, "Consumer-002", species_color(consumer_traits), consumer_traits, planet.tick),
    ])
    planet.populations[0, 8:16, 18:30] = 0.32
    planet.populations[1, 10:14, 22:26] = 0.08
    planet.nutrients.fill(0.8)
    planet.light.fill(0.8)
    planet.fertility.fill(0.8)
    planet.temperature_c.fill(20.0)
    planet.humidity.fill(0.5)
    planet.toxicity.fill(0.0)
    planet._update_biomass_maps()
    consumer_before = planet.species_total_population(2)

    planet.step(160)

    assert planet.species_total_population(2) >= consumer_before
    assert float(planet.biotic_pressure.max()) > 0.0


def test_lineage_habitat_summary_includes_biotic_pressure():
    config = PlanetConfig(
        width=64,
        height=32,
        seed=2728,
        abiogenesis_rate=0.30,
        abiogenesis_fertility_threshold=0.18,
    )
    planet = Planet.generate(config)
    planet.step(900)
    species, _total = planet.top_species(limit=1)[0]
    summary = planet.lineage_habitat_summary(species.id)

    assert 0.0 <= summary.mean_biotic_pressure <= 1.0

def test_phase5_atmosphere_layers_render_and_drift():
    planet = Planet.generate(PlanetConfig(width=64, height=32, seed=2728))

    clouds_early = render_layer(planet, "clouds")
    rain_early = render_layer(planet, "rain")
    planet.step(420)
    clouds_later = render_layer(planet, "clouds")
    rain_later = render_layer(planet, "rain")

    assert clouds_early.shape == rain_early.shape == (32, 64, 3)
    assert clouds_early.dtype == np.uint8
    assert rain_early.dtype == np.uint8
    assert not should_apply_life_overlay("clouds", "biomass")
    assert not should_apply_life_overlay("rain", "biomass")
    assert not np.array_equal(clouds_early, clouds_later)
    assert not np.array_equal(rain_early, rain_later)
    assert float(clouds_early.std()) > 0.0
    assert float(rain_early.std()) > 0.0



def test_weather_overlay_applies_only_to_biome_and_changes_render():
    planet = Planet.generate(PlanetConfig(width=64, height=32, seed=2829))

    plain = render_layer(planet, "biome", overlay_mode="off", weather_overlay_mode="off")
    cloudy = render_layer(planet, "biome", overlay_mode="off", weather_overlay_mode="clouds")
    rainy = render_layer(planet, "biome", overlay_mode="off", weather_overlay_mode="rain")
    clouds_layer = render_layer(planet, "clouds", weather_overlay_mode="rain")
    clouds_plain = render_layer(planet, "clouds", weather_overlay_mode="off")

    assert should_apply_weather_overlay("biome", "clouds")
    assert not should_apply_weather_overlay("clouds", "rain")
    assert not should_apply_weather_overlay("biomass", "rain")
    assert plain.shape == cloudy.shape == rainy.shape == (32, 64, 3)
    assert not np.array_equal(plain, cloudy)
    assert not np.array_equal(plain, rainy)
    np.testing.assert_array_equal(clouds_layer, clouds_plain)


def test_weather_overlay_drifts_over_time_on_biome():
    planet = Planet.generate(PlanetConfig(width=64, height=32, seed=2930))

    first = render_layer(planet, "biome", weather_overlay_mode="rain")
    planet.step(360)
    later = render_layer(planet, "biome", weather_overlay_mode="rain")

    assert not np.array_equal(first, later)


def test_season_labels_and_positions_are_available_for_observer_ui():
    planet = Planet.generate(PlanetConfig(width=64, height=32, seed=3031, seasonal_period_ticks=120))

    assert season_position(planet) == (1, 1, 120)
    assert season_label(planet)
    planet.step(121)
    assert season_position(planet) == (2, 2, 120)
    assert season_label(planet)

def test_default_year_length_is_seed_derived_and_reproducible():
    a = Planet.generate(PlanetConfig(width=64, height=32, seed=4142))
    b = Planet.generate(PlanetConfig(width=64, height=32, seed=4142))
    c = Planet.generate(PlanetConfig(width=64, height=32, seed=4243))

    assert 1500 <= a.config.seasonal_period_ticks <= 3600
    assert a.config.seasonal_period_ticks == b.config.seasonal_period_ticks
    assert a.config.seasonal_period_ticks != c.config.seasonal_period_ticks


def test_weather_all_overlay_and_globe_projection_render():
    planet = Planet.generate(PlanetConfig(width=64, height=32, seed=4344))

    plain = render_layer(planet, "biome", weather_overlay_mode="off")
    all_weather = render_layer(planet, "biome", weather_overlay_mode="all")
    globe = render_globe_texture(all_weather, (160, 120), rotation=0.4)

    assert should_apply_weather_overlay("biome", "all")
    assert plain.shape == all_weather.shape == (32, 64, 3)
    assert not np.array_equal(plain, all_weather)
    assert globe.shape == (120, 160, 3)
    assert globe.dtype == np.uint8
    assert float(globe.std()) > 0.0




def test_event_log_filters_keep_branches_visible_while_hiding_noise():
    planet = Planet.generate(PlanetConfig(width=32, height=16, seed=4445, seasonal_period_ticks=1200))
    planet.tick = 5000
    planet.event_log = [
        SimulationEvent(100, "birth", "Root-001 appears"),
        SimulationEvent(2200, "birth", "Late-002 appears"),
        SimulationEvent(2300, "volcanism", "volcanic pulse"),
        SimulationEvent(2400, "branch", "Child-003 branches from Root-001"),
        SimulationEvent(2500, "extinction", "Root-001 goes extinct"),
    ]

    viewer = PlanetViewer.__new__(PlanetViewer)
    viewer.planet = planet
    viewer.event_filter_show_volcanism = False
    viewer.event_birth_filter_mode = "early"

    visible = viewer._filtered_events_newest(limit=None)
    assert [event.kind for event in visible] == ["extinction", "branch", "birth"]
    assert visible[1].kind == "branch"

    viewer.event_birth_filter_mode = "hidden"
    visible = viewer._filtered_events_newest(limit=None)
    assert [event.kind for event in visible] == ["extinction", "branch"]

    viewer.event_filter_show_volcanism = True
    viewer.event_birth_filter_mode = "all"
    visible = viewer._filtered_events_newest(limit=None)
    assert [event.kind for event in visible] == ["extinction", "branch", "volcanism", "birth", "birth"]


def test_birth_event_cutoff_uses_at_least_one_seed_year():
    planet = Planet.generate(PlanetConfig(width=32, height=16, seed=4546, seasonal_period_ticks=1600))
    viewer = PlanetViewer.__new__(PlanetViewer)
    viewer.planet = planet

    assert viewer._birth_event_cutoff_tick() == 1600


def test_biomass_overlay_uses_different_land_and_ocean_tints():
    planet = Planet.generate(PlanetConfig(width=64, height=32, seed=5657))
    # Force visible biomass on one land and one ocean cell so the overlay can be
    # validated without depending on stochastic life spread.
    planet.biomass.fill(0.0)
    land_y, land_x = np.argwhere(planet.land)[0]
    ocean_y, ocean_x = np.argwhere(~planet.land)[0]
    planet.biomass[land_y, land_x] = 0.8
    planet.biomass[ocean_y, ocean_x] = 0.8

    plain = render_layer(planet, "biome", overlay_mode="off")
    overlay = render_layer(planet, "biome", overlay_mode="biomass")

    assert not np.array_equal(plain[land_y, land_x], overlay[land_y, land_x])
    assert not np.array_equal(plain[ocean_y, ocean_x], overlay[ocean_y, ocean_x])
    assert not np.array_equal(overlay[land_y, land_x], overlay[ocean_y, ocean_x])


def test_event_log_marks_descendant_events_in_yellow():
    viewer = PlanetViewer.__new__(PlanetViewer)
    branch_color = viewer._event_kind_color("branch")
    birth_color = viewer._event_kind_color("birth")

    assert branch_color[0] >= 240 and branch_color[1] >= 190 and branch_color[2] < 140
    assert branch_color != birth_color


def test_abiogenesis_reserves_lineage_capacity_for_descendants():
    planet = Planet.generate(
        PlanetConfig(
            width=48,
            height=24,
            seed=5758,
            max_species=20,
            min_branch_reserved_slots=6,
            abiogenesis_rate=1.0,
            abiogenesis_fertility_threshold=0.0,
            speciation_rate=0.0,
        )
    )
    planet.step(4000)

    assert len(planet.species) <= planet.config.max_species - planet.config.min_branch_reserved_slots


def test_branching_can_create_descendant_lineages_when_capacity_exists():
    planet = Planet.generate(
        PlanetConfig(
            width=48,
            height=24,
            seed=5859,
            max_species=20,
            abiogenesis_rate=0.0,
            speciation_rate=0.5,
            mutation_strength=0.12,
        )
    )
    y, x = np.unravel_index(np.argmax(planet.fertility), planet.fertility.shape)
    planet._create_seed_species(int(y), int(x))
    planet.populations[0] *= 0.0
    planet._add_population_blob(0, int(y), int(x), amount=2.5, radius=5.0)
    planet._maybe_branch_lineages(steps=5000)

    assert any(species.parent_id is not None for species in planet.species)
    assert any(event.kind == "branch" for event in planet.event_log)


def test_star_background_is_seeded_and_rotates():
    first = render_star_background((180, 120), seed=4242, rotation=0.0)
    repeat = render_star_background((180, 120), seed=4242, rotation=0.0)
    rotated = render_star_background((180, 120), seed=4242, rotation=0.8)

    assert first.shape == (120, 180, 3)
    assert first.dtype == np.uint8
    np.testing.assert_array_equal(first, repeat)
    assert not np.array_equal(first, rotated)
    assert int(first.max()) > 80


def test_globe_projection_can_render_rotating_star_background():
    planet = Planet.generate(PlanetConfig(width=64, height=32, seed=5151))
    texture = render_layer(planet, "biome")
    sky_a = render_globe_texture(texture, (180, 140), rotation=0.2, star_rotation=0.0, seed=planet.config.seed)
    sky_b = render_globe_texture(texture, (180, 140), rotation=0.2, star_rotation=1.0, seed=planet.config.seed)

    assert sky_a.shape == (140, 180, 3)
    assert sky_a.dtype == np.uint8
    assert not np.array_equal(sky_a, sky_b)
    # Outside the globe should no longer be a flat black background in 3D mode.
    corner = sky_a[:24, :24]
    assert float(corner.std()) > 0.0


def test_phase6_mobility_fields_are_bounded_and_renderable():
    config = PlanetConfig(
        width=64,
        height=32,
        seed=6263,
        abiogenesis_rate=0.30,
        abiogenesis_fertility_threshold=0.18,
        active_migration_rate=0.12,
    )
    planet = Planet.generate(config)
    for _ in range(6):
        planet.step(500)

    assert planet.total_biomass > 0.0
    assert float(planet.migration_pressure.min()) >= 0.0
    assert float(planet.migration_pressure.max()) <= 1.0
    assert float(planet.isolation_pressure.min()) >= 0.0
    assert float(planet.isolation_pressure.max()) <= 1.0
    assert float(planet.migration_pressure.max()) > 0.0

    migration_rgb = render_layer(planet, "migration_pressure")
    isolation_rgb = render_layer(planet, "isolation_pressure")
    assert migration_rgb.shape == isolation_rgb.shape == (32, 64, 3)
    assert migration_rgb.dtype == isolation_rgb.dtype == np.uint8


def test_phase6_isolation_biases_branch_locations_and_descendants():
    planet = Planet.generate(
        PlanetConfig(
            width=48,
            height=24,
            seed=6364,
            max_species=20,
            abiogenesis_rate=0.0,
            speciation_rate=0.5,
            active_migration_rate=0.0,
            mutation_strength=0.12,
        )
    )
    y, x = np.unravel_index(np.argmax(planet.fertility), planet.fertility.shape)
    planet._create_seed_species(int(y), int(x))
    planet.populations[0].fill(0.0)
    planet._add_population_blob(0, int(y), int(x), amount=1.0, radius=4.0)
    # Add a small isolated colony far away so frontier pressure has something to select.
    planet._add_population_blob(0, int(max(1, y // 2)), int((x + 17) % planet.config.width), amount=0.45, radius=1.2)
    planet._update_biomass_maps()

    assert planet._species_isolation_score(0) > 0.0
    planet._maybe_branch_lineages(steps=5000)

    assert any(species.parent_id == 1 for species in planet.species)
    assert any(event.kind == "branch" for event in planet.event_log)


def test_lineage_habitat_summary_includes_phase6_observer_fields():
    config = PlanetConfig(
        width=64,
        height=32,
        seed=6465,
        abiogenesis_rate=0.30,
        abiogenesis_fertility_threshold=0.18,
        active_migration_rate=0.12,
    )
    planet = Planet.generate(config)
    planet.step(1200)
    species, _total = planet.top_species(limit=1)[0]
    summary = planet.lineage_habitat_summary(species.id)

    assert 0.0 <= summary.mean_migration_pressure <= 1.0
    assert 0.0 <= summary.mean_isolation_pressure <= 1.0


def test_viewer_defaults_to_3d_and_all_weather_without_running_pygame_init():
    viewer = PlanetViewer.__new__(PlanetViewer)
    # Guard the source-level defaults without opening a Pygame window.
    source = PlanetViewer.__init__.__code__.co_consts
    assert "all" in source
    assert "3d" in source


def test_event_log_modal_click_selects_event_species_and_location():
    planet = Planet.generate(PlanetConfig(width=32, height=16, seed=6768))
    y, x = np.unravel_index(np.argmax(planet.fertility), planet.fertility.shape)
    planet._create_seed_species(int(y), int(x))
    species_id = planet.species[0].id

    viewer = PlanetViewer.__new__(PlanetViewer)
    viewer.planet = planet
    viewer.selected_species_id = None
    viewer.selected_cell = None
    viewer.event_log_modal_open = True
    viewer.event_log_modal_close_rect = pygame.Rect(900, 900, 10, 10)
    viewer.event_log_filter_button_rects = []
    viewer.event_log_modal_row_rects = [(pygame.Rect(10, 10, 120, 20), species_id, (int(x), int(y)))]
    viewer.event_log_modal_rect = pygame.Rect(0, 0, 400, 300)

    viewer._handle_event_log_modal_click((15, 15))

    assert viewer.selected_species_id == species_id
    assert viewer.selected_cell == (int(x), int(y))
    assert viewer.event_log_modal_open is False


def test_global_life_tree_rows_include_roots_and_descendants_in_tree_order():
    planet = Planet.generate(
        PlanetConfig(width=40, height=20, seed=6869, abiogenesis_rate=0.0, speciation_rate=1.0)
    )
    y, x = np.unravel_index(np.argmax(planet.fertility), planet.fertility.shape)
    planet._create_seed_species(int(y), int(x))
    planet.populations[0].fill(0.0)
    planet._add_population_blob(0, int(y), int(x), amount=2.0, radius=4.0)
    planet._maybe_branch_lineages(steps=4000)

    viewer = PlanetViewer.__new__(PlanetViewer)
    viewer.planet = planet

    rows = viewer._global_life_tree_rows()
    assert rows
    assert rows[0][0] == 0
    assert rows[0][1].parent_id is None
    assert any(depth >= 1 and species.parent_id == rows[0][1].id for depth, species in rows)


def test_life_tree_modal_click_selects_lineage_and_strongest_cell():
    planet = Planet.generate(PlanetConfig(width=32, height=16, seed=6970))
    y, x = np.unravel_index(np.argmax(planet.fertility), planet.fertility.shape)
    planet._create_seed_species(int(y), int(x))
    species_id = planet.species[0].id
    planet._update_biomass_maps()

    viewer = PlanetViewer.__new__(PlanetViewer)
    viewer.planet = planet
    viewer.selected_species_id = None
    viewer.selected_cell = None
    viewer.life_tree_modal_open = True
    viewer.life_tree_modal_close_rect = pygame.Rect(900, 900, 10, 10)
    viewer.life_tree_modal_rect = pygame.Rect(0, 0, 400, 300)
    viewer.life_tree_modal_row_rects = [(pygame.Rect(10, 10, 120, 20), species_id)]

    viewer._handle_life_tree_modal_click((15, 15))

    assert viewer.selected_species_id == species_id
    assert viewer.selected_cell is not None
    assert viewer.life_tree_modal_open is True


def test_specimen_keywords_are_trait_driven_observer_text():
    from alife.life import LifeSpecies, LifeTraits

    species = LifeSpecies(
        1,
        None,
        "Demo-001",
        (120, 210, 140),
        LifeTraits(
            photosynthesis=0.85,
            chemosynthesis=0.10,
            organic_absorption=0.20,
            living_consumption=0.05,
            defense=0.72,
            storage=0.18,
            size=0.30,
            structure=0.32,
            surface_area=0.72,
            armor=0.95,
            speed=0.12,
            longevity=0.28,
            fragility=0.20,
            complexity=0.22,
            temperature_optimum_c=21.0,
            temperature_tolerance_c=15.0,
            water_preference=0.5,
            water_tolerance=0.5,
            toxicity_tolerance=0.22,
            reproduction_rate=0.8,
            metabolism_cost=0.05,
            dispersal=0.12,
            mutation_rate=0.01,
        ),
        created_tick=0,
    )

    words = specimen_keywords(species)

    assert words
    assert "sun-catcher" in words
    assert "armored" in words
    assert len(words) <= 3


def test_panel_toggle_source_controls_exist_without_opening_window():
    source = PlanetViewer._handle_key.__code__.co_consts + PlanetViewer._draw_settings_row.__code__.co_consts

    assert "Panel: wide" in source
    assert "Panel: narrow" in source
    assert "Hide panel" in source


def test_panel_width_modes_control_section_visibility_without_opening_window():
    source = inspect.getsource(PlanetViewer.__init__) + inspect.getsource(PlanetViewer._toggle_panel_width) + inspect.getsource(PlanetViewer._draw_panel)

    assert "all_runtime_section_keys" in source
    assert 'self.collapsed_sections = {"legend"}' in source
    assert "self.collapsed_sections = set(self.all_runtime_section_keys)" in source
    assert "self._draw_life_summary" in source
    assert "self._draw_current_layer_legend" in source


def test_compact_event_log_click_selects_event_species_and_location():
    planet = Planet.generate(PlanetConfig(width=32, height=16, seed=7172))
    y, x = np.unravel_index(np.argmax(planet.fertility), planet.fertility.shape)
    planet._create_seed_species(int(y), int(x))
    species_id = planet.species[0].id

    viewer = PlanetViewer.__new__(PlanetViewer)
    viewer.planet = planet
    viewer.selected_species_id = None
    viewer.selected_cell = None
    viewer.panel_collapsed = False
    viewer.in_setup_screen = False
    viewer.panel_tab_rect = pygame.Rect(900, 900, 10, 10)
    viewer.fullscreen_button_rect = pygame.Rect(900, 900, 10, 10)
    viewer.projection_button_rect = pygame.Rect(900, 900, 10, 10)
    viewer.life_overlay_button_rect = pygame.Rect(900, 900, 10, 10)
    viewer.weather_overlay_button_rect = pygame.Rect(900, 900, 10, 10)
    viewer.panel_layout_button_rect = pygame.Rect(900, 900, 10, 10)
    viewer.panel_hide_button_rect = pygame.Rect(900, 900, 10, 10)
    viewer.life_tree_button_rect = pygame.Rect(900, 900, 10, 10)
    viewer.runtime_save_preset_button_rect = pygame.Rect(900, 900, 10, 10)
    viewer.genealogy_button_rect = pygame.Rect(900, 900, 10, 10)
    viewer.event_log_button_rect = pygame.Rect(900, 900, 10, 10)
    viewer.life_tree_modal_open = False
    viewer.event_log_modal_open = False
    viewer.genealogy_modal_species_id = None
    viewer.section_header_rects = []
    viewer.species_row_rects = []
    viewer.map_rect = pygame.Rect(900, 900, 10, 10)
    viewer.event_log_row_rects = [(pygame.Rect(10, 10, 120, 20), species_id, (int(x), int(y)))]

    viewer._handle_mouse_click((15, 15))

    assert viewer.selected_species_id == species_id
    assert viewer.selected_cell == (int(x), int(y))


def test_globe_texture_seam_feather_softens_wrapped_edges():
    from alife.viewer import _feather_horizontal_texture_seam

    texture = np.zeros((4, 20, 3), dtype=np.uint8)
    texture[:, :10] = (255, 0, 0)
    texture[:, 10:] = (0, 0, 255)

    feathered = _feather_horizontal_texture_seam(texture, columns=4)

    assert feathered.shape == texture.shape
    assert not np.array_equal(feathered[:, 0], texture[:, 0])
    assert not np.array_equal(feathered[:, -1], texture[:, -1])


def test_phase6_final_observer_source_polish_markers():
    source = inspect.getsource(PlanetViewer._draw_event_log) + inspect.getsource(PlanetViewer._draw_life_summary)

    assert "★ DESC" in source
    assert "event_log_row_rects" in source
    assert "branches" in source


def test_observer_migration_and_isolation_labels_are_readable():
    from alife.viewer import observer_isolation_label, observer_migration_label

    assert observer_migration_label(0.0) == "settled core"
    assert "frontier" in observer_migration_label(0.2)
    assert observer_isolation_label(0.0) == "connected range"
    assert "branch" in observer_isolation_label(0.2)



def test_phase7_morphology_traits_are_seeded_and_mutate_within_bounds():
    planet = Planet.generate(PlanetConfig(width=64, height=32, seed=7374))
    y, x = np.unravel_index(np.argmax(planet.fertility), planet.fertility.shape)
    planet._create_seed_species(int(y), int(x))
    parent = planet.species[0]
    planet.populations[0].fill(0.0)
    planet._add_population_blob(0, int(y), int(x), amount=2.0, radius=4.0)
    planet._create_mutant_species(0, int(y), int(x))
    child = planet.species[1]

    for traits in (parent.traits, child.traits):
        for value in (
            traits.size, traits.structure, traits.surface_area, traits.armor,
            traits.speed, traits.longevity, traits.fragility, traits.complexity,
        ):
            assert 0.0 <= value <= 1.0
    assert child.parent_id == parent.id


def test_phase7_body_plan_layer_renders_population_weighted_morphology():
    planet = Planet.generate(PlanetConfig(width=64, height=32, seed=7475, abiogenesis_rate=0.20, abiogenesis_fertility_threshold=0.20))
    planet.step(600)
    assert planet.total_biomass > 0.0
    assert float(planet.morphology_index.min()) >= 0.0
    assert float(planet.morphology_index.max()) <= 1.0

    rgb = render_layer(planet, "body_plan")
    assert rgb.shape == (32, 64, 3)
    assert rgb.dtype == np.uint8


def test_phase7_body_plan_source_markers_are_visible_to_observer():
    source = inspect.getsource(PlanetViewer._draw_selected_lineage) + inspect.getsource(PlanetViewer._draw_species_specimen)

    assert "Body plan" in source
    assert "traits.size" in source
    assert "traits.armor" in source
    assert "traits.speed" in source


def test_globe_renderer_rolls_texture_seam_to_hidden_side():
    from alife.viewer import _roll_texture_seam_behind

    texture = np.zeros((3, 12, 3), dtype=np.uint8)
    texture[:, :6] = (255, 0, 0)
    texture[:, 6:] = (0, 0, 255)

    rolled = _roll_texture_seam_behind(texture, rotation=0.0)

    assert rolled.shape == texture.shape
    # With rotation=0, the original texture center is placed at the visible
    # center of the globe; the hard rectangular cut stays at the hidden edge.
    np.testing.assert_array_equal(rolled[:, 6], texture[:, 0])


def test_phase6_final_polish_source_markers_are_kept_in_phase7():
    source = inspect.getsource(PlanetViewer._draw_event_log) + inspect.getsource(PlanetViewer._draw_selected_lineage) + inspect.getsource(render_globe_texture)

    assert "Event log ·" in source
    assert "★ DESC" in source
    assert "range:" in source
    assert "_roll_texture_seam_behind" in source
    assert "selected:" not in source


def test_simple_world_preset_round_trips_seed_and_setup(tmp_path):
    config = PlanetConfig(
        width=80,
        height=40,
        seed=9192,
        sea_level=0.57,
        continent_scale=7,
        detail_octaves=4,
        detail_gain=0.61,
        volcanic_activity_fraction=0.11,
        equator_temperature_c=35.0,
        pole_temperature_c=-22.0,
    )
    preset = planet_config_to_preset(config)
    restored = planet_config_from_preset(preset)

    assert restored.seed == config.seed
    assert restored.width == config.width
    assert restored.height == config.height
    assert restored.sea_level == config.sea_level
    assert restored.continent_scale == config.continent_scale
    assert restored.detail_gain == config.detail_gain
    assert restored.volcanic_activity_fraction == config.volcanic_activity_fraction

    path = save_world_preset(config, directory=tmp_path)
    assert path.exists()
    loaded = load_world_preset(path)
    assert loaded.seed == config.seed
    assert loaded.sea_level == config.sea_level
    assert list_world_presets(tmp_path, limit=5) == [path]


def test_setup_actions_can_save_and_open_load_modal_without_running_window(tmp_path, monkeypatch):
    planet = Planet.generate(PlanetConfig(width=32, height=16, seed=9293))
    viewer = PlanetViewer.__new__(PlanetViewer)
    viewer.planet = planet
    viewer.load_preset_modal_open = False
    viewer.setup_status_message = ""

    monkeypatch.setattr("alife.viewer.DEFAULT_PRESET_DIR", tmp_path)
    viewer._handle_setup_action("save_preset", "")

    presets = list_world_presets(tmp_path, limit=5)
    assert presets
    assert "Saved preset" in viewer.setup_status_message

    viewer._handle_setup_action("open_load_preset", "")
    assert viewer.load_preset_modal_open is True




def test_runtime_save_preset_button_saves_current_planet_without_running_window(tmp_path, monkeypatch):
    planet = Planet.generate(PlanetConfig(width=32, height=16, seed=9798, sea_level=0.57))
    viewer = PlanetViewer.__new__(PlanetViewer)
    viewer.planet = planet
    viewer.runtime_status_message = ""

    monkeypatch.setattr("alife.viewer.DEFAULT_PRESET_DIR", tmp_path)
    viewer._save_current_preset(status_target="runtime")

    presets = list_world_presets(tmp_path, limit=5)
    assert presets
    assert "Saved preset" in viewer.runtime_status_message
    loaded = load_world_preset(presets[0])
    assert loaded.seed == 9798
    assert loaded.sea_level == 0.57

def test_load_preset_modal_click_restores_planet_config(tmp_path):
    original = Planet.generate(PlanetConfig(width=32, height=16, seed=9394, sea_level=0.42))
    saved = save_world_preset(PlanetConfig(width=32, height=16, seed=9495, sea_level=0.62), directory=tmp_path)

    viewer = PlanetViewer.__new__(PlanetViewer)
    viewer.planet = original
    viewer.speed = 1
    viewer.in_setup_screen = True
    viewer.intro_active = False
    viewer.selected_cell = (1, 1)
    viewer.selected_species_id = 123
    viewer.setup_status_message = ""
    viewer.load_preset_modal_open = True
    viewer.load_preset_modal_close_rect = pygame.Rect(900, 900, 10, 10)
    viewer.load_preset_modal_rect = pygame.Rect(0, 0, 400, 300)
    viewer.load_preset_row_rects = [(pygame.Rect(10, 10, 120, 20), saved)]
    viewer._update_layout = lambda: None
    viewer._invalidate_cache = lambda: None

    viewer._handle_load_preset_modal_click((15, 15))

    assert viewer.planet.config.seed == 9495
    assert viewer.planet.config.sea_level == 0.62
    assert viewer.selected_cell is None
    assert viewer.selected_species_id is None
    assert viewer.load_preset_modal_open is False
    assert "Loaded" in viewer.setup_status_message



def test_world_name_is_seeded_editable_and_saved_with_thumbnail(tmp_path):
    config = PlanetConfig(width=48, height=24, seed=8081)
    name = random_world_name(config.seed)
    assert name == random_world_name(config.seed)
    assert name

    path = save_world_preset(config, directory=tmp_path, world_name="Gaia Test")
    loaded_config, loaded_name, data = read_world_preset_metadata(path)

    assert loaded_config.seed == config.seed
    assert loaded_name == "Gaia Test"
    assert "thumbnail" in data
    thumb = decode_world_thumbnail(data["thumbnail"])
    assert thumb is not None
    assert thumb.shape == (36, 72, 3)
    assert thumb.dtype == np.uint8
    assert float(thumb.std()) > 0.0


def test_viewer_world_name_editing_without_opening_window():
    planet = Planet.generate(PlanetConfig(width=32, height=16, seed=8182))
    viewer = PlanetViewer.__new__(PlanetViewer)
    viewer.planet = planet
    viewer.world_name = "Old"
    viewer.world_name_edit_active = True

    viewer._handle_world_name_key(pygame.K_BACKSPACE, "")
    viewer._handle_world_name_key(pygame.K_UNKNOWN, "X")
    viewer._handle_world_name_key(pygame.K_RETURN, "")

    assert viewer.world_name == "OlX"
    assert viewer.world_name_edit_active is False


def test_phase8_planetary_events_create_climate_stress_and_logs():
    config = PlanetConfig(
        width=64,
        height=32,
        seed=8283,
        planetary_event_rate=1.0,
        planetary_event_min_duration_fraction=0.05,
        planetary_event_max_duration_fraction=0.06,
    )
    planet = Planet.generate(config)
    planet.step(20)

    assert planet.planetary_event_label != "stable climate"
    assert planet.planetary_event_ticks_remaining > 0
    assert any(event.kind == "climate" for event in planet.event_log)
    assert float(planet.climate_stress.max()) >= 0.0
    rgb = render_layer(planet, "climate_stress")
    assert rgb.shape == (32, 64, 3)
    assert rgb.dtype == np.uint8


def test_phase8_source_markers_are_visible_to_observer():
    source = inspect.getsource(PlanetViewer._draw_panel) + inspect.getsource(PlanetViewer._draw_world_name_header) + inspect.getsource(PlanetViewer._draw_load_preset_modal)

    assert "Phase 8" in source
    assert "World name" in source
    assert "thumbnail" in source
