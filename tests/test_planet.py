import numpy as np

from alife import Planet, PlanetConfig
from alife.planet import SimulationEvent
from alife.viewer import DETAIL_SETUP_FIELDS, PLANET_SETUP_FIELDS, PRIMARY_SETUP_FIELDS, PlanetViewer, geological_intro_stage, render_geological_intro_layer, render_globe_texture, render_layer, render_star_background, season_label, season_position, should_apply_life_overlay, should_apply_weather_overlay


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
        temperature_optimum_c=20.0, temperature_tolerance_c=30.0,
        water_preference=0.5, water_tolerance=0.8, toxicity_tolerance=1.0,
        reproduction_rate=0.9, metabolism_cost=0.04, dispersal=0.0, mutation_rate=0.01,
    )
    consumer_traits = LifeTraits(
        photosynthesis=0.0, chemosynthesis=0.0, organic_absorption=0.0,
        living_consumption=1.0, defense=0.1, storage=0.1,
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
