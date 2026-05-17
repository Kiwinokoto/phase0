"""Phase 5 prototype for richer abstract proto-ecology."""

from .config import PlanetConfig
from .life import LifeSpecies, LifeTraits
from .planet import LineageHabitatSummary, Planet, SimulationEvent

__all__ = ["PlanetConfig", "Planet", "LineageHabitatSummary", "SimulationEvent", "LifeSpecies", "LifeTraits"]
