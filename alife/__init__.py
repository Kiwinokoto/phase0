"""Phase 4 prototype for a dynamic 2D planet with constrained proto-ecology."""

from .config import PlanetConfig
from .life import LifeSpecies, LifeTraits
from .planet import LineageHabitatSummary, Planet, SimulationEvent

__all__ = ["PlanetConfig", "Planet", "LineageHabitatSummary", "SimulationEvent", "LifeSpecies", "LifeTraits"]
