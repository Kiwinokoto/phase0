"""Phase 4 prototype for a dynamic 2D planet with constrained proto-ecology."""

from .config import PlanetConfig
from .life import LifeSpecies, LifeTraits
from .planet import Planet

__all__ = ["PlanetConfig", "Planet", "LifeSpecies", "LifeTraits"]
