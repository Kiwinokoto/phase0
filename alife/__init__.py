"""Phase 3 prototype for a dynamic 2D planet with first proto-life."""

from .config import PlanetConfig
from .life import LifeSpecies, LifeTraits
from .planet import Planet

__all__ = ["PlanetConfig", "Planet", "LifeSpecies", "LifeTraits"]
