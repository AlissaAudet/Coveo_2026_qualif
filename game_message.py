from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from enum import Enum, unique
from typing import Optional


@dataclass_json
@dataclass
class Position:
    """A position on the game map."""

    x: int
    y: int


@dataclass_json
@dataclass
class Colony:
    """A colony on the game map."""

    position: Position
    """Position of the colony on the map."""
    nutrients: int
    """Current value of the nutriments."""
    futureNutrients: list[int]
    """Future value of the nutriments."""


@dataclass_json
@dataclass
class GameMap:
    width: int
    """Width of the game map in tiles."""
    height: int
    """Height of the game map in tiles."""
    biomass: list[list[int]]
    """Biomass per tile."""
    colonies: list[Colony]
    """List of all colonies on the map."""


@dataclass_json
@dataclass
class TeamGameState:
    currentTickNumber: int
    """Current tick number in the game."""
    lastTickErrors: list[str]
    """List of errors from the last tick."""
    score: int
    """Current score of the game."""
    availableBiomass: int
    """Amount of biomass available to the player to be placed on the map."""
    maximumNumberOfBiomassPerTurn: int
    """Maximum amount of biomass that can be added or removed in a single turn."""
    maximumNumberOfBiomassOnMap: int
    """Maximum amount of biomass that can be on the map at once."""
    map: GameMap
    """The game map containing biomass and colonies."""


class Action:
    type: str


@dataclass_json
@dataclass
class AddBiomassAction(Action):
    """Add biomass to a specific position on the map. Amount must be greater than zero. Position must be inside the map boundaries."""

    amount: int
    position: Position
    type: str = "ADD_BIOMASS"


@dataclass_json
@dataclass
class RemoveBiomassAction(Action):
    """Remove biomass from a specific position on the map. Amount must be greater than zero. Position must be inside the map boundaries."""

    amount: int
    position: Position
    type: str = "REMOVE_BIOMASS"
