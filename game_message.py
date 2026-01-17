from dataclasses import dataclass, field
from enum import Enum, unique
from typing import Optional


@dataclass(slots=True)
class Constants:
    """Game constants. These will never change during the game."""

    neutralTeamId: str
    """Team id used for neutral tiles."""
    maxTicks: int
    """Maximum number of ticks before the game ends."""


@dataclass(slots=True)
class Position:
    """A two-dimensional point on the map."""

    x: int
    """X coordinate of the position. 0 is the column on the left."""
    y: int
    """Y coordinate of the position. 0 is the top row."""


@dataclass(slots=True)
class Spore:
    """A spore entity that can move and engage in combat."""

    id: str
    """Unique identifier of the spore."""
    teamId: str
    """Team that controls this spore."""
    position: Position
    """Current position of the spore."""
    biomass: int
    """Current biomass level determining combat strength."""


@dataclass(slots=True)
class Spawner:
    """A spawner that can produce spores."""

    id: str
    """Unique identifier of the spawner."""
    teamId: str
    """Team that controls this spawner."""
    position: Position
    """Position of the spawner."""


@dataclass(slots=True)
class TeamInfo:
    """Information for a team."""

    teamId: str
    """Team identifier."""
    isAlive: bool
    """True if the team is still alive in the game."""
    nutrients: int
    """Current nutrients available to the team."""
    spores: list[Spore]
    """List of all spores controlled by the team."""
    spawners: list[Spawner]
    """List of all spawners controlled by the team."""
    nextSpawnerCost: int
    """Cost in biomass to create the next spawner."""


@dataclass(slots=True)
class GameMap:
    """The game map."""

    width: int
    """Width of the map."""
    height: int
    """Height of the map."""
    nutrientGrid: list[list[int]]
    """The nutrients level per tile."""


@dataclass(slots=True)
class GameWorld:
    """The game world."""

    map: GameMap
    """The loaded game map. This map will never change during the game."""
    biomassGrid: list[list[int]]
    """The biomass level per tile. Use this in combination with ownershipGrid."""
    ownershipGrid: list[list[str]]
    """The owner id per tile. Will be the team id if owned by a player. Use this in combination with biomassGrid."""
    spores: list[Spore]
    """The spores in the map."""
    spawners: list[Spawner]
    """The spawners in the map."""
    teamInfos: dict[str, TeamInfo]
    """Current info of each team by team id."""


@dataclass(slots=True)
class TeamGameState:
    """State of the game for a specific team."""

    tick: int
    """Current tick number."""
    yourTeamId: str
    """Your team id."""
    lastTickErrors: list[str]
    """Errors that happened during the last tick."""
    constants: Constants
    """Game constants."""
    teamIds: list[str]
    """List of all the teams currently playing."""
    world: GameWorld
    """The game map, and objects"""


class Action:
    type: str


@dataclass
class SporeMoveAction(Action):
    """Move the spore one tile in the specified direction. Leaves biomass trail on empty tiles."""

    sporeId: str
    """ID of the spore to move."""
    direction: Position
    """Direction vector to move. Should be one of: {x:0,y:-1} (up), {x:0,y:1} (down), {x:-1,y:0} (left), {x:1,y:0} (right)."""
    type: str = "SPORE_MOVE"


@dataclass
class SporeMoveToAction(Action):
    """Move the spore towards the specified position using pathfinding. Moves one tile closer each turn."""

    sporeId: str
    """ID of the spore to move."""
    position: Position
    """Target position to move towards."""
    type: str = "SPORE_MOVE_TO"


@dataclass
class SporeCreateSpawnerAction(Action):
    """Create a spawner using part of the spore's biomass at its current position. Cost follows exponential sequence: 0, 1, 3, 7, 15, 31..."""

    sporeId: str
    """ID of the spore to create a spawner."""
    type: str = "SPORE_CREATE_SPAWNER"


@dataclass
class SpawnerProduceSporeAction(Action):
    """Create a new spore at the spawner location with specified biomass. Costs nutrients equal to biomass."""

    spawnerId: str
    """ID of the spawner that will produce the spore."""
    biomass: int
    """Amount of biomass for the new spore. Must be positive and team must have enough nutrients."""
    type: str = "SPAWNER_PRODUCE_SPORE"


@dataclass
class SporeSplitAction(Action):
    """Split a spore into two spores, distributing biomass between them. Original spore moves with specified biomass, new spore created at original position with remaining biomass."""

    sporeId: str
    """ID of the spore to split."""
    biomassForMovingSpore: int
    """Amount of biomass for the moving spore (must be at least 1 and less than current biomass)."""
    direction: Position
    """Direction for the original spore to move. Should be one of: {x:0,y:-1} (up), {x:0,y:1} (down), {x:-1,y:0} (left), {x:1,y:0} (right)."""
    type: str = "SPORE_SPLIT"
