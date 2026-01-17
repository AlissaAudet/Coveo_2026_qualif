from __future__ import annotations

import heapq
import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, TypeAlias

from game_message import *

PositionTuple: TypeAlias = Tuple[int, int]  # (x, y)

DIRS: list[PositionTuple] = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # U, D, L, R


# ============================================================
# TWEAKABLE CONSTANTS (MORE AGGRESSIVE)
# ============================================================

# Become aggressive earlier
AGGRO_TICK = 120

# Hub scoring
OWNED_BY_ME_PENALTY = 35
ENEMY_OWNED_BONUS = 180          # stronger pull onto enemy territory
NEUTRAL_OWNED_BONUS = 25
HUB_NUTRIENT_MIN = 1

# New: spawner hunting
SPAWNER_HUNT_BONUS = 800         # huge incentive to target enemy spawners
SPAWNER_HUNT_RADIUS = 30         # only consider spawners within this manhattan distance
CHASE_SPAWNER_BEFORE_TICK = 900  # still chase late game

# Ring behavior (area to fill around hub)
RING_MAX_EARLY = 4
RING_MAX_MID = 7
RING_MAX_LATE = 10

# Spawner creation
SPAWNER_COST_ALWAYS_OK_UNTIL = 7
SPAWNER_COST_OK_EARLY_UNTIL = 15
SPAWNER_EARLY_TICK = 220
NUTRIENT_FOR_SPAWNER = 45

# Spore production (more units + some fighters)
CLAIMER_BIOMASS = 2
WORKER_BIOMASS = 6
FIGHTER_BIOMASS = 12
FIGHTER_EVERY_N = 6
WORKER_EVERY_N = 9

DESIRED_SPORES_BASE = 10
DESIRED_SPORES_MAX_EXTRA = 30
DESIRED_SPORES_PER_NUTRIENTS = 20

# Combat safety margins:
# You only need to be strictly stronger to win (deterministic).
# Setting margin to 0 makes the bot commit more fights.
ENEMY_MARGIN_BEFORE_AGGRO = 1
ENEMY_MARGIN_AFTER_AGGRO = 0
NEUTRAL_MARGIN = 1


# ============================================================
# Strategy state per spore
# ============================================================
@dataclass
class SporePlan:
    hub: Optional[PositionTuple] = None
    mode: str = "SEEK"  # SEEK -> RING -> SEEK
    ring_r: int = 1
    ring_i: int = 0
    ring_max: int = RING_MAX_EARLY
    last_target: Optional[PositionTuple] = None


class Bot2:
    def __init__(self):
        print("Bot: hub -> ring fill, territory-first, aggression @ tick 250, neutral-friendly")
        self.plans: Dict[str, SporePlan] = {}
        self.reserved_hubs: Dict[PositionTuple, str] = {}  # hub -> sporeId

    # ----------------------------
    # Helpers map / grids (grid[y][x])
    # ----------------------------
    @staticmethod
    def _in_bounds(world: GameWorld, x: int, y: int) -> bool:
        return 0 <= x < world.map.width and 0 <= y < world.map.height

    @staticmethod
    def _nut(world: GameWorld, p: PositionTuple) -> int:
        x, y = p
        return world.map.nutrientGrid[y][x]

    @staticmethod
    def _tile_biomass(world: GameWorld, p: PositionTuple) -> int:
        x, y = p
        return world.biomassGrid[y][x]

    @staticmethod
    def _tile_owner(world: GameWorld, p: PositionTuple) -> str:
        x, y = p
        return world.ownershipGrid[y][x]

    def _owned_by_me(self, world: GameWorld, my_id: str, p: PositionTuple) -> bool:
        return self._tile_owner(world, p) == my_id and self._tile_biomass(world, p) >= 1

    @staticmethod
    def _has_spawner_at(world: GameWorld, p: PositionTuple) -> bool:
        x, y = p
        for spw in world.spawners:
            if spw.position.x == x and spw.position.y == y:
                return True
        return False

    # ----------------------------
    # Strength maps + safety checks
    # ----------------------------
    @staticmethod
    def _build_strength_maps(
        world: GameWorld,
        my_id: str,
        neutral_id: str,
    ) -> tuple[Dict[PositionTuple, int], Dict[PositionTuple, int]]:
        """
        Returns:
            enemy_strength[pos] = max biomass of any enemy spore on pos
            neutral_strength[pos] = max biomass of any neutral spore on pos
        """
        enemy_strength: Dict[PositionTuple, int] = {}
        neutral_strength: Dict[PositionTuple, int] = {}

        for s in world.spores:
            pos = (s.position.x, s.position.y)
            if s.teamId == my_id:
                continue
            if s.teamId == neutral_id:
                neutral_strength[pos] = max(neutral_strength.get(pos, 0), s.biomass)
            else:
                enemy_strength[pos] = max(enemy_strength.get(pos, 0), s.biomass)

        return enemy_strength, neutral_strength

    @staticmethod
    def _safe_to_enter(
        pos: PositionTuple,
        spore_biomass: int,
        tick: int,
        enemy_strength: Dict[PositionTuple, int],
        neutral_strength: Dict[PositionTuple, int],
    ) -> bool:
        e = enemy_strength.get(pos, 0)
        n = neutral_strength.get(pos, 0)

        enemy_margin = ENEMY_MARGIN_AFTER_AGGRO if tick >= AGGRO_TICK else ENEMY_MARGIN_BEFORE_AGGRO

        # Need to be strictly stronger (plus margin) to not die
        if e > 0 and spore_biomass <= e + enemy_margin:
            return False
        if n > 0 and spore_biomass <= n + NEUTRAL_MARGIN:
            return False

        return True

    # ----------------------------
    # Hub selection (fast)
    # ----------------------------
    def _pick_best_hub(
        self,
        world: GameWorld,
        my_id: str,
        neutral_id: str,
        tick: int,
        enemy_strength: Dict[PositionTuple, int],
        neutral_strength: Dict[PositionTuple, int],
    ) -> Optional[PositionTuple]:
        best: Optional[PositionTuple] = None
        best_score = -10**9

        for y in range(world.map.height):
            row = world.map.nutrientGrid[y]
            for x in range(world.map.width):
                v = row[x]
                if v < HUB_NUTRIENT_MIN:
                    continue

                p = (x, y)

                # Avoid hubs reserved by another spore
                if p in self.reserved_hubs and self.reserved_hubs[p] != "":
                    continue

                owner = self._tile_owner(world, p)

                score = v

                # Prefer expanding, not re-centering on already-owned
                if owner == my_id and self._tile_biomass(world, p) >= 1:
                    score -= OWNED_BY_ME_PENALTY

                # Aggressive: after AGGRO_TICK, value enemy territory as hubs
                if tick >= AGGRO_TICK and owner not in (my_id, neutral_id):
                    score += ENEMY_OWNED_BONUS

                # Neutral territory is often cheap to clear, give a little incentive
                if owner == neutral_id:
                    score += NEUTRAL_OWNED_BONUS

                # If occupied by a strong enemy we can't beat, don't hub there at all
                # (prevents repeatedly seeking an impossible hub)
                # We don't know which spore will use it here, so we only hard-ban extreme stacks:
                if enemy_strength.get(p, 0) >= 15 and tick < AGGRO_TICK:
                    continue

                if score > best_score:
                    best_score = score
                    best = p

        return best

    # ----------------------------
    # Ring fill (square rings around hub)
    # ----------------------------
    def _ring_positions(self, world: GameWorld, hub: PositionTuple, r: int) -> List[PositionTuple]:
        hx, hy = hub
        pts: List[PositionTuple] = []
        if r <= 0:
            return pts

        # top
        for x in range(hx - r, hx + r + 1):
            y = hy - r
            if self._in_bounds(world, x, y):
                pts.append((x, y))
        # right
        for y in range(hy - r + 1, hy + r + 1):
            x = hx + r
            if self._in_bounds(world, x, y):
                pts.append((x, y))
        # bottom
        for x in range(hx + r - 1, hx - r - 1, -1):
            y = hy + r
            if self._in_bounds(world, x, y):
                pts.append((x, y))
        # left
        for y in range(hy + r - 1, hy - r, -1):
            x = hx - r
            if self._in_bounds(world, x, y):
                pts.append((x, y))

        return pts

    def _next_ring_target(
        self,
        world: GameWorld,
        my_id: str,
        neutral_id: str,
        tick: int,
        spore_biomass: int,
        plan: SporePlan,
        enemy_strength: Dict[PositionTuple, int],
        neutral_strength: Dict[PositionTuple, int],
    ) -> Optional[PositionTuple]:
        if plan.hub is None:
            return None

        while plan.ring_r <= plan.ring_max:
            ring = self._ring_positions(world, plan.hub, plan.ring_r)
            if not ring:
                plan.ring_r += 1
                plan.ring_i = 0
                continue

            n = len(ring)
            for k in range(n):
                idx = (plan.ring_i + k) % n
                p = ring[idx]

                # We only target tiles not owned by us (steal / expand)
                if self._owned_by_me(world, my_id, p):
                    continue

                # Safety filter: must be enterable for THIS spore
                if not self._safe_to_enter(p, spore_biomass, tick, enemy_strength, neutral_strength):
                    continue

                plan.ring_i = (idx + 1) % n
                return p

            plan.ring_r += 1
            plan.ring_i = 0

        return None

    # ----------------------------
    # Economy knobs
    # territory first, then total resources, then spawners-built
    # ----------------------------
    @staticmethod
    def _should_create_spawner(team: TeamInfo, tick: int) -> bool:
        if team.nextSpawnerCost <= SPAWNER_COST_ALWAYS_OK_UNTIL:
            return True
        if tick < SPAWNER_EARLY_TICK and team.nextSpawnerCost <= SPAWNER_COST_OK_EARLY_UNTIL:
            return True
        return False

    def _produce_spores(self, actions: List[Action], team: TeamInfo):
        nutrients = team.nutrients
        desired_spores = DESIRED_SPORES_BASE + min(DESIRED_SPORES_MAX_EXTRA, nutrients // DESIRED_SPORES_PER_NUTRIENTS)

        if len(team.spores) >= desired_spores:
            return

        for spawner in team.spawners:
            if len(team.spores) >= desired_spores:
                break

            want_worker = (len(team.spores) % WORKER_EVERY_N == 0)
            bio = WORKER_BIOMASS if want_worker else CLAIMER_BIOMASS

            if team.nutrients < bio:
                continue

            actions.append(SpawnerProduceSporeAction(spawnerId=spawner.id, biomass=bio))
            team.nutrients -= bio  # local budget

    # ----------------------------
    # Main decision
    # ----------------------------
    def get_next_move(self, game_message: TeamGameState) -> List[Action]:
        actions: List[Action] = []

        world = game_message.world
        my_id = game_message.yourTeamId
        my_team: TeamInfo = world.teamInfos[my_id]
        tick = game_message.tick
        neutral_id = game_message.constants.neutralTeamId

        enemy_strength, neutral_strength = self._build_strength_maps(world, my_id, neutral_id)

        # 0) No units -> nothing to do
        if len(my_team.spores) == 0 and len(my_team.spawners) == 0:
            return actions

        # 1) Avoid elimination: ensure at least 1 spawner
        if len(my_team.spawners) == 0:
            if len(my_team.spores) > 0:
                s = my_team.spores[0]
                p = (s.position.x, s.position.y)

                if not self._has_spawner_at(world, p):
                    actions.append(SporeCreateSpawnerAction(sporeId=s.id))
            return actions

        # 2) Produce spores (territory-first)
        self._produce_spores(actions, my_team)

        # 3) Cleanup plans for dead spores
        live_ids = {s.id for s in my_team.spores}
        for sid in list(self.plans.keys()):
            if sid not in live_ids:
                old_hub = self.plans[sid].hub
                if old_hub is not None and self.reserved_hubs.get(old_hub) == sid:
                    del self.reserved_hubs[old_hub]
                del self.plans[sid]

        # 4) Assign plans / hubs
        ring_max = RING_MAX_EARLY if tick < 200 else (RING_MAX_MID if tick < 600 else RING_MAX_LATE)

        for sp in my_team.spores:
            if sp.id not in self.plans:
                self.plans[sp.id] = SporePlan(ring_max=ring_max)

            plan = self.plans[sp.id]
            plan.ring_max = ring_max

            # If hub missing or stolen by another reservation, repick
            hub_invalid = (
                plan.hub is None
                or (plan.hub in self.reserved_hubs and self.reserved_hubs[plan.hub] != sp.id)
            )
            if hub_invalid:
                hub = self._pick_best_hub(world, my_id, neutral_id, tick, enemy_strength, neutral_strength)
                if hub is not None:
                    plan.hub = hub
                    plan.mode = "SEEK"
                    plan.ring_r = 1
                    plan.ring_i = 0
                    self.reserved_hubs[hub] = sp.id

        # 5) Create more spawners while cheap (tie-break: spawners built)
        if self._should_create_spawner(my_team, tick):
            for sp in my_team.spores:
                if sp.biomass < max(2, my_team.nextSpawnerCost + 1):
                    continue

                p = (sp.position.x, sp.position.y)
                plan = self.plans.get(sp.id)

                if (plan and plan.hub == p) or self._nut(world, p) >= NUTRIENT_FOR_SPAWNER:
                    if not self._has_spawner_at(world, p):
                        actions.append(SporeCreateSpawnerAction(sporeId=sp.id))
                    break

        # Track spores that already used an action (create spawner uses spore action)
        used_spores = {a.sporeId for a in actions if hasattr(a, "sporeId")}

        # 6) Move spores: SEEK hub -> RING fill -> new hub
        for sp in my_team.spores:
            if sp.id in used_spores:
                continue

            # Needs 2+ biomass to act (move/combat)
            if sp.biomass < 2:
                continue

            plan = self.plans.get(sp.id)
            if not plan or plan.hub is None:
                continue

            sp_pos = (sp.position.x, sp.position.y)

            # If hub is not safe for this spore, repick hub now (prevents suicide-seeking)
            if not self._safe_to_enter(plan.hub, sp.biomass, tick, enemy_strength, neutral_strength):
                if plan.hub is not None and self.reserved_hubs.get(plan.hub) == sp.id:
                    del self.reserved_hubs[plan.hub]
                new_hub = self._pick_best_hub(world, my_id, neutral_id, tick, enemy_strength, neutral_strength)
                if new_hub is not None:
                    plan.hub = new_hub
                    self.reserved_hubs[new_hub] = sp.id
                    plan.mode = "SEEK"
                    plan.ring_r = 1
                    plan.ring_i = 0

            if plan.mode == "SEEK":
                if sp_pos != plan.hub:
                    # Aggressive after AGGRO_TICK: keep seeking even if it’s enemy-owned,
                    # but still filtered by _safe_to_enter above.
                    actions.append(
                        SporeMoveToAction(
                            sporeId=sp.id,
                            position=Position(x=plan.hub[0], y=plan.hub[1]),
                        )
                    )
                    continue

                plan.mode = "RING"
                plan.ring_r = 1
                plan.ring_i = 0

            if plan.mode == "RING":
                target = self._next_ring_target(
                    world=world,
                    my_id=my_id,
                    neutral_id=neutral_id,
                    tick=tick,
                    spore_biomass=sp.biomass,
                    plan=plan,
                    enemy_strength=enemy_strength,
                    neutral_strength=neutral_strength,
                )

                if target is None:
                    # finished ring: pick next hub and repeat
                    if plan.hub is not None and self.reserved_hubs.get(plan.hub) == sp.id:
                        del self.reserved_hubs[plan.hub]

                    new_hub = self._pick_best_hub(world, my_id, neutral_id, tick, enemy_strength, neutral_strength)
                    if new_hub is not None:
                        plan.hub = new_hub
                        self.reserved_hubs[new_hub] = sp.id
                        plan.mode = "SEEK"
                        plan.ring_r = 1
                        plan.ring_i = 0
                    continue

                actions.append(
                    SporeMoveToAction(
                        sporeId=sp.id,
                        position=Position(x=target[0], y=target[1]),
                    )
                )

        return actions


# -------------------------------------------------------------------
# Optional: keep your Dijkstra utilities (not used by this bot now)
# -------------------------------------------------------------------
def dijkstra(
    start: PositionTuple,
    neighbors_fn: Callable[[PositionTuple], list[PositionTuple]],
    cost_fn: Callable[[PositionTuple, PositionTuple], int],
    is_goal_fn: Optional[Callable[[PositionTuple], bool]] = None,
    max_cost: Optional[int] = None,
) -> tuple[Dict[PositionTuple, int], Dict[PositionTuple, PositionTuple]]:
    distances: Dict[PositionTuple, int] = {start: 0}
    previous: Dict[PositionTuple, PositionTuple] = {}
    pq: list[tuple[int, PositionTuple]] = [(0, start)]

    while pq:
        current_dist, current = heapq.heappop(pq)

        if current_dist > distances[current]:
            continue

        if is_goal_fn and is_goal_fn(current):
            break

        if max_cost is not None and current_dist > max_cost:
            continue

        for neighbor in neighbors_fn(current):
            step_cost = cost_fn(current, neighbor)
            if step_cost < 0:
                continue

            new_dist = current_dist + step_cost
            if neighbor not in distances or new_dist < distances[neighbor]:
                distances[neighbor] = new_dist
                previous[neighbor] = current
                heapq.heappush(pq, (new_dist, neighbor))

    return distances, previous


def reconstruct_path(
    previous: Dict[PositionTuple, PositionTuple],
    start: PositionTuple,
    goal: PositionTuple,
) -> list[PositionTuple]:
    if goal not in previous and goal != start:
        return []

    path = [goal]
    current = goal
    while current != start:
        current = previous[current]
        path.append(current)

    path.reverse()
    return path