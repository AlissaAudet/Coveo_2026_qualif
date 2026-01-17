from __future__ import annotations
import random
from game_message import *

import heapq
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, TypeAlias


PositionTuple: TypeAlias = Tuple[int, int]  # (x, y)

DIRS: list[PositionTuple] = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # U, D, L, R

# -----------------------------
# Strategy state per spore
# -----------------------------
@dataclass
class SporePlan:
    hub: Optional[PositionTuple] = None
    mode: str = "SEEK"  # SEEK -> RING -> SEEK (next hub)
    ring_r: int = 1
    ring_i: int = 0
    ring_max: int = 4
    last_target: Optional[PositionTuple] = None

class Bot:
    def __init__(self):
        print("Bot: hub -> ring fill, territory-first, safe spawner creation")
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
    # Hub selection (fast)
    # ----------------------------
    def _pick_best_hub(self, world: GameWorld, my_id: str) -> Optional[PositionTuple]:
        best: Optional[PositionTuple] = None
        best_score = -10**9

        for y in range(world.map.height):
            row = world.map.nutrientGrid[y]
            for x in range(world.map.width):
                v = row[x]
                if v <= 0:
                    continue

                p = (x, y)

                # Eviter hubs deja reserves par une autre spore
                if p in self.reserved_hubs and self.reserved_hubs[p] != "":
                    continue

                # Penaliser un peu si deja a nous (on veut expand)
                score = v - (40 if self._owned_by_me(world, my_id, p) else 0)

                if score > best_score:
                    best_score = score
                    best = p

        return best

    # ----------------------------
    # Ring fill (circonference en anneaux carres)
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

    def _next_ring_target(self, world: GameWorld, my_id: str, plan: SporePlan) -> Optional[PositionTuple]:
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
                if not self._owned_by_me(world, my_id, p):
                    plan.ring_i = (idx + 1) % n
                    return p

            plan.ring_r += 1
            plan.ring_i = 0

        return None

    # ----------------------------
    # Economy knobs (victory conditions)
    # territory first, then total resources, then spawners-built
    # ----------------------------
    @staticmethod
    def _should_create_spawner(team: TeamInfo, tick: int) -> bool:
        # Build early when cheap; stop when cost explodes.
        if team.nextSpawnerCost <= 7:
            return True
        if tick < 200 and team.nextSpawnerCost <= 15:
            return True
        return False

    def _produce_spores(self, actions: List[Action], team: TeamInfo):
        # Many small spores = more territory markers and more actions over the game.
        # biomass=2 can move and then becomes a marker (1) on a new tile.
        claimer_biomass = 2
        worker_biomass = 6  # occasional spore that can keep moving more

        nutrients = team.nutrients
        desired_spores = 6 + min(18, nutrients // 30)  # 6..24

        if len(team.spores) >= desired_spores:
            return

        for spawner in team.spawners:
            if len(team.spores) >= desired_spores:
                break

            want_worker = (len(team.spores) % 7 == 0)
            bio = worker_biomass if want_worker else claimer_biomass

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

        # 0) No units -> nothing to do
        if len(my_team.spores) == 0 and len(my_team.spawners) == 0:
            return actions

        # 1) Avoid elimination: ensure at least 1 spawner
        if len(my_team.spawners) == 0:
            if len(my_team.spores) > 0:
                s = my_team.spores[0]
                p = (s.position.x, s.position.y)

                # IMPORTANT FIX: do not create if a spawner already exists there
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
        ring_max = 4 if tick < 200 else (6 if tick < 600 else 8)

        for sp in my_team.spores:
            if sp.id not in self.plans:
                self.plans[sp.id] = SporePlan(ring_max=ring_max)

            plan = self.plans[sp.id]
            plan.ring_max = ring_max

            if plan.hub is None or (plan.hub in self.reserved_hubs and self.reserved_hubs[plan.hub] != sp.id):
                hub = self._pick_best_hub(world, my_id)
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

                # Prefer: on its hub or on high nutrient
                if (plan and plan.hub == p) or self._nut(world, p) >= 50:
                    # IMPORTANT FIX: avoid spawner collision
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

            if plan.mode == "SEEK":
                if sp_pos != plan.hub:
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
                target = self._next_ring_target(world, my_id, plan)
                if target is None:
                    # finished ring: pick next hub and repeat
                    if plan.hub is not None and self.reserved_hubs.get(plan.hub) == sp.id:
                        del self.reserved_hubs[plan.hub]

                    new_hub = self._pick_best_hub(world, my_id)
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


def dijkstra(
    start: PositionTuple,
    neighbors_fn: Callable[[PositionTuple], list[PositionTuple]],
    cost_fn: Callable[[PositionTuple, PositionTuple], int],
    is_goal_fn: Optional[Callable[[PositionTuple], bool]] = None,
    max_cost: Optional[int] = None,
) -> tuple[Dict[PositionTuple, int], Dict[PositionTuple, PositionTuple]]:
    """
    Generic Dijkstra algorithm.

    Returns:
        distances: shortest distance from start to each visited position
        previous: parent map for path reconstruction
    """

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
                continue  # impassable

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
