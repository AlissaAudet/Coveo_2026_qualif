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
        print("Bot: territory-first, hub+ring expansion, many small claim spores")
        self.plans: Dict[str, SporePlan] = {}
        self.reserved_hubs: Dict[PositionTuple, str] = {}  # hub -> sporeId

    # ---------- Map helpers ----------
    @staticmethod
    def _dirs() -> List[PositionTuple]:
        return [(0, -1), (0, 1), (-1, 0), (1, 0)]

    @staticmethod
    def _dir_pos(dx: int, dy: int) -> Position:
        return Position(x=dx, y=dy)

    @staticmethod
    def _in_bounds(world: GameWorld, x: int, y: int) -> bool:
        return 0 <= x < world.map.width and 0 <= y < world.map.height

    def _neighbors(self, world: GameWorld, p: PositionTuple) -> List[PositionTuple]:
        x, y = p
        out: List[PositionTuple] = []
        for dx, dy in self._dirs():
            nx, ny = x + dx, y + dy
            if self._in_bounds(world, nx, ny):
                out.append((nx, ny))
        return out

    def _nut(self, world: GameWorld, p: PositionTuple) -> int:
        x, y = p
        return world.map.nutrientGrid[y][x]

    def _owner(self, world: GameWorld, p: PositionTuple) -> str:
        x, y = p
        return world.ownershipGrid[y][x]

    def _bio_tile(self, world: GameWorld, p: PositionTuple) -> int:
        x, y = p
        return world.biomassGrid[y][x]

    def _owned_by_me(self, world: GameWorld, my_id: str, p: PositionTuple) -> bool:
        # "Controlled tile" = biomass>=1 + ownership==my_id (per docs)
        return self._owner(world, p) == my_id and self._bio_tile(world, p) >= 1

    def _enemy_spores(self, world: GameWorld, my_id: str) -> set[PositionTuple]:
        s: set[PositionTuple] = set()
        for sp in world.spores:
            if sp.teamId != my_id:
                s.add((sp.position.x, sp.position.y))
        return s

    # move cost: 0 on our trace, 1 otherwise; avoid enemy-occupied tile
    def _cost01(self, world: GameWorld, my_id: str, enemy_pos: set[PositionTuple], a: PositionTuple, b: PositionTuple) -> int:
        if b in enemy_pos:
            return -1
        if self._owned_by_me(world, my_id, b):
            return 0
        return 1

    # ---------- Hub selection ----------
    def _top_hubs(self, world: GameWorld, my_id: str, limit: int = 60) -> List[PositionTuple]:
        # Prefer high nutrient tiles not already reserved; not caring about distance here (fast).
        cand: List[Tuple[int, int, PositionTuple]] = []
        for y in range(world.map.height):
            row = world.map.nutrientGrid[y]
            for x in range(world.map.width):
                v = row[x]
                if v <= 0:
                    continue
                p = (x, y)
                if p in self.reserved_hubs and self.reserved_hubs[p] != "":
                    continue
                # Penalize already-owned tiles a bit so we keep expanding territory
                owned_pen = 40 if self._owned_by_me(world, my_id, p) else 0
                cand.append((v - owned_pen, v, p))
        cand.sort(reverse=True)
        return [p for _, __, p in cand[:limit]]

    def _pick_hub_reachable(
        self,
        world: GameWorld,
        my_id: str,
        start: PositionTuple,
        max_cost: int = 40,
    ) -> Optional[PositionTuple]:
        hubs = self._top_hubs(world, my_id, limit=60)
        if not hubs:
            return None

        enemy_pos = self._enemy_spores(world, my_id)

        def neighbors_fn(p: PositionTuple) -> List[PositionTuple]:
            return self._neighbors(world, p)

        def cost_fn(a: PositionTuple, b: PositionTuple) -> int:
            return self._cost01(world, my_id, enemy_pos, a, b)

        dist, _ = dijkstra(start, neighbors_fn, cost_fn, is_goal_fn=None, max_cost=max_cost)

        # Choose: reachable with lowest cost; tie: higher nutrient
        best: Optional[PositionTuple] = None
        best_key = (10**9, -10**9)  # (cost, -nutrient)
        for p in hubs:
            if p not in dist:
                continue
            key = (dist[p], -self._nut(world, p))
            if key < best_key:
                best_key = key
                best = p

        return best if best is not None else hubs[0]

    # ---------- Ring fill ----------
    def _ring_positions(self, world: GameWorld, hub: PositionTuple, r: int) -> List[PositionTuple]:
        # Square ring: max(|dx|,|dy|) == r
        hx, hy = hub
        pts: List[PositionTuple] = []
        if r <= 0:
            return pts

        for x in range(hx - r, hx + r + 1):
            y = hy - r
            if self._in_bounds(world, x, y):
                pts.append((x, y))
        for y in range(hy - r + 1, hy + r + 1):
            x = hx + r
            if self._in_bounds(world, x, y):
                pts.append((x, y))
        for x in range(hx + r - 1, hx - r - 1, -1):
            y = hy + r
            if self._in_bounds(world, x, y):
                pts.append((x, y))
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
                # Territory-first: try to take any not-controlled tile (even nutrient 0) for score.
                if not self._owned_by_me(world, my_id, p):
                    plan.ring_i = (idx + 1) % n
                    return p

            plan.ring_r += 1
            plan.ring_i = 0

        return None

    # ---------- One-step safe move (fast) ----------
    def _move_one_step(self, spore: Spore, target: PositionTuple) -> Optional[Action]:
        sx, sy = spore.position.x, spore.position.y
        tx, ty = target
        dx = 0 if tx == sx else (1 if tx > sx else -1)
        dy = 0 if ty == sy else (1 if ty > sy else -1)

        # Prefer x move first then y (deterministic)
        if dx != 0:
            return SporeMoveAction(sporeId=spore.id, direction=self._dir_pos(dx, 0))
        if dy != 0:
            return SporeMoveAction(sporeId=spore.id, direction=self._dir_pos(0, dy))
        return None

    # ---------- When to make more spawners ----------
    def _should_create_spawner(
        self,
        team: TeamInfo,
        tick: int,
    ) -> bool:
        # Spawners-built is a tie-breaker, but cost grows fast.
        # We aggressively create spawners while cheap, then slow down.
        if team.nextSpawnerCost <= 7:
            return True
        if tick < 200 and team.nextSpawnerCost <= 15:
            return True
        return False

    # ---------- Produce lots of small claim spores ----------
    def _produce_spores(self, actions: List[Action], team: TeamInfo):
        # Territory-first: many biomass=2 spores (can move once to claim a tile).
        # Keep some medium spores for repeated ring-fill and bridging between hubs.
        # We also want many actions (tie-break), but without heavy computations.

        # Target counts scale slowly with nutrients
        nutrients = team.nutrients
        desired_spores = 6 + min(18, nutrients // 30)  # 6..24
        if len(team.spores) >= desired_spores:
            return

        # Prefer cheap claimers
        claimer_bio = 2
        worker_bio = 6

        # Produce at most one per spawner per tick
        for spawner in team.spawners:
            if len(team.spores) >= desired_spores:
                break

            # Alternate: one worker occasionally
            want_worker = (len(team.spores) % 7 == 0)

            bio = worker_bio if want_worker else claimer_bio
            if team.nutrients < bio:
                continue

            actions.append(SpawnerProduceSporeAction(spawnerId=spawner.id, biomass=bio))
            team.nutrients -= bio  # local bookkeeping
            team.spores.append(Spore(id="__virtual__", teamId=team.teamId, position=spawner.position, biomass=bio))  # virtual for loop limit only

        # Remove virtual spores (only used to cap production in this tick)
        team.spores = [s for s in team.spores if s.id != "__virtual__"]

    # ---------- Main ----------
    def get_next_move(self, game_message: TeamGameState) -> List[Action]:
        actions: List[Action] = []
        world = game_message.world
        my_id = game_message.yourTeamId
        my_team: TeamInfo = world.teamInfos[my_id]
        tick = game_message.tick

        # Safety: no units
        if len(my_team.spores) == 0 and len(my_team.spawners) == 0:
            return actions

        # 1) Avoid elimination: ensure at least 1 spawner if possible
        if len(my_team.spawners) == 0:
            if len(my_team.spores) > 0:
                # Make a spawner ASAP (tiebreak spawners-built too)
                actions.append(SporeCreateSpawnerAction(sporeId=my_team.spores[0].id))
            return actions

        # 2) Produce many small spores for territory control
        # (also increases actions executed later)
        self._produce_spores(actions, my_team)

        # 3) Clean dead plans / free hubs
        live_ids = {s.id for s in my_team.spores}
        for sid in list(self.plans.keys()):
            if sid not in live_ids:
                old = self.plans[sid].hub
                if old is not None and self.reserved_hubs.get(old) == sid:
                    del self.reserved_hubs[old]
                del self.plans[sid]

        # 4) Ensure each spore has a hub + plan
        # Ring radius grows later to maximize territory by tick 1000
        default_ring_max = 4 if tick < 200 else (6 if tick < 600 else 8)

        for sp in my_team.spores:
            if sp.id not in self.plans:
                self.plans[sp.id] = SporePlan(ring_max=default_ring_max)
            plan = self.plans[sp.id]
            plan.ring_max = default_ring_max

            if plan.hub is None or (plan.hub in self.reserved_hubs and self.reserved_hubs[plan.hub] != sp.id):
                hub = self._pick_hub_reachable(world, my_id, start=(sp.position.x, sp.position.y), max_cost=45)
                if hub is not None:
                    plan.hub = hub
                    plan.mode = "SEEK"
                    plan.ring_r = 1
                    plan.ring_i = 0
                    plan.last_target = None
                    self.reserved_hubs[hub] = sp.id

        # 5) Create more spawners while cheap (spawners-built tie-break)
        # Convert a spore that is standing on a good nutrient tile (prefer hub tile).
        if self._should_create_spawner(my_team, tick) and len(my_team.spores) > 0:
            made = False
            for sp in my_team.spores:
                if sp.biomass < max(2, my_team.nextSpawnerCost + 1):
                    continue
                p = (sp.position.x, sp.position.y)
                # Prefer converting on high nutrient (helps resources tie-break too)
                if self._nut(world, p) >= 50 or (self.plans.get(sp.id) and self.plans[sp.id].hub == p):
                    actions.append(SporeCreateSpawnerAction(sporeId=sp.id))
                    made = True
                    break
            if made:
                # Don't overload this tick; movement still handled for other spores below.
                pass

        # Track which spores already acted (create spawner uses spore action)
        used_spores = {a.sporeId for a in actions if hasattr(a, "sporeId")}

        # 6) Movement: maximize territory (always try to move if biomass>=2)
        # - SEEK: go to hub
        # - RING: fill ring (takes all tiles around hub)
        # - After ring done: pick next hub and repeat (expands to next high nutrient)
        for sp in my_team.spores:
            if sp.id in used_spores:
                continue

            # Needs 2+ biomass to act
            if sp.biomass < 2:
                continue

            plan = self.plans.get(sp.id)
            if plan is None or plan.hub is None:
                continue

            sp_pos = (sp.position.x, sp.position.y)

            if plan.mode == "SEEK":
                if sp_pos != plan.hub:
                    # Use server pathfinding for speed + simplicity
                    actions.append(SporeMoveToAction(sporeId=sp.id, position=Position(x=plan.hub[0], y=plan.hub[1])))
                    continue
                plan.mode = "RING"
                plan.ring_r = 1
                plan.ring_i = 0
                plan.last_target = None

            if plan.mode == "RING":
                target = self._next_ring_target(world, my_id, plan)
                if target is None:
                    # Free hub and choose another top nutrient to keep expanding territory
                    if plan.hub is not None and self.reserved_hubs.get(plan.hub) == sp.id:
                        del self.reserved_hubs[plan.hub]
                    plan.hub = self._pick_hub_reachable(world, my_id, start=sp_pos, max_cost=55)
                    if plan.hub is not None:
                        self.reserved_hubs[plan.hub] = sp.id
                        plan.mode = "SEEK"
                        plan.ring_r = 1
                        plan.ring_i = 0
                        plan.last_target = None
                    continue

                # If we're already there, try to move to next ring tile (for actions tie-break + territory)
                if sp_pos == target:
                    # move to next ring target if possible
                    plan.last_target = None
                    target2 = self._next_ring_target(world, my_id, plan)
                    if target2 is None:
                        continue
                    target = target2

                # Use server pathing; it should take cheapest path.
                actions.append(SporeMoveToAction(sporeId=sp.id, position=Position(x=target[0], y=target[1])))

        # 7) If no movement actions were possible, at least "do something" with any active spore
        # (actions-executed tie-break) by nudging them deterministically.
        if len(actions) == 0 and len(my_team.spores) > 0:
            for sp in my_team.spores:
                if sp.biomass >= 2:
                    # move right if possible, else down
                    x, y = sp.position.x, sp.position.y
                    if x + 1 < world.map.width:
                        actions.append(SporeMoveAction(sporeId=sp.id, direction=Position(x=1, y=0)))
                    elif y + 1 < world.map.height:
                        actions.append(SporeMoveAction(sporeId=sp.id, direction=Position(x=0, y=1)))
                    break

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
