from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, TypeAlias

from game_message import *

PositionTuple: TypeAlias = Tuple[int, int]  # (x, y)

DIRS: list[PositionTuple] = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # U, D, L, R


# ============================================================
# TWEAKABLE CONSTANTS
# ============================================================

# Aggression (steal enemy territory by selecting hubs on enemy-owned tiles)
AGGRO_TICK = 250
ENEMY_OWNED_BONUS = 110
NEUTRAL_OWNED_BONUS = 15
OWNED_BY_ME_PENALTY = 40
HUB_NUTRIENT_MIN = 1

# Ring sizes
RING_MAX_EARLY = 4
RING_MAX_MID = 6
RING_MAX_LATE = 8

# Spawner creation
SPAWNER_COST_ALWAYS_OK_UNTIL = 7
SPAWNER_COST_OK_EARLY_UNTIL = 15
SPAWNER_EARLY_TICK = 200
NUTRIENT_FOR_SPAWNER = 50

# Spore production
CLAIMER_BIOMASS = 2
WORKER_BIOMASS = 6
WORKER_EVERY_N = 7
DESIRED_SPORES_BASE = 6
DESIRED_SPORES_MAX_EXTRA = 18
DESIRED_SPORES_PER_NUTRIENTS = 30

# Combat safety margins (entering a tile with a spore on it)
# If rules are "tie loses", keep margins >= 0.
ENEMY_MARGIN_BEFORE_AGGRO = 2
ENEMY_MARGIN_AFTER_AGGRO = 0
NEUTRAL_MARGIN = 0  # lower = less averse to neutral spores

# Cost-aware movement (paid steps budget)
MIN_REMAINING_BIOMASS = 2
PATH_BUDGET_EXTRA = 2

# Assassin mode
ASSASSIN_ENABLE = True
ASSASSIN_TICK = 250
ASSASSIN_MAX_ASSASSINS = 2
ASSASSIN_MIN_BIOMASS = 10
ASSASSIN_KILL_MARGIN = 1              # must be > enemy + margin to step onto the kill tile
ASSASSIN_PATH_MARGIN = 0              # safety on intermediate tiles (enemy/neutral on the way)
ASSASSIN_MAX_PATH_STEPS = 18          # dijkstra budget for assassin reachability (steps)
ASSASSIN_TARGET_MIN_DIST = 0          # set to 6 to avoid assassin mode for very close targets
ASSASSIN_TARGET_MAX_DIST = 25

ASSASSIN_PREFER_BIG_TARGETS = True
ASSASSIN_TARGET_BONUS_PER_BIOMASS = 3


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


class Bot:
    def __init__(self):
        print("Bot: hub -> ring fill, cost-aware movement, aggro + assassin")
        self.plans: Dict[str, SporePlan] = {}
        self.reserved_hubs: Dict[PositionTuple, str] = {}  # hub -> sporeId

    # ----------------------------
    # Grid helpers (grid[y][x])
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

    def _neighbors4(self, world: GameWorld, p: PositionTuple) -> list[PositionTuple]:
        x, y = p
        out: list[PositionTuple] = []
        for dx, dy in DIRS:
            nx, ny = x + dx, y + dy
            if self._in_bounds(world, nx, ny):
                out.append((nx, ny))
        return out

    @staticmethod
    def _pos_tuple(pos: Position) -> PositionTuple:
        return (pos.x, pos.y)

    # ----------------------------
    # Strength maps + safety checks
    # ----------------------------
    @staticmethod
    def _build_strength_maps(
        world: GameWorld,
        my_id: str,
        neutral_id: str,
    ) -> tuple[Dict[PositionTuple, int], Dict[PositionTuple, int]]:
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
        my_occupied: set[PositionTuple],
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

                if p in my_occupied:
                    continue

                # Avoid hubs reserved by another spore
                if p in self.reserved_hubs and self.reserved_hubs[p] != "":
                    continue

                owner = self._tile_owner(world, p)
                score = v

                # Prefer expand over already-owned
                if owner == my_id and self._tile_biomass(world, p) >= 1:
                    score -= OWNED_BY_ME_PENALTY

                # Aggro: prefer enemy-owned hubs
                if tick >= AGGRO_TICK and owner not in (my_id, neutral_id):
                    score += ENEMY_OWNED_BONUS

                # Neutral is often cheap-ish
                if owner == neutral_id:
                    score += NEUTRAL_OWNED_BONUS

                # Light global sanity: if heavily occupied by something too big, do not hub there early
                if tick < AGGRO_TICK and enemy_strength.get(p, 0) >= 15:
                    continue
                if tick < AGGRO_TICK and neutral_strength.get(p, 0) >= 15:
                    continue

                if score > best_score:
                    best_score = score
                    best = p

        return best

    # ----------------------------
    # Ring fill around hub
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
        my_occupied: set[PositionTuple],
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

                if p in my_occupied:
                    continue

                # Only target tiles not owned by us (steal or expand)
                if self._owned_by_me(world, my_id, p):
                    continue

                if not self._safe_to_enter(p, spore_biomass, tick, enemy_strength, neutral_strength):
                    continue

                plan.ring_i = (idx + 1) % n
                return p

            plan.ring_r += 1
            plan.ring_i = 0

        return None

    # ----------------------------
    # Economy knobs
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
    # Cost-aware one-step movement (paid steps)
    # ----------------------------
    def _move_one_step_cost_aware(
        self,
        world: GameWorld,
        my_id: str,
        neutral_id: str,
        tick: int,
        sp: Spore,
        goal: PositionTuple,
        enemy_strength: Dict[PositionTuple, int],
        neutral_strength: Dict[PositionTuple, int],
        my_occupied: set[PositionTuple],
    ) -> Optional[SporeMoveToAction]:
        start = (sp.position.x, sp.position.y)
        if start == goal:
            return None

        max_paid = max(0, sp.biomass - MIN_REMAINING_BIOMASS - PATH_BUDGET_EXTRA)
        if max_paid <= 0:
            return None

        def neighbors_fn(p: PositionTuple) -> list[PositionTuple]:
            return self._neighbors4(world, p)

        def cost_fn(a: PositionTuple, b: PositionTuple) -> int:
            if b in my_occupied and b != goal:
                return -1
            if not self._safe_to_enter(b, sp.biomass, tick, enemy_strength, neutral_strength):
                return -1
            return 0 if self._owned_by_me(world, my_id, b) else 1

        dist, prev = dijkstra(
            start=start,
            neighbors_fn=neighbors_fn,
            cost_fn=cost_fn,
            is_goal_fn=lambda p: p == goal,
            max_cost=max_paid,
        )
        if goal not in dist:
            return None

        path = reconstruct_path(prev, start, goal)
        if len(path) < 2:
            return None

        nxt = path[1]
        return SporeMoveToAction(sporeId=sp.id, position=Position(x=nxt[0], y=nxt[1]))

    # ----------------------------
    # Assassin helpers
    # ----------------------------
    def _assassin_safe_to_enter(
        self,
        pos: PositionTuple,
        spore_biomass: int,
        enemy_strength: Dict[PositionTuple, int],
        neutral_strength: Dict[PositionTuple, int],
        my_occupied: set[PositionTuple],
    ) -> bool:
        if pos in my_occupied:
            return False
        e = enemy_strength.get(pos, 0)
        n = neutral_strength.get(pos, 0)
        if e > 0 and spore_biomass <= e + ASSASSIN_PATH_MARGIN:
            return False
        if n > 0 and spore_biomass <= n + ASSASSIN_PATH_MARGIN:
            return False
        return True

    @staticmethod
    def _assassin_can_kill_target(
        target_pos: PositionTuple,
        spore_biomass: int,
        enemy_strength: Dict[PositionTuple, int],
    ) -> bool:
        e = enemy_strength.get(target_pos, 0)
        return e > 0 and spore_biomass > e + ASSASSIN_KILL_MARGIN

    def _assassin_has_clear_path(
        self,
        world: GameWorld,
        start: PositionTuple,
        goal: PositionTuple,
        spore_biomass: int,
        enemy_strength: Dict[PositionTuple, int],
        neutral_strength: Dict[PositionTuple, int],
        my_occupied: set[PositionTuple],
    ) -> bool:
        def neighbors_fn(p: PositionTuple) -> list[PositionTuple]:
            return self._neighbors4(world, p)

        def cost_fn(a: PositionTuple, b: PositionTuple) -> int:
            if b == goal:
                return 1 if self._assassin_can_kill_target(goal, spore_biomass, enemy_strength) else -1
            return 1 if self._assassin_safe_to_enter(b, spore_biomass, enemy_strength, neutral_strength, my_occupied) else -1

        dist, _prev = dijkstra(
            start=start,
            neighbors_fn=neighbors_fn,
            cost_fn=cost_fn,
            is_goal_fn=lambda p: p == goal,
            max_cost=ASSASSIN_MAX_PATH_STEPS,
        )
        return goal in dist

    def _pick_assassin_targets(
        self,
        world: GameWorld,
        my_id: str,
        neutral_id: str,
        tick: int,
        enemy_strength: Dict[PositionTuple, int],
        neutral_strength: Dict[PositionTuple, int],
        my_occupied: set[PositionTuple],
    ) -> list[tuple[str, PositionTuple]]:
        if not ASSASSIN_ENABLE or tick < ASSASSIN_TICK:
            return []

        my_team = world.teamInfos[my_id]
        my_spores = [s for s in my_team.spores if s.biomass >= max(2, ASSASSIN_MIN_BIOMASS)]
        my_spores.sort(key=lambda s: s.biomass, reverse=True)
        my_spores = my_spores[:ASSASSIN_MAX_ASSASSINS]
        if not my_spores:
            return []

        enemy_spores: list[tuple[PositionTuple, int]] = []
        for s in world.spores:
            if s.teamId == my_id or s.teamId == neutral_id:
                continue
            enemy_spores.append(((s.position.x, s.position.y), s.biomass))
        if not enemy_spores:
            return []

        orders: list[tuple[str, PositionTuple]] = []
        reserved_targets: set[PositionTuple] = set()

        for hunter in my_spores:
            start = (hunter.position.x, hunter.position.y)

            best_target: Optional[PositionTuple] = None
            best_score = -10**9

            for pos, ebio in enemy_spores:
                if pos in reserved_targets:
                    continue
                if pos in my_occupied:
                    continue

                d = abs(pos[0] - start[0]) + abs(pos[1] - start[1])
                if d < ASSASSIN_TARGET_MIN_DIST:
                    continue
                if d > ASSASSIN_TARGET_MAX_DIST:
                    continue

                if hunter.biomass <= ebio + ASSASSIN_KILL_MARGIN:
                    continue

                if not self._assassin_has_clear_path(
                    world=world,
                    start=start,
                    goal=pos,
                    spore_biomass=hunter.biomass,
                    enemy_strength=enemy_strength,
                    neutral_strength=neutral_strength,
                    my_occupied=my_occupied,
                ):
                    continue

                score = 0
                if ASSASSIN_PREFER_BIG_TARGETS:
                    score += ebio * ASSASSIN_TARGET_BONUS_PER_BIOMASS
                score -= d * 2

                if score > best_score:
                    best_score = score
                    best_target = pos

            if best_target is not None:
                orders.append((hunter.id, best_target))
                reserved_targets.add(best_target)

        return orders

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

        # Occupied tiles by our stuff (avoid stepping onto them)
        my_occupied: set[PositionTuple] = set()
        for s in my_team.spores:
            my_occupied.add((s.position.x, s.position.y))
        for spw in my_team.spawners:
            my_occupied.add((spw.position.x, spw.position.y))

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

            hub_invalid = (
                plan.hub is None
                or (plan.hub in self.reserved_hubs and self.reserved_hubs[plan.hub] != sp.id)
                or (plan.hub in my_occupied and plan.hub != (sp.position.x, sp.position.y))
            )

            if hub_invalid:
                hub = self._pick_best_hub(
                    world=world,
                    my_id=my_id,
                    neutral_id=neutral_id,
                    tick=tick,
                    enemy_strength=enemy_strength,
                    neutral_strength=neutral_strength,
                    my_occupied=my_occupied,
                )
                if hub is not None:
                    plan.hub = hub
                    plan.mode = "SEEK"
                    plan.ring_r = 1
                    plan.ring_i = 0
                    self.reserved_hubs[hub] = sp.id

        # 5) Create more spawners while cheap
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

        # 5.5) Assassin override
        assassin_orders = self._pick_assassin_targets(
            world=world,
            my_id=my_id,
            neutral_id=neutral_id,
            tick=tick,
            enemy_strength=enemy_strength,
            neutral_strength=neutral_strength,
            my_occupied=my_occupied,
        )
        for sid, tgt in assassin_orders:
            if sid in used_spores:
                continue
            actions.append(
                SporeMoveToAction(
                    sporeId=sid,
                    position=Position(x=tgt[0], y=tgt[1]),
                )
            )
            used_spores.add(sid)

        # 6) Move spores: SEEK hub -> RING fill -> new hub
        for sp in my_team.spores:
            if sp.id in used_spores:
                continue

            if sp.biomass < 2:
                continue

            plan = self.plans.get(sp.id)
            if not plan or plan.hub is None:
                continue

            sp_pos = (sp.position.x, sp.position.y)

            # If hub is not safe for this spore, repick (avoids suicide SEEK)
            if not self._safe_to_enter(plan.hub, sp.biomass, tick, enemy_strength, neutral_strength):
                if plan.hub is not None and self.reserved_hubs.get(plan.hub) == sp.id:
                    del self.reserved_hubs[plan.hub]
                new_hub = self._pick_best_hub(
                    world=world,
                    my_id=my_id,
                    neutral_id=neutral_id,
                    tick=tick,
                    enemy_strength=enemy_strength,
                    neutral_strength=neutral_strength,
                    my_occupied=my_occupied,
                )
                if new_hub is not None:
                    plan.hub = new_hub
                    self.reserved_hubs[new_hub] = sp.id
                    plan.mode = "SEEK"
                    plan.ring_r = 1
                    plan.ring_i = 0

            if plan.mode == "SEEK":
                if sp_pos != plan.hub:
                    step = self._move_one_step_cost_aware(
                        world=world,
                        my_id=my_id,
                        neutral_id=neutral_id,
                        tick=tick,
                        sp=sp,
                        goal=plan.hub,
                        enemy_strength=enemy_strength,
                        neutral_strength=neutral_strength,
                        my_occupied=my_occupied,
                    )
                    if step is not None:
                        actions.append(step)
                    else:
                        # hub too expensive, repick
                        if plan.hub is not None and self.reserved_hubs.get(plan.hub) == sp.id:
                            del self.reserved_hubs[plan.hub]
                        new_hub = self._pick_best_hub(
                            world=world,
                            my_id=my_id,
                            neutral_id=neutral_id,
                            tick=tick,
                            enemy_strength=enemy_strength,
                            neutral_strength=neutral_strength,
                            my_occupied=my_occupied,
                        )
                        if new_hub is not None:
                            plan.hub = new_hub
                            self.reserved_hubs[new_hub] = sp.id
                            plan.mode = "SEEK"
                            plan.ring_r = 1
                            plan.ring_i = 0
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
                    my_occupied=my_occupied,
                )

                if target is None:
                    if plan.hub is not None and self.reserved_hubs.get(plan.hub) == sp.id:
                        del self.reserved_hubs[plan.hub]

                    new_hub = self._pick_best_hub(
                        world=world,
                        my_id=my_id,
                        neutral_id=neutral_id,
                        tick=tick,
                        enemy_strength=enemy_strength,
                        neutral_strength=neutral_strength,
                        my_occupied=my_occupied,
                    )
                    if new_hub is not None:
                        plan.hub = new_hub
                        self.reserved_hubs[new_hub] = sp.id
                        plan.mode = "SEEK"
                        plan.ring_r = 1
                        plan.ring_i = 0
                    continue

                step = self._move_one_step_cost_aware(
                    world=world,
                    my_id=my_id,
                    neutral_id=neutral_id,
                    tick=tick,
                    sp=sp,
                    goal=target,
                    enemy_strength=enemy_strength,
                    neutral_strength=neutral_strength,
                    my_occupied=my_occupied,
                )
                if step is not None:
                    actions.append(step)
                else:
                    # target too expensive, skip ahead in ring
                    plan.ring_i = (plan.ring_i + 1) % 1000000

        return actions


# ============================================================
# Dijkstra utilities
# ============================================================
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
