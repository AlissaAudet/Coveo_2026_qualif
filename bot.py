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

# Aggression timing
AGGRO_TICK = 80
CHASE_SPAWNER_BEFORE_TICK = 950

# Hub scoring
HUB_NUTRIENT_MIN = 1
OWNED_BY_ME_PENALTY = 60

# Prefer enemy tiles for hubs (territory stealing)
ENEMY_OWNED_BONUS = 260

# Avoid neutral tiles (ownership == neutralTeamId) unless we choose to farm them
NEUTRAL_OWNED_PENALTY = 120
NEUTRAL_HARD_AVOID_BEFORE = 450  # early/mid: avoid neutral

# Spawner hunting
SPAWNER_HUNT_BONUS = 1100
SPAWNER_HUNT_RADIUS = 35

# Ring behavior
RING_MAX_EARLY = 4
RING_MAX_MID = 8
RING_MAX_LATE = 12

# Spawner creation (still important, but not at the expense of attack/territory)
SPAWNER_COST_ALWAYS_OK_UNTIL = 7
SPAWNER_COST_OK_EARLY_UNTIL = 15
SPAWNER_EARLY_TICK = 220
NUTRIENT_FOR_SPAWNER = 55
MAX_SPAWNERS_SOFT = 6  # don't spam spawners endlessly

# Spore production (more fighters to contest)
CLAIMER_BIOMASS = 2
WORKER_BIOMASS = 6
FIGHTER_BIOMASS = 12
FIGHTER_EVERY_N = 4
WORKER_EVERY_N = 9

DESIRED_SPORES_BASE = 12
DESIRED_SPORES_MAX_EXTRA = 28
DESIRED_SPORES_PER_NUTRIENTS = 18

# Combat safety margins
ENEMY_MARGIN_BEFORE_AGGRO = 1
ENEMY_MARGIN_AFTER_AGGRO = 0
NEUTRAL_MARGIN = 2  # more cautious vs neutral (avoid getting stuck)

# Attack/recapture steering
ATTACKERS_MIN = 2
ATTACKERS_DIV = 4  # ~25% attackers
FRONTLINE_RADIUS_FROM_SPAWNERS = 24
ENEMY_TILE_FRONTLINE_BONUS = 180

# Movement scoring (one-step "greedy" to force recapture)
STEP_CLOSER_WEIGHT = 55
STEP_NUTRIENT_WEIGHT = 2
STEP_STEAL_BONUS = 220
STEP_ENEMY_TILE_BONUS = 260
STEP_NEUTRAL_TILE_PENALTY = 200
STEP_FREE_TRACE_BONUS = 18

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
    role: str = "EXPAND"  # EXPAND / ATTACK / DEFEND


class Bot:
    def __init__(self):
        print("Bot: territory+attack first, spawner hunting, avoid neutrals early")
        self.plans: Dict[str, SporePlan] = {}
        self.reserved_hubs: Dict[PositionTuple, str] = {}

    # ----------------------------
    # Helpers (grid[y][x])
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

    @staticmethod
    def _manhattan(a: PositionTuple, b: PositionTuple) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _neighbors(self, world: GameWorld, p: PositionTuple) -> List[PositionTuple]:
        x, y = p
        out: List[PositionTuple] = []
        for dx, dy in DIRS:
            nx, ny = x + dx, y + dy
            if self._in_bounds(world, nx, ny):
                out.append((nx, ny))
        return out

    @staticmethod
    def _dir_to_pos(from_p: PositionTuple, to_p: PositionTuple) -> Position:
        return Position(x=to_p[0] - from_p[0], y=to_p[1] - from_p[1])

    @staticmethod
    def _my_spawner_positions(team: TeamInfo) -> List[PositionTuple]:
        return [(s.position.x, s.position.y) for s in team.spawners]

    @staticmethod
    def _enemy_spawner_positions(world: GameWorld, my_id: str) -> List[PositionTuple]:
        out: List[PositionTuple] = []
        for spw in world.spawners:
            if spw.teamId != my_id:
                out.append((spw.position.x, spw.position.y))
        return out

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
    # Role assignment: more attackers
    # ----------------------------
    def _assign_roles(self, team: TeamInfo) -> Dict[str, str]:
        roles: Dict[str, str] = {}
        spores_sorted = sorted(team.spores, key=lambda s: s.biomass, reverse=True)

        desired_attackers = max(ATTACKERS_MIN, len(spores_sorted) // ATTACKERS_DIV) if len(spores_sorted) >= 3 else 1
        attackers = {s.id for s in spores_sorted[:desired_attackers]}

        for s in spores_sorted:
            roles[s.id] = "ATTACK" if s.id in attackers else "EXPAND"
        return roles

    # ----------------------------
    # Economy: still generate spores + sometimes spawners
    # ----------------------------
    @staticmethod
    def _should_create_spawner(team: TeamInfo, tick: int) -> bool:
        if len(team.spawners) >= MAX_SPAWNERS_SOFT:
            return False
        if team.nextSpawnerCost <= SPAWNER_COST_ALWAYS_OK_UNTIL:
            return True
        if tick < SPAWNER_EARLY_TICK and team.nextSpawnerCost <= SPAWNER_COST_OK_EARLY_UNTIL:
            return True
        return False

    def _produce_spores(self, actions: List[Action], team: TeamInfo, tick: int):
        nutrients = team.nutrients
        desired_spores = DESIRED_SPORES_BASE + min(DESIRED_SPORES_MAX_EXTRA, nutrients // DESIRED_SPORES_PER_NUTRIENTS)

        if len(team.spores) >= desired_spores:
            return

        for spawner in team.spawners:
            if len(team.spores) >= desired_spores:
                break

            want_fighter = (tick >= 10) and (len(team.spores) % FIGHTER_EVERY_N == 0)
            want_worker = (len(team.spores) % WORKER_EVERY_N == 0)

            if want_fighter:
                bio = FIGHTER_BIOMASS
            elif want_worker:
                bio = WORKER_BIOMASS
            else:
                bio = CLAIMER_BIOMASS

            if team.nutrients < bio:
                continue

            actions.append(SpawnerProduceSporeAction(spawnerId=spawner.id, biomass=bio))
            team.nutrients -= bio

    # ----------------------------
    # Hub selection: enemy tiles > nutrients, avoid neutral early
    # ----------------------------
    def _pick_best_hub(
        self,
        world: GameWorld,
        my_id: str,
        neutral_id: str,
        tick: int,
        enemy_spawners: List[PositionTuple],
        my_spawners: List[PositionTuple],
        ref_pos: PositionTuple,
        enemy_strength: Dict[PositionTuple, int],
        neutral_strength: Dict[PositionTuple, int],
    ) -> Optional[PositionTuple]:
        best: Optional[PositionTuple] = None
        best_score = -10**18

        for y in range(world.map.height):
            row = world.map.nutrientGrid[y]
            for x in range(world.map.width):
                v = row[x]
                if v < HUB_NUTRIENT_MIN:
                    continue

                p = (x, y)
                if p in self.reserved_hubs and self.reserved_hubs[p] != "":
                    continue

                owner = self._tile_owner(world, p)

                # Avoid neutral early (even if high nutrient) to not get blocked by neutral spores
                if tick < NEUTRAL_HARD_AVOID_BEFORE and owner == neutral_id:
                    continue

                # hard safety skip for insane neutral stacks
                if neutral_strength.get(p, 0) >= 10 and tick < 800:
                    continue

                score = v * 2

                if owner == my_id and self._tile_biomass(world, p) >= 1:
                    score -= OWNED_BY_ME_PENALTY

                # prefer enemy territory strongly (take soil / unblock)
                if owner not in (my_id, neutral_id) and owner != "":
                    score += ENEMY_OWNED_BONUS

                    # frontline bias (enemy tiles near our spawners)
                    if my_spawners:
                        dmin = min(self._manhattan(p, q) for q in my_spawners)
                        if dmin <= FRONTLINE_RADIUS_FROM_SPAWNERS:
                            score += ENEMY_TILE_FRONTLINE_BONUS + (FRONTLINE_RADIUS_FROM_SPAWNERS - dmin) * 4

                # penalize neutral (when allowed later)
                if owner == neutral_id:
                    score -= NEUTRAL_OWNED_PENALTY

                # spawner hunt (huge)
                if enemy_spawners and tick <= CHASE_SPAWNER_BEFORE_TICK:
                    dmin = min(self._manhattan(p, spw) for spw in enemy_spawners)
                    if dmin == 0:
                        score += SPAWNER_HUNT_BONUS
                    elif dmin <= SPAWNER_HUNT_RADIUS:
                        score += max(0, SPAWNER_HUNT_BONUS - dmin * 30)

                # distance bias (be responsive)
                score -= self._manhattan(ref_pos, p) * 2

                # avoid impossible hubs early
                if tick < AGGRO_TICK and enemy_strength.get(p, 0) >= 16:
                    continue
                if tick < NEUTRAL_HARD_AVOID_BEFORE and neutral_strength.get(p, 0) > 0:
                    # if it's occupied by neutral spores, avoid completely early
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

                if self._owned_by_me(world, my_id, p):
                    continue

                # Avoid neutral tiles early even during ring-fill
                if tick < NEUTRAL_HARD_AVOID_BEFORE and self._tile_owner(world, p) == neutral_id:
                    continue

                if not self._safe_to_enter(p, spore_biomass, tick, enemy_strength, neutral_strength):
                    continue

                plan.ring_i = (idx + 1) % n
                return p

            plan.ring_r += 1
            plan.ring_i = 0

        return None

    # ----------------------------
    # One-step greedy to force stealing (avoids being "blocked")
    # ----------------------------
    def _choose_one_step(
        self,
        world: GameWorld,
        my_id: str,
        neutral_id: str,
        tick: int,
        sp: Spore,
        target: PositionTuple,
        enemy_strength: Dict[PositionTuple, int],
        neutral_strength: Dict[PositionTuple, int],
    ) -> Optional[PositionTuple]:
        start = (sp.position.x, sp.position.y)
        dist0 = self._manhattan(start, target)

        best: Optional[PositionTuple] = None
        best_score = -10**18

        for nb in self._neighbors(world, start):
            if not self._safe_to_enter(nb, sp.biomass, tick, enemy_strength, neutral_strength):
                continue

            owner = self._tile_owner(world, nb)
            is_enemy_tile = owner not in (my_id, neutral_id) and owner != ""
            is_neutral_tile = owner == neutral_id

            # avoid neutral tiles early
            if tick < NEUTRAL_HARD_AVOID_BEFORE and is_neutral_tile:
                continue

            dist1 = self._manhattan(nb, target)
            closer = dist0 - dist1

            score = closer * STEP_CLOSER_WEIGHT
            score += self._nut(world, nb) * STEP_NUTRIENT_WEIGHT

            if not self._owned_by_me(world, my_id, nb):
                score += STEP_STEAL_BONUS

            if is_enemy_tile:
                score += STEP_ENEMY_TILE_BONUS

            if is_neutral_tile:
                score -= STEP_NEUTRAL_TILE_PENALTY

            if self._owned_by_me(world, my_id, nb):
                score += STEP_FREE_TRACE_BONUS

            # stepping onto enemy spore = instant win (safe_to_enter ensured)
            if nb in enemy_strength:
                score += 320

            if score > best_score:
                best_score = score
                best = nb

        return best

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
        enemy_spawners = self._enemy_spawner_positions(world, my_id)
        my_spawners = self._my_spawner_positions(my_team)

        # 0) No units
        if len(my_team.spores) == 0 and len(my_team.spawners) == 0:
            return actions

        # 1) Ensure at least 1 spawner
        if len(my_team.spawners) == 0:
            if len(my_team.spores) > 0:
                s = my_team.spores[0]
                p = (s.position.x, s.position.y)
                if not self._has_spawner_at(world, p):
                    actions.append(SporeCreateSpawnerAction(sporeId=s.id))
            return actions

        # 2) Roles + production
        roles = self._assign_roles(my_team)
        self._produce_spores(actions, my_team, tick)

        # 3) Cleanup plans
        live_ids = {s.id for s in my_team.spores}
        for sid in list(self.plans.keys()):
            if sid not in live_ids:
                old_hub = self.plans[sid].hub
                if old_hub is not None and self.reserved_hubs.get(old_hub) == sid:
                    del self.reserved_hubs[old_hub]
                del self.plans[sid]

        # 4) Assign hubs / ring sizes
        ring_max = RING_MAX_EARLY if tick < 200 else (RING_MAX_MID if tick < 600 else RING_MAX_LATE)

        for sp in my_team.spores:
            if sp.id not in self.plans:
                self.plans[sp.id] = SporePlan(ring_max=ring_max)

            plan = self.plans[sp.id]
            plan.ring_max = ring_max
            plan.role = roles.get(sp.id, "EXPAND")

            sp_pos = (sp.position.x, sp.position.y)
            hub_invalid = plan.hub is None or (plan.hub in self.reserved_hubs and self.reserved_hubs[plan.hub] != sp.id)

            if hub_invalid:
                hub = self._pick_best_hub(
                    world=world,
                    my_id=my_id,
                    neutral_id=neutral_id,
                    tick=tick,
                    enemy_spawners=enemy_spawners,
                    my_spawners=my_spawners,
                    ref_pos=sp_pos,
                    enemy_strength=enemy_strength,
                    neutral_strength=neutral_strength,
                )
                if hub is not None:
                    plan.hub = hub
                    plan.mode = "SEEK"
                    plan.ring_r = 1
                    plan.ring_i = 0
                    self.reserved_hubs[hub] = sp.id

        # 5) Create spawners (still), prefer enemy/high nutrient tiles
        if self._should_create_spawner(my_team, tick):
            for sp in sorted(my_team.spores, key=lambda s: s.biomass, reverse=True):
                if sp.biomass < max(2, my_team.nextSpawnerCost + 1):
                    continue

                p = (sp.position.x, sp.position.y)
                if self._has_spawner_at(world, p):
                    continue

                owner = self._tile_owner(world, p)
                is_enemy_tile = owner not in (my_id, neutral_id) and owner != ""

                if self._nut(world, p) >= NUTRIENT_FOR_SPAWNER or (tick >= AGGRO_TICK and is_enemy_tile):
                    actions.append(SporeCreateSpawnerAction(sporeId=sp.id))
                    break

        used_spores = {a.sporeId for a in actions if hasattr(a, "sporeId")}

        # 6) Moves:
        # - Attackers: hunt spawners, otherwise steal enemy soil (one-step)
        # - Expanders: hub->ring, but also one-step after aggro to force recapture
        for sp in my_team.spores:
            if sp.id in used_spores:
                continue
            if sp.biomass < 2:
                continue

            plan = self.plans.get(sp.id)
            if not plan:
                continue

            sp_pos = (sp.position.x, sp.position.y)

            # A) Adjacent enemy spawner capture
            if plan.role == "ATTACK" and enemy_spawners and tick <= CHASE_SPAWNER_BEFORE_TICK:
                for dx, dy in DIRS:
                    nb = (sp_pos[0] + dx, sp_pos[1] + dy)
                    if not self._in_bounds(world, nb[0], nb[1]):
                        continue
                    if nb in enemy_spawners and self._safe_to_enter(nb, sp.biomass, tick, enemy_strength, neutral_strength):
                        actions.append(SporeMoveAction(sporeId=sp.id, direction=Position(x=dx, y=dy)))
                        break
                else:
                    # B) If close to a spawner, chase it via one-step greedy (avoids neutral blocks)
                    target: Optional[PositionTuple] = None
                    if enemy_spawners:
                        nearest = min(enemy_spawners, key=lambda ep: self._manhattan(sp_pos, ep))
                        if self._manhattan(sp_pos, nearest) <= SPAWNER_HUNT_RADIUS:
                            target = nearest
                    if target is None:
                        target = plan.hub

                    if target is None:
                        continue

                    step = self._choose_one_step(world, my_id, neutral_id, tick, sp, target, enemy_strength, neutral_strength)
                    if step is None:
                        continue
                    actions.append(SporeMoveAction(sporeId=sp.id, direction=self._dir_to_pos(sp_pos, step)))
                continue

            # EXPAND behavior
            if plan.hub is None:
                continue

            # If hub is unsafe for this spore, repick
            if not self._safe_to_enter(plan.hub, sp.biomass, tick, enemy_strength, neutral_strength):
                if plan.hub is not None and self.reserved_hubs.get(plan.hub) == sp.id:
                    del self.reserved_hubs[plan.hub]
                new_hub = self._pick_best_hub(
                    world=world,
                    my_id=my_id,
                    neutral_id=neutral_id,
                    tick=tick,
                    enemy_spawners=enemy_spawners,
                    my_spawners=my_spawners,
                    ref_pos=sp_pos,
                    enemy_strength=enemy_strength,
                    neutral_strength=neutral_strength,
                )
                if new_hub is not None:
                    plan.hub = new_hub
                    self.reserved_hubs[new_hub] = sp.id
                    plan.mode = "SEEK"
                    plan.ring_r = 1
                    plan.ring_i = 0

            # After AGGRO_TICK, use one-step greedy even for expanders to steal territory / avoid blocks
            if tick >= AGGRO_TICK:
                # pick ring target first (steal nearby), else go hub
                target = self._next_ring_target(
                    world=world,
                    my_id=my_id,
                    neutral_id=neutral_id,
                    tick=tick,
                    spore_biomass=sp.biomass,
                    plan=plan,
                    enemy_strength=enemy_strength,
                    neutral_strength=neutral_strength,
                ) or plan.hub

                step = self._choose_one_step(world, my_id, neutral_id, tick, sp, target, enemy_strength, neutral_strength)
                if step is None:
                    continue
                actions.append(SporeMoveAction(sporeId=sp.id, direction=self._dir_to_pos(sp_pos, step)))
                continue

            # Early: SEEK hub -> RING using MoveTo
            if plan.mode == "SEEK":
                if sp_pos != plan.hub:
                    actions.append(SporeMoveToAction(sporeId=sp.id, position=Position(x=plan.hub[0], y=plan.hub[1])))
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
                    if plan.hub is not None and self.reserved_hubs.get(plan.hub) == sp.id:
                        del self.reserved_hubs[plan.hub]

                    new_hub = self._pick_best_hub(
                        world=world,
                        my_id=my_id,
                        neutral_id=neutral_id,
                        tick=tick,
                        enemy_spawners=enemy_spawners,
                        my_spawners=my_spawners,
                        ref_pos=sp_pos,
                        enemy_strength=enemy_strength,
                        neutral_strength=neutral_strength,
                    )
                    if new_hub is not None:
                        plan.hub = new_hub
                        self.reserved_hubs[new_hub] = sp.id
                        plan.mode = "SEEK"
                        plan.ring_r = 1
                        plan.ring_i = 0
                    continue

                actions.append(SporeMoveToAction(sporeId=sp.id, position=Position(x=target[0], y=target[1])))

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