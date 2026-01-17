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

# Aggression timing
AGGRO_TICK = 120

# Avoid neutrals (strong)
AVOID_NEUTRALS = True
NEUTRAL_HARD_AVOID_ALWAYS = True
NEUTRAL_HARD_AVOID_BEFORE = 1000  # used only if ALWAYS=Falsed

# Hub scoring
OWNED_BY_ME_PENALTY = 35
ENEMY_OWNED_BONUS = 180
NEUTRAL_OWNED_BONUS = 0  # set to 0 since we want to avoid neutrals
HUB_NUTRIENT_MIN = 1

# Spawner hunting
SPAWNER_HUNT_BONUS = 800
SPAWNER_HUNT_RADIUS = 30
CHASE_SPAWNER_BEFORE_TICK = 900

# Ring behavior
RING_MAX_EARLY = 4
RING_MAX_MID = 7
RING_MAX_LATE = 10

# Spawner creation
SPAWNER_COST_ALWAYS_OK_UNTIL = 7
SPAWNER_COST_OK_EARLY_UNTIL = 15
SPAWNER_EARLY_TICK = 220
NUTRIENT_FOR_SPAWNER = 45

# Spore production
CLAIMER_BIOMASS = 2
WORKER_BIOMASS = 6
FIGHTER_BIOMASS = 12
FIGHTER_EVERY_N = 6
WORKER_EVERY_N = 9

DESIRED_SPORES_BASE = 10
DESIRED_SPORES_MAX_EXTRA = 30
DESIRED_SPORES_PER_NUTRIENTS = 20

# Combat safety margins
ENEMY_MARGIN_BEFORE_AGGRO = 1
ENEMY_MARGIN_AFTER_AGGRO = 0
NEUTRAL_MARGIN = 2  # stronger avoid neutrals (need clearly bigger to enter)


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
        print("Bot: territory + attack + spawners, avoid neutrals, greedy move")
        self.plans: Dict[str, SporePlan] = {}
        self.reserved_hubs: Dict[PositionTuple, str] = {}  # hub -> sporeId

    # ----------------------------
    # Neutral avoidance switch
    # ----------------------------
    @staticmethod
    def _avoid_neutral_now(tick: int) -> bool:
        if not AVOID_NEUTRALS:
            return False
        if NEUTRAL_HARD_AVOID_ALWAYS:
            return True
        return tick < NEUTRAL_HARD_AVOID_BEFORE

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

    @staticmethod
    def _manhattan(a: PositionTuple, b: PositionTuple) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    @staticmethod
    def _dir_to_pos(from_p: PositionTuple, to_p: PositionTuple) -> Position:
        return Position(x=to_p[0] - from_p[0], y=to_p[1] - from_p[1])

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
    # Enemy spawner positions
    # ----------------------------
    @staticmethod
    def _enemy_spawner_positions(world: GameWorld, my_id: str) -> List[PositionTuple]:
        return [(spw.position.x, spw.position.y) for spw in world.spawners if spw.teamId != my_id]

    # ----------------------------
    # Hub selection (fast, aggressive, avoid neutrals)
    # ----------------------------
    def _pick_best_hub(
        self,
        world: GameWorld,
        my_id: str,
        neutral_id: str,
        tick: int,
        enemy_strength: Dict[PositionTuple, int],
        neutral_strength: Dict[PositionTuple, int],
        ref_pos: Optional[PositionTuple] = None,
    ) -> Optional[PositionTuple]:
        best: Optional[PositionTuple] = None
        best_score = -10**12

        enemy_spawners = self._enemy_spawner_positions(world, my_id)

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

                # HARD AVOID neutral hubs if enabled
                if self._avoid_neutral_now(tick) and owner == neutral_id:
                    continue

                score = v * 3

                if owner == my_id and self._tile_biomass(world, p) >= 1:
                    score -= OWNED_BY_ME_PENALTY

                if tick >= AGGRO_TICK and owner not in (my_id, neutral_id):
                    score += ENEMY_OWNED_BONUS

                # spawner hunting
                if tick <= CHASE_SPAWNER_BEFORE_TICK and enemy_spawners:
                    dmin = 10**9
                    for spw_pos in enemy_spawners:
                        d = self._manhattan(p, spw_pos)
                        if d < dmin:
                            dmin = d
                    if dmin == 0:
                        score += SPAWNER_HUNT_BONUS
                    elif dmin <= SPAWNER_HUNT_RADIUS:
                        score += max(0, SPAWNER_HUNT_BONUS - dmin * 30)

                if ref_pos is not None:
                    score -= self._manhattan(ref_pos, p) * 2

                # early: avoid extreme stacks
                if enemy_strength.get(p, 0) >= 18 and tick < AGGRO_TICK:
                    continue
                # also avoid neutral stacks as hubs (if not already avoided)
                if neutral_strength.get(p, 0) > 0 and self._avoid_neutral_now(tick):
                    continue

                if score > best_score:
                    best_score = score
                    best = p

        return best

    # ----------------------------
    # Ring fill (square rings)
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

                # HARD AVOID neutral tiles (do not even target)
                if self._avoid_neutral_now(tick) and self._tile_owner(world, p) == neutral_id:
                    continue

                if not self._safe_to_enter(p, spore_biomass, tick, enemy_strength, neutral_strength):
                    continue

                plan.ring_i = (idx + 1) % n
                return p

            plan.ring_r += 1
            plan.ring_i = 0

        return None

    # ----------------------------
    # Greedy step selection (avoid neutral traversal, opportunistic fights)
    # ----------------------------
    def _choose_one_step(
        self,
        world: GameWorld,
        my_id: str,
        neutral_id: str,
        tick: int,
        spore: Spore,
        target: PositionTuple,
        enemy_strength: Dict[PositionTuple, int],
        neutral_strength: Dict[PositionTuple, int],
    ) -> Optional[PositionTuple]:
        start = (spore.position.x, spore.position.y)
        best = None
        best_score = -10**18

        for dx, dy in DIRS:
            nb = (start[0] + dx, start[1] + dy)
            if not self._in_bounds(world, nb[0], nb[1]):
                continue

            owner = self._tile_owner(world, nb)

            # HARD avoid neutrals (do not enter neutral tiles at all)
            if self._avoid_neutral_now(tick) and owner == neutral_id:
                continue

            # must be survivable
            if not self._safe_to_enter(nb, spore.biomass, tick, enemy_strength, neutral_strength):
                continue

            # scoring
            dist_before = self._manhattan(start, target)
            dist_after = self._manhattan(nb, target)
            closer = dist_before - dist_after

            nutrient = self._nut(world, nb)
            steal_enemy = 1 if (owner not in (my_id, neutral_id) and owner != "") else 0
            claim_new = 1 if not self._owned_by_me(world, my_id, nb) else 0

            # If enemy spore there and we win => big bonus
            e = enemy_strength.get(nb, 0)
            attack_bonus = 0
            if e > 0 and spore.biomass > e:
                attack_bonus = 600 + (spore.biomass - e) * 5

            score = (
                closer * 60
                + nutrient * 2
                + steal_enemy * 200
                + claim_new * 80
                + attack_bonus
            )

            if score > best_score:
                best_score = score
                best = nb

        return best

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

    def _produce_spores(self, actions: List[Action], team: TeamInfo, tick: int):
        nutrients = team.nutrients
        desired_spores = DESIRED_SPORES_BASE + min(
            DESIRED_SPORES_MAX_EXTRA, nutrients // DESIRED_SPORES_PER_NUTRIENTS
        )

        if len(team.spores) >= desired_spores:
            return

        for spawner in team.spawners:
            if len(team.spores) >= desired_spores:
                break

            want_fighter = (tick >= 30) and (len(team.spores) % FIGHTER_EVERY_N == 0)
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

        # 0) No units
        if len(my_team.spores) == 0 and len(my_team.spawners) == 0:
            return actions

        # 1) Ensure at least 1 spawner (avoid elimination)
        if len(my_team.spawners) == 0:
            if len(my_team.spores) > 0:
                s = my_team.spores[0]
                p = (s.position.x, s.position.y)
                if not self._has_spawner_at(world, p):
                    actions.append(SporeCreateSpawnerAction(sporeId=s.id))
            return actions

        # 2) Produce spores
        self._produce_spores(actions, my_team, tick)

        # 3) Cleanup dead spores
        live_ids = {s.id for s in my_team.spores}
        for sid in list(self.plans.keys()):
            if sid not in live_ids:
                old_hub = self.plans[sid].hub
                if old_hub is not None and self.reserved_hubs.get(old_hub) == sid:
                    del self.reserved_hubs[old_hub]
                del self.plans[sid]

        # 4) Assign hubs / plans
        ring_max = RING_MAX_EARLY if tick < 200 else (RING_MAX_MID if tick < 600 else RING_MAX_LATE)

        for sp in my_team.spores:
            if sp.id not in self.plans:
                self.plans[sp.id] = SporePlan(ring_max=ring_max)

            plan = self.plans[sp.id]
            plan.ring_max = ring_max

            sp_pos = (sp.position.x, sp.position.y)

            hub_invalid = (
                plan.hub is None
                or (plan.hub in self.reserved_hubs and self.reserved_hubs[plan.hub] != sp.id)
            )
            if hub_invalid:
                hub = self._pick_best_hub(
                    world=world,
                    my_id=my_id,
                    neutral_id=neutral_id,
                    tick=tick,
                    enemy_strength=enemy_strength,
                    neutral_strength=neutral_strength,
                    ref_pos=sp_pos,
                )
                if hub is not None:
                    plan.hub = hub
                    plan.mode = "SEEK"
                    plan.ring_r = 1
                    plan.ring_i = 0
                    self.reserved_hubs[hub] = sp.id

        # 5) Create spawners while cheap, on good soil (avoid collisions)
        if self._should_create_spawner(my_team, tick):
            for sp in my_team.spores:
                if sp.biomass < max(2, my_team.nextSpawnerCost + 1):
                    continue

                p = (sp.position.x, sp.position.y)
                if self._has_spawner_at(world, p):
                    continue

                owner = self._tile_owner(world, p)

                # avoid creating on neutral tiles if we're in avoid-neutral mode
                if self._avoid_neutral_now(tick) and owner == neutral_id:
                    continue

                if self._nut(world, p) >= NUTRIENT_FOR_SPAWNER or (tick >= AGGRO_TICK and owner not in (my_id, neutral_id)):
                    actions.append(SporeCreateSpawnerAction(sporeId=sp.id))
                    break

        used_spores = {a.sporeId for a in actions if hasattr(a, "sporeId")}

        # 6) Movement: attack / steal / expand, avoid neutrals, greedy steps
        enemy_spawners = self._enemy_spawner_positions(world, my_id)

        for sp in my_team.spores:
            if sp.id in used_spores:
                continue
            if sp.biomass < 2:
                continue

            plan = self.plans.get(sp.id)
            if not plan:
                continue

            sp_pos = (sp.position.x, sp.position.y)

            # A) Adjacent enemy spawner: step onto it if safe (pressure)
            moved = False
            if enemy_spawners and tick <= CHASE_SPAWNER_BEFORE_TICK:
                for dx, dy in DIRS:
                    nb = (sp_pos[0] + dx, sp_pos[1] + dy)
                    if not self._in_bounds(world, nb[0], nb[1]):
                        continue
                    if nb in enemy_spawners:
                        if self._avoid_neutral_now(tick) and self._tile_owner(world, nb) == neutral_id:
                            continue
                        if self._safe_to_enter(nb, sp.biomass, tick, enemy_strength, neutral_strength):
                            actions.append(SporeMoveAction(sporeId=sp.id, direction=Position(x=dx, y=dy)))
                            moved = True
                            break
            if moved:
                continue

            # B) If enemy spawner nearby, chase it with greedy steps (avoid neutral traversal)
            if enemy_spawners and tick <= CHASE_SPAWNER_BEFORE_TICK:
                nearest = None
                dmin = 10**9
                for ep in enemy_spawners:
                    d = self._manhattan(sp_pos, ep)
                    if d < dmin:
                        dmin = d
                        nearest = ep
                if nearest is not None and dmin <= SPAWNER_HUNT_RADIUS:
                    step = self._choose_one_step(
                        world=world,
                        my_id=my_id,
                        neutral_id=neutral_id,
                        tick=tick,
                        spore=sp,
                        target=nearest,
                        enemy_strength=enemy_strength,
                        neutral_strength=neutral_strength,
                    )
                    if step is not None:
                        actions.append(SporeMoveAction(sporeId=sp.id, direction=self._dir_to_pos(sp_pos, step)))
                        continue

            # C) Normal plan: SEEK hub -> RING (greedy steps)
            if plan.hub is None:
                hub = self._pick_best_hub(world, my_id, neutral_id, tick, enemy_strength, neutral_strength, ref_pos=sp_pos)
                if hub is None:
                    continue
                plan.hub = hub
                self.reserved_hubs[hub] = sp.id
                plan.mode = "SEEK"
                plan.ring_r = 1
                plan.ring_i = 0

            # If hub is unsafe for this spore, repick
            if not self._safe_to_enter(plan.hub, sp.biomass, tick, enemy_strength, neutral_strength):
                if plan.hub is not None and self.reserved_hubs.get(plan.hub) == sp.id:
                    del self.reserved_hubs[plan.hub]
                new_hub = self._pick_best_hub(world, my_id, neutral_id, tick, enemy_strength, neutral_strength, ref_pos=sp_pos)
                if new_hub is not None:
                    plan.hub = new_hub
                    self.reserved_hubs[new_hub] = sp.id
                    plan.mode = "SEEK"
                    plan.ring_r = 1
                    plan.ring_i = 0

            if plan.mode == "SEEK":
                if sp_pos != plan.hub:
                    step = self._choose_one_step(world, my_id, neutral_id, tick, sp, plan.hub, enemy_strength, neutral_strength)
                    if step is None:
                        continue
                    actions.append(SporeMoveAction(sporeId=sp.id, direction=self._dir_to_pos(sp_pos, step)))
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
                    new_hub = self._pick_best_hub(world, my_id, neutral_id, tick, enemy_strength, neutral_strength, ref_pos=sp_pos)
                    if new_hub is not None:
                        plan.hub = new_hub
                        self.reserved_hubs[new_hub] = sp.id
                        plan.mode = "SEEK"
                        plan.ring_r = 1
                        plan.ring_i = 0
                    continue

                step = self._choose_one_step(world, my_id, neutral_id, tick, sp, target, enemy_strength, neutral_strength)
                if step is None:
                    continue
                actions.append(SporeMoveAction(sporeId=sp.id, direction=self._dir_to_pos(sp_pos, step)))

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
