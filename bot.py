import random
import heapq
from typing import Callable, Dict, Tuple, Optional, TypeAlias

from game_message import *

PositionTuple: TypeAlias = Tuple[int, int]  # (x, y)

DIRS: list[PositionTuple] = [(0, -1), (0, 1), (-1, 0), (1, 0)]


# ----------------------------
# Pathfinding
# ----------------------------
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


# ----------------------------
# Small helpers
# ----------------------------
def tpos(p: Position) -> PositionTuple:
    return (p.x, p.y)


def dir_to_position(dx: int, dy: int) -> Position:
    return Position(x=dx, y=dy)


def manhattan(a: PositionTuple, b: PositionTuple) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# ----------------------------
# Bot
# ----------------------------
class Bot:
    def __init__(self):
        print("Initializing your super mega duper bot")
        self.last_pos: dict[str, PositionTuple] = {}
        self.stuck: dict[str, int] = {}

    def get_next_move(self, game_message: TeamGameState) -> list[Action]:
        actions: list[Action] = []

        tick = game_message.tick
        world = game_message.world
        my_id = game_message.yourTeamId
        my_team: TeamInfo = world.teamInfos[my_id]
        neutral_id = game_message.constants.neutralTeamId

        width = world.map.width
        height = world.map.height
        nutrient_grid = world.map.nutrientGrid
        biomass_grid = world.biomassGrid
        owner_grid = world.ownershipGrid

        # ----------------------------
        # Tick parameters
        # ----------------------------
        if tick < 200:
            desired_spores = 8
            scout_biomass = 4
            dijkstra_budget = 12
            top_k_targets = 10
        elif tick < 600:
            desired_spores = 14
            scout_biomass = 6
            dijkstra_budget = 14
            top_k_targets = 10
        else:
            desired_spores = 10
            scout_biomass = 8
            dijkstra_budget = 16
            top_k_targets = 12

        # ----------------------------
        # Grid helpers
        # ----------------------------
        def in_bounds(x: int, y: int) -> bool:
            return 0 <= x < width and 0 <= y < height

        def neighbors4(t: PositionTuple) -> list[PositionTuple]:
            x, y = t
            out: list[PositionTuple] = []
            for dx, dy in DIRS:
                nx, ny = x + dx, y + dy
                if in_bounds(nx, ny):
                    out.append((nx, ny))
            return out

        # Tiles occupied by our own units (do not step onto)
        my_occupied: set[PositionTuple] = set()
        for s in my_team.spores:
            my_occupied.add(tpos(s.position))
        for sp in my_team.spawners:
            my_occupied.add(tpos(sp.position))

        # Spores by position to check fights
        spores_by_pos: dict[PositionTuple, list[Spore]] = {}
        for s in world.spores:
            spores_by_pos.setdefault(tpos(s.position), []).append(s)

        def strongest_enemy_biomass_at(pos: PositionTuple) -> int:
            best = 0
            for s in spores_by_pos.get(pos, []):
                if s.teamId != my_id:
                    best = max(best, s.biomass)
            return best

        def strongest_neutral_biomass_at(pos: PositionTuple) -> int:
            best = 0
            for s in spores_by_pos.get(pos, []):
                if s.teamId == neutral_id:
                    best = max(best, s.biomass)
            return best

        def is_owned_by_me(pos: PositionTuple) -> bool:
            x, y = pos
            return owner_grid[y][x] == my_id and biomass_grid[y][x] > 0

        def tile_nutrient(pos: PositionTuple) -> int:
            x, y = pos
            return nutrient_grid[y][x]

        def safe_to_enter(spore_biomass: int, pos: PositionTuple) -> bool:
            enemy_b = strongest_enemy_biomass_at(pos)
            neutral_b = strongest_neutral_biomass_at(pos)
            if enemy_b >= spore_biomass:
                return False
            if neutral_b >= spore_biomass:
                return False
            return True

        def move_cost_for_spore(spore_biomass: int, a: PositionTuple, b: PositionTuple) -> int:
            # Hard block: stepping onto our occupied tiles
            if b in my_occupied:
                return -1
            # Hard block: suicidal tiles
            if not safe_to_enter(spore_biomass, b):
                return -1
            # Trail rule: owned costs 0, new costs 1
            return 0 if is_owned_by_me(b) else 1

        # ----------------------------
        # Territory geometry
        # ----------------------------
        def territory_cells() -> list[PositionTuple]:
            out: list[PositionTuple] = []
            for y in range(height):
                for x in range(width):
                    if owner_grid[y][x] == my_id and biomass_grid[y][x] > 0:
                        out.append((x, y))
            return out

        def centroid(cells: list[PositionTuple]) -> PositionTuple:
            if not cells:
                return (width // 2, height // 2)
            sx = sum(p[0] for p in cells)
            sy = sum(p[1] for p in cells)
            return (sx // len(cells), sy // len(cells))

        def is_frontier_cell(pos: PositionTuple) -> bool:
            x, y = pos
            if owner_grid[y][x] == my_id:
                return False
            for nb in neighbors4(pos):
                if is_owned_by_me(nb):
                    return True
            return False

        def candidate_targets(max_targets: int = 220) -> list[PositionTuple]:
            front: list[PositionTuple] = []
            other: list[PositionTuple] = []

            for y in range(height):
                for x in range(width):
                    pos = (x, y)
                    if owner_grid[y][x] == my_id:
                        continue
                    if nutrient_grid[y][x] <= 0:
                        continue
                    if pos in my_occupied:
                        continue
                    if is_frontier_cell(pos):
                        front.append(pos)
                    else:
                        other.append(pos)

            front.sort(key=lambda p: nutrient_grid[p[1]][p[0]], reverse=True)
            other.sort(key=lambda p: nutrient_grid[p[1]][p[0]], reverse=True)

            return front[:max_targets] + other[: max_targets // 3]

        # ----------------------------
        # Movement primitives
        # ----------------------------
        def greedy_step(spore: Spore) -> Optional[SporeMoveAction]:
            cur = tpos(spore.position)

            best_dir: Optional[PositionTuple] = None
            best_score = -10**9

            for dx, dy in DIRS:
                nxt = (cur[0] + dx, cur[1] + dy)
                if not in_bounds(nxt[0], nxt[1]):
                    continue
                if nxt in my_occupied:
                    continue
                if not safe_to_enter(spore.biomass, nxt):
                    continue

                n = tile_nutrient(nxt)
                owned = owner_grid[nxt[1]][nxt[0]] == my_id

                # prefer unowned nutrient, small preference for outward
                score = n * 10 + (0 if owned else 50)
                score += manhattan(nxt, terr_center) * 2

                if score > best_score:
                    best_score = score
                    best_dir = (dx, dy)

            if best_dir is None:
                return None
            return SporeMoveAction(sporeId=spore.id, direction=dir_to_position(best_dir[0], best_dir[1]))

        def pick_move_toward_goal(spore: Spore, goal: PositionTuple, budget: int) -> Optional[SporeMoveAction]:
            cur = tpos(spore.position)

            def cost_fn(a: PositionTuple, b: PositionTuple) -> int:
                return move_cost_for_spore(spore.biomass, a, b)

            dist, prev = dijkstra(
                start=cur,
                neighbors_fn=neighbors4,
                cost_fn=cost_fn,
                is_goal_fn=lambda p: p == goal,
                max_cost=budget,
            )

            if goal not in dist:
                return None

            path = reconstruct_path(prev, cur, goal)
            if len(path) < 2:
                return None

            nxt = path[1]
            dx = nxt[0] - cur[0]
            dy = nxt[1] - cur[1]
            if (dx, dy) not in DIRS:
                return None

            return SporeMoveAction(sporeId=spore.id, direction=dir_to_position(dx, dy))

        # ----------------------------
        # 1) Ensure at least one spawner
        # ----------------------------
        if len(my_team.spawners) == 0:
            candidates = [s for s in my_team.spores if s.biomass >= my_team.nextSpawnerCost]
            candidates.sort(key=lambda s: (tile_nutrient(tpos(s.position)), s.biomass), reverse=True)
            for s in candidates:
                if safe_to_enter(s.biomass, tpos(s.position)):
                    actions.append(SporeCreateSpawnerAction(sporeId=s.id))
                    break

        # ----------------------------
        # 2) Produce spores to expand
        # ----------------------------
        # create scouts if under desired count
        if len(my_team.spawners) > 0 and my_team.nutrients >= scout_biomass and len(my_team.spores) < desired_spores:
            # spread production across spawners
            for sp in my_team.spawners:
                if my_team.nutrients < scout_biomass:
                    break
                actions.append(SpawnerProduceSporeAction(spawnerId=sp.id, biomass=scout_biomass))
                my_team.nutrients -= scout_biomass

        # late: occasionally spawn a stronger spore if rich
        if tick >= 600 and len(my_team.spawners) > 0 and my_team.nutrients >= 18 and (tick % 10 == 0):
            sp0 = my_team.spawners[0]
            actions.append(SpawnerProduceSporeAction(spawnerId=sp0.id, biomass=18))
            my_team.nutrients -= 18

        # ----------------------------
        # 3) Movement: expand, avoid clumping, stay connected via trail
        # ----------------------------
        owned_cells = territory_cells()
        terr_center = centroid(owned_cells)
        all_targets = candidate_targets()

        # active spores only (biomass >= 2 can move and not die instantly)
        active_spores = [s for s in my_team.spores if s.biomass >= 2]
        active_spores.sort(key=lambda s: s.biomass, reverse=True)

        # update stuck counter
        for s in active_spores:
            cur = tpos(s.position)
            last = self.last_pos.get(s.id)
            if last == cur:
                self.stuck[s.id] = self.stuck.get(s.id, 0) + 1
            else:
                self.stuck[s.id] = 0
            self.last_pos[s.id] = cur

        reserved: set[PositionTuple] = set()

        def target_score(spore: Spore, target: PositionTuple) -> float:
            tx, ty = target
            n = nutrient_grid[ty][tx]
            outward = manhattan(target, terr_center)

            # keep spores separated
            if my_team.spores:
                sep = min(manhattan(target, tpos(s.position)) for s in my_team.spores)
            else:
                sep = 0

            frontier_bonus = 60 if is_frontier_cell(target) else 0
            d0 = manhattan(target, tpos(spore.position))

            stuck_boost = min(6, self.stuck.get(spore.id, 0))
            outward_weight = 2 + stuck_boost  # if stuck, push outward harder

            if tick < 200:
                return n * 12 + outward * outward_weight + sep * 6 + frontier_bonus - d0 * 3
            if tick < 600:
                return n * 12 + outward * outward_weight + sep * 4 + frontier_bonus - d0 * 4
            return n * 14 + outward * outward_weight + sep * 3 + frontier_bonus - d0 * 4

        for spore in active_spores:
            # do not send two actions for the same spore
            if any(getattr(a, "sporeId", None) == spore.id for a in actions):
                continue

            # pick top K targets for this spore
            scored = []
            for tgt in all_targets:
                if tgt in reserved:
                    continue
                if tgt in my_occupied:
                    continue
                # goal must be safe for this spore to eventually enter
                if not safe_to_enter(spore.biomass, tgt):
                    continue
                scored.append((target_score(spore, tgt), tgt))

            scored.sort(key=lambda x: x[0], reverse=True)
            tried = 0
            chosen_action: Optional[SporeMoveAction] = None
            chosen_goal: Optional[PositionTuple] = None

            for _, goal in scored:
                tried += 1
                chosen_action = pick_move_toward_goal(spore, goal, dijkstra_budget)
                if chosen_action is not None:
                    chosen_goal = goal
                    break
                if tried >= top_k_targets:
                    break

            if chosen_action is None:
                chosen_action = greedy_step(spore)

            if chosen_action is not None:
                actions.append(chosen_action)
                if chosen_goal is not None:
                    reserved.add(chosen_goal)

        return actions
