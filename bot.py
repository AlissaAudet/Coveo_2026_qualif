from heapq import heappush, heappop
from itertools import combinations
from game_message import *

class Bot:
    def __init__(self):
        print("Initializing connect-all first bot")

    # ---------- effective colony value (current + near future) ----------
    def colony_value(self, colony: Colony, horizon: int = 3) -> float:
        fut = colony.futureNutrients or []
        if not fut:
            return float(colony.nutrients)
        k = max(1, min(horizon, len(fut)))
        return 0.6 * colony.nutrients + 0.4 * max(fut[:k])

    # ---------- Dijkstra shortest path over 4-neighbor grid ----------
    # Colonies are obstacles except the target colony.
    def dijkstra_path(self, start: Position, end: Position, width: int, height: int,
                      colony_positions: set[tuple[int, int]]):
        dirs = [(1,0),(-1,0),(0,1),(0,-1)]
        sx, sy = start.x, start.y
        tx, ty = end.x, end.y
        start_xy, end_xy = (sx, sy), (tx, ty)

        dist = {start_xy: 0}
        prev = {}
        pq = [(0, start_xy)]
        seen = set()

        while pq:
            cost, (x, y) = heappop(pq)
            if (x, y) in seen:
                continue
            seen.add((x, y))
            if (x, y) == end_xy:
                path = []
                cur = end_xy
                while cur in prev:
                    path.append(Position(x=cur[0], y=cur[1]))
                    cur = prev[cur]
                path.append(Position(x=sx, y=sy))
                path.reverse()
                return path, cost
            for dx, dy in dirs:
                nx, ny = x + dx, y + dy
                if nx < 0 or ny < 0 or nx >= width or ny >= height:
                    continue
                if (nx, ny) != end_xy and (nx, ny) in colony_positions:
                    continue
                nc = cost + 1
                if (nx, ny) not in dist or nc < dist[(nx, ny)]:
                    dist[(nx, ny)] = nc
                    prev[(nx, ny)] = (x, y)
                    heappush(pq, (nc, (nx, ny)))
        return [], float("inf")

    # ---------- compute biomass tile components and colony components ----------
    # A colony joins the component of any adjacent biomass tile (>0); else it is its own component.
    def colony_components(self, game_map: GameMap) -> list[int]:
        width, height = game_map.width, game_map.height
        bio = game_map.biomass
        dirs = [(1,0),(-1,0),(0,1),(0,-1)]
        def inb(x, y): return 0 <= x < width and 0 <= y < height

        comp = [[-1] * height for _ in range(width)]
        cid = 0
        for x in range(width):
            for y in range(height):
                if bio[x][y] > 0 and comp[x][y] == -1:
                    q = [(x, y)]
                    comp[x][y] = cid
                    qi = 0
                    while qi < len(q):
                        cx, cy = q[qi]; qi += 1
                        for dx, dy in dirs:
                            nx, ny = cx + dx, cy + dy
                            if inb(nx, ny) and bio[nx][ny] > 0 and comp[nx][ny] == -1:
                                comp[nx][ny] = cid
                                q.append((nx, ny))
                    cid += 1

        result = [-1] * len(game_map.colonies)
        next_id = cid
        for i, c in enumerate(game_map.colonies):
            assigned = False
            for dx, dy in dirs:
                nx, ny = c.position.x + dx, c.position.y + dy
                if inb(nx, ny) and comp[nx][ny] != -1:
                    result[i] = comp[nx][ny]
                    assigned = True
                    break
            if not assigned:
                result[i] = next_id
                next_id += 1
        return result

    # ---------- DSU ----------
    def find(self, parent, i):
        if parent[i] != i:
            parent[i] = self.find(parent, parent[i])
        return parent[i]
    def union(self, parent, rank, a, b):
        ra, rb = self.find(parent, a), self.find(parent, b)
        if ra == rb: return False
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1
        return True

    # ---------- build all pair paths and edge metrics ----------
    # For each pair (i,j), compute:
    #  - inner path tiles (exclude colony tiles)
    #  - needed = tiles with 0 biomass (actual cost to complete now)
    #  - value priority = (valA * valB) / path_len to prefer short, rich pairs
    def build_edges(self, game_map: GameMap):
        cols = game_map.colonies
        width, height = game_map.width, game_map.height
        colony_pos = {(c.position.x, c.position.y) for c in cols}

        edges = []   # (needed, -value_score, path_len, i, j)
        paths = {}   # (i,j) -> inner tiles list

        for (i, a), (j, b) in combinations(enumerate(cols), 2):
            path, dist = self.dijkstra_path(a.position, b.position, width, height, colony_pos)
            if not path or dist == float("inf"):
                continue
            inner = path[1:-1] if len(path) >= 2 else []
            if not inner:
                continue  # cannot place on colony tiles
            needed = sum(1 for p in inner if game_map.biomass[p.x][p.y] == 0)
            vA, vB = self.colony_value(a), self.colony_value(b)
            value_score = (vA * vB) / (len(inner) + 1e-6)
            edges.append((needed, -value_score, len(inner), i, j))
            paths[(i, j)] = inner

        # primary sort by minimal needed (to connect as many colonies as possible this turn)
        # tie-break by higher value_score then shorter path
        edges.sort(key=lambda e: (e[0], e[1], e[2]))
        return edges, paths

    # ---------- Phase A: connect all colonies with minimal needed first ----------
    # Guarantees no partial paths. Greedily grows one connected component until all join.
    def plan_connect_all(self, game_state: TeamGameState, remaining: int):
        actions = []
        gmap = game_state.map
        n = len(gmap.colonies)

        # seed DSU from existing biomass components
        comps = self.colony_components(gmap)
        parent = [i for i in range(n)]
        rank = [0] * n
        rep = {}
        for idx, cid in enumerate(comps):
            if cid not in rep:
                rep[cid] = idx
            else:
                self.union(parent, rank, rep[cid], idx)

        edges, paths = self.build_edges(gmap)

        # greedy: always pick the smallest-needed edge that actually connects two sets
        progress = True
        while progress and remaining > 0:
            progress = False
            for needed, neg_val, plen, i, j in edges:
                if remaining <= 0:
                    break
                if self.find(parent, i) == self.find(parent, j):
                    continue
                if needed == 0:
                    self.union(parent, rank, i, j)
                    progress = True
                    continue
                if needed > remaining:
                    continue  # no partials

                # lay full inner path
                inner = paths[(i, j)]
                laid = 0
                for p in inner:
                    if gmap.biomass[p.x][p.y] == 0:
                        actions.append(AddBiomassAction(position=p, amount=1))
                        laid += 1
                if laid != needed:
                    # defensive: should not happen, but do not union if mismatch
                    continue
                remaining -= needed
                self.union(parent, rank, i, j)
                progress = True

                # early exit if all in one component
                r0 = self.find(parent, 0)
                if all(self.find(parent, k) == r0 for k in range(n)):
                    return actions, remaining, paths, parent, rank

        return actions, remaining, paths, parent, rank

    # ---------- Phase B: reinforce best existing links (use leftover only) ----------
    # Raise force by +1 only if we can cover all bottleneck tiles this turn.
    def plan_reinforce_best(self, game_state: TeamGameState, remaining: int, paths: dict):
        if remaining <= 0:
            return [], remaining
        gmap = game_state.map
        cols = gmap.colonies

        candidates = []  # (gain_per_bio, gain, cost, bottleneck_positions)
        for (i, j), inner in paths.items():
            if not inner:
                continue
            # must be fully built
            if any(gmap.biomass[p.x][p.y] == 0 for p in inner):
                continue
            f = min(gmap.biomass[p.x][p.y] for p in inner)
            bottleneck = [p for p in inner if gmap.biomass[p.x][p.y] == f]
            cost = len(bottleneck)
            if cost == 0:
                continue

            vA = self.colony_value(cols[i])
            vB = self.colony_value(cols[j])
            def score(force): return min(int(vA), force) * min(int(vB), force)
            delta = score(f + 1) - score(f)
            if delta <= 0:
                continue
            candidates.append((delta / cost, delta, cost, bottleneck))

        if not candidates:
            return [], remaining

        candidates.sort(key=lambda t: (-t[0], -t[1]))
        actions = []
        for ratio, delta, cost, bottleneck in candidates:
            if cost > remaining:
                continue
            for p in bottleneck:
                actions.append(AddBiomassAction(position=p, amount=1))
            remaining -= cost
            if remaining <= 0:
                break
        return actions, remaining

    # ---------- Main ----------
    def get_next_move(self, game_message: TeamGameState):
        actions: list[Action] = []

        per_turn_cap = game_message.maximumNumberOfBiomassPerTurn
        total_left = game_message.maximumNumberOfBiomassOnMap - sum(sum(r) for r in game_message.map.biomass)
        remaining = max(0, min(per_turn_cap, total_left))
        if remaining == 0:
            print("No biomass capacity this turn.")
            return actions

        # Phase A: connect all colonies first with minimal needed edges (no partials)
        addA, remaining, paths, parent, rank = self.plan_connect_all(game_message, remaining)
        actions.extend(addA)

        # If all colonies are connected and we still have budget, reinforce best links
        n = len(game_message.map.colonies)
        if remaining > 0 and n > 1:
            r0 = self.find(parent, 0)
            all_one = all(self.find(parent, k) == r0 for k in range(n))
            if all_one:
                addB, remaining = self.plan_reinforce_best(game_message, remaining, paths)
                actions.extend(addB)

        print(f"Planned actions: {len(actions)}")
        return actions
