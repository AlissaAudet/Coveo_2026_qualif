import math
from game_message import *


class Bot:
    def __init__(self):
        self.skip_next_add = False
        self.init_edges = None
        self.init_paths = None
        self.init_progress = None
        self.initialized = False
        self.used_tiles = set()
        self.active_paths = []  # [(path, (c1, c2))]
        self.last_ranked = []
        self.last_recalc_tick = -1
        print("Initializing your super mega duper bot")

    # ---------------------------------------------------------
    # --- MAIN DECISION LOOP ---------------------------------
    # ---------------------------------------------------------
    def get_next_move(self, game_message: TeamGameState):
        tick = getattr(game_message, "currentTickNumber", getattr(game_message, "tick", 0))
        actions = []

        # --- 0. skip one add tick after freeing biomass ---
        if hasattr(self, "skip_next_add") and self.skip_next_add:
            self.skip_next_add = False
            print("⏸️ Skipping add phase (waiting for biomass removal to apply)")
            return []

        # --- 1. prevent total biomass overflow ---
        total_biomass = sum(sum(row) for row in game_message.map.biomass)
        if total_biomass >= game_message.maximumNumberOfBiomassOnMap:
            remove_actions, freed = self.remove_worst_path(game_message)
            if freed > 0:
                print(f"⚠️ Biomass cap reached ({total_biomass}), freeing {freed}")
                self.skip_next_add = True  # pause 1 tick before adding again
                return remove_actions

        # --- 2. connect all colonies initially ---
        if not self.initialized:
            if not self.init_edges:
                self.init_edges = self.mst_edges(game_message.map.colonies)
                self.init_paths = [self.line_between(c1.position, c2.position) for c1, c2 in self.init_edges]
                self.init_progress = [0] * len(self.init_paths)

            actions = self.connect_all_colonies(game_message)
            if self.initialized:
                self.active_paths = list(zip(self.init_paths, self.init_edges))
                print("✅ All colonies connected!")
            return actions

        # --- 3. reinforce the most valuable path ---
        ranked = self.rank_paths(game_message)
        if not ranked:
            return []

        best_score, best_path, best_pair = ranked[0]
        actions = self.reinforce_path(game_message, best_path)

        # --- 4. if nothing was done, reallocate weakest path ---
        if not actions:
            remove_actions, freed = self.remove_worst_path(game_message)
            if freed > 0:
                ranked = self.rank_paths(game_message)
                if ranked:
                    best_score, best_path, _ = ranked[0]
                    add_actions = self.build_best_new_path(game_message, freed)
                    actions = remove_actions + add_actions
                    print("♻️ Forced reallocation: removed weakest path to make progress")
                self.skip_next_add = True  # also pause after forced reallocation
                return actions

        # --- 5. safety: if all paths removed, rebuild top 3 ---
        if self.initialized and not self.active_paths:
            ranked = self.rank_paths(game_message)
            if ranked:
                best = ranked[:3]
                self.active_paths = [(p, pair) for _, p, pair in best]
                print("🌱 Rebuilt initial active paths to keep growing")

        return actions[:game_message.maximumNumberOfBiomassPerTurn]

    # ---------------------------------------------------------
    # --- INITIAL CONNECTION PHASE ----------------------------
    # ---------------------------------------------------------
    def connect_all_colonies(self, game_message: TeamGameState):
        """Gradually connect all MST paths with biomass."""
        actions = []
        remaining = game_message.maximumNumberOfBiomassPerTurn
        current_total = sum(sum(row) for row in game_message.map.biomass)
        max_total = game_message.maximumNumberOfBiomassOnMap
        colony_positions = {(c.position.x, c.position.y) for c in game_message.map.colonies}

        for i, path in enumerate(self.init_paths):
            if remaining <= 0 or current_total >= max_total:
                break
            idx = self.init_progress[i]
            while idx < len(path) and remaining > 0 and current_total < max_total:
                x, y = path[idx]
                if (x, y) in colony_positions or (x, y) in self.used_tiles:
                    idx += 1
                    continue
                if game_message.map.biomass[x][y] == 0:
                    actions.append(AddBiomassAction(position=Position(x, y), amount=1))
                    self.used_tiles.add((x, y))
                    remaining -= 1
                    current_total += 1
                idx += 1
            self.init_progress[i] = idx

        if all(p >= len(path) for p, path in zip(self.init_progress, self.init_paths)):
            self.initialized = True
        return actions

    # ---------------------------------------------------------
    # --- PATH GENERATION -------------------------------------
    # ---------------------------------------------------------
    def line_between(self, start: Position, end: Position):
        """Return Manhattan path (no diagonals)."""
        path = []
        x, y = start.x, start.y
        while x != end.x:
            path.append((x, y))
            x += 1 if end.x > x else -1
        while y != end.y:
            path.append((x, y))
            y += 1 if end.y > y else -1
        path.append((end.x, end.y))
        return path

    def mst_edges(self, colonies):
        """Basic Prim’s MST (connects all colonies minimally)."""
        if len(colonies) < 2:
            return []
        connected = [colonies[0]]
        remaining = colonies[1:]
        edges = []
        while remaining:
            best_pair, best_dist = None, float("inf")
            for c1 in connected:
                for c2 in remaining:
                    d = abs(c1.position.x - c2.position.x) + abs(c1.position.y - c2.position.y)
                    if d < best_dist:
                        best_dist, best_pair = d, (c1, c2)
            edges.append(best_pair)
            connected.append(best_pair[1])
            remaining.remove(best_pair[1])
        return edges

    # ---------------------------------------------------------
    # --- FUTURE VALUE + TREND PREDICTION ---------------------
    # ---------------------------------------------------------
    def colony_future_value(self, colony, horizon=15):
        """Estimate colony value based on its next few ticks."""
        if hasattr(colony, "futureNutrients") and colony.futureNutrients:
            lookahead = colony.futureNutrients[:horizon]
            return sum(lookahead) / len(lookahead)
        return colony.nutrients

    def colony_future_trend(self, colony, horizon=10):
        """Estimate if colony value is rising or falling."""
        if hasattr(colony, "futureNutrients") and len(colony.futureNutrients) > 1:
            now = colony.nutrients
            future = sum(colony.futureNutrients[:horizon]) / min(horizon, len(colony.futureNutrients))
            return future - now
        return 0

    # ---------------------------------------------------------
    # --- DYNAMIC SCORING + REINFORCEMENT ---------------------
    # ---------------------------------------------------------
    def rank_paths(self, game_message: TeamGameState):
        colonies = game_message.map.colonies
        results = []
        for i, c1 in enumerate(colonies):
            for j, c2 in enumerate(colonies):
                if j <= i:
                    continue
                path = self.line_between(c1.position, c2.position)
                path_length = len(path)
                c1_val = self.colony_future_value(c1)
                c2_val = self.colony_future_value(c2)
                trend = self.colony_future_trend(c1) + self.colony_future_trend(c2)
                path_strength = min(game_message.map.biomass[x][y] for (x, y) in path)
                max_strength = min(c1_val, c2_val)
                gain = max_strength - path_strength + 0.5 * trend
                if gain <= 0:
                    continue
                score = gain * (c1_val + c2_val) / (1 + path_length)
                results.append((score, path, (c1, c2)))
        results.sort(key=lambda x: x[0], reverse=True)
        return results

    def reinforce_path(self, game_message: TeamGameState, path):
        """Reinforce weakest tiles first along path, skipping colonies."""
        actions = []
        remaining = game_message.maximumNumberOfBiomassPerTurn
        colony_positions = {(c.position.x, c.position.y) for c in game_message.map.colonies}
        clean = [(x, y) for (x, y) in path if (x, y) not in colony_positions]
        if not clean:
            return actions
        min_strength = min(game_message.map.biomass[x][y] for (x, y) in clean)
        weak_tiles = [(x, y) for (x, y) in clean if game_message.map.biomass[x][y] == min_strength]
        for (x, y) in weak_tiles:
            if remaining <= 0:
                break
            actions.append(AddBiomassAction(position=Position(x, y), amount=1))
            remaining -= 1
        return actions

    # ---------------------------------------------------------
    # --- REALLOCATION (REMOVE/REBUILD PATHS) -----------------
    # ---------------------------------------------------------
    def path_future_value(self, game_message, path, pair, horizon=15):
        """Predict a path’s long-term value using future colony nutrients."""
        c1, c2 = pair
        c1_future = self.colony_future_value(c1, horizon)
        c2_future = self.colony_future_value(c2, horizon)
        path_strength = min(game_message.map.biomass[x][y] for (x, y) in path)
        return min(c1_future, path_strength) * min(c2_future, path_strength)

    def remove_worst_path(self, game_message):
        """Remove the path with the lowest predicted future value safely."""
        if not self.active_paths:
            return [], 0

        worst = min(self.active_paths, key=lambda p: self.path_future_value(game_message, *p))
        path, pair = worst

        actions = []
        freed = 0
        per_turn_limit = game_message.maximumNumberOfBiomassPerTurn
        used_this_turn = 0

        for (x, y) in path:
            if used_this_turn >= per_turn_limit:
                break

            tile_amount = game_message.map.biomass[x][y]
            if tile_amount <= 0:
                continue

            amount = min(tile_amount, per_turn_limit - used_this_turn)
            actions.append(RemoveBiomassAction(position=Position(x, y), amount=amount))
            used_this_turn += amount
            freed += amount

        if freed > 0:
            self.active_paths.remove(worst)
            p1, p2 = pair[0].position, pair[1].position
            print(f"🧹 Removed path ({p1.x},{p1.y}) ↔ ({p2.x},{p2.y}) — freed {freed}")

        return actions, freed

    def build_best_new_path(self, game_message, freed):
        """Rebuild a new high-value path after freeing biomass."""
        ranked = self.rank_paths(game_message)
        if not ranked:
            return []

        for score, path, pair in ranked:
            expected_value = self.path_future_value(game_message, path, pair)
            if expected_value > 0:
                self.active_paths.append((path, pair))
                p1, p2 = pair[0].position, pair[1].position
                print(f"🌱 Built new path ({p1.x},{p1.y}) ↔ ({p2.x},{p2.y}) (future value {expected_value:.2f})")

                max_tiles = min(freed, game_message.maximumNumberOfBiomassPerTurn)
                return [AddBiomassAction(position=Position(x, y), amount=1)
                        for (x, y) in path[:max_tiles]]

        return []
