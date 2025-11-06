import math
from game_message import *


class Bot:
    def __init__(self):
        self.mode = "add"
        self.connecting_phase = True
        self.init_edges = None
        self.init_paths = None
        self.init_progress = None
        self.initialized = False
        self.used_tiles = set()
        self.active_paths = []      # [(path, (c1, c2))]
        self.current_path = None    # (path, pair)
        self.path_queue = []        # [(x, y)] tiles left to process
        print("Initializing your super mega duper bot")

    # ---------------------------------------------------------
    # MAIN LOOP
    # ---------------------------------------------------------
    def get_next_move(self, game_message: TeamGameState):
        actions = []
        total_biomass = sum(sum(row) for row in game_message.map.biomass)
        cap = game_message.maximumNumberOfBiomassOnMap

        # --- 1. Connecting phase ---
        if self.connecting_phase:
            actions = self.connect_all_colonies(game_message)
            if self.initialized:
                self.connecting_phase = False
                self.active_paths = list(zip(self.init_paths, self.init_edges))
                print("✅ All colonies connected — entering strategic mode!")
            return actions

        # --- 2. Handle full biomass cap (force removal) ---
        if total_biomass >= cap:
            if self.mode != "remove" or not self.path_queue:
                self.mode = "remove"
                self.select_worst_path_for_remove(game_message)
                # If no valid path, remove the weakest non-colony tile
                if not self.path_queue:
                    weakest_tile = self.find_weakest_tile(game_message)
                    if weakest_tile:
                        x, y = weakest_tile
                        print(f"⚠️ Forced single-tile removal at ({x},{y}) due to biomass cap.")
                        return [RemoveBiomassAction(position=Position(x, y), amount=1)]
            return self.process_remove_phase(game_message)

        # --- 3. Choose new path only when previous done ---
        if not self.path_queue and not self.current_path:
            if self.mode == "add":
                self.select_best_path_for_add(game_message)
            else:
                self.select_worst_path_for_remove(game_message)

        # --- 4. Perform current mode work ---
        if self.mode == "add":
            actions = self.process_add_phase(game_message)
        else:
            actions = self.process_remove_phase(game_message)

        # --- 5. Fallback only if under capacity ---
        if not actions and total_biomass < cap - 2:
            actions = self.reinforce_random_tile(game_message)

        return actions[:game_message.maximumNumberOfBiomassPerTurn]

    # ---------------------------------------------------------
    # INITIAL CONNECTION PHASE
    # ---------------------------------------------------------
    def connect_all_colonies(self, game_message: TeamGameState):
        actions = []
        remaining = game_message.maximumNumberOfBiomassPerTurn
        total = sum(sum(row) for row in game_message.map.biomass)
        cap = game_message.maximumNumberOfBiomassOnMap
        colonies = {(c.position.x, c.position.y) for c in game_message.map.colonies}

        if not self.init_edges:
            self.init_edges = self.mst_edges(game_message.map.colonies)
            self.init_paths = [
                self.line_between(c1.position, c2.position)
                for c1, c2 in self.init_edges
            ]
            self.init_progress = [0] * len(self.init_paths)

        for i, path in enumerate(self.init_paths):
            if remaining <= 0 or total >= cap:
                break
            idx = self.init_progress[i]
            while idx < len(path) and remaining > 0 and total < cap:
                x, y = path[idx]
                if (x, y) in colonies or (x, y) in self.used_tiles:
                    idx += 1
                    continue
                if game_message.map.biomass[x][y] == 0:
                    actions.append(AddBiomassAction(position=Position(x, y), amount=1))
                    self.used_tiles.add((x, y))
                    remaining -= 1
                    total += 1
                idx += 1
            self.init_progress[i] = idx

        if all(p >= len(path) for p, path in zip(self.init_progress, self.init_paths)):
            self.initialized = True
        return actions

    # ---------------------------------------------------------
    # PATH SELECTION
    # ---------------------------------------------------------
    def select_best_path_for_add(self, game_message):
        ranked = self.rank_paths(game_message)
        if not ranked:
            print("⚠️ No valid ADD path found.")
            return
        best_score, best_path, best_pair = ranked[0]
        colonies = {(c.position.x, c.position.y) for c in game_message.map.colonies}
        self.path_queue = [(x, y) for (x, y) in best_path if (x, y) not in colonies]
        self.current_path = (best_path, best_pair)
        print(f"🌿 Selected ADD path {best_pair[0].position} ↔ {best_pair[1].position}")

    def select_worst_path_for_remove(self, game_message):
        if not self.active_paths:
            self.path_queue = []
            return
        worst = min(self.active_paths, key=lambda p: self.path_future_value(game_message, *p))
        path, pair = worst
        self.path_queue = path.copy()
        self.current_path = (path, pair)
        self.active_paths.remove(worst)
        print(f"🧹 Selected REMOVE path {pair[0].position} ↔ {pair[1].position}")

    # ---------------------------------------------------------
    # ADD / REMOVE EXECUTION
    # ---------------------------------------------------------
    def process_add_phase(self, game_message):
        actions = []
        remaining = game_message.maximumNumberOfBiomassPerTurn
        total = sum(sum(row) for row in game_message.map.biomass)
        cap = game_message.maximumNumberOfBiomassOnMap

        if not self.path_queue:
            return []

        while self.path_queue and remaining > 0 and total < cap:
            x, y = self.path_queue.pop(0)
            if game_message.map.biomass[x][y] < 10:
                actions.append(AddBiomassAction(position=Position(x, y), amount=1))
                remaining -= 1
                total += 1

        # Path finished → store it and prepare to switch mode
        if not self.path_queue:
            if self.current_path:
                self.active_paths.append(self.current_path)
                print(f"✅ Finished reinforcing path {self.current_path[1][0].position} ↔ {self.current_path[1][1].position}")
            self.current_path = None
            self.mode = "remove"
        return actions

    def process_remove_phase(self, game_message):
        actions = []
        remaining = game_message.maximumNumberOfBiomassPerTurn

        if not self.path_queue:
            return []

        while self.path_queue and remaining > 0:
            x, y = self.path_queue.pop(0)
            biomass_here = game_message.map.biomass[x][y]
            if biomass_here > 0:
                actions.append(RemoveBiomassAction(position=Position(x, y), amount=1))
                remaining -= 1

        # Path finished → prepare to add next time
        if not self.path_queue:
            print(f"🗑️ Finished removing path {self.current_path[1][0].position} ↔ {self.current_path[1][1].position}")
            self.current_path = None
            self.mode = "add"
        return actions

    # ---------------------------------------------------------
    # BACKUP ACTION
    # ---------------------------------------------------------
    def reinforce_random_tile(self, game_message):
        """Fallback only if biomass not full."""
        total = sum(sum(row) for row in game_message.map.biomass)
        cap = game_message.maximumNumberOfBiomassOnMap
        if total >= cap - 1:
            return []
        for x in range(game_message.map.width):
            for y in range(game_message.map.height):
                if game_message.map.biomass[x][y] == 0:
                    print(f"🧩 Reinforcing backup tile ({x},{y})")
                    return [AddBiomassAction(position=Position(x, y), amount=1)]
        return []

    # ---------------------------------------------------------
    # UTILITIES
    # ---------------------------------------------------------
    def line_between(self, start: Position, end: Position):
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
    # SCORING
    # ---------------------------------------------------------
    def colony_future_value(self, colony, horizon=15):
        if hasattr(colony, "futureNutrients") and colony.futureNutrients:
            lookahead = colony.futureNutrients[:horizon]
            return sum(lookahead) / len(lookahead)
        return colony.nutrients

    def colony_future_trend(self, colony, horizon=10):
        if hasattr(colony, "futureNutrients") and len(colony.futureNutrients) > 1:
            now = colony.nutrients
            future = sum(colony.futureNutrients[:horizon]) / min(horizon, len(colony.futureNutrients))
            return future - now
        return 0

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
                gain = max(0.1, max_strength - path_strength + 0.5 * trend)
                score = gain * (c1_val + c2_val) / (1 + path_length)
                results.append((score, path, (c1, c2)))
        results.sort(key=lambda x: x[0], reverse=True)
        return results

    def path_future_value(self, game_message, path, pair, horizon=15):
        c1, c2 = pair
        c1_future = self.colony_future_value(c1, horizon)
        c2_future = self.colony_future_value(c2, horizon)
        path_strength = min(game_message.map.biomass[x][y] for (x, y) in path)
        return min(c1_future, path_strength) * min(c2_future, path_strength)

    def find_weakest_tile(self, game_message):
        """Find a non-colony tile with some biomass to remove when stuck."""
        colonies = {(c.position.x, c.position.y) for c in game_message.map.colonies}
        weakest = None
        weakest_val = float("inf")
        for x in range(game_message.map.width):
            for y in range(game_message.map.height):
                if (x, y) not in colonies:
                    val = game_message.map.biomass[x][y]
                    if 0 < val < weakest_val:
                        weakest_val = val
                        weakest = (x, y)
        return weakest
