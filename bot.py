import math
from game_message import *

class Bot:
    def __init__(self):
        self.init_progress = None
        self.init_paths = None
        self.init_edges = None
        self.initialized = False
        self.used_tiles = set()
        print("Initializing your super mega duper bot")

    def get_next_move(self, game_message: TeamGameState):
        # Initialize MST and paths if not done yet
        if not self.init_edges or not self.init_paths or not self.init_progress:
            self.init_edges = self.mst_edges(game_message.map.colonies)
            self.init_paths = [self.line_between(c1.position, c2.position) for c1, c2 in self.init_edges]
            self.init_progress = [0] * len(self.init_paths)
            self.initialized = False

        # Continue initial connection phase until all paths complete
        if not self.initialized:
            actions = self.continue_connect_all_paths(game_message)
            if self.initialized:
                print("✅ All colonies connected!")
            return actions

        # --- Reinforcement phase ---
        ranked = self.rank_paths(game_message)
        if not ranked:
            return []

        best_score, best_path, _ = ranked[0]
        return self.reinforce_path(game_message, best_path)

    # -------------------------------------------------------
    # --- INITIAL CONNECTION PHASE ---------------------------
    # -------------------------------------------------------

    def continue_connect_all_paths(self, game_message: TeamGameState):
        """Continue connecting all MST paths across multiple ticks until done."""
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

                # skip colonies or already-used tiles
                if (x, y) in colony_positions or (x, y) in self.used_tiles:
                    idx += 1
                    continue

                if game_message.map.biomass[x][y] == 0:
                    actions.append(AddBiomassAction(position=Position(x, y), amount=1))
                    remaining -= 1
                    current_total += 1
                    self.used_tiles.add((x, y))

                idx += 1

            # remember how far we got for this path
            self.init_progress[i] = idx

        # mark as done when all paths complete
        if all(p >= len(path) for p, path in zip(self.init_progress, self.init_paths)):
            self.initialized = True

        return actions

    # -------------------------------------------------------
    # --- PATH GENERATION -----------------------------------
    # -------------------------------------------------------

    def line_between(self, start: Position, end: Position):
        """Return a Manhattan path (no diagonals) between two points."""
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
        """Compute MST edges and close the loop only if it adds a new unique connection."""
        if len(colonies) < 2:
            return []

        connected = [colonies[0]]
        remaining = colonies[1:]
        edges = []

        # --- Prim’s algorithm ---
        while remaining:
            best_pair = None
            best_dist = float("inf")
            for c1 in connected:
                for c2 in remaining:
                    d = abs(c1.position.x - c2.position.x) + abs(c1.position.y - c2.position.y)
                    if d < best_dist:
                        best_dist = d
                        best_pair = (c1, c2)
            edges.append(best_pair)
            connected.append(best_pair[1])
            remaining.remove(best_pair[1])

        # --- Try to close the loop only if it's unique ---
        first_colony = colonies[0]
        last_colony = colonies[-1]
        close_path = self.line_between(first_colony.position, last_colony.position)

        # Get all existing path tiles
        existing_tiles = set()
        for c1, c2 in edges:
            path = self.line_between(c1.position, c2.position)
            existing_tiles.update(path)

        # Compute overlap ratio
        overlap = sum(1 for tile in close_path if tile in existing_tiles)
        overlap_ratio = overlap / len(close_path)

        # Add closing edge only if it creates a new distinct route
        if overlap_ratio < 0.4:
            edges.append((first_colony, last_colony))

        return edges

    # -------------------------------------------------------
    # --- REINFORCEMENT PHASE -------------------------------
    # -------------------------------------------------------

    def rank_paths(self, game_message: TeamGameState):
        """Rank all paths based on potential gain (strong colonies, short paths, weak links)."""
        paths_with_scores = []

        for c1, c2 in self.mst_edges(game_message.map.colonies):
            path = self.line_between(c1.position, c2.position)
            path_length = len(path)

            c1_val = c1.nutrients
            c2_val = c2.nutrients
            path_tiles = [(x, y) for (x, y) in path
                          if 0 <= x < game_message.map.width and 0 <= y < game_message.map.height]

            # Compute current strength (min biomass along path)
            path_strength = min(game_message.map.biomass[x][y] for (x, y) in path_tiles)
            max_strength = min(c1_val, c2_val)
            potential_gain = max_strength - path_strength

            # Skip paths already maxed out
            if potential_gain <= 0:
                continue

            # Weighted score (higher for high-value, short, weak paths)
            score = potential_gain * (c1_val + c2_val) / (1 + path_length)

            paths_with_scores.append((score, path, (c1, c2)))

        paths_with_scores.sort(key=lambda x: x[0], reverse=True)
        return paths_with_scores

    def reinforce_path(self, game_message: TeamGameState, path):
        """Reinforce weakest tiles first along a path, skipping colonies."""
        print(f'number of resources left: {game_message.availableBiomass}')
        actions = []
        remaining = game_message.maximumNumberOfBiomassPerTurn
        current_total = sum(sum(row) for row in game_message.map.biomass)
        max_total = game_message.maximumNumberOfBiomassOnMap

        colony_positions = {(c.position.x, c.position.y) for c in game_message.map.colonies}
        clean_path = [(x, y) for (x, y) in path if (x, y) not in colony_positions]

        # Find weakest biomass value in this path
        if not clean_path:
            return actions

        min_strength = min(game_message.map.biomass[x][y] for (x, y) in clean_path)
        weakest_tiles = [(x, y) for (x, y) in clean_path if game_message.map.biomass[x][y] == min_strength]

        # Reinforce the weakest tiles first
        for (x, y) in weakest_tiles:
            if remaining <= 0 or current_total >= max_total:
                break
            actions.append(AddBiomassAction(position=Position(x, y), amount=1))
            remaining -= 1
            current_total += 1

        return actions
