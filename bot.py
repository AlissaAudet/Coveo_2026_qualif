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

        # After network is complete → reinforcement phase
        ranked = self.rank_paths(game_message)
        if not ranked:
            return []

        best_score, best_path, _ = ranked[0]
        return self.reinforce_path(game_message, best_path)

    def continue_connect_all_paths(self, game_message: TeamGameState):
        """Continue connecting all MST paths across multiple ticks until done."""
        actions = []
        remaining = game_message.maximumNumberOfBiomassPerTurn
        current_total = sum(sum(row) for row in game_message.map.biomass)
        max_total = game_message.maximumNumberOfBiomassOnMap

        colony_positions = {(c.position.x, c.position.y) for c in game_message.map.colonies}

        # Persistent global memory for used tiles
        if not hasattr(self, "used_tiles"):
            self.used_tiles = set()

        for i, path in enumerate(self.init_paths):
            if remaining <= 0 or current_total >= max_total:
                break

            idx = self.init_progress[i]
            while idx < len(path) and remaining > 0 and current_total < max_total:
                x, y = path[idx]

                # skip colony or previously used tiles
                if (x, y) in colony_positions or (x, y) in self.used_tiles:
                    idx += 1
                    continue

                if game_message.map.biomass[x][y] == 0:
                    actions.append(AddBiomassAction(position=Position(x, y), amount=1))
                    remaining -= 1
                    current_total += 1
                    self.used_tiles.add((x, y))  # track globally

                idx += 1

            # remember how far we got for this path
            self.init_progress[i] = idx

        # check if all paths are done
        if all(p >= len(path) for p, path in zip(self.init_progress, self.init_paths)):
            self.initialized = True

        return actions

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

        # --- Prim’s algorithm to connect all colonies ---
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

        # --- Try to close the loop only if it's new (not overlapping existing paths) ---
        first_colony = colonies[0]
        last_colony = colonies[-1]

        # Generate the potential closing path
        close_path = self.line_between(first_colony.position, last_colony.position)

        # Flatten existing MST paths into a set of coordinates
        existing_tiles = set()
        for c1, c2 in edges:
            path = self.line_between(c1.position, c2.position)
            existing_tiles.update(path)

        # Compute overlap ratio
        overlap = sum(1 for tile in close_path if tile in existing_tiles)
        overlap_ratio = overlap / len(close_path)

        # Add closing edge only if it's mostly unique (less than 40% overlap)
        if overlap_ratio < 0.4:
            edges.append((first_colony, last_colony))

        return edges

    def rank_paths(self, game_message: TeamGameState):
        """Return a sorted list of (score, path, (colony1, colony2)) from best to worst."""
        paths_with_scores = []

        for c1, c2 in self.mst_edges(game_message.map.colonies):
            path = self.line_between(c1.position, c2.position)
            path_length = len(path)
            c1_val = c1.nutrients
            c2_val = c2.nutrients

            # basic efficiency score
            score = (c1_val + c2_val) / (1 + path_length)

            # only add if we have enough biomass to build or reinforce
            needed_biomass = sum(1 for (x, y) in path if game_message.map.biomass[x][y] < 5)
            if needed_biomass <= game_message.availableBiomass:
                paths_with_scores.append((score, path, (c1, c2)))

        # sort by score descending
        paths_with_scores.sort(key=lambda x: x[0], reverse=True)
        return paths_with_scores

    def reinforce_path(self, game_message: TeamGameState, path):
        """Add +1 biomass everywhere along a path (excluding colony tiles)."""
        print(f'remaining biomass to place: {game_message.availableBiomass}')
        actions = []
        remaining = game_message.maximumNumberOfBiomassPerTurn
        current_total = sum(sum(row) for row in game_message.map.biomass)
        max_total = game_message.maximumNumberOfBiomassOnMap

        colony_positions = {(c.position.x, c.position.y) for c in game_message.map.colonies}

        # exclude colonies from reinforcement path
        clean_path = [(x, y) for (x, y) in path if (x, y) not in colony_positions]

        for (x, y) in clean_path:
            if remaining <= 0 or current_total >= max_total:
                break
            actions.append(AddBiomassAction(position=Position(x, y), amount=1))
            remaining -= 1
            current_total += 1

        return actions

