import random
from game_message import *


class Bot:
    def __init__(self):
        print("Initializing your super mega duper bot")

    def get_next_move(self, game_message: TeamGameState):
        actions = []

        # Pick a number of biomass to move this turn.
        remaining_biomass_to_move_this_turn = random.randint(
            1, game_message.maximumNumberOfBiomassPerTurn
        )

        while remaining_biomass_to_move_this_turn > 0:
            random_position = Position(
                x=random.randint(0, game_message.map.width - 1),
                y=random.randint(0, game_message.map.height - 1),
            )

            # Randomly decide whether to add or remove biomass
            should_add_biomass = random.choice([True, False])

            if should_add_biomass:
                biomass_to_move_in_this_action = random.randint(
                    1, remaining_biomass_to_move_this_turn
                )
                remaining_biomass_to_move_this_turn -= biomass_to_move_in_this_action

                actions.append(
                    AddBiomassAction(
                        position=random_position, amount=biomass_to_move_in_this_action
                    )
                )
            else:
                biomass_to_move_in_this_action = min(
                    remaining_biomass_to_move_this_turn,
                    game_message.map.biomass[random_position.x][random_position.y],
                )
                remaining_biomass_to_move_this_turn -= biomass_to_move_in_this_action

                actions.append(
                    RemoveBiomassAction(
                        position=random_position, amount=biomass_to_move_in_this_action
                    )
                )

        # You can clearly do better than the random actions above. Have fun!!
        return actions
