import random
from game_message import *


class Bot:
    def __init__(self):
        print("Initializing your super mega duper bot")

    def get_next_move(self, game_message: TeamGameState) -> list[Action]:
        """
        Here is where the magic happens, for now the moves are not very good. I bet you can do better ;)
        """
        actions = []

        my_team: TeamInfo = game_message.world.teamInfos[game_message.yourTeamId]
        if len(my_team.spawners) == 0:
            actions.append(SporeCreateSpawnerAction(sporeId=my_team.spores[0].id))

        elif len(my_team.spores) == 0:
            actions.append(
                SpawnerProduceSporeAction(spawnerId=my_team.spawners[0].id, biomass=20)
            )

        else:
            actions.append(
                SporeMoveToAction(
                    sporeId=my_team.spores[0].id,
                    position=Position(
                        x=random.randint(0, game_message.world.map.width - 1),
                        y=random.randint(0, game_message.world.map.height - 1),
                    ),
                )
            )

        # You can clearly do better than the random actions above. Have fun!!
        return actions
