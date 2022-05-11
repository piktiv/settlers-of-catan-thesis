import numpy as np
import random as r
from src.agents import AbstractAgent


class RandomAgent(AbstractAgent):
    def __init__(self, id, train=False):
        super(RandomAgent, self).__init__(id)

        if id == 1:     # id must be > 1
            raise ValueError("Agent id need to be > 1")

    def step(self, obs):     # Obs -> observer
        buildable_road_locations, buildable_village_locations = obs.get_buildable_locations(self)
        available_actions = self.get_available_actions(buildable_road_locations, buildable_village_locations)

        action = r.choice(available_actions)
        location = self.take_action(action, buildable_road_locations, buildable_village_locations)
        print("RA", action, location)

        return action, location

    '''def take_action(self, action, buildable_road_locations, buildable_village_locations):
        location = (-1, -1)  # Undefined

        if action == "build_village":
            location = r.choice(buildable_village_locations)
            getattr(self, action)(location)
        elif action == "build_road":
            location = r.choice(buildable_road_locations)
            getattr(self, action)(location)
        elif action == "build_city":
            location = r.choice(self.villages)
            getattr(self, action)(location)
        elif "trade" in action:
            getattr(self, action[:5])(action[6:])
        return location'''

    # TODO add ports and in env
    '''def trade(self, trade_in):  # Trade with bank
        new_resources = list(self.resources.keys())
        new_resources.remove(trade_in)
        traded_out = r.choice(new_resources)
        self.resources[traded_out] += 1
        self.resources[trade_in] -= 4'''

    def free_village_build(self, obs):   # Round 1
        buildable_locations = obs.free_build_village()
        village_location = r.choice(buildable_locations)
        self.villages.append(village_location)
        self.score += 1
        return "build_village", village_location

    def free_road_build(self, obs, village_location):
        buildable_locations = obs.get_adjacent_roads(village_location)
        road_location = r.choice(buildable_locations)
        self.roads.append(road_location)
        return "build_road", road_location

    def reset_agent(self):
        self.__init__(self.id)

    def save_model(self, path):
        pass
