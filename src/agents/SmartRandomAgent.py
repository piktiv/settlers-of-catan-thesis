import numpy as np
import random as r
from src.agents import AbstractAgent


class SmartRandomAgent(AbstractAgent):
    def __init__(self, id, train=False):
        super(SmartRandomAgent, self).__init__(id)

        if id == 1:  # id must be > 1
            raise ValueError("Agent id need to be > 1")

    def key_with_max_resource(self):
        v = list(self.resources.values())
        k = list(self.resources.keys())
        return k[v.index(max(v))]

    def key_with_min_resource(self):
        v = list(self.resources.values())
        k = list(self.resources.keys())
        return k[v.index(min(v))]

    def get_available_actions(self, buildable_road_locations, buildable_village_locations):
        available_actions = ["pass"]

        if buildable_road_locations:
            if self.resources['brick'] >= 1 and self.resources['lumber'] >= 1:
                available_actions.append("build_road")
        if buildable_village_locations:
            if self.resources['brick'] >= 1 and self.resources['lumber'] >= 1 \
                    and self.resources['wool'] >= 1 and self.resources['grain'] >= 1:
                available_actions.append("build_village")
        if self.resources['grain'] >= 2 and self.resources['ore'] >= 3 and self.villages:
            available_actions.append("build_city")

        key = self.key_with_max_resource()  # Always trades resource that is most abundant
        if self.resources[key] >= 4:
            available_actions.append("trade_" + key)

        # Drops win rate to 60% from 70%
        '''if not available_actions:
            available_actions.append("pass")'''
        # Trade bank/player
        # Cards use/buy

        return available_actions

    def step(self, obs):  # Obs -> observer

        buildable_road_locations, buildable_village_locations = obs.get_buildable_locations(self)

        if buildable_village_locations or len(self.villages) > 2:  # Prioritize building villages
            buildable_road_locations.clear()
        # TODO For smarter agent only villages and roads that are not on the outskirts of map
        available_actions = self.get_available_actions(buildable_road_locations, buildable_village_locations)

        action = r.choice(available_actions)
        location = self.take_action(action, buildable_road_locations, buildable_village_locations)

        return action, location

    def take_action(self, action, buildable_road_locations, buildable_village_locations):
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
        elif "trade_" in action:
            getattr(self, action[:5])(action[6:])
        return location

    # TODO add ports and in env
    def trade(self, trade_in):  # Trade with bank max resource for min resource
        traded_out = self.key_with_min_resource()
        self.resources[traded_out] += 1
        self.resources[trade_in] -= 4

    def free_village_build(self, obs):  # Round 1   # TODO make agent choose village with better odds
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
