import numpy as np
import random as r


class AbstractAgent:

    if id == 1:     # id must be > 1
        raise ValueError("Agent id need to be > 1")

    def __init__(self, id, train=False):
        self.id = id
        self.score = 0
        self.train = train
        self.resources = {
            'brick': 0,
            'lumber': 0,
            'wool': 0,
            'grain': 0,
            'ore': 0
        }
        self.villages = []      # TODO add constraints max 5 villages, 15 roads, 4 cities
        self.cities = []
        self.roads = []

    def get_available_actions(self, buildable_road_locations, buildable_village_locations):
        available_actions = ["pass"]

        # TODO Change available actions into dict where action is key and value is positions
        if buildable_road_locations:
            if self.resources['brick'] >= 1 and self.resources['lumber'] >= 1:
                available_actions.append("build_road")
        if buildable_village_locations:
            if self.resources['brick'] >= 1 and self.resources['lumber'] >= 1 \
                    and self.resources['wool'] >= 1 and self.resources['grain'] >= 1:
                available_actions.append("build_village")
        if self.resources['grain'] >= 2 and self.resources['ore'] >= 3 and self.villages:
            available_actions.append("build_city")
        for key, item in self.resources.items():
            if self.resources[key] >= 4:
                available_actions.append("trade")
        # Trade bank/player
        # Cards use/buy

        return available_actions

    def step(self, obs):     # step is one action for a actor, Obs -> observer
        ...

    def take_action(self, action, buildable_road_locations, buildable_village_locations):
        if action == "build_village":
            location = r.choice(buildable_village_locations)
            getattr(self, action)(location)
        elif action == "pass":
            location = (-1, -1)
        elif action == "build_road":
            location = r.choice(buildable_road_locations)
            getattr(self, action)(location)
        elif action == "build_city":
            location = r.choice(self.villages)
            getattr(self, action)(location)
        elif "trade" in action:
            tradeable_resources = [x for x, resource in self.resources.items() if resource >= 4]
            trade_in = r.choice(tradeable_resources)
            new_resources = list(self.resources.keys())
            new_resources.remove(trade_in)
            trade_out = r.choice(new_resources)
            location = (trade_in, trade_out)
            getattr(self, action)(trade_in, trade_out)

        return location

    def trade(self, trade_in, trade_out):  # Trade with bank
        self.resources[trade_out] += 1
        self.resources[trade_in] -= 4

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

    def build_village(self, location):
        self.score += 1
        self.villages.append(location)
        self.resources['brick'] -= 1
        self.resources['lumber'] -= 1
        self.resources['wool'] -= 1
        self.resources['grain'] -= 1

    def build_road(self, location):
        self.roads.append(location)
        self.resources['brick'] -= 1
        self.resources['lumber'] -= 1

    def build_city(self, location):
        self.score += 1
        self.villages.remove(location)
        self.cities.append(location)
        self.resources['grain'] -= 2
        self.resources['ore'] -= 3

    def reset_agent(self):
        self.score = 0
        for resource in self.resources.keys():
            self.resources[resource] = 0
        self.villages = []
        self.cities = []
        self.roads = []

    def save_model(self, path):
        pass
