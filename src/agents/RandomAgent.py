import numpy as np
import random as r


class BasicAgent:
    def __init__(self, id, train):
        if id == 1:     # id must be > 1
            raise ValueError("Agent id need to be > 1")

        self.id = id
        self.train = train
        self.score = 0
        self.resources = {'brick': 0,
                          'lumber': 0,
                          'wool': 0,
                          'grain': 0,
                          'ore': 0}
        self.villages = []
        self.cities = []
        self.roads = []
        self.action_cards = []

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
        for key, item in self.resources.items():
            if self.resources[key] >= 4:
                available_actions.append("trade_" + key)
        # Trade bank/player
        # Cards use/buy

        return available_actions

    def step(self, obs):     # Obs -> observer
        buildable_road_locations, buildable_village_locations = obs.get_buildable_locations(self)

        for village in self.villages:   # TODO temporary solution sometimes village gets built twice
            if village in buildable_village_locations:
                buildable_village_locations.remove(village)
            if village in self.cities:
                self.villages.remove(village)

        for road in self.roads:
            if road in buildable_road_locations:
                buildable_road_locations.remove(road)

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
    def trade(self, trade_in):  # Trade with bank
        new_resources = list(self.resources.keys())
        new_resources.remove(trade_in)
        self.resources[r.choice(new_resources)] += 1
        self.resources[trade_in] -= 4

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
        print(f'building city {location}, current {self.villages}')
        self.villages.remove(location)
        print(f'removed {location}, from {self.villages}')
        self.cities.append(location)
        self.resources['grain'] -= 2
        self.resources['ore'] -= 3

    def reset_agent(self):
        pass

    def save_model(self, path):
        pass
