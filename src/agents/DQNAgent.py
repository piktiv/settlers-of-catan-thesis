import numpy as np
import random as r
from src.agents import AbstractAgent
from src.expreplay import ExperienceReplay, Experience
from src.qmodel import q_model
from src.environment import Environment

ENV = Environment([AbstractAgent(0)])


class DQNAgent(AbstractAgent):
    def __init__(self, id, train):
        super(DQNAgent, self).__init__(id, train)
        if id == 1:  # id must be > 1
            raise ValueError("Agent id need to be > 1")

        self.network = q_model((ENV.board.shape[0], ENV.board.shape[1], 1), action_space=ENV.action_space)
        self._EPSILON = 0.99
        self._EPSILON_DECAY = 0.0001
        self.experience_replay = ExperienceReplay()
        self.min_exp_len = 6000
        self.batch_size = 32
        self.state = None
        self.last_action = None

    def key_with_max_resource(self):
        v = list(self.resources.values())
        k = list(self.resources.keys())
        return k[v.index(max(v))]

    def key_with_min_resource(self):
        v = list(self.resources.values())
        k = list(self.resources.keys())
        return k[v.index(min(v))]

    def action_masking(self, q_values, obs, buildable_village_locations, buildable_road_locations):
        for i, (action, location) in enumerate(obs.action_space):
            if action == "build_village":
                if location not in buildable_village_locations:
                    q_values[i] = -999
            if action == "build_city":
                if location not in self.villages:
                    q_values[i] = -999
            if action == "build_road":
                if location not in buildable_road_locations:
                    q_values[i] = -999
            if action == "trade":
                trade_in, trade_out = location
                if self.resources[trade_in] < 4:
                    q_values[i] = -999

        return q_values

    def step(self, obs):  # Obs -> observer
        buildable_road_locations, buildable_village_locations = obs.get_buildable_locations(self)

        available_actions = self.get_available_actions(buildable_road_locations, buildable_village_locations)
        print(available_actions)

        # Exploration
        num = r.random()


        if num > self._EPSILON and self.train:  # Choose random direction
            action = np.random.choice(available_actions)    # Action is chosen uniformly
            location = self.take_action(action, buildable_road_locations, buildable_village_locations)
            self._EPSILON *= 1 - self._EPSILON_DECAY
        else:
            q_values = self.network.predict(np.array(obs.board.reshape([1, obs.board.shape[0], obs.board.shape[1], 1])))
            q_values = np.squeeze(q_values)

            masked_q_values = self.action_masking(q_values, obs, buildable_village_locations, buildable_road_locations)
            action, location = obs.action_space[np.argmax(masked_q_values)]

            if "build" in action:
                getattr(self, action)(location)
            elif "trade" in action:
                trade_in, trade_out = location
                self.trade(trade_in, trade_out)

            print(action, location)
            print(f'villages {self.villages} cities {self.cities} roads {self.roads}')

        if self.experience_replay.__len__() > self.min_exp_len and self.train:
            self.train_agent()

        # Add experience and check if train
        self.state = obs.board
        self.last_action = action
        print("Epsilon", self._EPSILON)
        return action, location

    def train_agent(self):
        experiences = self.experience_replay.sample(self.batch_size)

    def save_experience(self, state, action, reward, done, next_state):
        self.experience_replay.append(
            Experience(state, action, reward, done, next_state)
            )

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
            self.random_trade(action[6:])

        return location

    def random_trade(self, trade_in):  # Trade with bank
        new_resources = list(self.resources.keys())
        new_resources.remove(trade_in)
        traded_out = r.choice(new_resources)
        self.resources[traded_out] += 1
        self.resources[trade_in] -= 4

    def trade(self, trade_in, traded_out):  # Trade with bank max resource for min resource
        self.resources[traded_out] += 1
        self.resources[trade_in] -= 4

    def free_village_build(self, obs):  # Round 1
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
