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
        self.target_network = q_model((ENV.board.shape[0], ENV.board.shape[1], 1), action_space=ENV.action_space)

        self.save = './models/DQNAgent_weights.h5'
        self.actions = ENV.action_space
        self.update_interval = 300
        self.learning_rate = 0.95
        self.steps = 0
        self._EPSILON = 0.99
        self._EPSILON_DECAY = 0.000_1
        self._MIN_EPSILON = 0.1
        self.experience_replay = ExperienceReplay()
        self.min_exp_len = 100
        self.batch_size = 32
        self.state = None
        self.last_action = None

    # TODO Agent takes illegal actions think its because output exceeds -999 (-inf)
    # TODO Create test set illegal action to None see if np.argmax still works
    def action_masking(self, q_values, obs, buildable_village_locations, buildable_road_locations, available_actions):
        for i, (action, location) in enumerate(obs.action_space):
            if action == "build_village":
                if action not in available_actions or location not in buildable_village_locations:
                    q_values[i] = -np.Inf
            if action == "build_city":
                if action not in available_actions or location not in self.villages:
                    q_values[i] = -np.Inf
            if action == "build_road":
                if action not in available_actions or location not in buildable_road_locations:
                    q_values[i] = -np.Inf
            if action == "trade":
                trade_in, trade_out = location
                if self.resources[trade_in] < 4:
                    q_values[i] = -np.Inf

        return q_values

    def step(self, obs):  # Obs -> observer
        buildable_road_locations, buildable_village_locations = obs.get_buildable_locations(self)

        available_actions = self.get_available_actions(buildable_road_locations, buildable_village_locations)

        if r.random() < self._EPSILON and self.train:   # Exploration
            action = np.random.choice(available_actions)    # Action is chosen uniformly
            location = self.take_action(action, buildable_road_locations, buildable_village_locations)
            if self._EPSILON > self._MIN_EPSILON:
                self._EPSILON -= self._EPSILON_DECAY
        else:   # Exploitation
            q_values = self.network.predict(np.array(obs.board.reshape([1, obs.board.shape[0], obs.board.shape[1], 1])))
            q_values = np.squeeze(q_values)

            masked_q_values = self.action_masking(
                q_values, obs, buildable_village_locations, buildable_road_locations, available_actions)
            print(np.argmax(masked_q_values))
            print(f'{obs.action_space[np.argmax(masked_q_values)]} value {np.max(masked_q_values)}')
            action, location = obs.action_space[np.argmax(masked_q_values)]
            print(f'informed {action} here {location}')
            if "build" in action:
                getattr(self, action)(location)
            elif "trade" in action:
                trade_in, trade_out = location
                self.trade(trade_in, trade_out)

        if self.experience_replay.__len__() > self.min_exp_len and self.train:
            self.train_agent()

        self.state = obs.board
        self.last_action = (action, location)
        self.steps += 1

        if self.steps % self.update_interval == 0 and self.steps > 0:
            self.update_target()
        return action, location

    def train_agent(self):
        experiences = self.experience_replay.sample(self.batch_size)
        states = np.array([exp.state for exp in experiences], dtype=np.float64)
        new_states = np.array([exp.new_state for exp in experiences])
        y = self.network.predict(states)
        y_next = self.network.predict(new_states)
        y_target = self.target_network.predict(new_states)
        for i, exp in enumerate(experiences):
            if exp.done:
                y[i][self.actions.index(exp.action)] = exp.reward
            else:
                y[i][self.actions.index(exp.action)] = (
                        exp.reward + self.learning_rate * y_target[i][np.argmax(y_next[i])]
                )

        self.network.fit(states, y, batch_size=self.batch_size)

    def update_target(self):
        self.target_network.set_weights(self.network.get_weights())

    def save_experience(self, state, action, reward, done, next_state):
        self.experience_replay.append(
            Experience(state, action, reward, done, next_state)
            )

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

        if r.random() < self._EPSILON and self.train:   # Exploration
            village_location = r.choice(buildable_locations)
            self._EPSILON -= self._EPSILON_DECAY

        else:   # Exploitation
            q_values = self.network.predict(np.array(obs.board.reshape([1, obs.board.shape[0], obs.board.shape[1], 1])))
            q_values = np.squeeze(q_values)

            masked_q_values = self.action_masking(q_values, obs, buildable_locations, [], "build_village")
            masked_q_values[-1] = -999  # Pass unavailable

            action, village_location = obs.action_space[np.argmax(masked_q_values)]

        self.state = obs.board
        self.last_action = ("build_village", village_location)

        self.villages.append(village_location)
        self.score += 1
        return "build_village", village_location

    def free_road_build(self, obs, village_location):
        buildable_locations = obs.get_adjacent_roads(village_location)

        if r.random() < self._EPSILON and self.train:   # Exploration
            road_location = r.choice(buildable_locations)
            self._EPSILON -= self._EPSILON_DECAY

        else:   # Exploitation
            q_values = self.network.predict(np.array(obs.board.reshape([1, obs.board.shape[0], obs.board.shape[1], 1])))
            q_values = np.squeeze(q_values)

            masked_q_values = self.action_masking(q_values, obs, [], buildable_locations, "build_road")
            masked_q_values[-1] = -999  # Pass unavailable

            action, road_location = obs.action_space[np.argmax(masked_q_values)]

        self.state = obs.board
        self.last_action = ("build_road", road_location)

        self.roads.append(road_location)
        return "build_road", road_location

    def save_model(self, filename='models'):
        self.model.save_weights(self.save)

    def load_model(self, directory, filename='models'):
        self.model.load_weights(self.save)
