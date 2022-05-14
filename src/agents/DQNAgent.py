import datetime
import numpy as np
import random as r
from src.agents import AbstractAgent
from src.expreplay import ExperienceReplay, Experience, PrioritizedReplayBuffer
from src.qmodel import q_model
from src.environment import Environment
from src.decorators import timer

ENV = Environment([AbstractAgent(0)])


class DQNAgent(AbstractAgent):
    def __init__(self, id, train):
        super(DQNAgent, self).__init__(id, train)
        if id == 1:  # id must be > 1
            raise ValueError("Agent id need to be > 1")

        self.network = q_model((ENV.board.shape[0], ENV.board.shape[1], 1), action_space=ENV.action_space)
        self.target_network = q_model((ENV.board.shape[0], ENV.board.shape[1], 1), action_space=ENV.action_space)
        self.batch_size = 64
        self.history = None
        self.save = 'models/DQNAgent_weights_reduced.h5'
        self.verbose = 0

        self.actions = ENV.action_space
        self.update_interval = 300
        self.train_interval = 1
        self.gamma = 0.95
        self.steps = 0
        self._EPSILON = 0.99
        self._EPSILON_DECAY = 0.000_003
        self._MIN_EPSILON = 0.05

        self.alpha = 0.7
        self.beta = 0.5
        self.beta_inc = 0.000_003
        self.epsilon_replay = 0.000_001
        self.experience_replay = PrioritizedReplayBuffer(100_000, self.beta)
        #self.experience_replay = ExperienceReplay()
        self.min_exp_len = 300

        self.state = None
        self.last_action = None

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
                if self.resources[trade_in] < 4 or self.resources[trade_out] > 11:
                    q_values[i] = -np.Inf

        return q_values

    def step(self, obs):  # Obs -> observer
        buildable_road_locations, buildable_village_locations = obs.get_buildable_locations(self)

        available_actions = self.get_available_actions(buildable_road_locations, buildable_village_locations)

        if r.random() < self._EPSILON and self.train:   # Exploration
            action = np.random.choice(available_actions)    # Action is chosen uniformly
            location = self.take_action(action, buildable_road_locations, buildable_village_locations)

        else:   # Exploitation
            norm_board = np.copy(obs.board)
            norm_board = norm_board / 12.0
            q_values = self.network.predict(np.array(norm_board.reshape([1, norm_board.shape[0], norm_board.shape[1], 1])))
            q_values = np.squeeze(q_values)

            masked_q_values = self.action_masking(
                q_values, obs, buildable_village_locations, buildable_road_locations, available_actions)

            action, location = obs.action_space[np.argmax(masked_q_values)]

            if "build" in action:
                getattr(self, action)(location)
            elif "trade" in action:
                trade_in, trade_out = location
                self.trade(trade_in, trade_out)

        if self.experience_replay.__len__() > self.min_exp_len and self.train:
            self.train_agent()

        if self.beta < 1:
            self.beta += self.beta_inc
        else:
            self.beta = 1

        if self._EPSILON > self._MIN_EPSILON:
            self._EPSILON -= self._EPSILON_DECAY
        else:
            self._EPSILON = self._MIN_EPSILON

        self.state = obs.board
        self.last_action = (action, location)
        self.steps += 1

        if self.steps % self.update_interval == 0 and self.steps > 0 and self.train:
            self.update_target()

        return action, location

    def train_agent(self):
        experiences = self.experience_replay.sample(self.batch_size, self.beta)
        states, actions, rewards, new_states, dones, weights, batch_idxes = experiences

        y = self.network.predict(states)
        old_y = y
        y_next = self.network.predict(new_states)
        y_target = self.target_network.predict(new_states)
        actions = [tuple(action) for action in actions]

        for state in range(self.batch_size):
            if dones[state]:
                y[state][self.actions.index(actions[state])] = rewards[state]
            else:   # Bellman equation to update rewards
                y[state][self.actions.index(actions[state])] = (
                        rewards[state] + self.gamma * y_target[state][np.argmax(y_next[state])]
                )
                if self.steps % 600 == 0:
                    print("LABELS",
                        rewards[state] + self.gamma * y_target[state][np.argmax(y_next[state])]
                    )
        # error y gets to large. try normalize input
        # Try normalizing y

        self.history = self.network.fit(states, y, batch_size=self.batch_size, verbose=self.verbose, sample_weight=weights)
        self.verbose = 0

        priorities = []
        for state, action in enumerate(actions):
            error = self.epsilon_replay + np.abs(old_y[state][self.actions.index(action)] - y[state][self.actions.index(action)])
            priorities.append(error)

        self.experience_replay.update_priorities(batch_idxes, priorities)

    def update_target(self):
        self.target_network.set_weights(self.network.get_weights())
        self.verbose = 1
        print("BETA", self.beta)

    def save_experience(self, state, action, reward, done, next_state):
        for resource in self.resources.values():
            if resource > 12:
                raise ValueError("More resources than 12")

        next_state = self.encode_resources(next_state)
        norm_state = np.copy(state)
        norm_state = norm_state / 12.0

        norm_next_state = np.copy(next_state)
        norm_next_state = norm_next_state / 12.0

        a = np.where(norm_next_state > 12.0)
        b = np.where(norm_state > 12.0)
        x = np.where(norm_state < -1.0)
        y = np.where(norm_next_state < -1.0)
        if len(b[0]) > 0 or len(a[0]) > 0:
            print(norm_state)
            print(norm_next_state)
            raise ValueError("state larger than 12")

        if len(x[0]) > 0 or len(y[0]) > 0:
            print(norm_state)
            print(norm_next_state)
            raise ValueError("state less than -1.0")


        self.experience_replay.add(norm_state, action, reward, norm_next_state, done)

    def random_trade(self, trade_in):  # Trade with bank
        new_resources = list(self.resources.keys())
        new_resources.remove(trade_in)
        traded_out = r.choice(new_resources)
        self.resources[traded_out] += 1
        self.resources[trade_in] -= 4

    def trade(self, trade_in, traded_out):  # Trade with bank max resource for min resource
        if self.resources[traded_out] < 12:
            self.resources[traded_out] += 1
            self.resources[trade_in] -= 4

    def free_village_build(self, obs):  # Round 1
        buildable_locations = obs.free_build_village()

        if r.random() < self._EPSILON and self.train:   # Exploration
            village_location = r.choice(buildable_locations)
            self._EPSILON -= self._EPSILON_DECAY

        else:   # Exploitation
            norm_board = np.copy(obs.board)
            norm_board = norm_board / 12.0
            q_values = self.network.predict(np.array(norm_board.reshape([1, norm_board.shape[0], norm_board.shape[1], 1])))
            q_values = np.squeeze(q_values)

            masked_q_values = self.action_masking(q_values, obs, buildable_locations, [], "build_village")
            masked_q_values[-1] = -np.Inf  # Pass unavailable

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
            norm_board = np.copy(obs.board)
            norm_board = norm_board / 12.0
            q_values = self.network.predict(np.array(norm_board.reshape([1, norm_board.shape[0], norm_board.shape[1], 1])))
            q_values = np.squeeze(q_values)

            masked_q_values = self.action_masking(q_values, obs, [], buildable_locations, "build_road")
            masked_q_values[-1] = -np.Inf  # Pass unavailable

            action, road_location = obs.action_space[np.argmax(masked_q_values)]

        self.state = self.encode_resources(obs.board)
        self.last_action = ("build_road", road_location)

        self.roads.append(road_location)
        return "build_road", road_location

    def encode_resources(self, board):
        agent_board = np.copy(board)
        for row, value in enumerate(self.resources.values()):
            agent_board[row][1] = value
        return agent_board

    def save_model(self, filename='models'):
        self.network.save_weights(self.save)

    def load_model(self, directory, filename='models'):
        self.network.load_weights(self.save)

    def get_epsilon(self):
        return self._EPSILON
