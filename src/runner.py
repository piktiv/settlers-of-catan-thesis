import datetime
import os
import tensorflow as tf
import numpy as np
from itertools import chain
from itertools import permutations

from src.environment import Environment
from src.decorators import timer


class Runner:
    def __init__(self, agent, env, train, load_path):
        tf.config.run_functions_eagerly(False)
        self.agent = agent
        self.env = env
        self.train = train  # run only or train_model model?
        self.scores_batch = []
        self.victory_points_batch = []
        self.score = 0  # store for the scores of an episode
        self.episode = 1  # episode counter
        self.best_win_rate = 0

        self.path = './graphs/' + datetime.datetime.now().strftime("%y%m%d_%H%M") \
                    + ('_train_' if self.train else '_run_') \
                    + type(agent).__name__

        self.writer = tf.summary.create_file_writer(logdir=self.path)

        if not self.train and load_path is not None and os.path.isdir(load_path):
            self.agent.load_model(load_path)

    def summarize(self):
        print(f'Episode {self.episode}')

        self.victory_points_batch.append(self.agent.score)
        self.scores_batch.append(self.score)

        if len(self.scores_batch) == 50:
            with self.writer.as_default():
                tf.summary.scalar('Win rate average (50) per Episode', np.mean(self.scores_batch), self.episode - 50)
                tf.summary.scalar('Average VP (50) per Episode', np.mean(self.victory_points_batch), self.episode - 50)
                tf.summary.scalar('Epsilon per Episode', self.agent.get_epsilon(), self.episode)
                tf.summary.scalar('Loss', np.squeeze(self.agent.history.history['loss']), self.episode)

                if self.best_win_rate < np.mean(self.scores_batch):
                    self.best_win_rate = np.mean(self.scores_batch)
                    self.network.save_weights('models/DQNAgent_best_weights.h5')
                # self.writer.flush()
            self.scores_batch.pop(0)
            self.victory_points_batch.pop(0)
        if self.train and self.episode % 10 == 0:
            print(f'saving weights')
            self.agent.save_model(self.path)
        self.episode += 1
        self.score = 0

    def get_ranking(self):
        return sorted(self.env.players, key=lambda x: x.score, reverse=True)

    @timer
    def run(self, episodes):
        while self.episode <= episodes:
            self.env.reset_env()

            for player in self.env.players:
                player.reset_agent()

            obs = self.env

            start_order = chain(self.env.players, list(reversed(self.env.players)))
            for player in start_order:  # Game start
                action, location = player.free_village_build(obs)
                obs = self.env.step(action, location, player.id)
                if player == self.agent and self.train:
                    self.agent.save_experience(
                        self.agent.state, self.agent.last_action, obs.reward(self.agent),
                        obs.last(), self.agent.encode_resources(obs.board)
                    )

                action, location = player.free_road_build(obs, location)
                obs = self.env.step(action, location, player.id)

            while True:
                for player in self.env.players:
                    obs.environment_step()
                    while action != "pass" or not obs.last():

                        if player == self.agent and self.train:
                            self.agent.save_experience(
                                self.agent.state, self.agent.last_action, obs.reward(self.agent),
                                obs.last(), self.agent.encode_resources(obs.board)
                            )

                        action, location = player.step(obs)
                        obs = self.env.step(action, location, player.id)

                        if action == "pass" or obs.last():
                            break

                    if obs.last():
                        break

                if obs.last():
                    break

            # Last step, catches if agent don't take action leading to terminal state
            if self.train:
                if player == self.agent and self.train:
                    self.agent.save_experience(
                        self.agent.state, self.agent.last_action, obs.reward(self.agent),
                        obs.last(), self.agent.encode_resources(obs.board)
                    )

            ranking = self.get_ranking()
            if ranking[0] == self.agent:
                self.score = 1


            for player in self.env.players:
                print(f'{type(player).__name__} {player.villages} {player.cities}')
                print(len(player.roads))

            print(f'Epsilon {self.agent.get_epsilon}')
            self.summarize()

            self.env.print_board()
