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
        self.score = 0  # store for the scores of an episode
        self.episode = 1  # episode counter
        self.current_average = 0
        self.path = './graphs/' + datetime.datetime.now().strftime("%y%m%d_%H%M") \
                    + ('_train_' if self.train else 'run_') \
                    + type(agent).__name__

        # self.writer = tf.summary.FileWriter(self.path, tf.get_default_graph())
        #self.writer = tf.compat.v1.summary.FileWriter(self.path)
        #self.writer = tf.summary.create_file_writer(self.path)

        if not self.train and load_path is not None and os.path.isdir(load_path):
            self.agent.load_model(load_path)

    def summarize(self):
        print(f'Episode {self.episode}')
        self.scores_batch.append(self.score)
        '''if len(self.scores_batch) == 50:
            average = np.mean(self.scores_batch)
            self.writer.add_summary(tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(tag='Moving Average Score (50) per Episode', simple_value=average)]), self.episode - 50)
            self.scores_batch.pop(0)
            self.current_average = average'''
        if self.train and self.episode % 10 == 0:
            print(f'saving weights')
            self.agent.save_model(self.path)
        self.episode += 1
        self.score = 0

    def get_ranking(self):
        return sorted(self.env.players, key=lambda x: x.score, reverse=True)

    @timer
    def run(self, episodes):
        results = {
            type(self.agent).__name__: 0,
            "SmartRandomAgent": 0
        }

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
                        player.state, (action, location), obs.reward(self.agent), obs.last(), obs.board
                    )

                action, location = player.free_road_build(obs, location)
                obs = self.env.step(action, location, player.id)

            while True:
                for player in self.env.players:
                    obs.environment_step()
                    while action != "pass" or not obs.last():

                        if player == self.agent and self.train:
                            self.agent.save_experience(
                                player.state, (action, location), obs.reward(self.agent), obs.last(), obs.board
                            )

                        action, location = player.step(obs)
                        obs = self.env.step(action, location, player.id)

                        if action == "pass" or obs.last():
                            break

                    if obs.last():
                        break

                if obs.last():
                    break

            # Last step, catches if agent don't take action leading to terminal state aswell
            if player == self.agent and self.train:
                self.agent.save_experience(
                    player.state, (action, location), obs.reward(self.agent), obs.last(), obs.board
                )

            self.summarize()

            self.env.print_board()
            ranking = self.get_ranking()

            for i, player in enumerate(reversed(ranking)):
                results[type(player).__name__] += i

            for player in self.env.players:
                print(f'{type(player).__name__} {player.villages} {player.cities}')
            for key, item in results.items():
                print(f'{key} wins {(item / self.episode) * 100}% won {item} times')
            print(f'Epsilon {self.agent._EPSILON}')

        for key, item in results.items():
            print(f'{key} wins {(item / episodes) * 100}%')





