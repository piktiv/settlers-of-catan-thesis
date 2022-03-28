import datetime
import os
import tensorflow as tf
import numpy as np
from src.environment import Environment


class Runner:
    def __init__(self, agent, env, train, load_path):

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

        if not self.train and load_path is not None and os.path.isdir(load_path):
            self.agent.load_model(load_path)

    def summarize(self):
        self.scores_batch.append(self.score)
        if len(self.scores_batch) == 50:
            average = np.mean(self.scores_batch)
            self.writer.add_summary(tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(tag='Moving Average Score (50) per Episode', simple_value=average)]), self.episode - 50)
            self.scores_batch.pop(0)
            self.current_average = average
        if self.train and self.episode % 10 == 0:
            self.agent.save_model(self.path)
        self.episode += 1
        self.score = 0

    def run(self, episodes):
        self.env.players[0].roads += [(11, 8)]
        self.env.players[0].villages += [(4, 8)]
        self.env.board[4][8] = self.env.players[0].id
        self.env.board[11][8] = self.env.players[0].id
        self.env.players[0].score = 1

        self.env.print_board()
        while self.episode <= episodes:
            obs = self.env      # Reset env

            #obs = self.env.reset()
            # reset agents
            while True:
                # TODO round 1 place village and road

                # For each player
                for player in self.env.players:

                    # Throw dices, all gather resources
                    obs.environment_step()
                    while True:
                        # Agent take action
                        print(self.env.players[0].resources)
                        print(self.agent.score)
                        action, location = player.step(obs)
                        obs = self.env.step(action, location, player.id)  # Take action on env
                        if action == "pass" or obs.last():
                            break


                        #self.score += obs.reward
                    if obs.last():
                        break
                if obs.last():
                    break
            if len(self.env.players[0].roads) != len(set(self.env.players[0].roads)) or obs.last():
                print(self.env.players[0].score)
                print("villages", self.env.players[0].villages)
                print("cities", self.env.players[0].cities)
                print("roads", self.agent.roads)
                print("roads", len(self.env.players[0].roads))
                print("roads set", len(set(self.env.players[0].roads)))
                self.env.print_board()
                break
            print(len(self.env.players[0].roads))
            print(len(list(set(self.env.players[0].roads))))

            #self.summarize()
            print('Average: ', self.current_average)
