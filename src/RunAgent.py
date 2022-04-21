from absl import app
from src.environment import Environment
from src.runner import Runner
from src.agents import *


_CONFIG = dict(
    episodes=100,
    visualize=False,
    train=False,
    agent=DQNAgent,
    load_path='./pickles/',
    shuffle=False
)


def main(unused_argv):

    agent = _CONFIG['agent'](
        id=7,
        train=_CONFIG['train']
    )

    env = Environment(
        players=[agent, SmartRandomAgent(9)],
        visualize=_CONFIG['visualize'],
        shuffle=_CONFIG['shuffle']
    )

    runner = Runner(
        agent=agent,
        env=env,
        train=_CONFIG['train'],
        load_path=_CONFIG['load_path']
    )

    runner.run(episodes=_CONFIG['episodes'])


if __name__ == "__main__":
    app.run(main)