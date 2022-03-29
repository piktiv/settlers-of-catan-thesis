from absl import app
from src.environment import Environment
from src.runner import Runner
from src.agents import BasicAgent

_CONFIG = dict(
    episodes=500,
    visualize=False,
    train=True,
    agent=BasicAgent,
    load_path='./pickles/'
)


def main(unused_argv):

    agent = BasicAgent(
        id=2,
        train=_CONFIG['train']
    )

    env = Environment(
        players=[agent, BasicAgent(9, False)],
        visualize=_CONFIG['visualize']
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
