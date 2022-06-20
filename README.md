# settlers-of-catan-thesis
Reinforcement Learning project with Settlers of Catan as environment

This RL project implements a Deep Reinforcement Learning (DRL) agent using Double Deep Q-Learning with Dueling architecture and Prioritised Experience Replay (D3QN+PER) 
and Catan implementation which the agent acts upon

**Training the agent:** Run the TrainDQNAgent.py script which initialises all the other modules with specified parameters. Weights are saved to the "models" folder.
Tensorboard event file to follow training is saved in "graphs" folder. If "graphs" and "models" folders doesn't exsist create them in src directory.

**Running the agent:** Run the RunDQNAgent.py script which initialises all the other modules with specified parameters. 
The DQNAgent loads weights from specified path (DQNAgent.save attribute) and performs only best estimated action (purely exploitation)


• **Runner** - Game loop, summarises training, mediates between agents and environment
• **Environment** - Game logic, current board state and reward function
• **Agents** - Contains different agents, currently DQNAgent, RandomAgent and SmartRandomAgent
• **Prioritised Experience Replay** - Buffer saving and sampling transition tuples (st, at, rt, st+1)
• **Q-Model** - Returns a model with current neural network structure 
• **RunDQNAgent** - Initialises all other modules with parameters to load pre-trained weights and
only do informed actions and then start the runner
• **TrainDQNAgent** - Initialises all other modules with parameters to train the Agent and only do
informed actions and then start the runner
