import gym
import pickle

from agent import *
from continuousagent import *
from trainer import *
from environment import *

# Just a simple test to verify everything is working correctly on a very simple environnement
env = SimpleEnv()
agent = Agent(1, 2, hidden_units=3)
train(agent,env,100, max_t=50)

# CartPole environment
env = gym.make('CartPole-v0')
agent = Agent(4, 2, hidden_units=10)
train(agent, env, 100, max_t=195)

# Continuous agent tested on a custom environment, results are saved in hist4.p, and can be loaded by running plotter.py
env = RocketEnv()
agent = ContinuousAgent(2, hidden_units = 3, discount = 0.9,
                        learning_rate_pi = 0.001, learning_rate_v = 0.01, keras = False, bias = False)

hist = train(agent, env, 500, render=False)
with open( 'hist.p', 'wb' ) as file:
    pickle.dump(hist,file)

# Simple tests with curiosity added on an environment which would recquire a lot of luck to be solve by usual methods
# But which should be farily easy to solve with curiosity. However, it doesn't work.
env = SimpleEnv2()
agent = Agent(1, 2, hidden_units=3)
train(agent,env,10, max_t=30)
agent = Agent(1, 2, hidden_units=3, curiosity=0.02, opimtimizer='adam')
train(agent,env,10, max_t=30)
agent = Agent(1, 2, hidden_units=3, curiosity=0.02, opimtimizer='sgd')
train(agent,env,10, max_t=30)

# Doesn't work either on the Mountain car problem
env = gym.make('MountainCar-v0')
agent = Agent(2, 3, epsilon= 0.5, curiosity=1000, opimtimizer='sgd')
train(agent, env, 100, max_t=195)

