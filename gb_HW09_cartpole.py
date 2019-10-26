import random
import gym
import math
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


ENV_NAME = 'CartPole-v0'


GAMMA = 0.95 #aka decay or discount rate, to calculate the future discounted reward
LEARNING_RATE = 0.001 #how much neural net learns in each iteration ....dflt 0.001


EXPLORATION_RATE=1.0 #the rate in which an agent randomly decides its action rather than prediction
EXPLORATION_MIN = 0.01 # agent to explore at least this amount....dflt 0.01
EXPLORATION_DECAY = 0.995  # decrease the number of explorations as it gets good at playing game

MEMORY_SIZE = 1000000

BATCH_SIZE = 32
EPISODES = 500

class GB_cartpole_agent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        #parameters
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.gamma = GAMMA
        self.learning_rate = LEARNING_RATE
        self.epsilon = EXPLORATION_RATE
        self.epsilon_min = EXPLORATION_MIN
        self.epsilon_decay = EXPLORATION_DECAY
        self.model = self.model_setup()

    def model_setup(self):
        #setting neural network
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(48, activation='relu'))
        #model.add(Dense(48, activation='tanh'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss="mse", optimizer=Adam(lr=self.learning_rate)) #,decay=LEARNING_RATE_DECAY))

        return model


    def remember(self, state, action, reward, next_state, done):
        # store past experiences for using it in retrain process
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # decide whether to take random action at EXPLORATION_RATE to try "all kinds of things"
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        # otherwise predict the reward on current state
        act_values = self.model.predict(state)  #proba of moves
        return np.argmax(act_values[0])  # returns action which move to take

    def replay_experiences(self):
        # train model from past experiences - BATCH_SIZE is the number of used 'experiences'

        if len(self.memory) < BATCH_SIZE:
            return
        # select experiences
        minibatch = random.sample(self.memory, BATCH_SIZE)
        states, targets_f = [], []

        for state, action, reward, next_state, done in minibatch:
            # if done set target reward
            target=reward
            if not done:
                # used gamma /discount_rate to maximize the discounted future reward on hte given state
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])

            # approximately map the current state to future discounted reward (i.e. target_f)
            target_f = self.model.predict(state)
            target_f[0][action] = target

            states.append(state[0])
            targets_f.append(target_f[0])

        history = self.model.fit(np.array(states), np.array(targets_f), epochs=1, verbose=0)
        loss = history.history['loss'][0]

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss


    def load(self, name):
        self.model.load_weights(name)


    def save(self, name):
        self.model.save_weights(name)



def GB_cartpole():
    env = gym.make(ENV_NAME)
    env._max_episode_steps = 250  #or None
    # parameters for class GB_cartpole_agent from environment
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # create agent
    agent=GB_cartpole_agent(state_size, action_size)

    done=False

    for ep in range(EPISODES):
        state=env.reset()
        state = np.reshape(state, [1, state_size])

        for t_score in range(300):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)

            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])

            agent.remember(state, action, reward, next_state, done)

            state = next_state


            if len(agent.memory) > BATCH_SIZE:
                loss = agent.replay_experiences()

            if done:
                print("episode: {}/{}, score: {}, e: {:.4}".format(ep, EPISODES, t_score, agent.epsilon))
                break



if __name__ == "__main__":
    GB_cartpole()


