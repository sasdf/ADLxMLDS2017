import multiprocessing as mp
import queue
import copy
from tqdm import *
import time

class Player(mp.Process):
    def __init__(self, actQueue, obsQueue, reward):
        super().__init__()
        self.actQueue = actQueue
        self.obsQueue = obsQueue
        self.reward = reward

    def run(self):
        tqdm.write("[*] Player Start")
        env = self.actQueue.get()
        tqdm.write("[*] Got env")
        while True:
            state = env.reset().transpose(2, 0, 1)
            self.obsQueue.put([None, state, None, 0, False])
            done = False
            #playing one game
            epReward = 0
            while not done:
                action = self.actQueue.get()
                if action is None:
                    print("[*] Stopping Player")
                    return
                next_state, reward, done, info = env.step(action)
                epReward += reward
                next_state = next_state.transpose(2, 0, 1)
                self.obsQueue.put([state, next_state, action, reward, done])
                state = next_state
            self.reward.value = self.reward.value * 0.95 + epReward * 0.05

