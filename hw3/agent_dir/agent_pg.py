from agent_dir.agent import Agent
import scipy.misc
import numpy as np
from agent_dir.pg.network import PG
import random
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch import LongTensor, FloatTensor, optim, autograd
import time, sys
from tqdm import *
import multiprocessing as mp
import queue
import copy

def prepro(o,image_size=[80,80]):
    """
    Call this function to preprocess RGB image to grayscale image if necessary
    This preprocessing code is from
        https://github.com/hiwonjoon/tf-a3c-gpu/blob/master/async_agent.py
    
    Input: 
    RGB image: np.array
        RGB screen of game, shape: (210, 160, 3)
    Default return: np.array 
        Grayscale image, shape: (80, 80, 1)
    
    """
    y = 0.2126 * o[:, :, 0] + 0.7152 * o[:, :, 1] + 0.0722 * o[:, :, 2]
    y = y.astype(np.uint8)
    resized = scipy.misc.imresize(y, image_size)
    return np.expand_dims(resized.astype(np.float32),axis=2)


class Agent_PG(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_PG,self).__init__(env)
        
        self.lastState = None
        self.saved_actions = []

        if args.test_pg:
            #you can load your model here
            print('loading trained model')
            with open('pg.pt', 'rb') as f:
                self.pg = torch.load(f)
            env = self.env
            env.seed(42)
            random.seed(42)
            torch.manual_seed(42)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(42)
        else:
            self.reward = 0
            self.pg = PG().cuda()

        ##################
        # YOUR CODE HERE #
        ##################


    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        ##################
        # YOUR CODE HERE #
        ##################
        self.lastState = None
        self.saved_actions = []
        pass


    def train(self):
        """
        Implement your training algorithm here
        """
        ##################
        # YOUR CODE HERE #
        ##################
        log = open('reward_pg.log', 'w')
        optimizer = optim.RMSprop((p for p in self.pg.parameters() if p.requires_grad), lr=1e-4)
        env = self.env
        env.seed(42)
        random.seed(42)
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)

        self.pg.share_memory()

        avg_loss = None
        epReward = 0
        done = True
        try:
            bar = trange(10000000)
            for iters in bar:
                self.init_game_setting()
                sstate = env.reset()
                self.reward = self.reward * 0.95 + epReward * 0.05
                epReward = 0
                done = False
                rewards = []
                for i in range(10000):
                    action = self.make_action(sstate)
                    next_state, reward, done, info = env.step(action)
                    epReward += reward
                    if done:
                        reward = -10
                        rewards.append(reward)
                        break
                    else:
                        rewards.append(reward)
                    sstate = next_state
                rewards.pop(0)
                R = 0
                drewards = []
                for r in rewards[::-1]:
                    R = r + 0.99 * R
                    drewards.insert(0, R)
                rewards = torch.Tensor(drewards).cuda()
                rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
                for action, r in zip(self.saved_actions, rewards):
                    action.reinforce(r)
                optimizer.zero_grad()
                autograd.backward(self.saved_actions, [None for _ in self.saved_actions])
                optimizer.step()

                log.write(str(epReward) + '\n')
                log.flush()
                tqdm.write('[Epoch %5d] reward: %.5f, epReward: %.5f' % (iters, self.reward, epReward))
        except KeyboardInterrupt:
            pass
        log.close()
        tqdm.write("[*] Save Model")
        with open('pg.pt', 'wb') as f:
            torch.save(self.pg, f)
            f.flush()
        tqdm.write("[*] Save Model Done")
        pass


    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent

        Input:
            observation: np.array
                current RGB screen of game, shape: (210, 160, 3)

        Return:
            action: int
                the predicted action from trained model
        """
        ##################
        # YOUR CODE HERE #
        ##################
        observation = prepro(observation).transpose(2,0,1)
        state = torch.from_numpy(observation).float().unsqueeze(0)

        if self.lastState is None:
            self.lastState = state
            return self.env.get_random_action()
        diff = state - self.lastState
        self.lastState = state

        vstate = Variable(diff).cuda()
        probs = self.pg(vstate)
        action = probs.multinomial()
        self.saved_actions.append(action)
        ret = action.data.cpu().numpy().tolist()[0][0]
        return ret#self.env.get_random_action()

