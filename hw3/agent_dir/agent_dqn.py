from agent_dir.agent import Agent
from agent_dir.dqn.network import DDQN
from agent_dir.dqn.dataset import Experience
from agent_dir.dqn.memory import Memory
from agent_dir.dqn.player import Player
import random
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch import LongTensor, FloatTensor, optim
import time, sys
from tqdm import *
import multiprocessing as mp
import queue
import copy

class Worker(mp.Process):
    def __init__(self, paramQueue, expQueue, reward, wtf):
        super().__init__()
        self.paramQueue = paramQueue
        self.expQueue = expQueue
        self.reward = reward
        self.wtf = wtf

    def run(self):
        tqdm.write("[*] Worker Start")
        env = self.paramQueue.get()
        model = DDQN().cuda()
        param = self.paramQueue.get()
        if param is None: return
        model.load_state_dict(param)

class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_DQN,self).__init__(env)

        if args.test_dqn:
            #you can load your model here
            print('loading trained model')
            with open('dqn.pt', 'rb') as f:
                self.dqn = torch.load(f)
            env = self.env
            env.seed(42)
            random.seed(42)
            torch.manual_seed(42)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(42)
        else:
            mp.set_start_method('spawn')
            self.reward = mp.Value('d', 0)
            self.wtf = mp.Value('d', 0)
            self.dqn = DDQN().cuda()
            self.memory = Memory()
            self.experience = Experience(self.memory, 1 * 32)
            self.dataLoader = DataLoader(self.experience, batch_size=32, collate_fn=self.experience.collate_fn)

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
        pass



    def train(self):
        """
        Implement your training algorithm here
        """
        ##################
        # YOUR CODE HERE #
        ##################

        log = open('reward_dqn.log', 'w')
        #  optimizer = optim.Adam((p for p in self.dqn.parameters() if p.requires_grad))
        optimizer = optim.RMSprop((p for p in self.dqn.parameters() if p.requires_grad), lr=1e-4)
        env = self.env
        env.seed(42)
        random.seed(42)
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)

        self.dqn.share_memory()

        bar = trange(10000, desc='Warmup')
        while len(self.memory) < 10000: # warmup
            state = env.reset().transpose(2, 0, 1)
            done = False
            #playing one game
            while not done:
                action = env.get_random_action()
                next_state, reward, done, info = env.step(action)
                next_state = next_state.transpose(2, 0, 1)
                self.memory.append([FloatTensor(state), FloatTensor(next_state), LongTensor([action]), FloatTensor([reward])], 1)
                bar.update()
                if len(self.memory) >= 10000:
                    break
                state = next_state
        bar.close()

        avg_loss = None
        epReward = 0
        newReward = 0
        done = True
        try:
            for iters in trange(10000000):
                for _ in range(4):
                    if done:
                        sstate = env.reset().transpose(2, 0, 1)
                        self.reward.value = self.reward.value * 0.95 + epReward * 0.05
                        log.write(str(epReward) + '\n')
                        log.flush()
                        epReward = 0
                        done = False
                    vstate = Variable(FloatTensor(sstate).unsqueeze(0), volatile=True).cuda()
                    action = self.dqn.make_action(vstate).data.cpu().numpy().tolist()[0]
                    next_state, reward, done, info = env.step(action)
                    epReward += reward
                    newReward += reward
                    if done:
                        reward = -10
                    next_state = next_state.transpose(2, 0, 1)
                    batch = [FloatTensor(next_state), LongTensor([action]), FloatTensor([reward])]
                    batch = [vstate] + [Variable(z.unsqueeze(0), volatile=True).cuda() for z in batch]
                    error = self.dqn.loss(*batch).data.cpu().numpy().tolist()[0]
                    obs = [z.data.cpu().squeeze(0) for z in batch]
                    self.memory.append(obs, error)
                    sstate = next_state
                bar = self.dataLoader#tqdm(self.dataLoader)
                for i, data in enumerate(bar):
                    index = data[-1]
                    (state, next_state, action, reward, importance) = [Variable(z).cuda() for z in data[:-1]]
                    loss = self.dqn.loss(state, next_state, action, reward)
                    optimizer.zero_grad()
                    loss = (loss * importance).sum(0)
                    loss.backward()
                    #  torch.nn.utils.clip_grad_norm(self.dqn.parameters(), 0.1)
                    optimizer.step()
                    (state, next_state, action, reward, importance) = [Variable(z, volatile=True).cuda() for z in data[:-1]]
                    loss = self.dqn.loss(state, next_state, action, reward)
                    for i, e in zip(index, loss.data.cpu().numpy().tolist()):
                        e = abs(e)
                        self.memory.update(i, e)
                    if avg_loss is None: avg_loss = loss.data[0]
                    avg_loss = avg_loss * 0.99 + loss.data[0] * 0.01
                #  bar.close()

                if iters % 25 == 0:
                    tqdm.write('[Epoch %5d] Loss: %.5f, reward: %d, count: %d, epReward: %.5f, wtf: %.5f' % (iters, avg_loss, newReward, len(self.memory.errors), self.reward.value, self.memory.tree[1]))
                    newReward = 0
                if iters % 250 == 0:
                    self.dqn.sync()
        except KeyboardInterrupt:
            pass
        tqdm.write("[*] Save Model")
        with open('dqn.pt', 'wb') as f:
            torch.save(self.dqn, f)
            f.flush()
        tqdm.write("[*] Save Model Done")

    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent

        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)

        Return:
            action: int
                the predicted action from trained model
        """
        ##################
        # YOUR CODE HERE #
        ##################
        observation = observation.transpose(2, 0, 1)
        vstate = Variable(FloatTensor(observation).unsqueeze(0), volatile=True).cuda()
        action = self.dqn.make_action(vstate).data.cpu().numpy().tolist()[0]
        return action#self.env.get_random_action()

