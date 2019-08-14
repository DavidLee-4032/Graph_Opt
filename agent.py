import numpy as np
import random
import time
import os
import logging
import models
import copy
from utils.config import load_model_config

import torch.nn.functional as F
import torch

import torch.nn.parallel
import torch.backends.cudnn as cudnn

import environment as envir

# Set up logger
logging.basicConfig(
    format='%(asctime)s:%(levelname)s:%(message)s',
    level=logging.INFO
)

"""
Contains the definition of the agent that will run in an
environment.
"""




class DQAgent:


    def __init__(self,model,lr,bs,n_step,env_name,node_max):

        self.graphs = None
        self.embed_dim = 64
        self.model_name = model
        self.node_max = node_max
        self.alpha = 0.1
        self.gamma = 1 #0.99

        self.lambd = 0.
        self.n_step=n_step

        self.epsilon_=1
        self.epsilon_min=0.05
        self.discount_factor =0.999990
        #self.eps_end=0.02
        #self.eps_start=1
        #self.eps_step=20000
        
        self.memory = []
        self.memory_n=[]
        self.minibatch_length = bs

        self.env=envir.Environment(env_name)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.zpad=torch.zeros(1, self.node_max, 1, dtype=torch.float)

        if self.model_name == 'S2V_QN_1':

            args_init = load_model_config()[self.model_name]
            self.model = models.S2V_QN_1(**args_init)

        elif self.model_name == 'S2V_QN_2':
            args_init = load_model_config()[self.model_name]
            self.model = models.S2V_QN_2(**args_init)


        elif self.model_name== 'GCN_QN_1':

            args_init = load_model_config()[self.model_name]
            self.model = models.GCN_QN_1(**args_init)

        elif self.model_name == 'LINE_QN':

            args_init = load_model_config()[self.model_name]
            self.model = models.LINE_QN(**args_init)

        elif self.model_name == 'W2V_QN':

            args_init = load_model_config()[self.model_name]
            self.model = models.W2V_QN(G=self.graphs[self.games], **args_init)


        self.T = 5
        

        if torch.cuda.device_count() >= 1:
            print("Using", torch.cuda.device_count(), "GPUs!")
            self.model.to(self.device)
            torch.backends.cudnn.benchmark = True
            
        else: print("Using CPU")

        self.criterion = torch.nn.MSELoss(reduction='sum').to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        if torch.cuda.device_count() > 1: self.model = torch.nn.DataParallel(self.model)



    """
    p : embedding dimension
       
    """

    def reset(self, g):

        self.games = g
        self.env.reset(g)
        if (len(self.memory_n) != 0) and (len(self.memory_n) % 500000 == 0): #once memory comes to 500000, cut it down to 300000. Better if we make sure the RECENT datas to be preserved.
            tmp_mem = self.memory_n[-300000:].copy()
            self.memory_n.clear()
            self.memory_n=tmp_mem
            #random.sample(self.memory_n,300000)  random sample is not great

        self.nodes = self.graphs[self.games].nodes()
        self.adj = self.graphs[self.games].adj()
        self.last_action = 0
        self.last_observation = torch.zeros(1, self.nodes, 1, dtype=torch.float)
        self.last_reward = -0.01
        self.last_done = 0
        self.action = 0
        self.observation = torch.zeros(1, self.nodes, 1, dtype=torch.float)
        self.reward = -0.01
        self.done = 0
        self.iter = 0
        self.memory.clear()
        # you can use the auxiliary temp memory of the game and reset it here.

    def act(self, observation): # eps-greedy
        with torch.no_grad():
            if self.epsilon_ > np.random.rand():
                action = np.random.choice(np.where(observation.numpy()[0,:,0] == 0)[0])
            else:
                adj_max = torch.from_numpy(self.adj).type(torch.FloatTensor).view(1,self.node_max,self.node_max)
                q_a = self.model.forward(observation, adj_max).cpu() #forward propagate only, ADJ here should be ADJ_subgraph
                q_a=q_a.numpy()
                action = np.where((q_a[0, :, 0] == np.max(q_a[0, :, 0][observation.numpy()[0, :, 0] == 0])))[0][0]
            obs_tmp = self.env.observe().clone() # Using LOCAL variables to prevent unexpected changes of variables
            (reward,done) = self.env.act(action)
            self.remember(obs_tmp, action, reward)
            self.iter += 1
        return (reward, done)

    def adj_sub(self):
        actpt=self.env.active_pts_sol()
        I=np.zeros(self.nodes, self.nodes)
        for i in range(self.nodes):
            if actpt[i]==1:
                I[i,i]=1
        return I
        
    def permutation(self, xv):
        permu1=list(range(self.node_max))
        permu2=random.sample(permu1, len(xv))
        P=np.zeros(len(xv), self.node_max)
        for i in range(len(xv)):
            P[i,permu2[i]]=1
        return P
        #p:permutation matrix

    def renew(self,recent):
            # Warning: you should play the game several times (such as 1000) to start the optimizing process.
        if recent:#choose the recent rounds?
            exp_sam = random.sample(self.memory_n[:-20], self.minibatch_length-20)
            exp_sam_2=self.memory_n[-20:]
            exp_sam=exp_sam+exp_sam_2
        else:
            exp_sam = random.sample(self.memory_n, self.minibatch_length)
        l_obs_tens=torch.empty(self.minibatch_length, self.node_max,2)
        action_tens=torch.empty(self.minibatch_length)
        reward_tens=torch.empty(self.minibatch_length)
        obs_tens=torch.empty(self.minibatch_length, self.node_max,2)
        done_tens=torch.empty(self.minibatch_length)
        adj_tens=torch.empty(self.minibatch_length, self.node_max, self.node_max)
        target=torch.empty(self.minibatch_length)
        for i in range(self.minibatch_length):
            l_obs_tens[i]=exp_sam[i][0]#torch.zeros(1, 2).scatter_(1, exp_sam[i][0], 1)
            action_tens[i]=exp_sam[i][1].item()
            reward_tens[i]=exp_sam[i][2]
            obs_tens[i]=exp_sam[i][3]#torch.zeros(1, 2).scatter_(1, exp_sam[i][3], 1)
            done_tens[i]=int(exp_sam[i][4])
            adj_tens[i] = torch.from_numpy(self.graphs[exp_sam[i][5]].adj()).type(torch.FloatTensor)
        self.optimizer.zero_grad()
        with torch.no_grad():
            m1=self.model(obs_tens.to(self.device), adj_tens.to(self.device)).cpu()
            target = reward_tens + self.gamma*(1-done_tens)*(torch.max(m1 + obs_tens * (-1e5), dim=1)[0].view(self.minibatch_length))
            target_p=torch.zeros_like(target)
        p_tensor=self.model(l_obs_tens.to(self.device), adj_tens.to(self.device)).cpu()
        for i in range(self.minibatch_length):
            target_p[i] = p_tensor[i,exp_sam[i][1],:]
        loss=self.criterion(target_p, target)
        if(self.games%100==0):
            print(loss)
        loss.backward()
        self.optimizer.step()
        if self.epsilon_ > self.epsilon_min:
           self.epsilon_ *= self.discount_factor

    def remember(self, observation, action, reward): #You can change it to TEMPORAL data!!
        self.memory.append((observation, action, reward))

    def remember_n(self):#save n-step experience
        cum_reward=0
        for i in range(self.n_step):
            cum_reward += self.memory[i][2] #r0+...+r(i-1)
        for i in range(self.iter):
            done = (i+self.n_step > self.iter - 1)
            if done:
                step_init = (self.memory[i][0], self.memory[i][1], cum_reward, self.zpad, done, self.games)
            else:
                step_init = (self.memory[i][0], self.memory[i][1], cum_reward, self.memory[i+self.n_step][0], done, self.games)
            self.memory_n.append(step_init)
            if not done: cum_reward += self.memory[i+self.n_step][2]
            cum_reward -= self.memory[i][2]

    def save_model(self):
        cwd = os.getcwd()
        torch.save(self.model.state_dict(), cwd+'/model.pt')

    def graph_reset(self,graphs):
        self.graphs = graphs
        self.env.graph_reset(graphs)

    def change_to_test(self, test_graphs):
        self.graph_reset(test_graphs)
        self.epsilon_=0


Agent = DQAgent #alias
