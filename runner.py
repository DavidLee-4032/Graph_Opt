"""
This is the machinery that runs your agent in an environment.

"""
import matplotlib.pyplot as plt
import numpy as np
import torch
#from tqdm import tqdm

import agent
import graph


class Runner:
    def __init__(self,graphs, agent,verbose=True, test=False):
        self.agent = agent
        self.verbose = verbose
        self.test = test
        self.agent.graph_reset(graphs)

    def act(self):
        observation = self.agent.env.observe()#.clone()
        (reward, done) = self.agent.act(observation)#.copy()
        return (reward, done)



    def loop(self, games, nbr_epoch, max_iter):
        cumul_reward = 0.0
        list_cumul_reward=[]
        list_optimal_ratio = []
        list_aprox_ratio =[]
        if self.test:
            cumul_opti=0.0
            cumul_sol=0.0
            max_ratio=0.0
        cumul_sol_train=0.0
            #self.agent.graph_reset(self.graph_list)
        for g in range(games):
            self.agent.reset(g)
            cumul_reward = 0.0
            
            for i in range(1, max_iter + 1): #mat iter:
                (rew, done) = self.act()
                cumul_reward += rew
                if self.test and done:
                        #optimal solution
                    optimal_sol = self.agent.env.get_optimal_sol()
                        # print cumulative reward of one play, it is actually the solution found by the NN algorithm
                        #print(" ->    Terminal event: cumulative rewards = {}".format(-cumul_reward))
                        #print optimal solution
                    cumul_opti += optimal_sol
                        #print(" ->    Optimal solution = {}".format(optimal_sol))
                    assert(optimal_sol!=0)
                        #we add in a list the solution found by the NN algorithm
                    cumul_sol-=cumul_reward
                        #list_cumul_reward.append(-cumul_reward)
                    ratio_1 = -cumul_reward/optimal_sol
                    if ratio_1 > max_ratio:
                        max_ratio = ratio_1
                if done:
                    break
            cumul_sol_train-=cumul_reward
            self.agent.remember_n()
            if g > 1000:
                self.agent.renew(True)
                for i in range(4):
                    self.agent.renew(False)
            if self.verbose:
                if (g+1) % 100 == 0:
                    print(" <=> Finished game number: {} <=>".format(g))
                    print("reward:{}".format(cumul_sol_train/100))
                    cumul_sol_train=0
        if self.test:
            print("Obtained a final reward of {}".format(cumul_sol/(games+1)))
            print("Obtained a performance of {}".format(max_ratio))
            print("Obtained an average performance of {}".format(cumul_sol/cumul_opti))
    
    def change_to_test(self,graph_test):
        self.agent.change_to_test(graph_test)
        self.test = True
        self.graph_cnt=0
