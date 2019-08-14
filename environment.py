import numpy as np
import torch
import pulp
import networkx as nx
"""
This file contains the definition of the environment
in which the agents are run.
"""


class Environment:
    def __init__(self,name):
        self.graphs = None
        self.name = name

    def graph_reset(self,graphs):
        self.graphs = graphs
        
    def reset(self, g):
        self.games = g
        self.graph_init = self.graphs[self.games]
        self.nodes = self.graph_init.nodes()
        self.nodes_count = 0
        self.edge_cover_count = 0
        self.last_reward = 0
        self.observation = torch.zeros(1,self.nodes,1,dtype=torch.float)
        self.active_array=np.zeros(self.nodes,self.nodes)
    def observe(self):
        """Returns the current observation that the agent can make
                 of the environment, if applicable.
        """
        #This is a non-random instance
        return self.observation

    def fully_covered(self):
        done=True
        edge_sum=0
        for edge in self.graph_init.edges():
            if self.observation[:,edge[0],:]==0 and self.observation[:,edge[1],:]==0:
                done=False
            else:
                edge_sum += 1
        return(done,edge_sum)

    def active_pts_sol(self):#return the active points
        actpts=np.zeros(self.graph_init.number_of_nodes(),dtype=torch.int32)
        for node in self.graph_init.nodes():
            if node in self.observation:
                continue
            else:
                if nx.all_neighbors(self.graph_init,node) in self.observation:
                    continue
            actpts[node]=1
        return actpts

    def act(self,node):

        self.observation[:,node,:]=1
        reward,done = self.get_reward(node)
        return reward,done

    def get_reward(self, node, observation=None):
        if not observation: #This function can use inner or outer parameters
            observation=self.observation

        if self.name == "MVC":
            nodes_count=np.sum(observation[0].numpy())
            if self.nodes_count != nodes_count:
                reward = -1
            else:
                reward = 0
            self.nodes_count=nodes_count
            """
            done = True
            edge_sum = 0
            for edge in self.graph_init.edges():
                if observation[:,edge[0],:]==0 and observation[:,edge[1],:]==0:
                    done=False
                else:
                    edge_sum += 1
            """
            (done,self.edge_cover_count)=self.fully_covered()
            return (reward,done)

        elif self.name=="MAXCUT" :

            reward=0
            done=False

            adj= self.graph_init.edges()
            select_node=np.where(self.observation[0, :, 0].numpy() == 1)[0]
            for nodes in adj:
                if ((nodes[0] in select_node) & (nodes[1] not in select_node)) | ((nodes[0] not in select_node) & (nodes[1] in select_node))  :
                    reward += 1#/20.0
            change_reward = reward-self.last_reward
            if change_reward<=0:
                done=True

            self.last_reward = reward

            return (change_reward,done)

    def get_approx(self):

        if self.name=="MVC":
            cover_edge=[]
            edges= list(self.graph_init.edges())
            while len(edges)>0:
                edge = edges[np.random.choice(len(edges))]
                cover_edge.append(edge[0])
                cover_edge.append(edge[1])
                to_remove=[]
                for edge_ in edges:
                    if edge_[0]==edge[0] or edge_[0]==edge[1]:
                        to_remove.append(edge_)
                    else:
                        if edge_[1]==edge[1] or edge_[1]==edge[0]:
                            to_remove.append(edge_)
                for i in to_remove:
                    edges.remove(i)
            return len(cover_edge)

        elif self.name=="MAXCUT":
            return 1

        else:
            return 'you pass a wrong environment name'

    def get_optimal_sol(self):

        if self.name =="MVC":
            
            x = list(range(self.graph_init.g.number_of_nodes()))
            
            xv = pulp.LpVariable.dicts('is_opti', x,
                                       lowBound=0,
                                       upBound=1,
                                       cat=pulp.LpInteger)

            mdl = pulp.LpProblem("MVC", pulp.LpMinimize)

            mdl += sum(xv[k] for k in xv)

            for edge in self.graph_init.edges():
                mdl += xv[edge[0]] + xv[edge[1]] >= 1, "constraint :" + str(edge)
            mdl.solve()

            #print("Status:", pulp.LpStatus[mdl.status])
            optimal=0
            for x in xv:
                optimal += xv[x].value()
                #print(xv[x].value())
            return optimal

        elif self.name=="MAXCUT":

            x = list(range(self.graph_init.g.number_of_nodes()))
            e = list(self.graph_init.edges())
            xv = pulp.LpVariable.dicts('is_opti', x,
                                       lowBound=0,
                                       upBound=1,
                                       cat=pulp.LpInteger)
            ev = pulp.LpVariable.dicts('ev', e,
                                       lowBound=0,
                                       upBound=1,
                                       cat=pulp.LpInteger)

            mdl = pulp.LpProblem("MVC", pulp.LpMaximize)

            mdl += sum(ev[k] for k in ev)

            for i in e:
                mdl+= ev[i] <= xv[i[0]]+xv[i[1]]

            for i in e:
                mdl+= ev[i]<= 2 -(xv[i[0]]+xv[i[1]])

            #pulp.LpSolverDefault.msg = 1
            mdl.solve()

            # print("Status:", pulp.LpStatus[mdl.status])

            return mdl.objective.value()


        
