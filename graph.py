import numpy as np
import networkx as nx
import torch
import collections


# seed = np.random.seed(120)

class Graph:
    def __init__(self, graph_type, min_n, max_n, p, m=None, seed=None):

        cur_n=np.random.randint(max_n-min_n+1)+min_n
        if graph_type == 'erdos_renyi':
            self.g = nx.erdos_renyi_graph(n=cur_n, p=p, seed=seed)
        elif graph_type == 'powerlaw':
            self.g = nx.powerlaw_cluster_graph(n=cur_n, m=m, p=p, seed=seed)
        elif graph_type == 'barabasi_albert':
            self.g = nx.barabasi_albert_graph(n=cur_n, m=m, seed=seed)
        elif graph_type =='gnp_random_graph':
            self.g = nx.gnp_random_graph(n=cur_n, p=p, seed=seed)
        
        while self.g.number_of_edges == 0:
            if graph_type == 'erdos_renyi':
                self.g = nx.erdos_renyi_graph(n=cur_n, p=p, seed=seed)
            elif graph_type == 'powerlaw':
                self.g = nx.powerlaw_cluster_graph(n=cur_n, m=m, p=p, seed=seed)
            elif graph_type == 'barabasi_albert':
                self.g = nx.barabasi_albert_graph(n=cur_n, m=m, seed=seed)
            elif graph_type =='gnp_random_graph':
                self.g = nx.gnp_random_graph(n=cur_n, p=p, seed=seed)
        
        self.adj_dense = nx.to_numpy_matrix(self.g,dtype=np.int32)


    def nodes_count(self):
        return nx.number_of_nodes(self.g)

    def edges(self):

        return self.g.edges()

    def neighbors(self, node):

        return nx.all_neighbors(self.g,node)

    def average_neighbor_degree(self, node):

        return nx.average_neighbor_degree(self.g, nodes=node)

    def adj(self):
        return self.adj_dense