import argparse
import agent
#import environment
import runner
import graph
import logging
import numpy as np
import networkx as nx
import sys

# # 2to3 compatibility
# try:
#     input = raw_input
# except NameError:
#     pass

# Set up logger
logging.basicConfig(
    format='%(asctime)s:%(levelname)s:%(message)s',
    level=logging.INFO
)

parser = argparse.ArgumentParser(description='RL running machine')
parser.add_argument('--environment_name', metavar='ENV_CLASS', type=str, default='MVC', help='Class to use for the environment. Must be in the \'environment\' module')
parser.add_argument('--agent', metavar='AGENT_CLASS', default='Agent', type=str, help='Class to use for the agent. Must be in the \'agent\' module.')
parser.add_argument('--graph_type',metavar='GRAPH', default='barabasi_albert',help ='Type of graph to optimize')
parser.add_argument('--graph_nbr', type=int, default='201000', help='number of differente graph to generate for the training sample per epoch')#201000
parser.add_argument('--model', type=str, default='S2V_QN_1', help='model name')
parser.add_argument('--ngames', type=int, metavar='n', default='500', help='number of games to simulate')
parser.add_argument('--niter', type=int, metavar='n', default='1000', help='max number of iterations per game')
parser.add_argument('--epoch', type=int, metavar='nepoch',default=1, help="number of epochs")#change to 201
parser.add_argument('--lr',type=float, default=1e-4,help="learning rate")
parser.add_argument('--bs',type=int,default=128,help="minibatch experience size for training")
parser.add_argument('--n_step',type=int, default='5',help="n steps in RL")
parser.add_argument('--node_min', type=int, metavar='nnode',default='18', help="number of node in generated graphs")
parser.add_argument('--node_max', type=int, metavar='nnode',default='20', help="number of node in generated graphs")
parser.add_argument('--p',default=0.14,help="p, parameter in graph degree distribution")
parser.add_argument('--m',default=4,help="m, parameter in graph degree distribution")
parser.add_argument('--n_agents', type=int, metavar='nagent', default=None, help='batch run several agent at the same time')
parser.add_argument('--verbose', action='store_true', default=True, help='Display cumulative results at each step')
parser.add_argument('--trainedmodel', default=False, help='Using the trained model and skip the training process')

def main():
    n_test = 1000
    args = parser.parse_args()
    logging.info('Loading graph %s' % args.graph_type)
    graph_train = [None]*(args.graph_nbr)
    for graph_ in range(args.graph_nbr):
        graph_train[graph_]=graph.Graph(graph_type=args.graph_type, min_n=args.node_min, max_n=args.node_max, p=args.p,m=args.m,seed=120+graph_)
    graph_test = [None]*n_test
    for graph_ in range(n_test):
        graph_test[graph_]=graph.Graph(graph_type=args.graph_type, min_n=args.node_min, max_n=args.node_max, p=args.p,m=args.m,seed=1+graph_)
    logging.info('Loading agent and environment')
    agent_class = agent.Agent(args.model, args.lr, args.bs, args.n_step, args.environment_name, args.node_max)


    print("Running a single instance simulation...")
    my_runner = runner.Runner(graph_train, agent_class, args.verbose)
    my_runner.loop(args.graph_nbr, 1, args.niter)
    agent_class.save_model()
    my_runner.change_to_test(graph_test)
    my_runner.loop(n_test, 1, args.niter)

"""
to do some change
1)<runner.py> rule out the inactive points (not necessary selected)
2)<agent.py> <model.py> using the induced subgraph of active points instead of the entire graph to calculate
3)<model.py> using 2*n xv character instead of 1*n xv character
4)<agent.py> maintain the batch operation
5)matplotlib, other algorithms, prioritized replay and more.
"""






if __name__ == "__main__":
    main()
