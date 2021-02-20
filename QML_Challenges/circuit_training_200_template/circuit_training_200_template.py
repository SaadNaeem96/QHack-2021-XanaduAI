#! /usr/bin/python3
import json
import sys
import networkx as nx
import numpy as np
import pennylane as qml


# DO NOT MODIFY any of these parameters
NODES = 6
N_LAYERS = 10


def find_max_independent_set(graph, params):
    """Find the maximum independent set of an input graph given some optimized QAOA parameters.

    The code you write for this challenge should be completely contained within this function
    between the # QHACK # comment markers. You should create a device, set up the QAOA ansatz circuit
    and measure the probabilities of that circuit using the given optimized parameters. Your next
    step will be to analyze the probabilities and determine the maximum independent set of the
    graph. Return the maximum independent set as an ordered list of nodes.

    Args:
        graph (nx.Graph): A NetworkX graph
        params (np.ndarray): Optimized QAOA parameters of shape (2, 10)

    Returns:
        list[int]: the maximum independent set, specified as a list of nodes in ascending order
    """

    max_ind_set = []

    # QHACK #
    WIRES = range(NODES)
    dev = qml.device('default.qubit', wires= NODES)
    qaoa = qml.qaoa
    cost_h, mixer_h = qaoa.max_independent_set(graph)
    
    def qaoa_layer(gamma, alpha):
        qaoa.cost_layer(gamma, cost_h)
        qaoa.mixer_layer(alpha, mixer_h)
    
    @qml.qnode(dev)
    def circuit(params):
        #for w in range(NODES):
            #qml.Hadamard(wires=w)
            
        qml.layer(qaoa_layer,N_LAYERS,params[0],params[1])
        return qml.probs(wires = [0,1,2,3,4,5])
    
    prob = circuit(params)
    
    for i in range(len(prob)):
        if prob[i] == max(prob):
            answer = i
            
    #max_ind_set.append(prob)
    #max_ind_set.append(max(prob))
    
    unordered_max_set =[]
    
    for i in WIRES:
        if answer%2 == 1:
            unordered_max_set.append(NODES-i-1)
        answer =int(answer/2)
        
    L = len(unordered_max_set)
        
    for m in range(L):
        max_ind_set.append(unordered_max_set[L-m-1])
    
    # QHACK #

    return max_ind_set


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block

    # Load and process input
    graph_string = sys.stdin.read()
    graph_data = json.loads(graph_string)

    params = np.array(graph_data.pop("params"))
    graph = nx.json_graph.adjacency_graph(graph_data)

    max_independent_set = find_max_independent_set(graph, params)

    print(max_independent_set)
