#! /usr/bin/python3
"""
Created on Fri Feb 26 00:32:18 2021

@author: Joseph
"""

# ToDos:
# 1. sanity check of qcircuit_picker() input parameters
#    for example, how do we handle the case when the length of features is not 3?
# 2. currently AnnLayer() doesn't return anything
#    should we make it return any measurement, or other attributes such as
#    number of gates (or more generally, resource consumption)?
# 3. do we keep simulator and interface in the list of input parameters?
# 4. play with the for loops in AnnLayer(). For each layer, now we are 
#    first building the ROT gates that embed features and then the
#    ROT gates that embed params. What if, for each layer, we pick a wire
#    and build a ROT gate that embeds the features first and then a
#    ROT gate that embeds params. Then we repeat this process for another
#    wire...would the accuracy drop? why? 
    
#import sys
import pennylane as qml
#import numpy as np

def qcircuit_picker(nntype, params, features, wires, depth, simulator=None, interface=None, seed=None):
    
    if simulator is None: 
        simulator = "default.qubit"
    if interface is None:
        interface = "torch"
        
    if nntype == "ann":
        return AnnLayer(params, features, wires, depth)
    elif nntype == "cnn":
        return "CNN quantum circuit picker is under development..."
    elif nntype == "anntf":
        return "ANN quantum circuit picker for transfer learning is under development..."
    elif nntype == "gan":
        return "GAN quantum circuit picker is under development..."
    elif nntype == "gnn":
        return "GNN quantum circuit picker is under development..."
    
# a layer in an ANN
# This layer instantiates quantum gates
# For each wire, a ROT gate embedds the feature (i.e. the data sample of length 3) followed by
# another ROT gate that takes the params (i.e. the weights to optimize) as input
# The features could be the output of a preceeding layer (classical or quantum)

def AnnLayer(params, features, wires, depth):
    
        """A variational quantum circuit representing the Universal classifier.

        Args:
            params (array[float]): array of parameters
            features (array[float]): single input vector
            wires: wire indices
            depth (int): number of layers
        Returns: now void
            
        """

        f = features.flatten()

        plist = list(params)

        for layer in range(depth):

            for w in range(len(wires)):

                qml.Rot(*f, wires=w)

            for w in range(len(wires)):
                qml.Rot(plist.pop(), plist.pop(), plist.pop(), wires=w)
 