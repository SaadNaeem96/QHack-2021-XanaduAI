#! /usr/bin/python3

import sys
import pennylane as qml
import numpy as np


def gradient_200(weights, dev):
    r"""This function must compute the gradient *and* the Hessian of the variational
    circuit using the parameter-shift rule, using exactly 51 device executions.
    The code you write for this challenge should be completely contained within
    this function between the # QHACK # comment markers.

    Args:
        weights (array): An array of floating-point numbers with size (5,).
        dev (Device): a PennyLane device for quantum circuit execution.

    Returns:
        tuple[array, array]: This function returns a tuple (gradient, hessian).

            * gradient is a real NumPy array of size (5,).

            * hessian is a real NumPy array of size (5, 5).
    """

    @qml.qnode(dev, interface=None)
    def circuit(w):
        for i in range(3):
            qml.RX(w[i], wires=i)

        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.CNOT(wires=[2, 0])

        qml.RY(w[3], wires=1)

        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.CNOT(wires=[2, 0])

        qml.RX(w[4], wires=2)

        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(2))

    gradient = np.zeros([5], dtype=np.float64)
    hessian = np.zeros([5, 5], dtype=np.float64)

    # QHACK #
    def no_shift_term(qnode, params):
        f = qnode(params.copy())
        return f
    
    def single_forward(qnode, params):
        shifted = params.copy()
        FORWARD = []
        for i in range(len(params)):
            shifted[i] += np.pi/2
            FORWARD.append(qnode(shifted))
            shifted[i] -= np.pi/2
        return FORWARD
 

    def single_backward(qnode, params):
        shifted = params.copy()
        BACKWARD = []
        for i in range(len(params)):
            shifted[i] -= np.pi/2
            BACKWARD.append(qnode(shifted))
            shifted[i] += np.pi/2
        return BACKWARD
  
    
    
    def parameter_shift_term(qnode, params, i, j):
        
        shifted = params.copy()
        shifted[j] += np.pi/2
        shifted[i] += np.pi/2
        forward = qnode(shifted)  # forward evaluation
        
        shifted[i] -= np.pi
        forward2 = qnode(shifted)  # forward2 evaluation  
        
        shifted[j] -= np.pi
        shifted[i] += np.pi
        backward2 = qnode(shifted)  # backward2 evaluation
        
        shifted[i] -= np.pi
        backward = qnode(shifted) # backward evaluation
            

        return 0.25 * (forward - forward2 - backward2 + backward)
    
    F = no_shift_term(circuit, weights)
    FORWARD = single_forward(circuit, weights)
    BACKWARD = single_backward(circuit, weights)
    
        
    for i in range(len(weights)):
        for j in range(i, len(weights)):
            if i != j:
                hessian[i][j] = parameter_shift_term(circuit, weights, i, j)
                hessian[j][i] = hessian[i][j]
            else:
                hessian[i][j] = (FORWARD[i]-2*F+BACKWARD[i])/2
                gradient[i] = (FORWARD[i]-BACKWARD[i])/2
                
            

    # QHACK #

    return gradient, hessian, circuit.diff_options["method"]


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    weights = sys.stdin.read()
    weights = weights.split(",")
    weights = np.array(weights, float)

    dev = qml.device("default.qubit", wires=3)
    gradient, hessian, diff_method = gradient_200(weights, dev)

    print(
        *np.round(gradient, 10),
        *np.round(hessian.flatten(), 10),
        dev.num_executions,
        diff_method,
        sep=","
    )
