#! /usr/bin/python3
import sys
import pennylane as qml
from pennylane import numpy as np

# DO NOT MODIFY any of these parameters
a = 0.7
b = -0.3
dev = qml.device("default.qubit", wires=3)


def natural_gradient(params):
    """Calculate the natural gradient of the qnode() cost function.

    The code you write for this challenge should be completely contained within this function
    between the # QHACK # comment markers.

    You should evaluate the metric tensor and the gradient of the QNode, and then combine these
    together using the natural gradient definition. The natural gradient should be returned as a
    NumPy array.

    The metric tensor should be evaluated using the equation provided in the problem text. Hint:
    you will need to define a new QNode that returns the quantum state before measurement.

    Args:
        params (np.ndarray): Input parameters, of dimension 6

    Returns:
        np.ndarray: The natural gradient evaluated at the input parameters, of dimension 6
    """

    natural_grad = np.zeros(6)

    # QHACK #
    F = np.zeros([6, 6], dtype=np.float64)
    gradient = np.zeros_like(params)
    
    def parameter_shift_term(qnode_, params_, h):
        shifted_ = params_.copy()
        shifted_[h] += np.pi/2
        forward = qnode_(shifted_)  # forward evaluation

        shifted_[h] -= np.pi
        backward = qnode_(shifted_) # backward evaluation

        return 0.5 * (forward - backward)
    
    for i in range(len(params)):
        gradient[i] = parameter_shift_term(qnode, params, i)
        
    @qml.qnode(dev)
    def qnode_state(params_):
        """A PennyLane QNode that pairs the variational_circuit with an state vector.
        
        """
        variational_circuit(params_)
        return qml.state()
    
    state_0 = qnode_state(params)
    
    def projective_product(state_0_, state_x):
        return (np.absolute((np.conj(state_0_)@state_x)).item())**2
    
    for i in range(len(params)):
        for j in range(i,len(params)):
            shifted = params.copy()
            shifted[i] += np.pi/2
            shifted[j] += np.pi/2
            F[i][j] -= projective_product(state_0, qnode_state(shifted))
            
            shifted[j] -= np.pi
            F[i][j] += projective_product(state_0, qnode_state(shifted))         

            shifted[i] -= np.pi
            shifted[j] += np.pi
            F[i][j] += projective_product(state_0, qnode_state(shifted)) 
            
            shifted[j] -= np.pi
            F[i][j] -= projective_product(state_0, qnode_state(shifted)) 
            
            F[i][j] /= 8
            
            F[j][i] = F[i][j]
            
    inv_F = np.linalg.inv(F)

    
    natural_grad = inv_F@gradient
    
    #FF = qml.metric_tensor(qnode_state)
    #print(FF(params))
    #print(F)
    #print(inv_F)
        
    

    # QHACK #

    return natural_grad


def non_parametrized_layer():
    """A layer of fixed quantum gates.

    # DO NOT MODIFY anything in this function! It is used to judge your solution.
    """
    qml.RX(a, wires=0)
    qml.RX(b, wires=1)
    qml.RX(a, wires=1)
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    qml.RZ(a, wires=0)
    qml.Hadamard(wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RZ(b, wires=1)
    qml.Hadamard(wires=0)


def variational_circuit(params):
    """A layered variational circuit composed of two parametrized layers of single qubit rotations
    interleaved with non-parameterized layers of fixed quantum gates specified by
    ``non_parametrized_layer``.

    The first parametrized layer uses the first three parameters of ``params``, while the second
    layer uses the final three parameters.

    # DO NOT MODIFY anything in this function! It is used to judge your solution.
    """
    non_parametrized_layer()
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=1)
    qml.RZ(params[2], wires=2)
    non_parametrized_layer()
    qml.RX(params[3], wires=0)
    qml.RY(params[4], wires=1)
    qml.RZ(params[5], wires=2)


@qml.qnode(dev)
def qnode(params):
    """A PennyLane QNode that pairs the variational_circuit with an expectation value
    measurement.

    # DO NOT MODIFY anything in this function! It is used to judge your solution.
    """
    variational_circuit(params)
    return qml.expval(qml.PauliX(1))


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block

    # Load and process inputs
    params = sys.stdin.read()
    params = params.split(",")
    params = np.array(params, float)

    updated_params = natural_gradient(params)

    print(*updated_params, sep=",")
