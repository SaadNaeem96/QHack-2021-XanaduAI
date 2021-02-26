import pennylane as qml
import numpy as np
from qcircuit_picker import qcircuit_picker
from wrapper_function import wrapper_function


with open('test_input.in', 'r') as f:
    input = f.readlines()[0]

def cost(nntype, params, x, y, qcircuit, num_qubits, depth):
    batch_loss = []
    label_vecs = {
        1: [1, 0, 0],
        0: [0, 1, 0],
        -1: [0, 0, 1]
    }
    for i in range(len(x)):
        # can we simplify how x is used by qcircuitann()?
        sample = np.zeros((1,3))
        sample[0,0] = x[i][0]
        sample[0,1] = x[i][1]
        sample[0,2] = x[i][2]
        f = qcircuit(nntype, params, sample, range(num_qubits), depth)
        label = label_vecs[y[i]]
        s = 0
        for e in range(3):
            s += abs(f[e] - label[e])**2
        batch_loss.append(s)
    m = 0
    for s in batch_loss:
        m += s
    return m / len(x)

print(wrapper_function('ann', input, 3, cost, labels={0: 1, 1: 0, 2: -1}))