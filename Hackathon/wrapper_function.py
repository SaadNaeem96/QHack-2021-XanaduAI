import pennylane as qml
import numpy as np
from qcircuit_picker import qcircuit_picker

def wrapper_function(nntype, test_and_train_data, n_qubits, cost, n_layers=1, opt=None, stepsize=0.01, n_epoch=10, batch_size = 10, labels=None, dev=None):

    if not isinstance(n_qubits, int):
        print('n_qubits must be int')
        return 'error'
    if n_qubits < 1:
        print('n_qubits must greater than 0')
        return 'error'

    if not isinstance(n_epoch, int):
        print('n_epoch must be int')
        return 'error'
    if n_epoch < 0:
        print('n_epoch must greater than (equal to) 0')
        return 'error'
    
    if not isinstance(batch_size, int):
        print('batch_size must be int')
        return 'error'
    if batch_size < 1:
        print('batch_size must greater than 0')
        return 'error'

    """
    test_and_train_data must be a string: train_inputXXXtrain_outputXXXtest_input
    train_input's and test_input's instances must be connected using "S": train_data=td1Std2Std3S...StdN
    """

    # The project is under development. Not all nntypes have been implemented yet.
    developed = ['ann', 'cnn']
    nntypes = ['ann', 'cnn', 'anntf', 'gan']
    if nntype not in developed:
        if nntype in nntypes:
            print(f"{nntype} wrapper under development...")
            return 'error'
        else:
            print(f"We provide only the following nntypes: {', '.join([t for t in nntypes])}")
            return 'error'

    optimizers = ['Adam', 'Adagrad', 'GradienDescent', 'Momentum', 'NesterovMomentum', 'QNG', 'RMSProp', 'Rotosolve', 'Rotoselect']
    if opt == None or opt == 'Adam':
        opt = qml.AdamOptimizer(stepsize)
    elif opt == 'Adagrad':
        opt = qml.AdagradOptimizer(stepsize)
    elif opt == 'GradienDescent':
        opt = qml.GradientDescentOptimizer(stepsize)
    elif opt == 'Momentum':
        opt = qml.MomentumOptimizer(stepsize)
    elif opt == 'NesterovMomentum':
        opt = qml.NesterovMomentumOptimizer(stepsize)
    elif opt == 'QNG':
        opt = qml.QNGOptimizer(stepsize)
    elif opt == 'RMSProp':
        opt = qml.RMSPropOptimizer(stepsize)
    elif opt == 'Rotosolve':
        opt = qml.RotosolveOptimizer()
    elif opt == 'Rotoselect':
        opt = qml.RotoselectOptimizer()
    else:
        print(f"Only pennylane's: {', '.join([o for o in optimizers])} Optimizers are provided.")
        return "error"
    
    if labels==None:
        labels = {}
        for i in ranga(n_qubits):
            labels[i] = i

    if dev == None:
        dev = qml.device("default.qubit", wires=n_qubits)
          

    # helper functions, ref: circuit_training_500, QHack 2021
    def array_to_concatenated_string(array):
        """DO NOT MODIFY THIS FUNCTION.
        Turns an array of integers into a concatenated string of integers
        separated by commas. (Inverse of concatenated_string_to_array).
        """
        return ",".join(str(x) for x in array)


    def concatenated_string_to_array(string):
        """DO NOT MODIFY THIS FUNCTION.
        Turns a concatenated string of integers separated by commas into
        an array of integers. (Inverse of array_to_concatenated_string).
        """
        return np.array([int(x) for x in string.split(",")])

    
    @qml.qnode(dev)
    def qml_circuit(nntype, params, sample, wires, depth):
        qcircuit_picker(nntype, params, sample, wires, depth)
    
        # currently qcircuit_picker() does not return any measurement
        # we take measurements after calling qcircuit_picker()
        # should we revise qcircuit_picker() and make it return
        # measurements?
        ret = []
        for i in range(len(wires)):
            ret.append(qml.expval(qml.PauliZ(wires=i)))
        return ret

    def parse_input(giant_string):
        """DO NOT MODIFY THIS FUNCTION.
        Parse the input data into 3 arrays: the training data, training labels,
        and testing data.
        Dimensions of the input data are:
          - X_train: (250, 3)
          - Y_train: (250,)
          - X_test:  (50, 3)
        """
        X_train_part, Y_train_part, X_test_part = giant_string.split("XXX")

        X_train_row_strings = X_train_part.split("S")
        X_train_rows = [[float(x) for x in row.split(",")] for row in X_train_row_strings]
        X_train = np.array(X_train_rows)

        Y_train = concatenated_string_to_array(Y_train_part)

        X_test_row_strings = X_test_part.split("S")
        X_test_rows = [[float(x) for x in row.split(",")] for row in X_test_row_strings]
        X_test = np.array(X_test_rows)

        return X_train, Y_train, X_test

    def iterate_minibatches(inputs, targets, batch_size):
        """
        A generator for batches of the input data
        Args:
        inputs (array[float]): input data
        targets (array[float]): targets
        Returns:
        inputs (array[float]): one batch of input data of length `batch_size`
        targets (array[float]): one batch of targets of length `batch_size`
        """
        for start_idx in range(0, inputs.shape[0] - batch_size + 1, batch_size):
            idxs = slice(start_idx, start_idx + batch_size)
            yield inputs[idxs], targets[idxs]


    X_train, Y_train, X_test = parse_input(test_and_train_data)

    params = [np.random.uniform(0, np.pi) for _ in range(n_layers*n_qubits*3)]
    for Xbatch, ybatch in iterate_minibatches(X_train, Y_train, batch_size=batch_size):
        for it in range(n_epoch):
            for Xbatch, ybatch in iterate_minibatches(X_train, Y_train, batch_size=batch_size):
                params = opt.step(lambda v: cost(nntype, v, Xbatch, ybatch, qml_circuit, n_qubits, n_layers), params)

    predictions = []
    for x in X_test:
        pred = qml_circuit(nntype, params, x, range(n_qubits), n_layers)
        predictions.append(labels[np.argmax(pred)])
    return predictions