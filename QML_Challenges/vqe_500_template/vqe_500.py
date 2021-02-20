#! /usr/bin/python3

import sys
import pennylane as qml
import numpy as np

def find_excited_states(H):
    """
    Fill in the missing parts between the # QHACK # markers below. Implement
    a variational method that can find the three lowest energies of the provided
    Hamiltonian.

    Args:
        H (qml.Hamiltonian): The input Hamiltonian

    Returns:
        The lowest three eigenenergies of the Hamiltonian as a comma-separated string,
        sorted from smallest to largest.
    """

    energies = np.zeros(3)

    # QHACK #
    # import time
    # clock = time.time()

    def variational_ansatz(params, wires):
        """
        Args:
            params (np.ndarray): An array of floating-point numbers with size (n, 3),
                where n is the number of parameter sets required (this is determined by
                the problem Hamiltonian).
            wires (qml.Wires): The device wires this circuit will run on.
        """
        n_qubits = len(wires)
        n_rotations = len(params)

        if n_rotations > 1:
            n_layers = n_rotations // n_qubits
            n_extra_rots = n_rotations - n_layers * n_qubits

            # Alternating layers of unitary rotations on every qubit followed by a
            # ring cascade of CNOTs.
            for layer_idx in range(n_layers):
                layer_params = params[layer_idx *
                                      n_qubits: layer_idx * n_qubits + n_qubits, :]
                qml.broadcast(qml.Rot, wires, pattern="single",
                              parameters=layer_params)
                # qml.broadcast(qml.RY, wires, pattern="single",
                #               parameters=layer_params[:, 0])
                # qml.broadcast(qml.RZ, wires, pattern="single",
                #               parameters=layer_params[:, 1])
                qml.broadcast(qml.CNOT, wires, pattern="ring")

            if n_extra_rots > 0:
                # There may be "extra" parameter sets required for which it's not necessarily
                # to perform another full alternating cycle. Apply these to the qubits as needed.
                extra_params = params[-n_extra_rots:, :]
                extra_wires = wires[: n_qubits - 1 - n_extra_rots: -1]
                qml.broadcast(qml.Rot, extra_wires, pattern="single",
                              parameters=extra_params)
                # qml.broadcast(qml.RY, extra_wires, pattern="single",
                #               parameters=extra_params[:, 0])
                # qml.broadcast(qml.RZ, extra_wires, pattern="single",
                #               parameters=extra_params[:, 1])
        else:
            # For 1-qubit case, just a single rotation to the qubit
            qml.Rot(*params[0], wires=wires[0])

    # SET UP

    num_qubits = len(H.wires)

    dev = qml.device('default.qubit', wires=num_qubits)

    num_param_sets = num_qubits * 2

    opt = qml.AdamOptimizer(stepsize=0.4)
    # print('opt = qml.AdamOptimizer(stepsize=0.4)')

    #np.random.seed(0)
    params = np.random.uniform(low=-np.pi / 2, high=np.pi / 2,
                               size=(num_param_sets, 3))
    # params = np.array(
    #     [[-7.59242872e-01, -1.57079693e+00,  3.14161520e+00],
    #      [ 1.60832297e-01, -1.57169232e+00, -3.35098934e-05],
    #      [ 5.62685800e-01, -1.57074646e+00, -3.14152854e+00],
    #      [-2.39391497e-02, -3.14159454e+00, -2.37179610e-02],
    #      [ 1.57735149e+00,  3.17535756e+00, -1.56409095e+00],
    #      [ 8.89491742e-01, -1.56405184e-01,  2.24609453e+00]])


    ##### first opimization for GS

    cost_fn = qml.ExpvalCost(variational_ansatz, H, dev)

    max_iterations = 300
    rel_conv_tol = 1e-6

    for n in range(max_iterations):
        params, prev_cost = opt.step_and_cost(cost_fn, params)
        cost = cost_fn(params)
        conv = np.abs((cost - prev_cost) / cost)

        # DEBUG PRINT
        if n % 20 == 0:
            energies[0] = cost
            # print(f'Iteration = {n}, cost = {cost}, energies = ', energies,
            #       f'time {time.time() - clock:.0f}s')

        if (conv <= rel_conv_tol) and (n % 20 == 1):
            break
    energies[0] = cost

    ground_state = dev.state


    ##### second opimization for FES
    # params = np.random.uniform(low=-np.pi / 2, high=np.pi / 2,
    #                            size=(num_param_sets, 3))

    opt = qml.AdamOptimizer(stepsize=0.4)

    def costs(params):
        energy_cost = qml.ExpvalCost(variational_ansatz, H, dev)(params)
        state = dev.state
        gs_cost = abs(sum(a * np.conj(b)
                      for a, b in zip(state, ground_state))) ** 2
        return energy_cost, gs_cost
    cost_fn = lambda params: sum(costs(params))

    max_iterations = 300
    rel_conv_tol = 1e-6

    for n in range(max_iterations):
        params, prev_cost = opt.step_and_cost(cost_fn, params)
        cost = cost_fn(params)
        conv = np.abs((cost - prev_cost) / cost)

        # DEBUG PRINT
        if n % 20 == 0:
            energies[1] = costs(params)[0]
            # print(f'Iteration = {n}, cost = {cost}, energies = ', energies,
            #       f'time {time.time() - clock:.0f}s')

        if (conv <= rel_conv_tol) and (n % 20 == 1):
            break
    energies[1] = costs(params)[0]
    first_excited_state = dev.state

    # print('FES', first_excited_state)


    ##### third opimization for SES
    # params = np.random.uniform(low=-np.pi / 2, high=np.pi / 2,
    #                            size=(num_param_sets, 3))

    opt = qml.AdamOptimizer(stepsize=0.4)

    def costs(params):
        energy_cost = qml.ExpvalCost(variational_ansatz, H, dev)(params)
        state = dev.state
        gs_cost = abs(sum(a * np.conj(b)
                      for a, b in zip(state, ground_state))) ** 2
        fes_cost = abs(sum(a * np.conj(b)
                      for a, b in zip(state, first_excited_state))) ** 2
        return energy_cost, gs_cost, fes_cost
    cost_fn = lambda params: sum(costs(params))

    max_iterations = 300
    rel_conv_tol = 1e-6

    for n in range(max_iterations):
        params, prev_cost = opt.step_and_cost(cost_fn, params)
        cost = cost_fn(params)
        conv = np.abs((cost - prev_cost) / cost)

        # DEBUG PRINT
        if n % 20 == 0:
            energies[2] = costs(params)[0]
            # print(f'Iteration = {n}, cost = {cost}, energies = ', energies,
            #       f'time {time.time() - clock:.0f}s')

        if (conv <= rel_conv_tol) and (n % 20 == 1):
            break
    energies[2] = costs(params)[0]


    # print(f'tot time: ', time.time() - clock)

    # QHACK #

    return ",".join([str(E) for E in energies])


def pauli_token_to_operator(token):
    """
    DO NOT MODIFY anything in this function! It is used to judge your solution.

    Helper function to turn strings into qml operators.

    Args:
        token (str): A Pauli operator input in string form.

    Returns:
        A qml.Operator instance of the Pauli.
    """
    qubit_terms = []

    for term in token:
        # Special case of identity
        if term == "I":
            qubit_terms.append(qml.Identity(0))
        else:
            pauli, qubit_idx = term[0], term[1:]
            if pauli == "X":
                qubit_terms.append(qml.PauliX(int(qubit_idx)))
            elif pauli == "Y":
                qubit_terms.append(qml.PauliY(int(qubit_idx)))
            elif pauli == "Z":
                qubit_terms.append(qml.PauliZ(int(qubit_idx)))
            else:
                print("Invalid input.")

    full_term = qubit_terms[0]
    for term in qubit_terms[1:]:
        full_term = full_term @ term

    return full_term


def parse_hamiltonian_input(input_data):
    """
    DO NOT MODIFY anything in this function! It is used to judge your solution.

    Turns the contents of the input file into a Hamiltonian.

    Args:
        filename(str): Name of the input file that contains the Hamiltonian.

    Returns:
        qml.Hamiltonian object of the Hamiltonian specified in the file.
    """
    # Get the input
    coeffs = []
    pauli_terms = []

    # Go through line by line and build up the Hamiltonian
    for line in input_data.split("S"):
        line = line.strip()
        tokens = line.split(" ")

        # Parse coefficients
        sign, value = tokens[0], tokens[1]

        coeff = float(value)
        if sign == "-":
            coeff *= -1
        coeffs.append(coeff)

        # Parse Pauli component
        pauli = tokens[2:]
        pauli_terms.append(pauli_token_to_operator(pauli))

    return qml.Hamiltonian(coeffs, pauli_terms)


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block

    # Turn input to Hamiltonian
    H = parse_hamiltonian_input(sys.stdin.read())

    # Send Hamiltonian through VQE routine and output the solution
    lowest_three_energies = find_excited_states(H)
    print(lowest_three_energies)
