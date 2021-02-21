#! /usr/bin/python3

import sys
import pennylane as qml
import numpy as np


def variational_ansatz(params, wires):
    """The variational ansatz circuit.

    Fill in the details of your ansatz between the # QHACK # comment markers. Your
    ansatz should produce an n-qubit state of the form

        a_0 |10...0> + a_1 |01..0> + ... + a_{n-2} |00...10> + a_{n-1} |00...01>

    where {a_i} are real-valued coefficients.

    Args:
         params (np.array): The variational parameters.
         wires (qml.Wires): The device wires that this circuit will run on.
    """
    n_qubits = len(wires)

    qml.RY(params[0], wires=wires[0:1])

    for i in range(1, n_qubits - 1):
        qml.CRY(params[i], wires=wires[i - 1: i + 1])

    for i in range(n_qubits - 1, 0, -1):
        qml.CNOT(wires=wires[i - 1: i + 1],)

    qml.PauliX(wires=wires[0:1])


def run_vqe(H):
    """Runs the variational quantum eigensolver on the problem Hamiltonian using the
    variational ansatz specified above.

    Fill in the missing parts between the # QHACK # markers below to run the VQE.

    Args:
        H (qml.Hamiltonian): The input Hamiltonian

    Returns:
        The ground state eng of the Hamiltonian.
    """
    eng = 0
    num_qubits = len(H.wires)
    dev = qml.device('default.qubit', wires=num_qubits)
    params = np.random.uniform(low=-np.pi / 2, high=np.pi / 2,
                               size=(num_qubits - 1))

    cost_func = qml.ExpvalCost(variational_ansatz, H, dev)

    opt = qml.GradientDescentOptimizer(stepsize=0.01)
    max_iterations = 500
    conv_tol = 1e-06

    for n in range(max_iterations):
        params, prev_eng = opt.step_and_cost(cost_func, params)
        eng = cost_func(params)
        conv = np.abs(eng - prev_eng)
        if conv <= conv_tol:
            break
    return eng


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
    ground_state_eng = run_vqe(H)
    print(f"{ground_state_eng:.6f}")
