# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 03:08:53 2023

@author: dpope
"""

import json
import pennylane as qml
import pennylane.numpy as np

dev = qml.device('default.qubit', wires = 2)

@qml.qnode(dev)
def trotterize(alpha, beta, time, depth):
    """This quantum circuit implements the Trotterization of a Hamiltonian given by a linear combination
    of tensor products of X and Z Pauli gates.

    Args:
        alpha (float): The coefficient of the XX term in the Hamiltonian, as in the statement of the problem.
        beta (float): The coefficient of the YY term in the Hamiltonian, as in the statement of the problem.
        time (float): Time interval during which the quantum state evolves under the interactions specified by the Hamiltonian.
        depth (int): The Trotterization depth.

    Returns:
        (numpy.array): The probabilities of each measuring each computational basis state.
    """


    # Put your code here #
    #create the XX Hamiltonian
    coeffs_xx = [alpha]
    obs_xx = [qml.PauliX(0) @ qml.PauliX(1)]
    H_xx = qml.Hamiltonian(coeffs_xx,obs_xx)

    #create the ZZ Hamiltonian
    coeffs_zz = [beta]
    obs_zz = [qml.PauliZ(0) @ qml.PauliZ(1)]
    H_zz = qml.Hamiltonian(coeffs_zz,obs_zz)

    #execute evolution under XX term & ZZ term for "depth" number of times
    for i in range(depth):
        qml.ops.Evolution(H_xx,-time/depth)
        qml.ops.Evolution(H_zz,-time/depth)
        
    return qml.probs(wires=[0,1])


# These functions are responsible for testing the solution.
def run(test_case_input: str) -> str:
    dev = qml.device("default.qubit", wires=2)
    ins = json.loads(test_case_input)
    output = list(trotterize(*ins).numpy())

    return str(output)

def check(solution_output: str, expected_output: str) -> None:
    solution_output = json.loads(solution_output)
    expected_output = json.loads(expected_output)
    assert np.allclose(
        solution_output, expected_output, rtol=1e-4
    ), "Your circuit does not give the correct probabilities."

    tape = trotterize.qtape

    names = [op.name for op in tape.operations]
    
    assert names.count('ApproxTimeEvolution') == 0, "Your circuit is using the built-in PennyLane Trotterization"
    assert names.count('QubitUnitary') == 0, "Can't use custom-built gates!"


test_cases = [['[0.5,0.8,0.2,1]', '[0.99003329, 0, 0, 0.00996671]'], ['[0.9,1.0,0.4,2]', '[0.87590286, 0, 0, 0.12409714]']]

for i, (input_, expected_output) in enumerate(test_cases):
    print(f"Running test case {i} with input '{input_}'...")

    try:
        output = run(input_)

    except Exception as exc:
        print(f"Runtime Error. {exc}")

    else:
        if message := check(output, expected_output):
            print(f"Wrong Answer. Have: '{output}'. Want: '{expected_output}'.")

        else:
            print("Correct!")