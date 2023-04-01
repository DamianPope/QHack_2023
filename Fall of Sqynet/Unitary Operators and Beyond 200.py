# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 03:10:03 2023

@author: dpope
"""

import json
import pennylane as qml
import pennylane.numpy as np

def W(alpha, beta):
    """ This function returns the matrix W in terms of
    the coefficients alpha and beta

    Args:
        - alpha (float): The prefactor alpha of U in the linear combination, as in the
        challenge statement.
        - beta (float): The prefactor beta of V in the linear combination, as in the
        challenge statement.
    Returns 
        -(numpy.ndarray): A 2x2 matrix representing the operator W,
        as defined in the challenge statement
    """


    # Put your code here #
    #create & return W
    W_matrix = np.array([[np.sqrt(alpha),-np.sqrt(beta)],[np.sqrt(beta),np.sqrt(alpha)]])
    W_matrix = W_matrix*1/np.sqrt(alpha+beta)
    return W_matrix

dev = qml.device('default.qubit', wires = 2)

@qml.qnode(dev)
def linear_combination(U, V,  alpha, beta):
    """This circuit implements the circuit that probabilistically calculates the linear combination 
    of the unitaries.

    Args:
        - U (list(list(float))): A 2x2 matrix representing the single-qubit unitary operator U.
        - V (list(list(float))): A 2x2 matrix representing the single-qubit unitary operator U.
        - alpha (float): The prefactor alpha of U in the linear combination, as above.
        - beta (float): The prefactor beta of V in the linear combination, as above.

    Returns:
        -(numpy.tensor): Probabilities of measuring the computational
        basis states on the auxiliary wire. 
    """


    # Put your code here #
    qml.QubitUnitary(W(alpha,beta),wires=0)
    qml.ControlledQubitUnitary(U, control_wires=[0], wires=1)
    qml.ControlledQubitUnitary(V, control_wires=[0], wires=1)

    #build W_dag, the adjoint of W. 
    #Note that we don't need to swap the two diagonal elements & as they're equal
    W_dag = W(alpha,beta)
    W_dag[0,1] =  -W_dag[0,1]
    W_dag[1,0] =  -W_dag[1,0]
    qml.QubitUnitary(W_dag,wires=0)

    return qml.probs(wires=0)



# These functions are responsible for testing the solution.

def run(test_case_input: str) -> str:
    dev = qml.device('default.qubit', wires = 2)
    ins = json.loads(test_case_input)
    output = linear_combination(*ins)[0].numpy()

    return str(output)

def check(solution_output: str, expected_output: str) -> None:
    solution_output = json.loads(solution_output)
    expected_output = json.loads(expected_output)
    assert np.allclose(
        solution_output, expected_output, rtol=1e-4
    ), "Your circuit doesn't look quite right "


test_cases = [['[[[ 0.70710678,  0.70710678], [ 0.70710678, -0.70710678]],[[1, 0], [0, -1]], 1, 3]', '0.8901650422902458']]

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