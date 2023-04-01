# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 02:38:11 2023

@author: dpope
"""

import json
import pennylane as qml
import pennylane.numpy as np

dev = qml.device("default.qubit", wires=3)


@qml.qnode(dev)
def model(alpha):
    """In this qnode you will define your model in such a way that there is a single 
    parameter alpha which returns each of the basic states.

    Args:
        alpha (float): The only parameter of the model.

    Returns:
        (numpy.tensor): The probability vector of the resulting quantum state.
    """
    # Put your code here #

    '''
    This solution is based on the fact that we can access all 8 computational basis states
    using the quantum Fourier transform. It works by:
        1. generating one of the 8 Fourier states for 3 qubits
        2. Applying the inverse quantum Fourier transform to go from the alpha-th Fourier state 
        to the alpha-th computational basis state
    '''

    #generate the alpha-th (i.e., 0th, 1st, 2nd, 3rd etc.) Fourier state for 3 quits in a bitwise fashion
    for i in range(3):
        qml.Hadamard(wires=i)
        qml.PhaseShift((2*np.pi/8)*alpha*(2**(2-i)),wires=i)

    #Apply the inverse QFT to go from a Fourier state to a computational basis state    
    qml.adjoint(qml.QFT(wires=[0,1,2]))

    return qml.probs(wires=range(3))


def generate_coefficients():
    """This function must return a list of 8 different values of the parameter that
    generate the states 000, 001, 010, ..., 111, respectively, with your ansatz.

    Returns:
        (list(int)): A list of eight real numbers.
    """
    # Put your code here #
    return [0,1,2,3,4,5,6,7]


# These functions are responsible for testing the solution.
def run(test_case_input: str) -> str:
    return None

def check(solution_output, expected_output: str) -> None:
    coefs = generate_coefficients()
    output = np.array([model(c) for c in coefs])
    epsilon = 0.001

    for i in range(len(coefs)):
        assert np.isclose(output[i][i], 1)

    def is_continuous(function, point):
        limit = calculate_limit(function, point)

        if limit is not None and sum(abs(limit - function(point))) < epsilon:
            return True
        else:
            return False

    def is_continuous_in_interval(function, interval):
        for point in interval:
            if not is_continuous(function, point):
                return False
        return True

    def calculate_limit(function, point):
        x_values = [point - epsilon, point, point + epsilon]
        y_values = [function(x) for x in x_values]
        average = sum(y_values) / len(y_values)

        return average

    assert is_continuous_in_interval(model, np.arange(0,10,0.001))

    for coef in coefs:
        assert coef >= 0 and coef <= 10


test_cases = [['No input', 'No output']]

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