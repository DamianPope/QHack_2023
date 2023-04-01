# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 19:48:58 2023

@author: dpope
"""

import json
import pennylane as qml
import pennylane.numpy as np
import scipy

def abs_dist(rho, sigma):
    """A function to compute the absolute value |rho - sigma|."""
    polar = scipy.linalg.polar(rho - sigma)
    return polar[1]

def word_dist(word):
    """A function which counts the non-identity operators in a Pauli word"""
    return sum(word[i] != "I" for i in range(len(word)))


# Produce the Pauli density for a given Pauli word and apply noise

def noisy_Pauli_density(word, lmbda):
    """
       A subcircuit which prepares a density matrix (I + P)/2**n for a given Pauli
       word P, and applies depolarizing noise to each qubit. Nothing is returned.

    Args:
            word (str): A Pauli word represented as a string with characters I,  X, Y and Z.
            lmbda (float): The probability of replacing a qubit with something random.
    """


    #
    #start code here
    #
    #convert a letter (i.e., a string consisting of a single character) to a matrix
    #that represents a Pauli operator
    def convert_letter_to_Pauli_Word_matrix(letter,i):   
        pw1 = qml.pauli.PauliWord({i:letter})
        pw1_matrix = pw1.to_mat(wire_order=[i])
        return pw1_matrix

    #create the density matrix for a Pauli word
    def create_noiseless_density_matrix():
        #initialize rho_Pauli_word_total
        rho_Pauli_word_total = np.identity(2)
        
        for j in range(len(word)):
            #create density_matrix for the j^th Pauli letter
            rho_single_qubit = convert_letter_to_Pauli_Word_matrix(word[j],j)
    
            #add normalization factor of 1/2
            rho_single_qubit = rho_single_qubit*0.5
           
            if j == 0:
                rho_Pauli_word_total = rho_single_qubit            
            else:
                rho_Pauli_word_total = np.kron(rho_Pauli_word_total,rho_single_qubit)        
        
        #create density matrix for identity part of density matrix
        rho_identity_part = np.identity(2**len(word))*2**(-len(word))
    
        #create the noiseless density matrix, rho_P
        density_matrix = rho_Pauli_word_total + rho_identity_part      
        return density_matrix

    dev = qml.device("default.mixed", wires=range(len(word)))

    @qml.qnode(dev)
    def noise():  
        qml.QubitDensityMatrix(create_noiseless_density_matrix(), wires=range(len(word)))
        #apply noise to noiseless density matrix
        for k in range(len(word)):
            qml.DepolarizingChannel(lmbda, wires=k) 
        return qml.density_matrix(wires=range(len(word)))

    return noise()
    
# Compute the trace distance from a noisy Pauli density to the maximally mixed density

def maxmix_trace_dist(word, lmbda):
    """
       A function compute the trace distance between a noisy density matrix, specified
       by a Pauli word, and the maximally mixed matrix.

    Args:
            word (str): A Pauli word represented as a string with characters I, X, Y and Z.
            lmbda (float): The probability of replacing a qubit with something random.

    Returns:
            float: The trace distance between two matrices encoding Pauli words.
    """


    # Put your code here #
    rho_Q = noisy_Pauli_density(word,lmbda)
    rho_max_mixed = np.identity(2**len(word))*(2**(-len(word)))
    trace_abs_dist = np.trace(abs_dist(rho_Q,rho_max_mixed))
    return 0.5*trace_abs_dist


def bound_verifier(word, lmbda):
    """
       A simple check function which verifies the trace distance from a noisy Pauli density
       to the maximally mixed matrix is bounded by (1 - lambda)^|P|.

    Args:
            word (str): A Pauli word represented as a string with characters I, X, Y and Z.
            lmbda (float): The probability of replacing a qubit with something random.

    Returns:
            float: The difference between (1 - lambda)^|P| and T(rho_P(lambda), rho_0).
    """


    # Put your code here #
    exponent = len(word) - word.count("I")
    diff = (1-lmbda)**exponent - maxmix_trace_dist(word,lmbda)
    return diff


# These functions are responsible for testing the solution.
def run(test_case_input: str) -> str:

    word, lmbda = json.loads(test_case_input)
    output = np.real(bound_verifier(word, lmbda))

    return str(output)


def check(solution_output: str, expected_output: str) -> None:

    solution_output = json.loads(solution_output)
    expected_output = json.loads(expected_output)
    assert np.allclose(
        solution_output, expected_output, rtol=1e-4
    ), "Your trace distance isn't quite right!"


test_cases = [['["XXI", 0.7]', '0.0877777777777777'], ['["XXIZ", 0.1]', '0.4035185185185055'], ['["YIZ", 0.3]', '0.30999999999999284'], ['["ZZZZZZZXXX", 0.1]', '0.22914458207245006']]
#test_cases = [['["XXI", 0.0]', '0.0877777777777777'], ['["XXIZ", 0.1]', '0.4035185185185055'], ['["YIZ", 0.3]', '0.30999999999999284'], ['["ZZZZZZZXXX", 0.1]', '0.22914458207245006']]


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