# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 02:59:53 2023

@author: dpope
"""

import json
import pennylane as qml
import pennylane.numpy as np

def U_psi(theta):
    """
    Quantum function that generates |psi>, Zenda's state wants to send to Reece.

    Args:
        theta (float): Parameter that generates the state.

    """
    qml.Hadamard(wires = 0)
    qml.CRX(theta, wires = [0,1])
    qml.CRZ(theta, wires = [0,1])

def is_unsafe(alpha, beta, epsilon):
    """
    Boolean function that we will use to know if a set of parameters is unsafe.

    Args:
        alpha (float): parameter used to encode the state.
        beta (float): parameter used to encode the state.
        epsilon (float): unsafe-tolerance.

    Returns:
        (bool): 'True' if alpha and beta are epsilon-unsafe coefficients. 'False' in the other case.

    """
    # Put your code here #
    #create 2 separate devices so that they're easily distinguishable
    dev = qml.device("default.qubit", wires=2)
    dev2 = qml.device("default.qubit", wires=2)

    #Define a quantum function/QNODE that generates psi_U(theta)
    #note that it returns a qml.state() 

    @qml.qnode(dev)
    def circuit_U(theta):
        qml.BasisState(np.array([0, 0]), wires=range(2))
        qml.Hadamard(wires=0)
        qml.CRX(theta,wires=[0,1])
        qml.CRZ(theta,wires=[0,1])
        return qml.state()

    #Define a function that generates the encoded state, (R_x(alpha) R_z(beta))^2  U(theta) |0 0>
    #Note that it returns a qml.state() 
    
    @qml.qnode(dev2)
    def circuit_encoded_state(theta,alpha,beta):    
        qml.BasisState(np.array([0, 0]), wires=range(2))        

        #generate U(theta)
        qml.Hadamard(wires=0)
        qml.CRX(theta,wires=[0,1])
        qml.CRZ(theta,wires=[0,1])
    
        #implement state encoding
        qml.RZ(alpha,wires=0)
        qml.RX(beta,wires=0)
        qml.RZ(alpha,wires=1)
        qml.RX(beta,wires=1)
        
        return qml.state()

    #
    #set up gradient descent optimization scheme & execute it
    #
    
    #define a cost function
    def cost_fn(theta):
        state_0 = circuit_U(theta)
        state_1 = circuit_encoded_state(theta,alpha,beta)        
        temp = 0.0
        
        #calculate <state_0 | state_1 >
        for i in range(4):
            temp += np.conjugate(state_0[i])*state_1[i]
            
        fidelity = (np.abs(temp))**2
        local_fidelity_error = 1 - fidelity
    
        #We want to see what the smallest possible "fidelity error" is for any value of theta & a fixed alpha & beta values encoding scheme
        return local_fidelity_error
    
    opt = qml.GradientDescentOptimizer(stepsize=0.2)
    
    #parameter to be optimized
    theta = np.array(7.1, requires_grad=True)
    
    fidelity_error = cost_fn(theta)
    max_iterations = 100
    conv_tol = 1e-08
    
    for n in range(max_iterations):
        #Note that step_and_cost method calculates cost_fn first & then does an optimization step
        theta, old_cost_fn_value = opt.step_and_cost(cost_fn, theta)
    
        #store value of fidelity_error associated with the most recent theta value
        fidelity_error = cost_fn(theta)
    
        conv = np.abs(old_cost_fn_value - fidelity_error)   
        if conv <= conv_tol:
            break

    if (fidelity_error < epsilon):
        #alpha & beta are epsilon *unsafe*
        return True
    else:
        return False

# These functions are responsible for testing the solution.
def run(test_case_input: str) -> str:
    ins = json.loads(test_case_input)
    output = is_unsafe(*ins)
    return str(output)

def check(solution_output: str, expected_output: str) -> None:
    
    def bool_to_int(string):
        if string == "True":
            return 1
        return 0

    solution_output = bool_to_int(solution_output)
    expected_output = bool_to_int(expected_output)
    assert solution_output == expected_output, "The solution is not correct."


test_cases = [['[0.1, 0.2, 0.3]', 'True'], ['[1.1, 1.2, 0.3]', 'False'], ['[1.1, 1.2, 0.4]', 'True'], ['[0.5, 1.9, 0.7]', 'True']]

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