# Technical Report: Implementing Grover Adaptive Search for QUBO Problems in Cirq

This document outlines the prompts for a coding agent to implement Grover Adaptive Search (GAS) for solving Quadratic Unconstrained Binary Optimization (QUBO) problems using the Cirq library.

## Prompt 1: Project Setup and QUBO Definition

**Goal:** Initialize the Python project and define the structure for our QUBO problem.

**Prompt:**

"Create a Python script named `gas_for_qubo.py`. In this script, import `cirq` and `numpy`. Define a sample QUBO problem as a NumPy 2D array. For now, you can use a simple 2x2 matrix like:

```python
import cirq
import numpy as np

# Define the QUBO matrix Q
qubo_matrix = np.array([
    [-1, 2],
    [0, -1]
])

print('QUBO Matrix defined:')
print(qubo_matrix)
```

This will serve as the input for our GAS implementation."

## Prompt 2: The Oracle for the QUBO Problem

**Goal:** Create a quantum oracle that encodes the QUBO problem. The oracle will mark the states that represent solutions with a cost below a certain threshold.

**Prompt:**

"Now, create a function `create_qubo_oracle(qubits, qubo_matrix, threshold)` that constructs a Cirq oracle.

This function should:
1.  Take a list of `qubits`, the `qubo_matrix`, and a `threshold` as input.
2.  The oracle should apply a phase flip (Z gate) to the states `|x>` for which the QUBO cost `x^T * Q * x` is less than the `threshold`.
3.  To implement this, you will need to create a circuit that calculates the cost `x^T * Q * x` into an ancilla register, compares it with the threshold, and then applies the phase flip if the condition is met.
4.  Return the oracle as a `cirq.Gate` or a list of `cirq.Operation`s."

## Prompt 3: The Grover Diffusion Operator

**Goal:** Implement the standard Grover diffusion operator (amplitude amplification).

**Prompt:**

"Implement a function `create_diffusion_operator(qubits)` that creates the Grover diffusion operator for a given set of `qubits`.

The diffusion operator should:
1.  Apply a Hadamard gate to each qubit.
2.  Apply a multi-controlled Z gate (or equivalent using X and CNOTs) that flips the phase of the `|00...0>` state.
3.  Apply a Hadamard gate to each qubit again.
4.  Return the diffusion operator as a `cirq.Gate` or a list of `cirq.Operation`s."

## Prompt 4: The Grover Adaptive Search Algorithm

**Goal:** Implement the main GAS loop that uses the oracle and diffusion operator to find the best solution.

**Prompt:**

"Now, let's tie everything together in a function `grover_adaptive_search(qubo_matrix)`.

This function should:
1.  Initialize the number of qubits based on the size of the `qubo_matrix`.
2.  Start with an initial `threshold` for the QUBO cost. A good starting point could be the maximum possible cost or a sufficiently large number.
3.  Loop until a satisfactory solution is found or a maximum number of iterations is reached:
    a. Create the oracle for the current `threshold` using `create_qubo_oracle`.
    b. Create the diffusion operator using `create_diffusion_operator`.
    c. Construct the full Grover circuit by applying the oracle and the diffusion operator a number of times (you can start with a single iteration).
    d. Simulate the circuit and measure the results.
    e. Find the state with the highest probability and calculate its QUBO cost.
    f. If the cost is an improvement, update the `threshold` to this new best cost.
    g. If no improvement is found after a certain number of Grover iterations, you might need to adjust the number of iterations or stop.
4.  Return the best solution found and its cost."

## Prompt 5: Example and Testing

**Goal:** Test the implementation with a sample QUBO problem and verify the results.

**Prompt:**

"Finally, add a `main` block to your script to test the implementation.

In the `main` block:
1.  Define a more complex QUBO matrix (e.g., a 4x4 matrix for a MAX-CUT problem on a square graph).
2.  Call `grover_adaptive_search` with this matrix.
3.  Print the best solution (the bitstring) and its corresponding QUBO cost.
4.  Compare the result with the classically computed optimal solution to verify the correctness of your implementation."

## Prompt 6: Refinement and Documentation

**Goal:** Improve the code quality by adding documentation and structuring it into a class.

**Prompt:**

"To make the code more reusable and understandable, please perform the following refactorings:

1.  Encapsulate the entire logic into a class `GroverAdaptiveSearchSolver`.
    - The `__init__` method should take the `qubo_matrix` as input.
    - The core logic should be in a `solve()` method.
    - Helper functions like `create_qubo_oracle` and `create_diffusion_operator` should become private methods of the class.
2.  Add comprehensive docstrings to the class and all its methods, explaining their purpose, arguments, and return values.
3.  Add type hints to all function signatures.
4.  Include comments to clarify any complex parts of the implementation, especially within the oracle creation."
