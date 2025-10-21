# Grover Adaptive Search for QUBO Problems

This implementation provides a quantum algorithm for solving Quadratic Unconstrained Binary Optimization (QUBO) problems using Grover's Adaptive Search in the Cirq framework.

## Overview

The `gas_for_qubo.py` script implements Grover Adaptive Search (GAS), a quantum algorithm that iteratively searches for the minimum of a QUBO problem by:

1. Creating a quantum oracle that marks states with cost below a threshold
2. Applying the Grover diffusion operator to amplify marked states
3. Measuring to find candidate solutions
4. Adaptively updating the threshold based on improvements

## Installation

### Prerequisites

Install the required Python packages:

```bash
pip install cirq numpy
```

Or using conda:

```bash
conda install -c conda-forge cirq
conda install numpy
```

## Usage

### Basic Usage with the Class

```python
import numpy as np
from gas_for_qubo import GroverAdaptiveSearchSolver

# Define a QUBO problem
qubo_matrix = np.array([
    [-1, 2],
    [0, -1]
])

# Create solver and find solution
solver = GroverAdaptiveSearchSolver(qubo_matrix)
solution, cost = solver.solve(max_iterations=10, verbose=True)

print(f"Solution: {solution}")
print(f"Cost: {cost}")

# Compare with classical solution
optimal_solution, optimal_cost = solver.classical_solve()
print(f"Optimal: {optimal_solution}, Cost: {optimal_cost}")
```

### Running the Tests

The script includes three test cases that demonstrate the algorithm:

```bash
python gas_for_qubo.py
```

This will run:
1. **Test 1**: Simple 2x2 QUBO problem
2. **Test 2**: MAX-CUT problem on a 4-node square graph
3. **Test 3**: Custom 3x3 QUBO problem

## Class Documentation

### `GroverAdaptiveSearchSolver`

Main class for solving QUBO problems with Grover Adaptive Search.

#### Methods

- **`__init__(qubo_matrix: np.ndarray)`**: Initialize the solver with a QUBO matrix
- **`solve(max_iterations: int = 10, verbose: bool = True)`**: Run GAS to find minimum
- **`classical_solve()`**: Find optimal solution using brute force (for verification)

### Private Methods

- **`_create_qubo_oracle(threshold: float)`**: Create oracle marking states below threshold
- **`_create_diffusion_operator()`**: Create Grover diffusion operator
- **`_calculate_cost(solution: np.ndarray)`**: Calculate QUBO cost for a solution

## Algorithm Details

### QUBO Formulation

A QUBO problem seeks to minimize:

```
f(x) = x^T * Q * x
```

where:
- `x` is a binary vector (values 0 or 1)
- `Q` is the QUBO matrix (can be upper triangular)

### Grover Adaptive Search

The algorithm works as follows:

1. **Initialization**: Set threshold to infinity
2. **For each iteration**:
   - Create superposition of all states
   - Apply oracle to mark states with cost < threshold
   - Apply diffusion operator to amplify marked states
   - Measure to get candidate solution
   - If candidate is better, update threshold and best solution
3. **Termination**: Stop after max iterations or when no improvement

### Oracle Implementation

The oracle marks states by:
1. For each state with cost < threshold:
   - Apply X gates to transform state to |11...1⟩
   - Apply multi-controlled Z for phase flip
   - Undo X gates to restore state

### Diffusion Operator

Implements the transformation `2|ψ⟩⟨ψ| - I`:
1. Apply Hadamard to all qubits
2. Apply X to all qubits
3. Apply multi-controlled Z
4. Apply X to all qubits
5. Apply Hadamard to all qubits

## Example Problems

### MAX-CUT on Square Graph

Partitioning nodes to maximize cut edges:

```python
# Graph: 0--1
#        |  |
#        3--2
maxcut_qubo = np.array([
    [-2,  2,  0,  2],
    [ 0, -2,  2,  0],
    [ 0,  0, -2,  2],
    [ 0,  0,  0, -2]
])

solver = GroverAdaptiveSearchSolver(maxcut_qubo)
solution, cost = solver.solve()
```

## Limitations

1. **Oracle Complexity**: Current implementation uses classical computation to determine which states to mark. A true quantum implementation would require quantum arithmetic circuits.

2. **Scalability**: The oracle currently enumerates all 2^n states, limiting practical use to small problems (n < 10 qubits).

3. **Simulation**: Uses Cirq's simulator. Results may differ on real quantum hardware due to noise and gate errors.

## Advanced Usage

### Custom Test Cases

```python
# Create your own QUBO
my_qubo = np.array([
    [-3,  1,  2],
    [ 0, -2,  1],
    [ 0,  0, -1]
])

solver = GroverAdaptiveSearchSolver(my_qubo)

# Run with custom parameters
solution, cost = solver.solve(
    max_iterations=15,
    verbose=True
)

# Verify correctness
optimal, optimal_cost = solver.classical_solve()
print(f"Found optimal: {cost == optimal_cost}")
```

### Silent Mode

For production use, disable verbose output:

```python
solution, cost = solver.solve(max_iterations=10, verbose=False)
```

## Theory and Background

### Grover's Algorithm

Grover's algorithm provides quadratic speedup for unstructured search problems. The adaptive search variant iteratively refines the search space.

### Quantum Amplitude Amplification

The diffusion operator amplifies the amplitude of marked states while suppressing unmarked ones, increasing measurement probability for good solutions.

### QUBO Problems

QUBO problems are NP-hard and appear in:
- Graph partitioning (MAX-CUT)
- Portfolio optimization
- Machine learning
- Protein folding
- Scheduling problems

## References

1. Grover, L. K. (1996). A fast quantum mechanical algorithm for database search.
2. Boyer, M., et al. (1998). Tight bounds on quantum searching.
3. Dürr, C., & Høyer, P. (1996). A quantum algorithm for finding the minimum.

## Contributing

Feel free to extend this implementation with:
- True quantum arithmetic for the oracle
- Variable Grover iterations based on search space
- Integration with real quantum hardware backends
- Additional QUBO problem examples

## License

This code is provided for educational purposes as part of quantum optimization tutorials.
