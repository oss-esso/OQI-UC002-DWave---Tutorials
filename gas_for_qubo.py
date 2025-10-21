import cirq
import numpy as np
from typing import List, Tuple, Optional


class GroverAdaptiveSearchSolver:
    """
    A quantum solver for QUBO problems using Grover Adaptive Search.
    
    This class implements Grover's Adaptive Search algorithm to find the minimum
    of a Quadratic Unconstrained Binary Optimization (QUBO) problem. The algorithm
    iteratively searches for better solutions by adjusting a threshold and using
    quantum amplitude amplification.
    
    Attributes:
        qubo_matrix (np.ndarray): The QUBO problem matrix Q, where the objective
            is to minimize x^T * Q * x for binary vector x.
        n (int): Number of qubits (dimension of the problem).
        qubits (List[cirq.LineQubit]): List of qubits used in the circuit.
        simulator (cirq.Simulator): Cirq simulator for running quantum circuits.
    
    Example:
        >>> qubo = np.array([[-1, 2], [0, -1]])
        >>> solver = GroverAdaptiveSearchSolver(qubo)
        >>> solution, cost = solver.solve(max_iterations=10)
        >>> print(f"Solution: {solution}, Cost: {cost}")
    """
    
    def __init__(self, qubo_matrix: np.ndarray):
        """
        Initialize the Grover Adaptive Search Solver.
        
        Args:
            qubo_matrix: A square numpy array representing the QUBO problem.
                The matrix should be of shape (n, n) where n is the number
                of binary variables.
        
        Raises:
            ValueError: If qubo_matrix is not a 2D square matrix.
        """
        if len(qubo_matrix.shape) != 2 or qubo_matrix.shape[0] != qubo_matrix.shape[1]:
            raise ValueError("QUBO matrix must be a square 2D array")
        
        self.qubo_matrix = qubo_matrix
        self.n = qubo_matrix.shape[0]
        self.qubits = cirq.LineQubit.range(self.n)
        self.simulator = cirq.Simulator()
    
    def _create_qubo_oracle(self, threshold: float) -> List[cirq.Operation]:
        """
        Create a quantum oracle that marks states with QUBO cost below threshold.
        
        This is a simplified implementation that uses classical computation
        to determine which states to mark with a phase flip. In a real quantum
        implementation on hardware, this would need to be implemented using
        quantum arithmetic circuits.
        
        The oracle applies a phase flip (Z gate) to all states |x⟩ where
        the QUBO cost x^T * Q * x is less than the given threshold.
        
        Args:
            threshold: The cost threshold. States with cost below this value
                are marked with a phase flip.
        
        Returns:
            A list of Cirq operations that implement the oracle.
        """
        operations = []
        
        # For each possible state, compute QUBO cost and mark if below threshold
        for state_int in range(2**self.n):
            # Convert integer to binary vector
            state = np.array([int(b) for b in format(state_int, f'0{self.n}b')])
            
            # Calculate QUBO cost: x^T * Q * x
            cost = state.T @ self.qubo_matrix @ state
            
            # If cost is below threshold, mark this state with a phase flip
            if cost < threshold:
                # Apply X gates to flip qubits where state bit is 0
                # This transforms the target state to |11...1⟩
                for i, bit in enumerate(state):
                    if bit == 0:
                        operations.append(cirq.X(self.qubits[i]))
                
                # Apply multi-controlled Z (phase flip) for |11...1⟩ state
                if self.n == 1:
                    operations.append(cirq.Z(self.qubits[0]))
                elif self.n == 2:
                    operations.append(cirq.CZ(self.qubits[0], self.qubits[1]))
                else:
                    # For n > 2, use multi-controlled Z
                    operations.append(cirq.Z(self.qubits[-1]).controlled_by(*self.qubits[:-1]))
                
                # Undo the X gates to restore original state
                for i, bit in enumerate(state):
                    if bit == 0:
                        operations.append(cirq.X(self.qubits[i]))
        
        return operations
    
    def _create_diffusion_operator(self) -> List[cirq.Operation]:
        """
        Create the Grover diffusion operator (amplitude amplification).
        
        The diffusion operator reflects the amplitudes of all states about
        their average, which has the effect of amplifying the amplitude of
        marked states while reducing the amplitude of unmarked states.
        
        The operator is defined as: 2|ψ⟩⟨ψ| - I, where |ψ⟩ is the uniform
        superposition state.
        
        Returns:
            A list of Cirq operations that implement the diffusion operator.
        """
        operations = []
        
        # Step 1: Apply Hadamard to all qubits (transforms |+⟩^n to |0⟩^n)
        for qubit in self.qubits:
            operations.append(cirq.H(qubit))
        
        # Step 2: Apply X to all qubits (transforms |0⟩^n to |1⟩^n)
        for qubit in self.qubits:
            operations.append(cirq.X(qubit))
        
        # Step 3: Apply multi-controlled Z to flip phase of |1⟩^n state
        if self.n == 1:
            operations.append(cirq.Z(self.qubits[0]))
        elif self.n == 2:
            operations.append(cirq.CZ(self.qubits[0], self.qubits[1]))
        else:
            # Multi-controlled Z for n > 2
            operations.append(cirq.Z(self.qubits[-1]).controlled_by(*self.qubits[:-1]))
        
        # Step 4: Apply X to all qubits (undo the flip)
        for qubit in self.qubits:
            operations.append(cirq.X(qubit))
        
        # Step 5: Apply Hadamard to all qubits (transforms |0⟩^n back to |+⟩^n)
        for qubit in self.qubits:
            operations.append(cirq.H(qubit))
        
        return operations
    
    def _calculate_cost(self, solution: np.ndarray) -> float:
        """
        Calculate the QUBO cost for a given binary solution.
        
        Args:
            solution: A binary vector representing the solution.
        
        Returns:
            The QUBO cost x^T * Q * x.
        """
        return solution.T @ self.qubo_matrix @ solution
    
    def solve(self, max_iterations: int = 10, verbose: bool = True) -> Tuple[np.ndarray, float]:
        """
        Perform Grover Adaptive Search to find the minimum of the QUBO problem.
        
        The algorithm works by iteratively:
        1. Creating an oracle that marks states with cost below current threshold
        2. Applying the Grover diffusion operator to amplify marked states
        3. Measuring to find a candidate solution
        4. Updating the threshold if a better solution is found
        
        Args:
            max_iterations: Maximum number of adaptive search iterations.
                More iterations may find better solutions but take longer.
            verbose: If True, print progress information during search.
        
        Returns:
            A tuple (best_solution, best_cost) where:
                - best_solution is a binary numpy array of shape (n,)
                - best_cost is the QUBO cost of the best solution found
        """
        # Initialize threshold to infinity (accept any solution initially)
        threshold = float('inf')
        
        best_solution = None
        best_cost = float('inf')
        
        if verbose:
            print(f"Starting Grover Adaptive Search for {self.n}-qubit QUBO problem")
            print(f"QUBO Matrix:\n{self.qubo_matrix}\n")
        
        for iteration in range(max_iterations):
            if verbose:
                print(f"Iteration {iteration + 1}/{max_iterations}")
                print(f"Current threshold: {threshold}")
            
            # Create circuit
            circuit = cirq.Circuit()
            
            # Initialize qubits in uniform superposition
            circuit.append([cirq.H(q) for q in self.qubits])
            
            # Determine number of Grover iterations
            # For adaptive search, typically use 1 iteration per phase
            num_grover_iterations = 1
            
            # Apply Grover iterations: Oracle followed by Diffusion
            for _ in range(num_grover_iterations):
                # Apply oracle to mark states with cost < threshold
                oracle_ops = self._create_qubo_oracle(threshold)
                circuit.append(oracle_ops)
                
                # Apply diffusion operator to amplify marked states
                diffusion_ops = self._create_diffusion_operator()
                circuit.append(diffusion_ops)
            
            # Measure all qubits
            circuit.append(cirq.measure(*self.qubits, key='result'))
            
            # Simulate the circuit
            result = self.simulator.run(circuit, repetitions=1000)
            measurements = result.measurements['result']
            
            # Find the most common measurement result
            unique, counts = np.unique(measurements, axis=0, return_counts=True)
            most_common_idx = np.argmax(counts)
            candidate_solution = unique[most_common_idx]
            
            # Calculate cost of candidate solution
            candidate_cost = self._calculate_cost(candidate_solution)
            
            if verbose:
                print(f"  Candidate solution: {candidate_solution}")
                print(f"  Candidate cost: {candidate_cost}")
            
            # Check if this is an improvement
            if candidate_cost < best_cost:
                best_cost = candidate_cost
                best_solution = candidate_solution
                
                if verbose:
                    print(f"  *** New best solution found! Cost: {best_cost}")
                
                # Update threshold to search for even better solutions
                threshold = best_cost
            else:
                if verbose:
                    print(f"  No improvement found")
                
                # Early stopping: if no improvement for several iterations
                if iteration > 2 and candidate_cost >= best_cost:
                    if verbose:
                        print(f"No improvement for multiple iterations. Stopping search.")
                    break
            
            if verbose:
                print()
        
        return best_solution, best_cost
    
    def classical_solve(self) -> Tuple[np.ndarray, float]:
        """
        Find the optimal solution using classical brute force search.
        
        This method exhaustively searches all 2^n possible solutions
        and returns the one with minimum cost. Useful for verifying
        the quantum solver's results on small problems.
        
        Returns:
            A tuple (optimal_solution, optimal_cost) where:
                - optimal_solution is the binary vector with minimum cost
                - optimal_cost is the minimum QUBO cost
        
        Note:
            This method has exponential complexity O(2^n) and should
            only be used for small problems (n < 20).
        """
        best_solution = None
        best_cost = float('inf')
        
        # Try all possible solutions
        for state_int in range(2**self.n):
            state = np.array([int(b) for b in format(state_int, f'0{self.n}b')])
            cost = self._calculate_cost(state)
            
            if cost < best_cost:
                best_cost = cost
                best_solution = state
        
        return best_solution, best_cost


def classical_brute_force(qubo_matrix: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Find the optimal solution by classical brute force search.
    
    This is a standalone function for backward compatibility.
    For class-based usage, use GroverAdaptiveSearchSolver.classical_solve().
    
    Args:
        qubo_matrix: QUBO problem matrix
    
    Returns:
        Tuple of (optimal_solution, optimal_cost)
    """
    solver = GroverAdaptiveSearchSolver(qubo_matrix)
    return solver.classical_solve()


if __name__ == "__main__":
    print("=" * 60)
    print("Grover Adaptive Search for QUBO Problems")
    print("=" * 60)
    print()
    
    # Test 1: Simple 2x2 QUBO
    print("Test 1: Simple 2x2 QUBO")
    print("-" * 60)
    simple_qubo = np.array([
        [-1, 2],
        [0, -1]
    ])
    
    solver = GroverAdaptiveSearchSolver(simple_qubo)
    solution, cost = solver.solve(max_iterations=5, verbose=True)
    optimal_solution, optimal_cost = solver.classical_solve()
    
    print(f"GAS Solution: {solution}, Cost: {cost}")
    print(f"Optimal Solution: {optimal_solution}, Optimal Cost: {optimal_cost}")
    print(f"Match: {cost == optimal_cost}")
    print()
    
    # Test 2: 4x4 MAX-CUT problem on a square graph
    print("Test 2: 4x4 MAX-CUT on Square Graph")
    print("-" * 60)
    # MAX-CUT on a square: minimize edges between same-colored vertices
    # Graph edges: (0,1), (1,2), (2,3), (3,0)
    # QUBO formulation: minimize sum of (1 - x_i - x_j + 2*x_i*x_j) for each edge
    # This simplifies to: Q_ii = -degree, Q_ij = 2 for edges
    maxcut_qubo = np.array([
        [-2,  2,  0,  2],
        [ 0, -2,  2,  0],
        [ 0,  0, -2,  2],
        [ 0,  0,  0, -2]
    ])
    
    solver = GroverAdaptiveSearchSolver(maxcut_qubo)
    solution, cost = solver.solve(max_iterations=10, verbose=True)
    optimal_solution, optimal_cost = solver.classical_solve()
    
    print(f"GAS Solution: {solution}, Cost: {cost}")
    print(f"Optimal Solution: {optimal_solution}, Optimal Cost: {optimal_cost}")
    print(f"Match: {cost == optimal_cost}")
    print()
    
    # Test 3: 3x3 custom QUBO
    print("Test 3: 3x3 Custom QUBO")
    print("-" * 60)
    custom_qubo = np.array([
        [-5,  2,  1],
        [ 0, -3,  2],
        [ 0,  0, -4]
    ])
    
    solver = GroverAdaptiveSearchSolver(custom_qubo)
    solution, cost = solver.solve(max_iterations=8, verbose=True)
    optimal_solution, optimal_cost = solver.classical_solve()
    
    print(f"GAS Solution: {solution}, Cost: {cost}")
    print(f"Optimal Solution: {optimal_solution}, Optimal Cost: {optimal_cost}")
    print(f"Match: {cost == optimal_cost}")
    print()
    
    print("=" * 60)
    print("Testing Complete!")
    print("=" * 60)

