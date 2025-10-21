import cirq
import numpy as np
from typing import List, Tuple, Optional
import math


class ImprovedGroverAdaptiveSearchSolver:
    """
    An improved quantum solver for QUBO problems using Grover Adaptive Search.
    
    This enhanced version includes:
    - Dynamic Grover iteration calculation
    - Multiple random restarts to escape local minima
    - Configurable measurement repetitions
    - Better convergence detection
    
    Attributes:
        qubo_matrix (np.ndarray): The QUBO problem matrix Q
        n (int): Number of qubits
        qubits (List[cirq.LineQubit]): List of qubits
        simulator (cirq.Simulator): Cirq simulator
    """
    
    def __init__(self, qubo_matrix: np.ndarray):
        """Initialize the improved solver."""
        if len(qubo_matrix.shape) != 2 or qubo_matrix.shape[0] != qubo_matrix.shape[1]:
            raise ValueError("QUBO matrix must be a square 2D array")
        
        self.qubo_matrix = qubo_matrix
        self.n = qubo_matrix.shape[0]
        self.qubits = cirq.LineQubit.range(self.n)
        self.simulator = cirq.Simulator()
    
    def _count_marked_states(self, threshold: float) -> int:
        """Count how many states have cost below threshold."""
        count = 0
        for state_int in range(2**self.n):
            state = np.array([int(b) for b in format(state_int, f'0{self.n}b')])
            cost = state.T @ self.qubo_matrix @ state
            if cost < threshold:
                count += 1
        return count
    
    def _calculate_optimal_grover_iterations(self, num_marked: int) -> int:
        """
        Calculate optimal number of Grover iterations.
        
        Formula: π/4 * √(N/M) where N = total states, M = marked states
        """
        if num_marked == 0:
            return 1
        
        N = 2**self.n
        optimal = math.pi / 4 * math.sqrt(N / num_marked)
        return max(1, int(optimal))
    
    def _create_qubo_oracle(self, threshold: float) -> List[cirq.Operation]:
        """Create oracle that marks states with cost below threshold."""
        operations = []
        
        for state_int in range(2**self.n):
            state = np.array([int(b) for b in format(state_int, f'0{self.n}b')])
            cost = state.T @ self.qubo_matrix @ state
            
            if cost < threshold:
                for i, bit in enumerate(state):
                    if bit == 0:
                        operations.append(cirq.X(self.qubits[i]))
                
                if self.n == 1:
                    operations.append(cirq.Z(self.qubits[0]))
                elif self.n == 2:
                    operations.append(cirq.CZ(self.qubits[0], self.qubits[1]))
                else:
                    operations.append(cirq.Z(self.qubits[-1]).controlled_by(*self.qubits[:-1]))
                
                for i, bit in enumerate(state):
                    if bit == 0:
                        operations.append(cirq.X(self.qubits[i]))
        
        return operations
    
    def _create_diffusion_operator(self) -> List[cirq.Operation]:
        """Create the Grover diffusion operator."""
        operations = []
        
        for qubit in self.qubits:
            operations.append(cirq.H(qubit))
        
        for qubit in self.qubits:
            operations.append(cirq.X(qubit))
        
        if self.n == 1:
            operations.append(cirq.Z(self.qubits[0]))
        elif self.n == 2:
            operations.append(cirq.CZ(self.qubits[0], self.qubits[1]))
        else:
            operations.append(cirq.Z(self.qubits[-1]).controlled_by(*self.qubits[:-1]))
        
        for qubit in self.qubits:
            operations.append(cirq.X(qubit))
        
        for qubit in self.qubits:
            operations.append(cirq.H(qubit))
        
        return operations
    
    def _calculate_cost(self, solution: np.ndarray) -> float:
        """Calculate QUBO cost for a solution."""
        return solution.T @ self.qubo_matrix @ solution
    
    def solve(self, 
              max_iterations: int = 15, 
              num_restarts: int = 3,
              repetitions: int = 2000,
              verbose: bool = True) -> Tuple[np.ndarray, float]:
        """
        Improved solve with multiple restarts and dynamic iterations.
        
        Args:
            max_iterations: Max iterations per restart
            num_restarts: Number of random restarts to try
            repetitions: Measurement repetitions per iteration
            verbose: Print progress
        
        Returns:
            Tuple of (best_solution, best_cost)
        """
        global_best_solution = None
        global_best_cost = float('inf')
        
        if verbose:
            print(f"Improved Grover Adaptive Search for {self.n}-qubit QUBO")
            print(f"Using {num_restarts} restarts with {repetitions} measurements each")
            print(f"QUBO Matrix:\n{self.qubo_matrix}\n")
        
        for restart in range(num_restarts):
            if verbose and num_restarts > 1:
                print(f"\n{'='*60}")
                print(f"Restart {restart + 1}/{num_restarts}")
                print(f"{'='*60}")
            
            threshold = float('inf')
            best_solution = None
            best_cost = float('inf')
            no_improvement_count = 0
            
            for iteration in range(max_iterations):
                if verbose:
                    print(f"\nIteration {iteration + 1}/{max_iterations}")
                    print(f"Current threshold: {threshold}")
                
                # Count marked states and calculate optimal iterations
                num_marked = self._count_marked_states(threshold)
                if verbose:
                    print(f"States below threshold: {num_marked}/{2**self.n}")
                
                if num_marked == 0:
                    if verbose:
                        print("No states below threshold. Stopping.")
                    break
                
                # Calculate optimal Grover iterations
                num_grover_iterations = self._calculate_optimal_grover_iterations(num_marked)
                if verbose:
                    print(f"Using {num_grover_iterations} Grover iteration(s)")
                
                # Build circuit
                circuit = cirq.Circuit()
                
                # Initialize with superposition
                circuit.append([cirq.H(q) for q in self.qubits])
                
                # Add random phase for this restart (helps explore different regions)
                if restart > 0:
                    for i, qubit in enumerate(self.qubits):
                        # Random rotation based on restart number
                        angle = (restart * 0.1 + i * 0.05) * 2 * math.pi
                        circuit.append(cirq.rz(angle)(qubit))
                
                # Apply Grover iterations
                for _ in range(num_grover_iterations):
                    oracle_ops = self._create_qubo_oracle(threshold)
                    circuit.append(oracle_ops)
                    
                    diffusion_ops = self._create_diffusion_operator()
                    circuit.append(diffusion_ops)
                
                # Measure
                circuit.append(cirq.measure(*self.qubits, key='result'))
                
                # Simulate
                result = self.simulator.run(circuit, repetitions=repetitions)
                measurements = result.measurements['result']
                
                # Find most common measurement
                unique, counts = np.unique(measurements, axis=0, return_counts=True)
                most_common_idx = np.argmax(counts)
                candidate_solution = unique[most_common_idx]
                candidate_cost = self._calculate_cost(candidate_solution)
                
                if verbose:
                    print(f"Candidate: {candidate_solution}, Cost: {candidate_cost}")
                    print(f"Probability: {counts[most_common_idx]/repetitions:.3f}")
                
                # Check for improvement
                if candidate_cost < best_cost:
                    best_cost = candidate_cost
                    best_solution = candidate_solution
                    no_improvement_count = 0
                    
                    if verbose:
                        print(f"*** New best for this restart! Cost: {best_cost}")
                    
                    threshold = best_cost
                else:
                    no_improvement_count += 1
                    if verbose:
                        print(f"No improvement (count: {no_improvement_count})")
                    
                    # Early stopping if stuck
                    if no_improvement_count >= 3:
                        if verbose:
                            print("Stopping early - no improvement")
                        break
            
            # Update global best
            if best_cost < global_best_cost:
                global_best_cost = best_cost
                global_best_solution = best_solution
                if verbose:
                    print(f"\n*** NEW GLOBAL BEST! Cost: {global_best_cost}")
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Search Complete!")
            print(f"Best solution: {global_best_solution}")
            print(f"Best cost: {global_best_cost}")
            print(f"{'='*60}\n")
        
        return global_best_solution, global_best_cost
    
    def classical_solve(self) -> Tuple[np.ndarray, float]:
        """Find optimal solution by brute force."""
        best_solution = None
        best_cost = float('inf')
        
        for state_int in range(2**self.n):
            state = np.array([int(b) for b in format(state_int, f'0{self.n}b')])
            cost = self._calculate_cost(state)
            
            if cost < best_cost:
                best_cost = cost
                best_solution = state
        
        return best_solution, best_cost


if __name__ == "__main__":
    print("="*70)
    print("IMPROVED Grover Adaptive Search for QUBO Problems")
    print("="*70)
    print()
    
    # Test 1: Simple 2x2 QUBO
    print("\n" + "="*70)
    print("Test 1: Simple 2x2 QUBO")
    print("="*70)
    simple_qubo = np.array([
        [-1, 2],
        [0, -1]
    ])
    
    solver = ImprovedGroverAdaptiveSearchSolver(simple_qubo)
    solution, cost = solver.solve(max_iterations=10, num_restarts=2, verbose=True)
    optimal_solution, optimal_cost = solver.classical_solve()
    
    print(f"\nGAS Solution: {solution}, Cost: {cost}")
    print(f"Optimal Solution: {optimal_solution}, Optimal Cost: {optimal_cost}")
    print(f"Match: {'✅ YES' if cost == optimal_cost else '❌ NO'}")
    
    # Test 2: 4x4 MAX-CUT
    print("\n" + "="*70)
    print("Test 2: 4x4 MAX-CUT on Square Graph")
    print("="*70)
    maxcut_qubo = np.array([
        [-2,  2,  0,  2],
        [ 0, -2,  2,  0],
        [ 0,  0, -2,  2],
        [ 0,  0,  0, -2]
    ])
    
    solver = ImprovedGroverAdaptiveSearchSolver(maxcut_qubo)
    solution, cost = solver.solve(max_iterations=15, num_restarts=3, verbose=True)
    optimal_solution, optimal_cost = solver.classical_solve()
    
    print(f"\nGAS Solution: {solution}, Cost: {cost}")
    print(f"Optimal Solution: {optimal_solution}, Optimal Cost: {optimal_cost}")
    print(f"Match: {'✅ YES' if cost == optimal_cost else '❌ NO'}")
    
    # Test 3: 3x3 Custom
    print("\n" + "="*70)
    print("Test 3: 3x3 Custom QUBO")
    print("="*70)
    custom_qubo = np.array([
        [-5,  2,  1],
        [ 0, -3,  2],
        [ 0,  0, -4]
    ])
    
    solver = ImprovedGroverAdaptiveSearchSolver(custom_qubo)
    solution, cost = solver.solve(max_iterations=12, num_restarts=3, verbose=True)
    optimal_solution, optimal_cost = solver.classical_solve()
    
    print(f"\nGAS Solution: {solution}, Cost: {cost}")
    print(f"Optimal Solution: {optimal_solution}, Optimal Cost: {optimal_cost}")
    print(f"Match: {'✅ YES' if cost == optimal_cost else '❌ NO'}")
    
    print("\n" + "="*70)
    print("All Tests Complete!")
    print("="*70)
