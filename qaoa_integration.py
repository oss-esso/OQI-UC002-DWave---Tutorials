"""
QAOA Python Integration

This file demonstrates how to use the Q# QAOA implementation from Python.
It provides the classical optimization loop and visualization.

Prerequisites:
    pip install azure-quantum qsharp numpy scipy matplotlib
"""

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict

# Try to import qsharp (may not be available without proper setup)
try:
    import qsharp
    QSHARP_AVAILABLE = True
    print("Q# integration available!")
except ImportError:
    QSHARP_AVAILABLE = False
    print("Q# not available. Install with: pip install qsharp")
    print("This module will run in demonstration mode.")


class QAOASolver:
    """
    QAOA solver with classical optimization loop.
    """
    
    def __init__(self, edges: List[Tuple[int, int]], n_qubits: int):
        """
        Initialize QAOA solver.
        
        Args:
            edges: List of edges in the graph
            n_qubits: Number of vertices (qubits)
        """
        self.edges = edges
        self.n_qubits = n_qubits
        self.optimization_history = []
        
    def calculate_maxcut_cost(self, bitstring: List[int]) -> float:
        """
        Calculate MaxCut cost for a given bitstring.
        
        Args:
            bitstring: Binary assignment of vertices
            
        Returns:
            Number of cut edges (negative = cost to minimize)
        """
        cuts = 0
        for i, j in self.edges:
            if bitstring[i] != bitstring[j]:
                cuts += 1
        return -cuts  # Negative because we minimize
    
    def simulate_qaoa_circuit(self, gammas: np.ndarray, betas: np.ndarray, 
                             shots: int = 1000) -> Dict[str, int]:
        """
        Simulate QAOA circuit (or use Q# if available).
        
        Args:
            gammas: Cost Hamiltonian parameters
            betas: Mixer Hamiltonian parameters
            shots: Number of circuit executions
            
        Returns:
            Dictionary of bitstrings and their counts
        """
        if QSHARP_AVAILABLE:
            # Use actual Q# implementation
            try:
                from QAOAOptimization import QAOACircuitWithReset
                
                results = {}
                for _ in range(shots):
                    # Run Q# circuit
                    bitstring = QAOACircuitWithReset.simulate(
                        gammas=gammas.tolist(),
                        betas=betas.tolist(),
                        edges=self.edges,
                        nQubits=self.n_qubits
                    )
                    # Convert to string for dictionary key
                    key = ''.join(map(str, bitstring))
                    results[key] = results.get(key, 0) + 1
                
                return results
            except Exception as e:
                print(f"Q# simulation failed: {e}")
                print("Falling back to numpy simulation...")
        
        # Fallback: Simplified numpy simulation
        # This is NOT a real quantum simulation, just for demonstration
        return self._simulate_qaoa_numpy(gammas, betas, shots)
    
    def _simulate_qaoa_numpy(self, gammas: np.ndarray, betas: np.ndarray, 
                            shots: int) -> Dict[str, int]:
        """
        Simplified QAOA simulation using numpy (classical approximation).
        
        NOTE: This is not a real quantum simulation! It's a classical
        approximation to demonstrate the workflow.
        """
        results = {}
        
        # Generate random bitstrings weighted by approximate QAOA distribution
        # This is a VERY rough approximation
        p = len(gammas)
        
        for _ in range(shots):
            # Start with random bitstring
            bitstring = np.random.randint(0, 2, self.n_qubits)
            
            # Apply some bias toward good solutions based on parameters
            # (This is NOT how QAOA actually works!)
            cost = self.calculate_maxcut_cost(bitstring)
            
            # Accept with probability based on parameters
            if np.random.random() < np.exp(cost * np.sum(gammas)):
                key = ''.join(map(str, bitstring))
                results[key] = results.get(key, 0) + 1
        
        return results
    
    def evaluate_expectation(self, params: np.ndarray, shots: int = 1000) -> float:
        """
        Evaluate expected cost for given parameters.
        
        This is the function we optimize classically.
        
        Args:
            params: Concatenated [gammas, betas]
            shots: Number of circuit executions
            
        Returns:
            Expected cost value
        """
        p = len(params) // 2
        gammas = params[:p]
        betas = params[p:]
        
        # Run QAOA circuit
        results = self.simulate_qaoa_circuit(gammas, betas, shots)
        
        # Calculate expected cost
        total_cost = 0
        total_counts = 0
        
        for bitstring, count in results.items():
            bitlist = [int(b) for b in bitstring]
            cost = self.calculate_maxcut_cost(bitlist)
            total_cost += cost * count
            total_counts += count
        
        expected_cost = total_cost / total_counts
        
        # Store for history
        self.optimization_history.append(expected_cost)
        
        return expected_cost
    
    def optimize(self, p: int = 1, max_iter: int = 100) -> Tuple[np.ndarray, float]:
        """
        Run QAOA optimization.
        
        Args:
            p: Circuit depth (number of layers)
            max_iter: Maximum optimization iterations
            
        Returns:
            Tuple of (optimal_parameters, best_cost)
        """
        print(f"Starting QAOA optimization (p={p})...")
        
        # Initialize parameters randomly
        initial_params = np.random.uniform(0, 2*np.pi, 2*p)
        
        # Classical optimization
        result = minimize(
            self.evaluate_expectation,
            initial_params,
            args=(1000,),  # shots
            method='COBYLA',
            options={'maxiter': max_iter}
        )
        
        optimal_params = result.x
        best_cost = result.fun
        
        print(f"Optimization complete!")
        print(f"  Best cost: {best_cost:.4f}")
        print(f"  Optimal parameters: {optimal_params}")
        
        return optimal_params, best_cost
    
    def plot_optimization_history(self):
        """
        Plot the optimization history.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.optimization_history)
        plt.xlabel('Iteration')
        plt.ylabel('Expected Cost')
        plt.title('QAOA Optimization Progress')
        plt.grid(True)
        plt.show()


def example_triangle_graph():
    """
    Example: Solve MaxCut on triangle graph using QAOA.
    """
    print("="*70)
    print("QAOA Example: MaxCut on Triangle Graph")
    print("="*70)
    
    # Define triangle graph
    edges = [(0, 1), (1, 2), (0, 2)]
    n_qubits = 3
    
    print("\nProblem Setup:")
    print(f"  Vertices: {n_qubits}")
    print(f"  Edges: {edges}")
    print("\n  Graph visualization:")
    print("      0 --- 1")
    print("       \\   /")
    print("        \\ /")
    print("         2")
    
    # Create solver
    solver = QAOASolver(edges, n_qubits)
    
    # Evaluate all possible solutions (brute force for comparison)
    print("\n" + "-"*70)
    print("Brute Force Solution (Classical)")
    print("-"*70)
    
    best_cost = float('inf')
    best_bitstring = None
    
    for i in range(2**n_qubits):
        bitstring = [int(b) for b in format(i, f'0{n_qubits}b')]
        cost = solver.calculate_maxcut_cost(bitstring)
        
        print(f"  {''.join(map(str, bitstring))}: {-cost} cuts (cost={cost})")
        
        if cost < best_cost:
            best_cost = cost
            best_bitstring = bitstring
    
    print(f"\n  Optimal solution: {best_bitstring} with {-best_cost} cuts")
    
    # Run QAOA optimization
    print("\n" + "-"*70)
    print("QAOA Solution (Quantum)")
    print("-"*70)
    
    optimal_params, qaoa_cost = solver.optimize(p=2, max_iter=30)
    
    print(f"\n  QAOA found cost: {qaoa_cost:.4f}")
    print(f"  Classical optimal: {best_cost}")
    print(f"  Approximation ratio: {qaoa_cost / best_cost:.4f}")
    
    # Run final circuit with optimal parameters
    print("\n" + "-"*70)
    print("Final QAOA Circuit Execution")
    print("-"*70)
    
    p = len(optimal_params) // 2
    gammas = optimal_params[:p]
    betas = optimal_params[p:]
    
    final_results = solver.simulate_qaoa_circuit(gammas, betas, shots=1000)
    
    print("\n  Top 5 most frequent outcomes:")
    sorted_results = sorted(final_results.items(), key=lambda x: x[1], reverse=True)
    for bitstring, count in sorted_results[:5]:
        bitlist = [int(b) for b in bitstring]
        cost = solver.calculate_maxcut_cost(bitlist)
        probability = count / 1000
        print(f"    {bitstring}: {count:4d} times ({probability:.1%}) - {-cost} cuts")


def compare_with_dwave():
    """
    Compare QAOA approach with D-Wave quantum annealing.
    """
    print("\n" + "="*70)
    print("QAOA vs D-Wave Comparison")
    print("="*70)
    
    print("\nSame Problem, Different Quantum Approaches:")
    print("\n1. D-Wave Quantum Annealing (from Tutorial 2):")
    print("   - Formulate as QUBO/Ising")
    print("   - Submit to D-Wave QPU")
    print("   - Get samples in milliseconds")
    print("   - Requires embedding for complex connectivity")
    
    print("\n2. QAOA with Q# (this tutorial):")
    print("   - Formulate as cost Hamiltonian")
    print("   - Build parameterized quantum circuit")
    print("   - Optimize parameters classically")
    print("   - Run on gate-based quantum computer")
    
    print("\nKey Differences:")
    print("  Hardware: Specialized vs Universal")
    print("  Scale: Thousands vs Hundreds of qubits")
    print("  Time: Milliseconds vs Seconds-Minutes")
    print("  Flexibility: Fixed vs Programmable circuits")


def main():
    """
    Main demonstration of QAOA with Q#.
    """
    print("\n" + "="*70)
    print("QAOA WITH Q# - PYTHON INTEGRATION")
    print("="*70)
    
    if not QSHARP_AVAILABLE:
        print("\nWARNING: Q# not available!")
        print("Running in demonstration mode with classical simulation.")
        print("\nTo use real Q# integration:")
        print("  1. Install .NET SDK: https://dotnet.microsoft.com/download")
        print("  2. Install Q# package: pip install qsharp")
        print("  3. Install Azure Quantum: pip install azure-quantum")
    
    print("\n")
    
    # Run example
    example_triangle_graph()
    
    # Comparison
    compare_with_dwave()
    
    print("\n" + "="*70)
    print("Next Steps:")
    print("="*70)
    print("  1. Set up Q# development environment")
    print("  2. Test QAOA.qs file: dotnet run")
    print("  3. Try different p values (circuit depths)")
    print("  4. Implement larger graph problems")
    print("  5. Compare with D-Wave results from Tutorials 1-6")
    print("\n")


if __name__ == "__main__":
    main()
