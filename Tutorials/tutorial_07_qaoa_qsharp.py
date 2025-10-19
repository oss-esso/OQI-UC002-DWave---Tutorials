"""
Tutorial 7: QAOA (Quantum Approximate Optimization Algorithm) with Q#

This tutorial demonstrates how to solve optimization problems using QAOA,
a gate-based quantum algorithm that can solve the same types of problems
as quantum annealing but using a different quantum paradigm.

You'll learn:
1. The difference between quantum annealing and gate-based quantum computing
2. How QAOA works conceptually
3. How to formulate optimization problems for QAOA
4. How to implement QAOA using Q# and Azure Quantum
5. How to compare QAOA with D-Wave's quantum annealing approach

Key Differences:
- Quantum Annealing (D-Wave): Specialized hardware, adiabatic evolution
- QAOA (Gate-Based): Universal quantum computer, parameterized circuits
- Both solve: QUBO/Ising model optimization problems

Prerequisites:
- Install Azure Quantum SDK: pip install azure-quantum qsharp
- Azure Quantum account (free tier available)
"""

import sys
import os
import numpy as np
from typing import Dict, List, Tuple

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from src.scenarios import load_food_data


def example_1_qaoa_vs_quantum_annealing():
    """
    Example 1: Understanding QAOA vs Quantum Annealing
    
    Compare the two quantum approaches to optimization.
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: QAOA vs Quantum Annealing")
    print("="*70)
    
    print("\n" + "-"*70)
    print("Quantum Annealing (D-Wave) - Tutorials 1-6")
    print("-"*70)
    print("  Paradigm: Adiabatic Quantum Computing")
    print("  Hardware: Specialized (D-Wave systems)")
    print("  Process:")
    print("    1. Encode problem as QUBO/Ising Hamiltonian")
    print("    2. Initialize system in ground state of simple Hamiltonian")
    print("    3. Slowly evolve to problem Hamiltonian")
    print("    4. Measure final state = solution")
    print("\n  Advantages:")
    print("    - Specialized for optimization")
    print("    - Large number of qubits (5000+)")
    print("    - Direct problem encoding")
    print("\n  Limitations:")
    print("    - Limited connectivity between qubits")
    print("    - Requires embedding for complex problems")
    print("    - Single purpose (optimization only)")
    
    print("\n" + "-"*70)
    print("QAOA (Gate-Based Quantum) - This Tutorial")
    print("-"*70)
    print("  Paradigm: Gate-Based Quantum Computing")
    print("  Hardware: Universal quantum computers (IBM, Google, Azure)")
    print("  Process:")
    print("    1. Encode problem as cost Hamiltonian")
    print("    2. Apply parameterized quantum circuit (layers)")
    print("    3. Measure and evaluate cost")
    print("    4. Optimize circuit parameters classically")
    print("    5. Iterate until convergence")
    print("\n  Advantages:")
    print("    - Universal quantum computer (not just optimization)")
    print("    - Can run on various hardware")
    print("    - Tuneable depth vs quality tradeoff")
    print("\n  Limitations:")
    print("    - Fewer qubits (50-1000 currently)")
    print("    - Requires classical optimization loop")
    print("    - Circuit depth limited by decoherence")
    
    print("\n" + "-"*70)
    print("QAOA Circuit Structure")
    print("-"*70)
    print("  QAOA repeats p layers of:")
    print("    1. Problem (Cost) Hamiltonian: U_C(gamma)")
    print("       - Encodes the optimization objective")
    print("       - Applied with parameter gamma")
    print("    2. Mixer Hamiltonian: U_M(beta)")
    print("       - Enables exploration of solution space")
    print("       - Applied with parameter beta")
    print("\n  |psi> = U_M(beta_p) U_C(gamma_p) ... U_M(beta_1) U_C(gamma_1) |+>^n")
    print("\n  p = circuit depth (more layers = better approximation)")
    print("  (gamma, beta) = parameters optimized classically")


def example_2_max_cut_problem():
    """
    Example 2: MaxCut Problem - Classic QAOA Example
    
    MaxCut: Given a graph, partition vertices into two sets to maximize
    the number of edges between sets.
    
    This is equivalent to an Ising model, just like D-Wave problems!
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: MaxCut Problem for QAOA")
    print("="*70)
    
    print("\nProblem: MaxCut on a Triangle Graph")
    print("  Graph: 3 vertices (A, B, C) with 3 edges")
    print("         A --- B")
    print("          \\   /")
    print("           \\ /")
    print("            C")
    print("\n  Goal: Partition {A,B,C} into two sets to maximize cuts")
    print("  Example partitions:")
    print("    {A} | {B,C} -> 2 cuts (A-B, A-C)")
    print("    {B} | {A,C} -> 2 cuts (B-A, B-C)")
    print("    {A,B} | {C} -> 2 cuts (A-C, B-C)")
    print("    {A,C} | {B} -> 2 cuts (A-B, B-C)")
    
    # Encode as Ising/QUBO
    print("\n" + "-"*70)
    print("Encoding as Ising Model (same as D-Wave!)")
    print("-"*70)
    
    print("\n  Variables: z_i in {-1, +1} for each vertex")
    print("  Cost: -sum of (z_i * z_j) over edges")
    print("        Negative when vertices have different signs (cut!)")
    
    print("\n  Hamiltonian:")
    print("    H = -1/2 * [(1 - z_A*z_B) + (1 - z_B*z_C) + (1 - z_A*z_C)]")
    print("      = -3/2 + 1/2*(z_A*z_B + z_B*z_C + z_A*z_C)")
    
    print("\n  In Pauli operators (for quantum circuit):")
    print("    H = -3/2 + 1/2*(Z_A Z_B + Z_B Z_C + Z_A Z_C)")
    print("    where Z_i is Pauli-Z operator on qubit i")
    
    print("\n  This is exactly the Ising model from Tutorials 1-2!")
    print("  But now we solve it with quantum gates, not annealing")
    
    # Enumerate solutions
    print("\n" + "-"*70)
    print("All Possible Solutions (Brute Force)")
    print("-"*70)
    
    solutions = [
        ("000", [0, 0, 0], "All same partition"),
        ("001", [0, 0, 1], "C different"),
        ("010", [0, 1, 0], "B different"),
        ("011", [0, 1, 1], "A different"),
        ("100", [1, 0, 0], "A different"),
        ("101", [1, 0, 1], "B different"),
        ("110", [1, 1, 0], "C different"),
        ("111", [1, 1, 1], "All same partition"),
    ]
    
    print("\n  Bitstring | Partition | Cuts | Cost")
    print("  " + "-"*45)
    
    for bitstring, config, desc in solutions:
        # Convert 0/1 to -1/+1 for Ising
        z = [2*x - 1 for x in config]
        
        # Calculate cuts
        cuts = 0
        if z[0] != z[1]: cuts += 1  # A-B edge
        if z[1] != z[2]: cuts += 1  # B-C edge
        if z[0] != z[2]: cuts += 1  # A-C edge
        
        # Cost (to minimize) = -cuts
        cost = -cuts
        
        marker = " <- OPTIMAL" if cuts == 2 else ""
        print(f"  {bitstring}     | {desc:20} | {cuts}    | {cost:+2d}{marker}")
    
    print("\n  4 optimal solutions (2 cuts each)")
    print("  QAOA will find superposition favoring these states!")


def example_3_qaoa_circuit_explanation():
    """
    Example 3: QAOA Circuit Structure
    
    Explain the quantum circuit used in QAOA.
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: QAOA Circuit Structure")
    print("="*70)
    
    print("\n" + "-"*70)
    print("Step 1: Initialize in Equal Superposition")
    print("-"*70)
    print("  Apply Hadamard gates to all qubits:")
    print("    |psi_0> = H^(n) |0>^n = |+>^n")
    print("    |+> = 1/sqrt(2) * (|0> + |1>)")
    print("\n  This creates equal superposition over all bitstrings")
    print("  For 3 qubits: (|000> + |001> + ... + |111>) / sqrt(8)")
    
    print("\n" + "-"*70)
    print("Step 2: Apply p QAOA Layers")
    print("-"*70)
    print("  For layer l = 1 to p:")
    
    print("\n  2a. Cost Hamiltonian: U_C(gamma_l)")
    print("      Implements: exp(-i * gamma_l * H_cost)")
    print("      For MaxCut: exp(-i * gamma_l * (Z_i Z_j))")
    print("      Effect: Phases based on objective value")
    print("\n      Circuit: For each edge (i,j):")
    print("        CNOT(i, j)")
    print("        Rz(2*gamma_l, j)  # Rotate by 2*gamma")
    print("        CNOT(i, j)")
    
    print("\n  2b. Mixer Hamiltonian: U_M(beta_l)")
    print("      Implements: exp(-i * beta_l * sum(X_i))")
    print("      Effect: Enables exploration (quantum tunneling)")
    print("\n      Circuit: For each qubit i:")
    print("        Rx(2*beta_l, i)  # Rotate around X axis")
    
    print("\n" + "-"*70)
    print("Step 3: Measure in Computational Basis")
    print("-"*70)
    print("  Measure all qubits -> get bitstring")
    print("  Evaluate cost for this bitstring")
    print("  Repeat many times to get expectation value")
    
    print("\n" + "-"*70)
    print("Step 4: Classical Optimization")
    print("-"*70)
    print("  Use classical optimizer (e.g., COBYLA, Nelder-Mead)")
    print("  Optimize parameters (gamma, beta) to minimize cost")
    print("  This is the 'hybrid' part of QAOA")
    
    print("\n" + "-"*70)
    print("Complete QAOA Algorithm")
    print("-"*70)
    print("""
  1. Choose circuit depth p
  2. Initialize parameters (gamma, beta) randomly
  3. Repeat until convergence:
     a. Build quantum circuit with current parameters
     b. Execute circuit multiple times (shots)
     c. Measure outcomes and calculate average cost
     d. Update parameters using classical optimizer
  4. Return best bitstring found
    """)
    
    print("  Typical values: p = 1-10, shots = 1000-10000")


def example_4_qaoa_pseudocode():
    """
    Example 4: QAOA Implementation (Pseudocode)
    
    Show how to implement QAOA step-by-step.
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: QAOA Implementation (Pseudocode)")
    print("="*70)
    
    print("\nQ# Implementation Structure:")
    print("-"*70)
    
    qsharp_code = """
// File: QAOA.qs
namespace QAOAOptimization {
    open Microsoft.Quantum.Canon;
    open Microsoft.Quantum.Intrinsic;
    open Microsoft.Quantum.Measurement;
    open Microsoft.Quantum.Math;
    
    // Cost Hamiltonian: Apply phase based on edge
    operation ApplyCostHamiltonian(
        gamma : Double, 
        edges : (Int, Int)[], 
        qubits : Qubit[]
    ) : Unit {
        for edge in edges {
            let (i, j) = edge;
            // exp(-i * gamma * Z_i Z_j)
            CNOT(qubits[i], qubits[j]);
            Rz(2.0 * gamma, qubits[j]);
            CNOT(qubits[i], qubits[j]);
        }
    }
    
    // Mixer Hamiltonian: Apply X rotations
    operation ApplyMixerHamiltonian(
        beta : Double, 
        qubits : Qubit[]
    ) : Unit {
        for qubit in qubits {
            Rx(2.0 * beta, qubit);
        }
    }
    
    // Complete QAOA circuit
    operation QAOACircuit(
        gammas : Double[], 
        betas : Double[], 
        edges : (Int, Int)[], 
        nQubits : Int
    ) : Result[] {
        use qubits = Qubit[nQubits];
        
        // Initialize in superposition
        ApplyToEach(H, qubits);
        
        // Apply p layers
        for p in 0..Length(gammas)-1 {
            ApplyCostHamiltonian(gammas[p], edges, qubits);
            ApplyMixerHamiltonian(betas[p], qubits);
        }
        
        // Measure
        return ForEach(M, qubits);
    }
}
"""
    
    print(qsharp_code)
    
    print("\nPython Driver Code:")
    print("-"*70)
    
    python_code = """
import qsharp
from scipy.optimize import minimize
import numpy as np

# Load Q# operations
from QAOAOptimization import QAOACircuit

def evaluate_cost(params, edges, n_qubits, p, shots=1000):
    '''Evaluate expected cost for given parameters'''
    gammas = params[:p]
    betas = params[p:]
    
    # Run circuit multiple times
    costs = []
    for _ in range(shots):
        result = QAOACircuit.simulate(
            gammas=gammas, 
            betas=betas, 
            edges=edges, 
            nQubits=n_qubits
        )
        # Convert Result[] to bitstring
        bitstring = [1 if r else 0 for r in result]
        # Calculate cost for this bitstring
        cost = calculate_maxcut_cost(bitstring, edges)
        costs.append(cost)
    
    return np.mean(costs)

def run_qaoa(edges, n_qubits, p=2):
    '''Run QAOA optimization'''
    # Initialize parameters
    initial_params = np.random.random(2*p) * 2 * np.pi
    
    # Classical optimization
    result = minimize(
        evaluate_cost,
        initial_params,
        args=(edges, n_qubits, p),
        method='COBYLA'
    )
    
    return result.x  # Optimal parameters

# Example: Triangle graph
edges = [(0,1), (1,2), (0,2)]
optimal_params = run_qaoa(edges, n_qubits=3, p=2)
print(f"Optimal parameters: {optimal_params}")
"""
    
    print(python_code)


def example_5_azure_quantum_setup():
    """
    Example 5: Setting Up Azure Quantum
    
    Guide for running QAOA on real quantum hardware or simulators.
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: Azure Quantum Setup")
    print("="*70)
    
    print("\n" + "-"*70)
    print("Installation")
    print("-"*70)
    print("  1. Install Azure Quantum SDK:")
    print("     pip install azure-quantum qsharp")
    print("\n  2. Install .NET SDK (for Q#):")
    print("     https://dotnet.microsoft.com/download")
    print("\n  3. Create Azure Quantum workspace:")
    print("     https://portal.azure.com")
    print("     - Free tier available (10 hours/month)")
    
    print("\n" + "-"*70)
    print("Available Quantum Backends")
    print("-"*70)
    print("  Simulators (Free, Local):")
    print("    - Resource Estimator: Analyze circuit requirements")
    print("    - Full State Simulator: Simulate perfect quantum computer")
    print("    - Noise Simulator: Simulate with realistic noise")
    
    print("\n  Azure Quantum Targets:")
    print("    - IonQ: Trapped ion quantum computer")
    print("    - Quantinuum: Trapped ion, high fidelity")
    print("    - Rigetti: Superconducting qubits")
    
    print("\n  Note: Each provider has different:")
    print("    - Number of qubits available")
    print("    - Gate sets supported")
    print("    - Pricing structure")
    
    print("\n" + "-"*70)
    print("Azure Quantum Python Example")
    print("-"*70)
    
    azure_code = """
from azure.quantum import Workspace
from azure.quantum.qiskit import AzureQuantumProvider

# Connect to Azure Quantum workspace
workspace = Workspace(
    resource_id="/subscriptions/.../Microsoft.Quantum/Workspaces/...",
    location="eastus"
)

# Get available targets
print("Available targets:")
for target in workspace.get_targets():
    print(f"  - {target.name}")

# Submit QAOA job
from qiskit import QuantumCircuit
circuit = build_qaoa_circuit(gammas, betas, edges)

# Choose backend
backend = workspace.get_targets("ionq.simulator")

# Submit job
job = backend.run(circuit, shots=1000)
result = job.result()
counts = result.get_counts()

print("Results:", counts)
"""
    
    print(azure_code)


def example_6_qaoa_vs_dwave_comparison():
    """
    Example 6: Direct Comparison - QAOA vs D-Wave
    
    Solve the same problem with both approaches.
    """
    print("\n" + "="*70)
    print("EXAMPLE 6: QAOA vs D-Wave Comparison")
    print("="*70)
    
    print("\nSame Problem, Different Quantum Approaches:")
    print("-"*70)
    
    print("\nProblem: MaxCut on Triangle Graph")
    print("  Goal: Partition vertices to maximize edge cuts")
    
    print("\n" + "-"*70)
    print("D-Wave Approach (Tutorials 1-4)")
    print("-"*70)
    print("""
# 1. Formulate as QUBO
Q = {
    (0, 0): 0.5,
    (1, 1): 0.5,
    (2, 2): 0.5,
    (0, 1): -0.5,
    (1, 2): -0.5,
    (0, 2): -0.5
}

# 2. Convert to BQM
bqm = dimod.BinaryQuadraticModel.from_qubo(Q)

# 3. Solve with quantum annealing
sampler = EmbeddingComposite(DWaveSampler())
sampleset = sampler.sample(bqm, num_reads=1000)

# 4. Get best solution
best = sampleset.first
print(f"Solution: {best.sample}")
print(f"Energy: {best.energy}")
    """)
    
    print("\n  Time: ~milliseconds (single anneal)")
    print("  Qubits used: 3 (embedded to ~15 on hardware)")
    print("  Result: Deterministic after many samples")
    
    print("\n" + "-"*70)
    print("QAOA Approach (This Tutorial)")
    print("-"*70)
    print("""
# 1. Define cost Hamiltonian (same as Ising!)
edges = [(0,1), (1,2), (0,2)]

# 2. Choose QAOA parameters
p = 2  # circuit depth
initial_params = [0.5, 0.5, 1.0, 1.0]  # gammas, betas

# 3. Optimize parameters
def cost_function(params):
    return evaluate_qaoa(params, edges, shots=1000)

result = minimize(cost_function, initial_params)
optimal_params = result.x

# 4. Run with optimal parameters
final_result = run_qaoa(optimal_params, edges, shots=10000)
best_bitstring = max(final_result, key=final_result.get)

print(f"Solution: {best_bitstring}")
print(f"Cost: {calculate_cost(best_bitstring, edges)}")
    """)
    
    print("\n  Time: ~seconds to minutes (optimization loop)")
    print("  Qubits used: 3 (direct, no embedding)")
    print("  Result: Probabilistic, improves with p")
    
    print("\n" + "-"*70)
    print("Trade-offs Summary")
    print("-"*70)
    
    comparison = [
        ["Aspect", "D-Wave (Annealing)", "QAOA (Gate-Based)"],
        ["-"*20, "-"*25, "-"*25],
        ["Hardware", "Specialized", "Universal"],
        ["Qubits", "5000+", "50-1000"],
        ["Problem Size", "Large", "Medium"],
        ["Embedding", "Required", "Not required"],
        ["Circuit Depth", "N/A", "Tunable (p)"],
        ["Time per run", "Milliseconds", "Varies"],
        ["Optimization", "Hardware", "Hybrid (classical)"],
        ["Accessibility", "D-Wave only", "Multiple vendors"],
        ["Cost", "Pay per QPU time", "Pay per shot/job"],
    ]
    
    for row in comparison:
        print(f"  {row[0]:20} | {row[1]:25} | {row[2]:25}")
    
    print("\n  Best use cases:")
    print("    D-Wave: Large optimization problems, many variables")
    print("    QAOA:   Medium problems, flexible hardware, learning/research")


def example_7_food_production_with_qaoa():
    """
    Example 7: Food Production Problem with QAOA
    
    Adapt our food production scenario for QAOA.
    """
    print("\n" + "="*70)
    print("EXAMPLE 7: Food Production with QAOA (Conceptual)")
    print("="*70)
    
    print("\nChallenges Adapting Food Production to QAOA:")
    print("-"*70)
    
    farms, foods, food_groups, config = load_food_data('simple')
    
    print(f"\n  Original problem:")
    print(f"    Farms: {len(farms)}")
    print(f"    Foods: {len(foods)}")
    print(f"    Decision variables: {len(farms) * len(foods)} (binary)")
    print(f"    Constraints: Land availability, diversity, etc.")
    
    print("\n  QAOA Limitations:")
    print("    1. Number of qubits limited (~100 on current hardware)")
    print("    2. Constraints difficult to encode in Hamiltonian")
    print("    3. Continuous/integer variables need discretization")
    print("    4. Circuit depth grows with problem complexity")
    
    print("\n  Simplified QAOA Formulation:")
    print("    - Reduce to 3 farms, 4 foods = 12 qubits")
    print("    - Binary variables only (plant or not)")
    print("    - Soft constraints (penalty terms in Hamiltonian)")
    
    print("\n" + "-"*70)
    print("QAOA Hamiltonian for Simplified Problem")
    print("-"*70)
    
    print("""
  H_cost = H_objective + lambda * H_constraints
  
  H_objective = -sum(weight_ij * Z_ij) for all farm-food pairs
                (Maximize weighted benefits)
  
  H_constraints = sum((sum(Z_ij over foods) - capacity)^2 for each farm)
                  (Penalize exceeding farm capacity)
  
  lambda = penalty weight (tuned like Tutorial 3!)
    """)
    
    print("\n  This is exactly the Ising formulation from Tutorial 2!")
    print("  But solved with QAOA instead of annealing")
    
    print("\n" + "-"*70)
    print("When to Use QAOA for This Problem")
    print("-"*70)
    
    print("\n  Advantages:")
    print("    ✓ Don't need D-Wave hardware")
    print("    ✓ Can run on various quantum computers")
    print("    ✓ Educational: understand gate-based quantum")
    print("    ✓ Flexible circuit design")
    
    print("\n  Disadvantages:")
    print("    ✗ Limited problem size (qubits)")
    print("    ✗ Slower (classical optimization loop)")
    print("    ✗ Requires circuit engineering")
    print("    ✗ Current hardware noisy")
    
    print("\n  Recommendation:")
    print("    - D-Wave/CQM: For production optimization (Tutorials 1-6)")
    print("    - QAOA: For learning gate-based quantum computing")
    print("    - Classical (PuLP): For exact solutions (Tutorial 6)")


def main():
    """
    Main function to run all examples.
    """
    print("\n" + "="*70)
    print("TUTORIAL 7: QAOA WITH Q#")
    print("="*70)
    print("\nThis tutorial demonstrates QAOA (Quantum Approximate Optimization")
    print("Algorithm), a gate-based alternative to D-Wave's quantum annealing.")
    print("\nQAOA uses universal quantum computers with quantum gates,")
    print("unlike D-Wave's specialized annealing hardware.")
    
    # Run examples
    example_1_qaoa_vs_quantum_annealing()
    
    example_2_max_cut_problem()
    
    example_3_qaoa_circuit_explanation()
    
    example_4_qaoa_pseudocode()
    
    example_5_azure_quantum_setup()
    
    example_6_qaoa_vs_dwave_comparison()
    
    example_7_food_production_with_qaoa()
    
    print("\n" + "="*70)
    print("TUTORIAL COMPLETE!")
    print("="*70)
    print("\nKey Takeaways:")
    print("  1. QAOA is a gate-based alternative to quantum annealing")
    print("  2. Solves same problems (QUBO/Ising) with different hardware")
    print("  3. Uses parameterized circuits + classical optimization")
    print("  4. More flexible but currently more limited in scale")
    print("  5. Runs on universal quantum computers (IonQ, Rigetti, etc.)")
    
    print("\nNext Steps:")
    print("  - Install Azure Quantum SDK: pip install azure-quantum qsharp")
    print("  - Create Q# project: dotnet new console -lang Q#")
    print("  - Implement QAOA for MaxCut problem")
    print("  - Compare results with D-Wave approach (Tutorial 2)")
    print("  - Explore other gate-based algorithms (VQE, Grover)")
    
    print("\nResources:")
    print("  - Azure Quantum: https://azure.microsoft.com/quantum")
    print("  - Q# Documentation: https://docs.microsoft.com/quantum")
    print("  - QAOA Paper: https://arxiv.org/abs/1411.4028")
    print("  - Qiskit QAOA: https://qiskit.org/textbook/ch-applications/qaoa.html")


if __name__ == "__main__":
    main()
