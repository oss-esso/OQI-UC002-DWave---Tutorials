"""
Tutorial 2: QUBO Formulation Basics

This tutorial teaches you the fundamentals of QUBO (Quadratic Unconstrained Binary 
Optimization) formulation and how to work with it in DIMOD. You'll learn:

1. What is QUBO and how it differs from BQM
2. How to formulate optimization problems as QUBO
3. How to convert QUBO to DIMOD BQM
4. Common patterns for encoding constraints as penalties
5. Using different samplers (ExactSolver, SimulatedAnnealing, RandomSampler)

QUBO Basics:
- QUBO is a matrix representation: minimize x^T Q x
- Q is an upper-triangular matrix
- Diagonal elements = linear coefficients
- Off-diagonal elements = quadratic coefficients (interaction/coupling)
- All variables are binary (0 or 1)
"""

import dimod
import numpy as np
from typing import Dict, Tuple, List


def example_1_qubo_matrix_basics():
    """
    Example 1: Understanding QUBO matrix representation
    
    Shows how to represent a problem using a QUBO matrix and convert it to BQM.
    
    Problem: minimize 2*x0 - 3*x1 + x2 + 4*x0*x1 - 2*x1*x2
    
    QUBO matrix Q:
        x0  x1  x2
    x0 [ 2   4   0 ]
    x1 [ 0  -3  -2 ]
    x2 [ 0   0   1 ]
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: QUBO Matrix Representation")
    print("="*70)
    
    # Define QUBO as dictionary (i, j): coefficient
    # Note: QUBO uses (i,j) where i <= j (upper triangular)
    Q = {
        (0, 0): 2,    # Linear term for x0
        (1, 1): -3,   # Linear term for x1
        (2, 2): 1,    # Linear term for x2
        (0, 1): 4,    # Quadratic term x0*x1
        (1, 2): -2,   # Quadratic term x1*x2
    }
    
    print("\nQUBO dictionary representation:")
    print(f"  Q = {Q}")
    
    # Convert QUBO to BQM
    bqm = dimod.BinaryQuadraticModel.from_qubo(Q)
    
    print("\nConverted to BQM:")
    print(f"  Linear terms: {dict(bqm.linear)}")
    print(f"  Quadratic terms: {dict(bqm.quadratic)}")
    print(f"  Offset: {bqm.offset}")
    
    # Alternative: Create QUBO as numpy array
    Q_matrix = np.array([
        [2,  4,  0],
        [0, -3, -2],
        [0,  0,  1]
    ])
    
    print("\nQUBO as numpy matrix:")
    print(Q_matrix)
    
    # Solve using exact solver (tries all 2^n combinations)
    print("\nSolving with ExactSolver (tries all combinations)...")
    sampler = dimod.ExactSolver()
    sampleset = sampler.sample(bqm)
    
    print("\nAll possible solutions (sorted by energy):")
    for i, sample_data in enumerate(list(sampleset.data(['sample', 'energy']))[:5]):
        sample = sample_data.sample
        energy = sample_data.energy
        print(f"  {i+1}. x0={sample[0]}, x1={sample[1]}, x2={sample[2]} -> energy = {energy}")
    
    print(f"\nOptimal solution:")
    print(f"  Variables: {sampleset.first.sample}")
    print(f"  Energy: {sampleset.first.energy}")
    
    return bqm, sampleset


def example_2_constraint_as_penalty():
    """
    Example 2: Encoding constraints as penalty terms
    
    Problem: Select exactly 2 out of 4 items
    - Constraint: x0 + x1 + x2 + x3 = 2
    
    We convert this to penalty: P * (x0 + x1 + x2 + x3 - 2)^2
    When expanded: P * (x0 + x1 + x2 + x3 - 2)^2
                 = P * (x0^2 + x1^2 + x2^2 + x3^2 + 
                        2*x0*x1 + 2*x0*x2 + 2*x0*x3 + 2*x1*x2 + 2*x1*x3 + 2*x2*x3 +
                        -4*x0 - 4*x1 - 4*x2 - 4*x3 + 4)
    
    Since x^2 = x for binary variables:
                 = P * (x0 + x1 + x2 + x3 + 
                        2*x0*x1 + 2*x0*x2 + 2*x0*x3 + 2*x1*x2 + 2*x1*x3 + 2*x2*x3 +
                        -4*x0 - 4*x1 - 4*x2 - 4*x3 + 4)
    
    Simplifying:
                 = P * (-3*x0 - 3*x1 - 3*x2 - 3*x3 + 
                        2*x0*x1 + 2*x0*x2 + 2*x0*x3 + 2*x1*x2 + 2*x1*x3 + 2*x2*x3 + 4)
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Constraint as Penalty Term")
    print("="*70)
    
    print("\nProblem: Select exactly 2 out of 4 items")
    print("Constraint: x0 + x1 + x2 + x3 = 2")
    
    # Each item has a different value (objective to maximize)
    item_values = [5, 3, 8, 4]  # We want to maximize value
    # Convert to minimization: negate the values
    
    print(f"\nItem values: {item_values}")
    print("Objective: Maximize total value of selected items")
    
    # Build QUBO with constraint penalty
    penalty_weight = 20  # Large penalty for constraint violation
    
    Q = {}
    
    # Add objective (maximize value = minimize negative value)
    for i, value in enumerate(item_values):
        Q[(i, i)] = -value  # Linear terms (negate to minimize)
    
    # Add constraint penalty: P * (-3*xi + 2*xi*xj)
    for i in range(4):
        Q[(i, i)] += penalty_weight * (-3)  # Linear part of penalty
    
    for i in range(4):
        for j in range(i+1, 4):
            Q[(i, j)] = penalty_weight * 2  # Quadratic part of penalty
    
    # Add constant offset from penalty
    offset = penalty_weight * 4
    
    print(f"\nQUBO with constraint penalty (weight={penalty_weight}):")
    print(f"  Q = {Q}")
    print(f"  Offset = {offset}")
    
    # Convert to BQM
    bqm = dimod.BinaryQuadraticModel.from_qubo(Q, offset=offset)
    
    # Solve
    print("\nSolving...")
    sampler = dimod.SimulatedAnnealingSampler()
    sampleset = sampler.sample(bqm, num_reads=100, seed=42)
    
    print("\nTop 5 solutions:")
    for i, sample_data in enumerate(list(sampleset.data(['sample', 'energy', 'num_occurrences']))[:5]):
        sample = sample_data.sample
        energy = sample_data.energy
        occurrences = sample_data.num_occurrences
        
        selected = [j for j in range(4) if sample[j] == 1]
        num_selected = len(selected)
        total_value = sum(item_values[j] for j in selected)
        
        violation = abs(num_selected - 2)
        
        print(f"\n  Solution {i+1}:")
        print(f"    Selected items: {selected}")
        print(f"    Total value: {total_value}")
        print(f"    Constraint satisfied: {violation == 0}")
        print(f"    Energy: {energy:.2f}")
        print(f"    Occurrences: {occurrences}")
    
    return bqm, sampleset


def example_3_number_partitioning():
    """
    Example 3: Number Partitioning Problem
    
    Problem: Partition a set of numbers into two subsets with equal sums.
    
    Given numbers: [3, 1, 5, 2, 7, 4]
    Find partition: S1 and S2 such that sum(S1) = sum(S2)
    
    Formulation:
    - Variable xi = 1 if number i is in S1, 0 if in S2
    - Minimize: (sum of numbers in S1 - sum of numbers in S2)^2
    - This equals: (sum of xi*ni - sum of (1-xi)*ni)^2
    -            = (2*sum(xi*ni) - sum(ni))^2
    
    Let S = sum(ni), then minimize: (2*sum(xi*ni) - S)^2
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Number Partitioning Problem")
    print("="*70)
    
    numbers = [3, 1, 5, 2, 7, 4]
    total_sum = sum(numbers)
    
    print(f"\nNumbers to partition: {numbers}")
    print(f"Total sum: {total_sum}")
    print(f"Target sum per partition: {total_sum / 2}")
    
    # Build QUBO
    # Objective: minimize (2*sum(xi*ni) - S)^2
    # Expanding: 4*sum(xi^2*ni^2) + 4*sum(sum(xi*xj*ni*nj)) - 4*S*sum(xi*ni) + S^2
    # Since xi^2 = xi: 4*sum(xi*ni^2) + 4*sum(sum(xi*xj*ni*nj)) - 4*S*sum(xi*ni) + S^2
    
    n = len(numbers)
    Q = {}
    
    # Linear terms: 4*ni^2 - 4*S*ni
    for i in range(n):
        Q[(i, i)] = 4 * numbers[i]**2 - 4 * total_sum * numbers[i]
    
    # Quadratic terms: 4*ni*nj
    for i in range(n):
        for j in range(i+1, n):
            Q[(i, j)] = 8 * numbers[i] * numbers[j]  # Factor of 2 because we only use upper triangle
    
    # Offset
    offset = total_sum**2
    
    print(f"\nQUBO formulation:")
    print(f"  Variables: x0 to x{n-1} (1=partition 1, 0=partition 2)")
    
    # Convert to BQM
    bqm = dimod.BinaryQuadraticModel.from_qubo(Q, offset=offset)
    
    # Solve
    print("\nSolving with Simulated Annealing...")
    sampler = dimod.SimulatedAnnealingSampler()
    sampleset = sampler.sample(bqm, num_reads=200, seed=42)
    
    print("\nTop 5 solutions:")
    for i, sample_data in enumerate(list(sampleset.data(['sample', 'energy', 'num_occurrences']))[:5]):
        sample = sample_data.sample
        energy = sample_data.energy
        occurrences = sample_data.num_occurrences
        
        partition_1 = [numbers[j] for j in range(n) if sample[j] == 1]
        partition_2 = [numbers[j] for j in range(n) if sample[j] == 0]
        sum_1 = sum(partition_1)
        sum_2 = sum(partition_2)
        difference = abs(sum_1 - sum_2)
        
        print(f"\n  Solution {i+1}:")
        print(f"    Partition 1: {partition_1} (sum={sum_1})")
        print(f"    Partition 2: {partition_2} (sum={sum_2})")
        print(f"    Difference: {difference}")
        print(f"    Energy: {energy:.2f}")
        print(f"    Occurrences: {occurrences}")
    
    return bqm, sampleset


def example_4_comparing_samplers():
    """
    Example 4: Comparing different samplers
    
    Shows the difference between:
    - ExactSolver: Tries all 2^n combinations (guaranteed optimal but slow for large n)
    - SimulatedAnnealingSampler: Heuristic approach (fast but not guaranteed optimal)
    - RandomSampler: Random sampling (baseline)
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Comparing Different Samplers")
    print("="*70)
    
    # Create a simple optimization problem
    # Minimize: -5*x0 - 3*x1 - 8*x2 - 6*x3 + 4*x0*x1 + 2*x0*x2 + 3*x1*x3
    Q = {
        (0, 0): -5,
        (1, 1): -3,
        (2, 2): -8,
        (3, 3): -6,
        (0, 1): 4,
        (0, 2): 2,
        (1, 3): 3,
    }
    
    bqm = dimod.BinaryQuadraticModel.from_qubo(Q)
    
    print("\nProblem:")
    print(f"  4 binary variables")
    print(f"  Objective coefficients: {Q}")
    
    # 1. ExactSolver
    print("\n1. ExactSolver (tries all 16 combinations):")
    exact_sampler = dimod.ExactSolver()
    exact_sampleset = exact_sampler.sample(bqm)
    print(f"   Best energy: {exact_sampleset.first.energy}")
    print(f"   Best solution: {exact_sampleset.first.sample}")
    print(f"   Total samples evaluated: {len(exact_sampleset)}")
    
    # 2. SimulatedAnnealingSampler
    print("\n2. SimulatedAnnealingSampler (heuristic optimization):")
    sa_sampler = dimod.SimulatedAnnealingSampler()
    sa_sampleset = sa_sampler.sample(bqm, num_reads=100, seed=42)
    print(f"   Best energy: {sa_sampleset.first.energy}")
    print(f"   Best solution: {sa_sampleset.first.sample}")
    print(f"   Unique solutions found: {len([tuple(s.values()) for s in sa_sampleset.samples()])}")
    
    # 3. RandomSampler
    print("\n3. RandomSampler (random baseline):")
    random_sampler = dimod.RandomSampler()
    random_sampleset = random_sampler.sample(bqm, num_reads=100, seed=42)
    print(f"   Best energy: {random_sampleset.first.energy}")
    print(f"   Best solution: {random_sampleset.first.sample}")
    
    # Compare
    print("\nComparison:")
    print(f"  Optimal energy (ExactSolver): {exact_sampleset.first.energy}")
    print(f"  SA found optimal: {sa_sampleset.first.energy == exact_sampleset.first.energy}")
    print(f"  Random found optimal: {random_sampleset.first.energy == exact_sampleset.first.energy}")
    
    return bqm, exact_sampleset, sa_sampleset, random_sampleset


def example_5_qubo_from_ising():
    """
    Example 5: Converting between QUBO and Ising formulations
    
    QUBO uses binary variables (0, 1)
    Ising uses spin variables (-1, +1)
    
    Conversion: s = 2*x - 1 (or x = (s + 1)/2)
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: QUBO and Ising Conversions")
    print("="*70)
    
    # Create a simple BQM in QUBO form
    Q = {
        (0, 0): 1,
        (1, 1): -1,
        (0, 1): 2,
    }
    
    print("\nOriginal QUBO:")
    print(f"  Q = {Q}")
    
    bqm_qubo = dimod.BinaryQuadraticModel.from_qubo(Q)
    print(f"\n  As BQM (BINARY vartype):")
    print(f"    Linear: {dict(bqm_qubo.linear)}")
    print(f"    Quadratic: {dict(bqm_qubo.quadratic)}")
    print(f"    Offset: {bqm_qubo.offset}")
    
    # Convert to Ising (creates a copy)
    bqm_ising = bqm_qubo.copy()
    bqm_ising.change_vartype('SPIN', inplace=True)
    print(f"\n  Converted to Ising (SPIN vartype):")
    print(f"    Linear (h): {dict(bqm_ising.linear)}")
    print(f"    Quadratic (J): {dict(bqm_ising.quadratic)}")
    print(f"    Offset: {bqm_ising.offset}")
    
    # Solve in both representations
    sampler = dimod.SimulatedAnnealingSampler()
    
    sampleset_qubo = sampler.sample(bqm_qubo, num_reads=50, seed=42)
    sampleset_ising = sampler.sample(bqm_ising, num_reads=50, seed=42)
    
    print("\nSolutions:")
    print(f"  QUBO best: {sampleset_qubo.first.sample}, energy={sampleset_qubo.first.energy}")
    print(f"  Ising best: {sampleset_ising.first.sample}, energy={sampleset_ising.first.energy}")
    
    # Convert Ising solution to QUBO variables
    ising_sample = sampleset_ising.first.sample
    qubo_from_ising = {k: (v + 1) // 2 for k, v in ising_sample.items()}
    print(f"  Ising solution as QUBO variables: {qubo_from_ising}")
    
    return bqm_qubo, bqm_ising


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("TUTORIAL 2: QUBO FORMULATION BASICS")
    print("="*70)
    print("\nThis tutorial demonstrates QUBO formulation techniques and")
    print("different solving approaches using D-Wave's DIMOD library.")
    
    # Run all examples
    example_1_qubo_matrix_basics()
    example_2_constraint_as_penalty()
    example_3_number_partitioning()
    example_4_comparing_samplers()
    example_5_qubo_from_ising()
    
    print("\n" + "="*70)
    print("TUTORIAL 2 COMPLETE")
    print("="*70)
    print("\nKey Takeaways:")
    print("1. QUBO is a matrix formulation: minimize x^T Q x")
    print("2. Constraints can be encoded as penalty terms (violation^2 * weight)")
    print("3. ExactSolver finds optimal solution but is slow for large problems")
    print("4. SimulatedAnnealing is a fast heuristic sampler")
    print("5. QUBO and Ising are equivalent formulations")
    print("\nNext: tutorial_03_scenario_to_qubo.py")


if __name__ == "__main__":
    main()
