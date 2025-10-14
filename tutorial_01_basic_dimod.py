"""
Tutorial 1: Basic DIMOD BQM Construction

This tutorial teaches you how to build a Binary Quadratic Model (BQM) using DIMOD,
D-Wave's library for expressing optimization problems. You'll learn:

1. What is a BQM and its components (linear and quadratic terms)
2. How to construct a BQM from scratch
3. How to solve it using a simulated annealer
4. How to interpret the results

DIMOD BQM Basics:
- BQM = Binary Quadratic Model
- Variables are binary (0 or 1)
- Objective = sum of linear terms + sum of quadratic terms + offset
- Linear terms: coefficients for individual variables
- Quadratic terms: coefficients for variable interactions
"""

import dimod
import numpy as np
from typing import Dict, Any


def example_1_simple_bqm():
    """
    Example 1: Create a simple BQM with 3 variables
    
    Problem: Minimize x0 - 2*x1 + 3*x2 - x0*x1 + 2*x1*x2
    
    This represents an energy landscape where:
    - x0 has coefficient 1 (increasing x0 increases energy)
    - x1 has coefficient -2 (increasing x1 decreases energy)
    - x2 has coefficient 3 (increasing x2 increases energy)
    - x0 and x1 interaction has coefficient -1 (they prefer different values)
    - x1 and x2 interaction has coefficient 2 (they prefer same values)
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Simple BQM Construction")
    print("="*70)
    
    # Method 1: Build BQM using add_variable and add_interaction
    bqm = dimod.BinaryQuadraticModel('BINARY')
    
    # Add linear terms (biases for individual variables)
    bqm.add_variable('x0', 1.0)   # Coefficient for x0
    bqm.add_variable('x1', -2.0)  # Coefficient for x1
    bqm.add_variable('x2', 3.0)   # Coefficient for x2
    
    # Add quadratic terms (interactions between variables)
    bqm.add_interaction('x0', 'x1', -1.0)  # Coefficient for x0*x1
    bqm.add_interaction('x1', 'x2', 2.0)   # Coefficient for x1*x2
    
    print("\nBQM Structure:")
    print(f"  Number of variables: {len(bqm.variables)}")
    print(f"  Variables: {list(bqm.variables)}")
    print(f"  Linear terms: {dict(bqm.linear)}")
    print(f"  Quadratic terms: {dict(bqm.quadratic)}")
    print(f"  Offset: {bqm.offset}")
    
    # Solve using simulated annealing
    print("\nSolving with Simulated Annealing...")
    sampler = dimod.SimulatedAnnealingSampler()
    sampleset = sampler.sample(bqm, num_reads=100)
    
    # Get the best solution
    best_sample = sampleset.first.sample
    best_energy = sampleset.first.energy
    
    print(f"\nBest solution found:")
    print(f"  Variables: {best_sample}")
    print(f"  Energy: {best_energy}")
    
    # Verify the energy calculation manually
    x0, x1, x2 = best_sample['x0'], best_sample['x1'], best_sample['x2']
    manual_energy = (1*x0 - 2*x1 + 3*x2 - x0*x1 + 2*x1*x2)
    print(f"  Manual verification: {manual_energy}")
    
    return bqm, sampleset


def example_2_different_construction_methods():
    """
    Example 2: Different ways to construct the same BQM
    
    Shows multiple equivalent methods to build BQMs in DIMOD.
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Different BQM Construction Methods")
    print("="*70)
    
    # Method A: Using dictionaries
    print("\nMethod A: Using dictionaries")
    linear = {'a': -1, 'b': 2, 'c': -3}
    quadratic = {('a', 'b'): 1.5, ('b', 'c'): -2.0}
    offset = 0.5
    
    bqm_a = dimod.BinaryQuadraticModel(linear, quadratic, offset, 'BINARY')
    print(f"  BQM created with {len(bqm_a.variables)} variables")
    
    # Method B: Starting empty and building up
    print("\nMethod B: Building incrementally")
    bqm_b = dimod.BinaryQuadraticModel('BINARY')
    
    for var, coeff in linear.items():
        bqm_b.add_variable(var, coeff)
    
    for (u, v), coeff in quadratic.items():
        bqm_b.add_interaction(u, v, coeff)
    
    bqm_b.offset = offset
    print(f"  BQM created with {len(bqm_b.variables)} variables")
    
    # Verify they're equivalent
    print("\nVerification:")
    print(f"  BQMs are equivalent: {bqm_a == bqm_b}")
    
    # Solve both
    sampler = dimod.SimulatedAnnealingSampler()
    sampleset_a = sampler.sample(bqm_a, num_reads=50, seed=42)
    
    print(f"\nBest solution:")
    print(f"  Variables: {sampleset_a.first.sample}")
    print(f"  Energy: {sampleset_a.first.energy}")
    
    return bqm_a, sampleset_a


def example_3_practical_problem():
    """
    Example 3: A practical optimization problem
    
    Problem: Resource allocation
    - We have 3 tasks that can be assigned to 2 processors
    - Each task has a cost on each processor
    - Tasks that run on the same processor have an interaction cost
    
    Variables:
    - t1_p1: Task 1 on Processor 1 (binary)
    - t1_p2: Task 1 on Processor 2 (binary)
    - t2_p1: Task 2 on Processor 1 (binary)
    - t2_p2: Task 2 on Processor 2 (binary)
    - t3_p1: Task 3 on Processor 1 (binary)
    - t3_p2: Task 3 on Processor 2 (binary)
    
    Constraints (as penalties):
    - Each task must be assigned to exactly one processor
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Practical Resource Allocation Problem")
    print("="*70)
    
    # Task costs on each processor
    task_costs = {
        ('t1', 'p1'): 2.0,
        ('t1', 'p2'): 3.0,
        ('t2', 'p1'): 1.5,
        ('t2', 'p2'): 2.5,
        ('t3', 'p1'): 3.0,
        ('t3', 'p2'): 1.0,
    }
    
    # Interaction costs (tasks on same processor)
    interaction_costs = {
        (('t1', 'p1'), ('t2', 'p1')): 0.5,
        (('t1', 'p1'), ('t3', 'p1')): 0.8,
        (('t2', 'p1'), ('t3', 'p1')): 0.3,
        (('t1', 'p2'), ('t2', 'p2')): 0.4,
        (('t1', 'p2'), ('t3', 'p2')): 0.6,
        (('t2', 'p2'), ('t3', 'p2')): 0.2,
    }
    
    print("\nProblem Setup:")
    print("  Tasks: t1, t2, t3")
    print("  Processors: p1, p2")
    print(f"  Task costs: {task_costs}")
    
    # Build BQM
    bqm = dimod.BinaryQuadraticModel('BINARY')
    
    # Add linear terms (task assignment costs)
    for (task, proc), cost in task_costs.items():
        var_name = f"{task}_{proc}"
        bqm.add_variable(var_name, cost)
    
    # Add quadratic terms (interaction costs)
    for ((t1, p1), (t2, p2)), cost in interaction_costs.items():
        var1 = f"{t1}_{p1}"
        var2 = f"{t2}_{p2}"
        bqm.add_interaction(var1, var2, cost)
    
    # Add constraints: each task must be assigned to exactly one processor
    # Constraint: t1_p1 + t1_p2 = 1 can be penalized as (t1_p1 + t1_p2 - 1)^2
    # Expanding: t1_p1^2 + t1_p2^2 + 2*t1_p1*t1_p2 - 2*t1_p1 - 2*t1_p2 + 1
    # Since variables are binary, x^2 = x, so: t1_p1 + t1_p2 + 2*t1_p1*t1_p2 - 2*t1_p1 - 2*t1_p2 + 1
    # Simplifying: -t1_p1 - t1_p2 + 2*t1_p1*t1_p2 + 1
    
    print("\nBQM BEFORE adding constraints:")
    print(f"  Linear terms sample: t1_p1={bqm.get_linear('t1_p1')}, t1_p2={bqm.get_linear('t1_p2')}")
    print(f"  Quadratic terms: {len(bqm.quadratic)}")
    print(f"  Offset: {bqm.offset}")
    
    penalty_weight = 10.0  # Large penalty for violating constraints
    
    for task in ['t1', 't2', 't3']:
        var1 = f"{task}_p1"
        var2 = f"{task}_p2"
        
        # Add penalty terms (these ADD TO existing coefficients)
        bqm.add_variable(var1, -penalty_weight)  # Adds to linear coefficient
        bqm.add_variable(var2, -penalty_weight)  # Adds to linear coefficient
        bqm.add_interaction(var1, var2, 2*penalty_weight)  # Adds quadratic term
        bqm.offset += penalty_weight
    
    print("\nBQM AFTER adding constraints:")
    print(f"  Linear terms sample: t1_p1={bqm.get_linear('t1_p1')}, t1_p2={bqm.get_linear('t1_p2')}")
    print(f"  Notice: Constraint penalties (-10) added to objective costs")
    print(f"  Quadratic terms: {len(bqm.quadratic)} (3 new constraint interactions added)")
    print(f"  Offset: {bqm.offset} (3 tasks x penalty = 30)")
    
    print(f"\nBQM with constraints:")
    print(f"  Total variables: {len(bqm.variables)}")
    print(f"  Constraint penalty weight: {penalty_weight}")

    
    # Solve
    print("\nSolving...")
    sampler = dimod.SimulatedAnnealingSampler()
    sampleset = sampler.sample(bqm, num_reads=200, seed=42)
    
    # Display top 5 solutions
    print("\nTop 5 solutions:")
    for i, sample in enumerate(list(sampleset.data(['sample', 'energy', 'num_occurrences']))[:5]):
        print(f"\n  Solution {i+1}:")
        print(f"    Energy: {sample.energy:.2f}")
        print(f"    Occurrences: {sample.num_occurrences}")
        
        # Show assignments
        assignments = sample.sample
        for task in ['t1', 't2', 't3']:
            if assignments[f"{task}_p1"] == 1:
                print(f"    {task} -> Processor 1")
            elif assignments[f"{task}_p2"] == 1:
                print(f"    {task} -> Processor 2")
            else:
                print(f"    {task} -> UNASSIGNED (constraint violation!)")
    
    return bqm, sampleset


def example_4_understanding_energy():
    """
    Example 4: Understanding energy landscapes
    
    Shows how different variable assignments lead to different energies.
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Understanding Energy Landscapes")
    print("="*70)
    
    # Simple BQM: minimize -x0 - x1 - 2*x0*x1
    # This encourages both variables to be 1
    linear = {'x0': -1, 'x1': -1}
    quadratic = {('x0', 'x1'): -2}
    
    bqm = dimod.BinaryQuadraticModel(linear, quadratic, 0, 'BINARY')
    
    print("\nObjective: minimize -x0 - x1 - 2*x0*x1")
    print("\nEnergy for all possible assignments:")
    
    # Evaluate all 4 possible assignments
    for x0 in [0, 1]:
        for x1 in [0, 1]:
            sample = {'x0': x0, 'x1': x1}
            energy = bqm.energy(sample)
            print(f"  x0={x0}, x1={x1}: energy = {energy}")
    
    print("\nObservation: The lowest energy (-4) occurs when both variables are 1.")
    print("This is the optimal solution.")
    
    # Solve to confirm
    sampler = dimod.SimulatedAnnealingSampler()
    sampleset = sampler.sample(bqm, num_reads=100)
    
    print(f"\nSolver found optimal solution:")
    print(f"  Variables: {sampleset.first.sample}")
    print(f"  Energy: {sampleset.first.energy}")
    
    return bqm, sampleset


def example_5_bqm_vs_cqm():
    """
    Example 5: BQM with Penalties vs CQM (Constrained Quadratic Model)
    
    Shows the difference between:
    - BQM: Must encode constraints as penalties (soft constraints)
    - CQM: Has built-in hard constraints (guaranteed satisfaction)
    
    Problem: Same resource allocation from Example 3
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: BQM with Penalties vs CQM")
    print("="*70)
    
    print("\n" + "-"*70)
    print("BQM APPROACH: Constraints as Penalties (Soft Constraints)")
    print("-"*70)
    
    print("\nPros:")
    print("  + Works with any BQM solver (SA, QPU)")
    print("  + Simple to implement")
    print("  + Can trade off constraint violation vs objective")
    
    print("\nCons:")
    print("  - Constraints may be violated if penalty too small")
    print("  - Must tune penalty weights carefully")
    print("  - Adds extra terms to the model (larger problem)")
    print("  - No guarantee of feasibility")
    
    print("\nExample from Tutorial 1, Example 3:")
    print("  Constraint: Each task assigned to exactly one processor")
    print("  Penalty: P * (x_p1 + x_p2 - 1)^2")
    print("  If P=10: Violation costs 10, 20, or 30 energy units")
    print("  If P too small: Solver might violate to improve objective")
    print("  If P too large: Solver ignores objective, just satisfies constraints")
    
    # Show BQM formulation
    bqm = dimod.BinaryQuadraticModel('BINARY')
    bqm.add_variable('t1_p1', 2.0)  # Task cost
    bqm.add_variable('t1_p2', 3.0)
    
    penalty = 10.0
    bqm.add_variable('t1_p1', -penalty)  # Constraint penalty
    bqm.add_variable('t1_p2', -penalty)
    bqm.add_interaction('t1_p1', 't1_p2', 2*penalty)
    bqm.offset = penalty
    
    print(f"\n  BQM linear coefficients:")
    print(f"    t1_p1: {bqm.get_linear('t1_p1')} (2.0 cost - 10.0 penalty)")
    print(f"    t1_p2: {bqm.get_linear('t1_p2')} (3.0 cost - 10.0 penalty)")
    print(f"  BQM quadratic: {dict(bqm.quadratic)}")
    print(f"  BQM offset: {bqm.offset}")
    
    print("\n" + "-"*70)
    print("CQM APPROACH: Built-in Hard Constraints")
    print("-"*70)
    
    print("\nPros:")
    print("  + Constraints are ALWAYS satisfied (hard constraints)")
    print("  + No penalty tuning needed")
    print("  + Cleaner formulation (constraints separate from objective)")
    print("  + Better for complex constraint types (inequalities, etc)")
    
    print("\nCons:")
    print("  - Requires CQM-capable solver (Hybrid CQM solver)")
    print("  - Not available on all D-Wave systems")
    print("  - May return 'infeasible' if constraints can't be satisfied")
    
    try:
        from dimod import ConstrainedQuadraticModel, Binary
        
        print("\nCQM formulation of same problem:")
        cqm = ConstrainedQuadraticModel()
        
        # Define variables
        t1_p1 = Binary('t1_p1')
        t1_p2 = Binary('t1_p2')
        
        # Set objective (just the costs, no penalties needed!)
        cqm.set_objective(2.0*t1_p1 + 3.0*t1_p2)
        
        # Add constraint as a HARD constraint
        cqm.add_constraint(t1_p1 + t1_p2 == 1, label='task1_assignment')
        
        print("  CQM objective: 2.0*t1_p1 + 3.0*t1_p2")
        print("  CQM constraints:")
        print("    task1_assignment: t1_p1 + t1_p2 == 1 (HARD CONSTRAINT)")
        print("\n  Note: No penalty weights needed!")
        print("  Note: Constraint MUST be satisfied by solver")
        
        print("\n" + "-"*70)
        print("COMPARISON SUMMARY")
        print("-"*70)
        
        print("\n+---------------------+-----------------------+-----------------------+")
        print("| Aspect              | BQM (Penalties)       | CQM (Hard Constraints)|")
        print("+---------------------+-----------------------+-----------------------+")
        print("| Constraint Type     | Soft (may violate)    | Hard (always met)     |")
        print("| Penalty Tuning      | Required              | Not needed            |")
        print("| Solver Availability | All (SA, QPU, Hybrid) | Hybrid CQM only       |")
        print("| Feasibility         | Always finds solution | May be infeasible     |")
        print("| Model Size          | Larger (added terms)  | Smaller (cleaner)     |")
        print("| Use Case            | Flexible, trade-offs  | Must satisfy rules    |")
        print("+---------------------+-----------------------+-----------------------+")
        
        print("\nWhen to use BQM with penalties:")
        print("  - Developing/testing with simulator")
        print("  - Using QPU directly")
        print("  - Soft constraints acceptable")
        print("  - Need flexibility in constraint satisfaction")
        
        print("\nWhen to use CQM:")
        print("  - Constraints MUST be satisfied")
        print("  - Complex constraints (inequalities, multiple types)")
        print("  - Production systems with Hybrid solver")
        print("  - Don't want to tune penalty weights")
        
        return bqm, cqm
        
    except ImportError:
        print("\n  [CQM not available in current dimod version]")
        print("  Install with: pip install dwave-system")
        print("\n  CQM features:")
        print("    - Built-in constraint support")
        print("    - No penalty tuning needed")
        print("    - Guaranteed constraint satisfaction")
        print("    - Cleaner problem formulation")
        
        return bqm, None


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("TUTORIAL 1: BASIC DIMOD BQM CONSTRUCTION")
    print("="*70)
    print("\nThis tutorial demonstrates the fundamentals of building and solving")
    print("Binary Quadratic Models using D-Wave's DIMOD library.")
    
    # Run all examples
    example_1_simple_bqm()
    example_2_different_construction_methods()
    example_3_practical_problem()
    example_4_understanding_energy()
    example_5_bqm_vs_cqm()
    
    print("\n" + "="*70)
    print("TUTORIAL 1 COMPLETE")
    print("="*70)
    print("\nKey Takeaways:")
    print("1. BQMs consist of linear terms, quadratic terms, and an offset")
    print("2. Variables are binary (0 or 1)")
    print("3. The goal is to minimize the energy function")
    print("4. Constraints can be added as penalty terms (soft constraints)")
    print("5. BQM penalties: flexible but need tuning")
    print("6. CQM hard constraints: guaranteed satisfaction (Hybrid solver)")
    print("7. SimulatedAnnealingSampler can solve BQMs without hardware")
    print("\nNext: tutorial_02_qubo_basics.py")


if __name__ == "__main__":
    main()
