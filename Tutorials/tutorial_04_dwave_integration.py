"""
Tutorial 4: D-Wave Solver Integration

This tutorial demonstrates the plug-and-play capability between different D-Wave
solvers. You'll learn:

1. How to set up configuration for different solver types
2. How to switch between SimulatedAnnealing (local, free) and QPU (cloud, requires token)
3. How to configure and use the Leap Hybrid Solver
4. How to manage API tokens and environment variables
5. Best practices for solver selection based on problem size

This enables you to develop locally with simulators and deploy to real quantum hardware
when ready, with minimal code changes.
"""

import os
import dimod
import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class SolverConfig:
    """Configuration for D-Wave solvers."""
    solver_type: str  # 'simulator', 'qpu', 'hybrid'
    num_reads: int = 100
    api_token: Optional[str] = None
    # QPU-specific parameters
    chain_strength: Optional[float] = None
    annealing_time: Optional[int] = None
    # Hybrid-specific parameters
    time_limit: Optional[int] = None


def check_dwave_token():
    """
    Check if D-Wave API token is available.
    
    Returns:
        tuple: (token_available: bool, token: str or None)
    """
    # Check environment variable
    token = os.environ.get('DWAVE_API_TOKEN')
    
    if token:
        print(f"  D-Wave API token found in environment (length: {len(token)})")
        return True, token
    
    # Check dwave config file
    try:
        from dwave.cloud import Client
        try:
            client = Client.from_config()
            print("  D-Wave configuration file found")
            client.close()
            return True, None
        except:
            pass
    except ImportError:
        pass
    
    print("  No D-Wave API token found")
    print("  To use real QPU or Hybrid solvers, set DWAVE_API_TOKEN environment variable")
    print("  or configure using: dwave config create")
    return False, None


def example_1_simulator_solver():
    """
    Example 1: Using the Simulated Annealing Sampler (Local, Free)
    
    This is the default solver for development and testing.
    - Runs locally on your machine
    - No API token required
    - Fast for small to medium problems
    - Good for testing QUBO formulations
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Simulated Annealing Sampler (Local)")
    print("="*70)
    
    print("\nAdvantages:")
    print("  + Free and unlimited use")
    print("  + Runs locally (no internet required)")
    print("  + Fast for development and testing")
    print("  + No API token needed")
    
    print("\nLimitations:")
    print("  - Heuristic solver (not guaranteed optimal)")
    print("  - May struggle with very large problems")
    print("  - Not true quantum annealing")
    
    # Create a simple problem
    Q = {
        (0, 0): -1,
        (1, 1): -1,
        (2, 2): -1,
        (0, 1): 2,
        (1, 2): 2,
    }
    
    bqm = dimod.BinaryQuadraticModel.from_qubo(Q)
    
    print(f"\nProblem: {len(bqm.variables)} variables")
    
    # Create sampler
    sampler = dimod.SimulatedAnnealingSampler()
    
    print("\nSolver: SimulatedAnnealingSampler")
    print("  Parameters:")
    print("    - num_reads: 100 (number of solutions to generate)")
    print("    - seed: 42 (for reproducibility)")
    
    # Solve
    sampleset = sampler.sample(bqm, num_reads=100, seed=42)
    
    print(f"\nResults:")
    print(f"  Best energy: {sampleset.first.energy}")
    print(f"  Best solution: {dict(sampleset.first.sample)}")
    print(f"  Unique solutions found: {len([tuple(s.values()) for s in sampleset.samples()])}")
    
    return sampleset


def example_2_solver_comparison():
    """
    Example 2: Comparing different local samplers
    
    Shows the difference between various sampling strategies.
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Comparing Local Samplers")
    print("="*70)
    
    # Create a problem
    Q = {
        (0, 0): -2,
        (1, 1): -3,
        (2, 2): -1,
        (3, 3): -4,
        (0, 1): 1,
        (0, 2): 2,
        (1, 3): 1,
        (2, 3): 3,
    }
    
    bqm = dimod.BinaryQuadraticModel.from_qubo(Q)
    
    print(f"\nTest problem: {len(bqm.variables)} variables")
    
    # 1. ExactSolver (guaranteed optimal)
    print("\n1. ExactSolver:")
    print("   Tries all 2^n combinations")
    exact_sampler = dimod.ExactSolver()
    exact_result = exact_sampler.sample(bqm)
    print(f"   Optimal energy: {exact_result.first.energy}")
    print(f"   Optimal solution: {dict(exact_result.first.sample)}")
    
    # 2. SimulatedAnnealingSampler
    print("\n2. SimulatedAnnealingSampler:")
    print("   Heuristic optimization")
    sa_sampler = dimod.SimulatedAnnealingSampler()
    sa_result = sa_sampler.sample(bqm, num_reads=100, seed=42)
    print(f"   Best energy: {sa_result.first.energy}")
    print(f"   Best solution: {dict(sa_result.first.sample)}")
    print(f"   Found optimal: {sa_result.first.energy == exact_result.first.energy}")
    
    # 3. RandomSampler
    print("\n3. RandomSampler:")
    print("   Random baseline")
    random_sampler = dimod.RandomSampler()
    random_result = random_sampler.sample(bqm, num_reads=100, seed=42)
    print(f"   Best energy: {random_result.first.energy}")
    print(f"   Best solution: {dict(random_result.first.sample)}")
    print(f"   Found optimal: {random_result.first.energy == exact_result.first.energy}")
    
    print("\nRecommendation:")
    print("  - ExactSolver: Problems with <= 20 variables")
    print("  - SimulatedAnnealing: Development and testing")
    print("  - QPU/Hybrid: Production, large problems")
    
    return exact_result, sa_result, random_result


def example_3_qpu_simulation():
    """
    Example 3: Simulating QPU workflow (without actual QPU access)
    
    Shows how to structure your code for QPU deployment.
    The actual QPU code is similar, just with different sampler.
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: QPU Workflow Simulation")
    print("="*70)
    
    print("\nThis example shows the code structure for QPU deployment.")
    print("Replace SimulatedAnnealingSampler with DWaveSampler for actual QPU.")
    
    # Check for token
    token_available, token = check_dwave_token()
    
    # Create problem
    Q = {
        (0, 0): -1,
        (1, 1): -1,
        (2, 2): -1,
        (0, 1): 0.5,
        (1, 2): 0.5,
    }
    
    bqm = dimod.BinaryQuadraticModel.from_qubo(Q)
    
    print(f"\nProblem: {len(bqm.variables)} variables")
    
    # Solver selection based on token availability
    if token_available:
        print("\nQPU Code Structure (with token):")
        print("  from dwave.system import DWaveSampler, EmbeddingComposite")
        print("  sampler = EmbeddingComposite(DWaveSampler())")
        print("  sampleset = sampler.sample(bqm, num_reads=100)")
        print("\n  Note: Actual QPU execution would happen here")
        print("  For this tutorial, using simulator instead...")
    else:
        print("\nSimulator (no token):")
    
    # Use simulator
    sampler = dimod.SimulatedAnnealingSampler()
    sampleset = sampler.sample(bqm, num_reads=100, seed=42)
    
    print(f"\nResults:")
    print(f"  Best energy: {sampleset.first.energy}")
    print(f"  Best solution: {dict(sampleset.first.sample)}")
    
    print("\nQPU-specific considerations:")
    print("  - Chain strength: Controls qubit coupling")
    print("  - Annealing time: Duration of quantum evolution")
    print("  - Embedding: Maps logical variables to physical qubits")
    print("  - Cost: Charged per problem submission")
    
    return sampleset


def example_4_hybrid_solver_simulation():
    """
    Example 4: Hybrid Solver workflow
    
    Hybrid solvers combine classical and quantum approaches.
    Best for large problems (1000+ variables).
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Hybrid Solver Workflow")
    print("="*70)
    
    print("\nHybrid Solver Benefits:")
    print("  + Handles large problems (1000+ variables)")
    print("  + Combines classical and quantum approaches")
    print("  + Automatic problem decomposition")
    print("  + Better for real-world applications")
    
    print("\nRequirements:")
    print("  - D-Wave Leap account (free tier available)")
    print("  - API token")
    
    # Check for token
    token_available, token = check_dwave_token()
    
    # Create a larger problem
    n = 20
    Q = {}
    for i in range(n):
        Q[(i, i)] = np.random.randn()
    for i in range(n):
        for j in range(i+1, min(i+4, n)):
            Q[(i, j)] = np.random.randn() * 0.5
    
    bqm = dimod.BinaryQuadraticModel.from_qubo(Q)
    
    print(f"\nProblem: {len(bqm.variables)} variables")
    
    if token_available:
        print("\nHybrid Solver Code Structure (with token):")
        print("  from dwave.system import LeapHybridSampler")
        print("  sampler = LeapHybridSampler()")
        print("  sampleset = sampler.sample(bqm, time_limit=5)")
        print("\n  Note: Actual hybrid solver would be used here")
        print("  For this tutorial, using simulator instead...")
    else:
        print("\nSimulator (no token):")
    
    # Use simulator
    sampler = dimod.SimulatedAnnealingSampler()
    sampleset = sampler.sample(bqm, num_reads=100, seed=42)
    
    print(f"\nResults:")
    print(f"  Best energy: {sampleset.first.energy:.3f}")
    print(f"  Number of variables in solution: {len(sampleset.first.sample)}")
    
    print("\nHybrid Solver Parameters:")
    print("  - time_limit: Maximum runtime in seconds (default: 5)")
    print("  - label: Tag for tracking submissions")
    
    return sampleset


def example_5_plug_and_play():
    """
    Example 5: Plug-and-play solver selection
    
    Demonstrates a unified interface that works with any solver type.
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: Plug-and-Play Solver Selection")
    print("="*70)
    
    print("\nThis example shows a flexible solver selection pattern.")
    
    # Create problem
    Q = {
        (0, 0): -2,
        (1, 1): -3,
        (2, 2): -1,
        (0, 1): 1,
        (1, 2): 2,
    }
    
    bqm = dimod.BinaryQuadraticModel.from_qubo(Q)
    
    def solve_with_config(bqm: dimod.BinaryQuadraticModel, 
                         config: SolverConfig) -> dimod.SampleSet:
        """
        Solve BQM using specified configuration.
        
        This function abstracts the solver selection logic.
        """
        print(f"\n  Solver type: {config.solver_type}")
        
        if config.solver_type == 'simulator':
            sampler = dimod.SimulatedAnnealingSampler()
            sampleset = sampler.sample(bqm, num_reads=config.num_reads)
            
        elif config.solver_type == 'qpu':
            print("    QPU solver requires:")
            print("      from dwave.system import DWaveSampler, EmbeddingComposite")
            print("      sampler = EmbeddingComposite(DWaveSampler(token=config.api_token))")
            print("\n    Using simulator as fallback...")
            sampler = dimod.SimulatedAnnealingSampler()
            sampleset = sampler.sample(bqm, num_reads=config.num_reads)
            
        elif config.solver_type == 'hybrid':
            print("    Hybrid solver requires:")
            print("      from dwave.system import LeapHybridSampler")
            print("      sampler = LeapHybridSampler(token=config.api_token)")
            print("\n    Using simulator as fallback...")
            sampler = dimod.SimulatedAnnealingSampler()
            sampleset = sampler.sample(bqm, num_reads=config.num_reads)
            
        else:
            raise ValueError(f"Unknown solver type: {config.solver_type}")
        
        return sampleset
    
    # Test different configurations
    configs = [
        SolverConfig(solver_type='simulator', num_reads=100),
        SolverConfig(solver_type='qpu', num_reads=100),
        SolverConfig(solver_type='hybrid', time_limit=5),
    ]
    
    print("\nTesting different solver configurations:")
    
    for config in configs:
        print(f"\nConfiguration: {config.solver_type}")
        sampleset = solve_with_config(bqm, config)
        print(f"  Best energy: {sampleset.first.energy}")
        print(f"  Best solution: {dict(sampleset.first.sample)}")
    
    print("\nConfiguration Management Tips:")
    print("  1. Store solver configs in environment variables or config files")
    print("  2. Use 'simulator' for development")
    print("  3. Switch to 'qpu' or 'hybrid' for production")
    print("  4. Handle token loading gracefully")
    print("  5. Implement fallback mechanisms")
    
    return configs


def example_6_best_practices():
    """
    Example 6: Best practices for solver selection
    
    Guidelines for choosing the right solver.
    """
    print("\n" + "="*70)
    print("EXAMPLE 6: Solver Selection Best Practices")
    print("="*70)
    
    guidelines = {
        "Problem Size": {
            "< 20 variables": "ExactSolver (guaranteed optimal)",
            "20-100 variables": "SimulatedAnnealing or QPU",
            "100-1000 variables": "QPU with embedding",
            "> 1000 variables": "Hybrid Solver",
        },
        "Development Phase": {
            "Prototyping": "SimulatedAnnealing (fast, free)",
            "Testing": "SimulatedAnnealing with multiple seeds",
            "Validation": "ExactSolver on small instances",
            "Production": "QPU or Hybrid (based on size)",
        },
        "Cost Considerations": {
            "Free": "Simulators (unlimited)",
            "Budget": "Hybrid solver (per-second pricing)",
            "Premium": "QPU (per-problem pricing)",
        },
    }
    
    for category, items in guidelines.items():
        print(f"\n{category}:")
        for key, value in items.items():
            print(f"  {key}: {value}")
    
    print("\nEnvironment Variables Setup:")
    print("  Windows PowerShell:")
    print("    $env:DWAVE_API_TOKEN = 'your-token-here'")
    print("\n  Linux/Mac:")
    print("    export DWAVE_API_TOKEN='your-token-here'")
    print("\n  Or use dwave CLI:")
    print("    dwave config create")
    print("    dwave ping")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("TUTORIAL 4: D-WAVE SOLVER INTEGRATION")
    print("="*70)
    print("\nThis tutorial demonstrates how to work with different D-Wave solvers")
    print("and implement plug-and-play solver selection.")
    
    # Run all examples
    example_1_simulator_solver()
    example_2_solver_comparison()
    example_3_qpu_simulation()
    example_4_hybrid_solver_simulation()
    example_5_plug_and_play()
    example_6_best_practices()
    
    print("\n" + "="*70)
    print("TUTORIAL 4 COMPLETE")
    print("="*70)
    print("\nKey Takeaways:")
    print("1. SimulatedAnnealing is perfect for development (free, local)")
    print("2. QPU requires API token and has usage costs")
    print("3. Hybrid solvers handle large problems (1000+ variables)")
    print("4. Use configuration objects for flexible solver selection")
    print("5. Implement fallback mechanisms for robustness")
    print("6. Choose solver based on problem size and budget")
    print("\nNext: tutorial_05_complete_workflow.py")


if __name__ == "__main__":
    main()
