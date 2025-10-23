"""Quick test of the benchmark with just 5 farms"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from benchmark_scalability_BQUBO import run_benchmark

# Test with just 5 farms, 1 run
result = run_benchmark(n_farms=5, run_number=1, total_runs=1, dwave_token=None)

if result:
    print("\n" + "="*80)
    print("TEST RESULT")
    print("="*80)
    print(f"PuLP Status: {result['pulp_status']}")
    print(f"PuLP Objective: {result['pulp_objective']}")
    print(f"Problem Size: {result['problem_size']}")
    print(f"Variables: {result['n_vars']}")
    print(f"Constraints: {result['n_constraints']}")
    
    if result['pulp_status'] == 'Optimal':
        print("\n✅ TEST PASSED: Problem is feasible!")
    else:
        print(f"\n❌ TEST FAILED: Problem is {result['pulp_status']}")
else:
    print("\n❌ TEST FAILED: No result returned")
