"""
Test MINLP Solvers for Fractional Programming
==============================================

This script tests all three solver implementations:
1. PuLP with Dinkelbach's algorithm (iterative linearization)
2. Pyomo with direct MINLP solvers (Ipopt, BARON, Couenne)
3. D-Wave with Charnes-Cooper transformation

Tests on simple scenario (3 farms, 6 foods) to verify correctness.
"""

import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(__file__))

from src.scenarios import load_food_data
from solver_runner_NLD import (
    solve_with_pulp,
    solve_with_pyomo,
    solve_with_dwave_charnes_cooper
)

def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)

def print_results(solver_name, results):
    """Print solver results in formatted manner."""
    print(f"\n{solver_name} Results:")
    print(f"  Status: {results.get('status', 'N/A')}")
    
    if results.get('objective_value') is not None:
        print(f"  Objective Value: {results['objective_value']:.6f}")
    
    if results.get('solve_time') is not None:
        print(f"  Solve Time: {results['solve_time']:.3f} seconds")
    
    if results.get('iterations') is not None:
        print(f"  Iterations: {results['iterations']}")
    
    if results.get('solver') is not None:
        print(f"  Solver: {results['solver']}")
    
    if results.get('error'):
        print(f"  Error: {results['error']}")
    
    # Show sample allocations
    if results.get('areas'):
        print(f"\n  Sample Area Allocations:")
        count = 0
        for key, value in results['areas'].items():
            if value > 0.01:  # Only show non-trivial allocations
                print(f"    {key}: {value:.2f}")
                count += 1
                if count >= 5:  # Limit to first 5
                    print(f"    ... ({len([v for v in results['areas'].values() if v > 0.01]) - 5} more)")
                    break

def main():
    """Run tests on all three solver implementations."""
    print_section("MINLP FRACTIONAL PROGRAMMING SOLVER TEST")
    
    print("\nObjective Function:")
    print("  MILP (old):  max (weighted_sum / total_area)")
    print("  MINLP (new): max (weighted_sum / sum(A_{f,c}))")
    print("\nThis transforms the problem from linear to nonlinear fractional programming.")
    
    # Load simple scenario
    print_section("Loading Test Scenario")
    scenario = 'simple'
    print(f"  Loading '{scenario}' scenario...")
    
    farms, foods, food_groups, config = load_food_data(scenario)
    
    print(f"\n  Scenario Details:")
    print(f"    Farms: {len(farms)} - {farms}")
    print(f"    Foods: {len(foods)} - {list(foods.keys())}")
    print(f"    Food Groups: {len(food_groups)}")
    print(f"    Variables: {len(farms) * len(foods) * 2} (continuous + binary)")
    
    # Track results for comparison
    all_results = {}
    
    # Test 1: PuLP with Dinkelbach's algorithm
    print_section("Test 1: PuLP with Dinkelbach's Algorithm")
    print("  Iterative linearization: max f(x) - lambda * g(x)")
    print("  Converges to optimal fractional solution")
    
    try:
        start = time.time()
        pulp_model, pulp_results = solve_with_pulp(farms, foods, food_groups, config)
        pulp_time = time.time() - start
        pulp_results['total_time'] = pulp_time
        
        print_results("PuLP (Dinkelbach)", pulp_results)
        all_results['pulp'] = pulp_results
        
        if pulp_results.get('convergence_history'):
            print(f"\n  Convergence History:")
            for i, hist in enumerate(pulp_results['convergence_history'][:5]):
                print(f"    Iter {i}: lambda={hist['lambda']:.6f}, residual={hist['residual']:.2e}")
            if len(pulp_results['convergence_history']) > 5:
                print(f"    ... ({len(pulp_results['convergence_history']) - 5} more iterations)")
    
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: Pyomo with direct MINLP solver
    print_section("Test 2: Pyomo with Direct MINLP Solver")
    print("  Direct fractional objective: max f(x) / g(x)")
    print("  Requires MINLP solver (Ipopt, BARON, Couenne, SCIP)")
    
    try:
        start = time.time()
        pyomo_model, pyomo_results = solve_with_pyomo(farms, foods, food_groups, config)
        pyomo_time = time.time() - start
        pyomo_results['total_time'] = pyomo_time
        
        print_results("Pyomo (MINLP)", pyomo_results)
        all_results['pyomo'] = pyomo_results
    
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: D-Wave with Charnes-Cooper transformation
    print_section("Test 3: D-Wave with Charnes-Cooper Transformation")
    print("  Variable substitution: z = x/g(x), t = 1/g(x)")
    print("  Converts to equivalent linear program for quantum annealing")
    print("\n  NOTE: This test requires D-Wave API access and may take longer")
    print("  Skipping D-Wave test by default. Set DWAVE_TEST=1 to enable.")
    
    if os.getenv('DWAVE_TEST') == '1':
        try:
            token = os.getenv('DWAVE_API_TOKEN', '45FS-23cfb48dca2296ed24550846d2e7356eb6c19551')
            start = time.time()
            sampleset, dwave_results = solve_with_dwave_charnes_cooper(
                farms, foods, food_groups, config, token
            )
            dwave_time = time.time() - start
            dwave_results['total_time'] = dwave_time
            
            print_results("D-Wave (Charnes-Cooper)", dwave_results)
            all_results['dwave'] = dwave_results
        
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("  (Skipped - set DWAVE_TEST=1 to enable)")
    
    # Comparison Summary
    print_section("Comparison Summary")
    
    if all_results:
        print("\n  Solver Comparison:")
        print(f"  {'Solver':<20} {'Status':<15} {'Objective':<12} {'Time (s)':<10}")
        print(f"  {'-'*20} {'-'*15} {'-'*12} {'-'*10}")
        
        for solver_name, results in all_results.items():
            status = results.get('status', 'N/A')
            obj = results.get('objective_value')
            obj_str = f"{obj:.6f}" if obj is not None else "N/A"
            time_val = results.get('total_time', results.get('solve_time', 0))
            
            print(f"  {solver_name:<20} {status:<15} {obj_str:<12} {time_val:<10.3f}")
        
        # Verify consistency
        print("\n  Consistency Check:")
        objectives = [r['objective_value'] for r in all_results.values() 
                     if r.get('objective_value') is not None]
        
        if len(objectives) >= 2:
            max_obj = max(objectives)
            min_obj = min(objectives)
            diff = abs(max_obj - min_obj)
            rel_diff = diff / max_obj if max_obj > 0 else 0
            
            print(f"    Max Objective: {max_obj:.6f}")
            print(f"    Min Objective: {min_obj:.6f}")
            print(f"    Difference: {diff:.6f} ({rel_diff*100:.2f}%)")
            
            if rel_diff < 0.01:
                print("    ✓ Solutions are consistent (< 1% difference)")
            elif rel_diff < 0.05:
                print("    ⚠ Solutions differ slightly (1-5% difference)")
            else:
                print("    ✗ Solutions differ significantly (> 5% difference)")
        else:
            print("    Not enough successful solvers to compare")
    
    # Save results
    print_section("Saving Results")
    
    results_file = 'minlp_test_results.json'
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n  Results saved to: {results_file}")
    
    print_section("Test Complete")
    print("\n  Summary:")
    print(f"    Tested {len(all_results)} solver(s)")
    successful = sum(1 for r in all_results.values() if r.get('status') == 'Optimal')
    print(f"    Successful: {successful}/{len(all_results)}")
    
    print("\n  Next Steps:")
    print("    1. Install Pyomo and Ipopt: conda install -c conda-forge pyomo ipopt")
    print("    2. Run benchmark: python benchmark_scalability_NLD.py")
    print("    3. Compare with original MILP: python solver_runner.py --scenario simple")
    
    return all_results

if __name__ == "__main__":
    main()
