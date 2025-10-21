"""
Batch Testing Script for Benders Decomposition

Runs Benders Decomposition on multiple scenarios and annealing modes,
collecting results for analysis.
"""

import os
import sys
import json
import time
from typing import Dict, List, Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from scenarios import load_food_data
from benders_decomposition import BendersDecomposition


def run_scenario_test(
    scenario: str,
    use_quantum: bool,
    max_iterations: int = 20,
    output_dir: str = "benders_results"
) -> Dict[str, Any]:
    """
    Run Benders Decomposition on a single scenario
    
    Returns:
        Dictionary with results
    """
    print(f"\n{'='*80}")
    print(f"Testing Scenario: {scenario}")
    print(f"Annealing Mode: {'Quantum' if use_quantum else 'Classical'}")
    print(f"{'='*80}")
    
    try:
        # Load scenario
        farms, crops, food_groups, config = load_food_data(scenario)
        
        # Configure
        config['benders_max_iterations'] = max_iterations
        config['benders_tolerance'] = 1e-3
        
        # Create solver
        benders = BendersDecomposition(
            farms=farms,
            crops=crops,
            food_groups=food_groups,
            config=config,
            use_quantum=use_quantum
        )
        
        # Solve
        start_time = time.time()
        solution = benders.solve()
        total_time = time.time() - start_time
        
        # Prepare results
        results = {
            'scenario': scenario,
            'annealing_mode': 'quantum' if use_quantum else 'classical',
            'success': True,
            'status': solution.status,
            'objective': solution.objective_value,
            'lower_bound': solution.lower_bound,
            'upper_bound': solution.upper_bound,
            'gap': solution.gap,
            'iterations': solution.iterations,
            'total_time': total_time,
            'problem_size': {
                'farms': len(farms),
                'crops': len(crops),
                'binary_vars': len(farms) * len(crops),
                'continuous_vars': len(farms) * len(crops)
            }
        }
        
        # Save detailed solution
        os.makedirs(output_dir, exist_ok=True)
        mode_str = 'quantum' if use_quantum else 'classical'
        filename = os.path.join(output_dir, f"solution_{scenario}_{mode_str}.json")
        
        solution_dict = {
            'status': solution.status,
            'objective_value': solution.objective_value,
            'lower_bound': solution.lower_bound,
            'upper_bound': solution.upper_bound,
            'gap': solution.gap,
            'iterations': solution.iterations,
            'total_time': solution.total_time,
            'binary_variables': {f"{k[0]}_{k[1]}": v for k, v in solution.binary_variables.items()},
            'area_variables': {f"{k[0]}_{k[1]}": v for k, v in solution.area_variables.items()},
            'iteration_history': [
                {
                    'iteration': it.iteration,
                    'lower_bound': it.lower_bound,
                    'upper_bound': it.upper_bound,
                    'gap': it.gap,
                    'num_cuts': it.num_cuts,
                    'cut_type': it.cut_type,
                    'is_feasible': it.is_feasible
                }
                for it in solution.iteration_history
            ]
        }
        
        with open(filename, 'w') as f:
            json.dump(solution_dict, f, indent=2)
        
        print(f"\n✓ Test completed successfully")
        print(f"  Objective: {solution.objective_value:.6f}")
        print(f"  Time: {total_time:.2f}s")
        print(f"  Iterations: {solution.iterations}")
        print(f"  Solution saved to: {filename}")
        
        return results
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return {
            'scenario': scenario,
            'annealing_mode': 'quantum' if use_quantum else 'classical',
            'success': False,
            'error': str(e)
        }


def run_batch_tests(
    scenarios: List[str] = None,
    annealing_modes: List[bool] = None,
    max_iterations: int = 20,
    output_dir: str = "benders_results"
):
    """
    Run batch tests across multiple scenarios and annealing modes
    """
    if scenarios is None:
        scenarios = ['simple', 'intermediate']
    
    if annealing_modes is None:
        annealing_modes = [False, True]  # Classical and Quantum
    
    print("="*80)
    print("BENDERS DECOMPOSITION BATCH TESTING")
    print("="*80)
    print(f"\nScenarios: {', '.join(scenarios)}")
    print(f"Annealing Modes: {['Classical', 'Quantum'][annealing_modes[0]:]} {['Classical', 'Quantum'][annealing_modes[-1]:]}")
    print(f"Max Iterations: {max_iterations}")
    print(f"Output Directory: {output_dir}")
    
    all_results = []
    
    for scenario in scenarios:
        for use_quantum in annealing_modes:
            result = run_scenario_test(
                scenario=scenario,
                use_quantum=use_quantum,
                max_iterations=max_iterations,
                output_dir=output_dir
            )
            all_results.append(result)
            
            # Brief pause between tests
            time.sleep(1)
    
    # Create summary report
    summary_file = os.path.join(output_dir, "batch_test_summary.json")
    with open(summary_file, 'w') as f:
        json.dump({
            'test_configuration': {
                'scenarios': scenarios,
                'annealing_modes': ['quantum' if m else 'classical' for m in annealing_modes],
                'max_iterations': max_iterations
            },
            'results': all_results
        }, f, indent=2)
    
    # Print summary
    print(f"\n{'='*80}")
    print("BATCH TEST SUMMARY")
    print(f"{'='*80}\n")
    
    print(f"{'Scenario':<15} {'Mode':<12} {'Status':<12} {'Objective':<12} {'Time (s)':<10} {'Iters':<8}")
    print("-" * 80)
    
    for result in all_results:
        if result['success']:
            print(f"{result['scenario']:<15} "
                  f"{result['annealing_mode']:<12} "
                  f"{result['status']:<12} "
                  f"{result['objective']:<12.6f} "
                  f"{result['total_time']:<10.2f} "
                  f"{result['iterations']:<8}")
        else:
            print(f"{result['scenario']:<15} "
                  f"{result['annealing_mode']:<12} "
                  f"{'FAILED':<12} "
                  f"{'-':<12} "
                  f"{'-':<10} "
                  f"{'-':<8}")
    
    print(f"\nSummary saved to: {summary_file}")
    
    # Calculate statistics
    successful = [r for r in all_results if r['success']]
    if successful:
        print(f"\n{'='*80}")
        print("STATISTICS")
        print(f"{'='*80}")
        print(f"Total Tests: {len(all_results)}")
        print(f"Successful: {len(successful)}")
        print(f"Failed: {len(all_results) - len(successful)}")
        
        classical_results = [r for r in successful if r['annealing_mode'] == 'classical']
        quantum_results = [r for r in successful if r['annealing_mode'] == 'quantum']
        
        if classical_results:
            avg_time_classical = sum(r['total_time'] for r in classical_results) / len(classical_results)
            avg_obj_classical = sum(r['objective'] for r in classical_results) / len(classical_results)
            print(f"\nClassical Annealing:")
            print(f"  Average Time: {avg_time_classical:.2f}s")
            print(f"  Average Objective: {avg_obj_classical:.6f}")
        
        if quantum_results:
            avg_time_quantum = sum(r['total_time'] for r in quantum_results) / len(quantum_results)
            avg_obj_quantum = sum(r['objective'] for r in quantum_results) / len(quantum_results)
            print(f"\nQuantum Annealing:")
            print(f"  Average Time: {avg_time_quantum:.2f}s")
            print(f"  Average Objective: {avg_obj_quantum:.6f}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Batch test Benders Decomposition')
    parser.add_argument(
        '--scenarios',
        type=str,
        nargs='+',
        default=['simple', 'intermediate'],
        help='Scenarios to test'
    )
    parser.add_argument(
        '--classical-only',
        action='store_true',
        help='Test only classical annealing'
    )
    parser.add_argument(
        '--quantum-only',
        action='store_true',
        help='Test only quantum annealing'
    )
    parser.add_argument(
        '--max-iter',
        type=int,
        default=20,
        help='Maximum Benders iterations'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='benders_results',
        help='Output directory for results'
    )
    
    args = parser.parse_args()
    
    # Determine annealing modes
    if args.classical_only:
        annealing_modes = [False]
    elif args.quantum_only:
        annealing_modes = [True]
    else:
        annealing_modes = [False, True]
    
    run_batch_tests(
        scenarios=args.scenarios,
        annealing_modes=annealing_modes,
        max_iterations=args.max_iter,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
