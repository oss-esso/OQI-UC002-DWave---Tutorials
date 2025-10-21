"""
Comparison Script: Benders Decomposition vs Standard MILP

Compares the Benders Decomposition approach (with annealing) against
standard PuLP MILP solver for the crop allocation problem.
"""

import sys
import os
import time
import pulp as pl
import json
from typing import Dict, List, Tuple, Any

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from scenarios import load_food_data
from benders_decomposition import BendersDecomposition


def solve_with_pulp(
    farms: List[str],
    crops: Dict[str, Dict[str, float]],
    food_groups: Dict[str, List[str]],
    config: Dict[str, Any]
) -> Tuple[str, float, float, Dict, Dict]:
    """
    Solve using standard PuLP MILP solver
    
    Returns:
        (status, objective, solve_time, Y_solution, A_solution)
    """
    print("\n" + "="*80)
    print("SOLVING WITH STANDARD PULP MILP SOLVER")
    print("="*80)
    
    params = config['parameters']
    crop_names = list(crops.keys())
    
    # Create model
    model = pl.LpProblem("Crop_Allocation_MILP", pl.LpMaximize)
    
    # Variables
    A = pl.LpVariable.dicts(
        "Area",
        [(f, c) for f in farms for c in crop_names],
        lowBound=0
    )
    Y = pl.LpVariable.dicts(
        "Choose",
        [(f, c) for f in farms for c in crop_names],
        cat='Binary'
    )
    
    # Objective
    total_area = sum(params['land_availability'].values())
    objective_terms = []
    
    weights = params['weights']
    for farm in farms:
        for crop in crop_names:
            N = crops[crop].get('nutritional_value', 0.5)
            D = crops[crop].get('nutrient_density', 0.5)
            E = crops[crop].get('environmental_impact', 0.5)
            P = crops[crop].get('sustainability', 0.5)
            
            obj_coeff = (
                weights.get('nutritional_value', 0.25) * N +
                weights.get('nutrient_density', 0.25) * D -
                weights.get('environmental_impact', 0.25) * E +
                weights.get('sustainability', 0.25) * P
            ) / total_area
            
            objective_terms.append(obj_coeff * A[(farm, crop)])
    
    model += pl.lpSum(objective_terms), "Objective"
    
    # Constraints
    # 1. Land availability
    for farm in farms:
        model += (
            pl.lpSum([A[(farm, crop)] for crop in crop_names]) 
            <= params['land_availability'][farm],
            f"Land_{farm}"
        )
    
    # 2. Minimum planting area
    A_min = params.get('minimum_planting_area', {})
    if not A_min:
        A_min = {crop: 5.0 for crop in crop_names}
    
    for farm in farms:
        for crop in crop_names:
            a_min = A_min.get(crop, 5.0)
            model += (
                A[(farm, crop)] >= a_min * Y[(farm, crop)],
                f"MinArea_{farm}_{crop}"
            )
            model += (
                A[(farm, crop)] <= params['land_availability'][farm] * Y[(farm, crop)],
                f"MaxArea_{farm}_{crop}"
            )
    
    # 3. Food group constraints
    food_group_constraints = params.get('food_group_constraints', {})
    for farm in farms:
        for group, crops_in_group in food_groups.items():
            fg_constraints = food_group_constraints.get(group, {})
            min_foods = fg_constraints.get('min_foods', 1)
            max_foods = fg_constraints.get('max_foods', len(crops_in_group))
            
            model += (
                pl.lpSum([Y[(farm, crop)] for crop in crops_in_group if crop in crop_names]) >= min_foods,
                f"MinFoodGroup_{farm}_{group}"
            )
            model += (
                pl.lpSum([Y[(farm, crop)] for crop in crops_in_group if crop in crop_names]) <= max_foods,
                f"MaxFoodGroup_{farm}_{group}"
            )
    
    # Solve
    start_time = time.time()
    solver = pl.PULP_CBC_CMD(msg=1, timeLimit=300)
    model.solve(solver)
    solve_time = time.time() - start_time
    
    status = pl.LpStatus[model.status]
    
    if status == 'Optimal':
        objective = pl.value(model.objective)
        
        # Extract solutions
        Y_solution = {}
        A_solution = {}
        for farm in farms:
            for crop in crop_names:
                Y_solution[(farm, crop)] = int(Y[(farm, crop)].varValue) if Y[(farm, crop)].varValue else 0
                A_solution[(farm, crop)] = float(A[(farm, crop)].varValue) if A[(farm, crop)].varValue else 0.0
        
        print(f"\nStatus: {status}")
        print(f"Objective: {objective:.6f}")
        print(f"Solve Time: {solve_time:.2f}s")
        
        return status, objective, solve_time, Y_solution, A_solution
    else:
        print(f"\nStatus: {status}")
        print(f"Solve Time: {solve_time:.2f}s")
        return status, float('-inf'), solve_time, {}, {}


def compare_solutions(scenario: str = 'simple', max_benders_iter: int = 20):
    """Compare Benders Decomposition against standard MILP solver"""
    
    print("="*80)
    print(f"COMPARISON: Benders Decomposition vs Standard MILP")
    print(f"Scenario: {scenario}")
    print("="*80)
    
    # Load scenario
    farms, crops, food_groups, config = load_food_data(scenario)
    crop_names = list(crops.keys())
    
    print(f"\nProblem Size:")
    print(f"  Farms: {len(farms)}")
    print(f"  Crops: {len(crop_names)}")
    print(f"  Binary Variables: {len(farms) * len(crop_names)}")
    print(f"  Continuous Variables: {len(farms) * len(crop_names)}")
    
    # Solve with standard MILP
    pulp_status, pulp_obj, pulp_time, pulp_Y, pulp_A = solve_with_pulp(
        farms, crops, food_groups, config
    )
    
    # Solve with Benders (Classical Annealing)
    print("\n" + "="*80)
    print("SOLVING WITH BENDERS DECOMPOSITION (Classical Annealing)")
    print("="*80)
    
    config['benders_max_iterations'] = max_benders_iter
    benders_classical = BendersDecomposition(
        farms=farms,
        crops=crops,
        food_groups=food_groups,
        config=config,
        use_quantum=False
    )
    
    solution_classical = benders_classical.solve()
    
    # Solve with Benders (Quantum Annealing)
    print("\n" + "="*80)
    print("SOLVING WITH BENDERS DECOMPOSITION (Quantum Annealing)")
    print("="*80)
    
    benders_quantum = BendersDecomposition(
        farms=farms,
        crops=crops,
        food_groups=food_groups,
        config=config,
        use_quantum=True
    )
    
    solution_quantum = benders_quantum.solve()
    
    # Create comparison report
    report = {
        'scenario': scenario,
        'problem_size': {
            'farms': len(farms),
            'crops': len(crop_names),
            'binary_variables': len(farms) * len(crop_names),
            'continuous_variables': len(farms) * len(crop_names)
        },
        'standard_milp': {
            'status': pulp_status,
            'objective': pulp_obj,
            'solve_time': pulp_time
        },
        'benders_classical': {
            'status': solution_classical.status,
            'objective': solution_classical.objective_value,
            'solve_time': solution_classical.total_time,
            'iterations': solution_classical.iterations,
            'gap': solution_classical.gap
        },
        'benders_quantum': {
            'status': solution_quantum.status,
            'objective': solution_quantum.objective_value,
            'solve_time': solution_quantum.total_time,
            'iterations': solution_quantum.iterations,
            'gap': solution_quantum.gap
        }
    }
    
    # Calculate quality metrics
    if pulp_obj > -float('inf'):
        report['benders_classical']['optimality_gap_vs_milp'] = (
            abs(solution_classical.objective_value - pulp_obj) / abs(pulp_obj) * 100
        )
        report['benders_quantum']['optimality_gap_vs_milp'] = (
            abs(solution_quantum.objective_value - pulp_obj) / abs(pulp_obj) * 100
        )
    
    # Print comparison
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    
    print(f"\n{'Method':<30} {'Objective':<15} {'Time (s)':<12} {'Status':<15}")
    print("-" * 75)
    print(f"{'Standard MILP':<30} {pulp_obj:<15.6f} {pulp_time:<12.2f} {pulp_status:<15}")
    print(f"{'Benders (Classical)':<30} {solution_classical.objective_value:<15.6f} "
          f"{solution_classical.total_time:<12.2f} {solution_classical.status:<15}")
    print(f"{'Benders (Quantum)':<30} {solution_quantum.objective_value:<15.6f} "
          f"{solution_quantum.total_time:<12.2f} {solution_quantum.status:<15}")
    
    if pulp_obj > -float('inf'):
        print(f"\nOptimality Gap vs MILP:")
        print(f"  Classical Annealing: {report['benders_classical']['optimality_gap_vs_milp']:.2f}%")
        print(f"  Quantum Annealing: {report['benders_quantum']['optimality_gap_vs_milp']:.2f}%")
    
    print(f"\nBenders Iterations:")
    print(f"  Classical: {solution_classical.iterations}")
    print(f"  Quantum: {solution_quantum.iterations}")
    
    # Save report
    filename = f"comparison_{scenario}.json"
    with open(filename, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nComparison report saved to: {filename}")
    
    return report


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare Benders vs Standard MILP')
    parser.add_argument(
        '--scenario',
        type=str,
        default='simple',
        choices=['simple', 'intermediate', 'custom'],
        help='Scenario to test'
    )
    parser.add_argument(
        '--benders-iter',
        type=int,
        default=20,
        help='Maximum Benders iterations'
    )
    
    args = parser.parse_args()
    
    compare_solutions(args.scenario, args.benders_iter)


if __name__ == "__main__":
    main()
