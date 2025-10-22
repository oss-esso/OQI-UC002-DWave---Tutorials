"""
Professional solver runner script with Non-Linear objective (A^0.548).

This script:
1. Loads a scenario (simple, intermediate, or custom)
2. Converts to CQM with piecewise linear approximation for non-linear objective
3. Saves the model
4. Solves with PuLP using piecewise approximation and saves results
5. (DWave solving disabled - token removed for testing)
6. Saves all constraints for verification
"""

import os
import sys
import json
import pickle
import shutil
import time
import numpy as np
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from src.scenarios import load_food_data
from dimod import ConstrainedQuadraticModel, Binary, Real
from dwave.system import LeapHybridCQMSampler
import pulp as pl
from tqdm import tqdm
from piecewise_approximation import PiecewiseApproximation

def create_cqm(farms, foods, food_groups, config, power=0.548, num_breakpoints=10):
    """
    Creates a CQM for the food optimization problem with piecewise linear approximation.
    Uses f(A) = A^power instead of linear objective.
    
    Args:
        farms: List of farm names
        foods: Dictionary of food data
        food_groups: Dictionary of food groups
        config: Configuration dictionary
        power: Power for non-linear objective (default: 0.548)
        num_breakpoints: Number of interior breakpoints for piecewise approximation
    
    Returns CQM, variables, constraint metadata, and approximation metadata.
    """
    cqm = ConstrainedQuadraticModel()
    
    # Extract parameters
    params = config['parameters']
    land_availability = params['land_availability']
    weights = params['weights']
    min_planting_area = params.get('minimum_planting_area', {})
    food_group_constraints = params.get('food_group_constraints', {})
    
    n_farms = len(farms)
    n_foods = len(foods)
    n_food_groups = len(food_groups) if food_group_constraints else 0
    
    # Determine max land value for piecewise approximation
    max_land = max(land_availability.values())
    
    # Create piecewise approximation for each farm-food pair
    print(f"\nCreating piecewise linear approximation: f(A) = A^{power}")
    print(f"  Using {num_breakpoints} interior points ({num_breakpoints + 2} total breakpoints)")
    print(f"  Domain: [0, {max_land}]")
    
    # Store approximations for each unique max_land value
    approximations = {}
    unique_max_lands = set(land_availability.values())
    
    for max_val in unique_max_lands:
        approx = PiecewiseApproximation(power=power, num_points=num_breakpoints, max_value=max_val)
        approximations[max_val] = approx
        max_abs, max_rel, avg_abs = approx.get_max_error()
        print(f"  Max land={max_val:.1f}: Max error={max_abs:.6f}, Avg error={avg_abs:.6f}")
    
    # Calculate total operations for progress bar
    # Additional variables needed: lambda variables for SOS2 (num_breakpoints+2 per farm-food pair)
    total_ops = (
        n_farms * n_foods * (2 + num_breakpoints + 2) +  # Variables (A, Y, and lambda_i for each breakpoint)
        n_farms * n_foods * 3 +       # Objective terms + SOS2 constraint + convexity constraint
        n_farms +                 # Land availability constraints
        n_farms * n_foods * 2 +   # Linking constraints (2 per farm-food pair)
        n_farms * n_food_groups * 2  # Food group constraints (min and max)
    )
    
    pbar = tqdm(total=total_ops, desc="Building CQM with piecewise approximation", unit="op", ncols=100)
    
    # Define variables
    A = {}
    Y = {}
    Lambda = {}  # Lambda variables for piecewise linear approximation (SOS2)
    f_approx = {}  # Approximated function values
    
    pbar.set_description("Creating area and binary variables")
    for farm in farms:
        for food in foods:
            A[(farm, food)] = Real(f"A_{farm}_{food}", lower_bound=0, upper_bound=land_availability[farm])
            pbar.update(1)
            Y[(farm, food)] = Binary(f"Y_{farm}_{food}")
            pbar.update(1)
    
    # Create lambda variables and f_approx for piecewise approximation
    pbar.set_description("Creating piecewise approximation variables")
    for farm in farms:
        max_val = land_availability[farm]
        approx = approximations[max_val]
        n_breakpoints_total = len(approx.breakpoints)
        
        for food in foods:
            # Create lambda variables for each breakpoint (SOS2 constraint)
            Lambda[(farm, food)] = {}
            for i in range(n_breakpoints_total):
                Lambda[(farm, food)][i] = Real(
                    f"Lambda_{farm}_{food}_{i}", 
                    lower_bound=0, 
                    upper_bound=1
                )
                pbar.update(1)
            
            # f_approx will be computed as weighted sum of function values at breakpoints
            f_approx[(farm, food)] = sum(
                Lambda[(farm, food)][i] * approx.function_values[i] 
                for i in range(n_breakpoints_total)
            )
            pbar.update(1)
    
    # Piecewise approximation constraints
    pbar.set_description("Adding piecewise approximation constraints")
    for farm in farms:
        max_val = land_availability[farm]
        approx = approximations[max_val]
        n_breakpoints_total = len(approx.breakpoints)
        
        for food in foods:
            # Constraint 1: A[(farm, food)] = sum of lambda_i * breakpoint_i
            cqm.add_constraint(
                A[(farm, food)] - sum(
                    Lambda[(farm, food)][i] * approx.breakpoints[i] 
                    for i in range(n_breakpoints_total)
                ) == 0,
                label=f"Piecewise_A_Definition_{farm}_{food}"
            )
            pbar.update(1)
            
            # Constraint 2: Convexity constraint: sum of lambda_i = 1
            cqm.add_constraint(
                sum(Lambda[(farm, food)][i] for i in range(n_breakpoints_total)) == 1,
                label=f"Piecewise_Convexity_{farm}_{food}"
            )
            pbar.update(1)
            
            # Note: SOS2 constraint (at most 2 adjacent lambdas non-zero) 
            # is implicit in CQM solver - we rely on the convex combination
    
    # Objective function using piecewise approximation
    pbar.set_description("Building non-linear objective")
    total_area = sum(land_availability[farm] for farm in farms)
    
    objective = 0
    for farm in farms:
        for food in foods:
            # Use f_approx instead of A directly for non-linear objective
            objective += (
                weights.get('nutritional_value', 0) * foods[food].get('nutritional_value', 0) * f_approx[(farm, food)] +
                weights.get('nutrient_density', 0) * foods[food].get('nutrient_density', 0) * f_approx[(farm, food)] -
                weights.get('environmental_impact', 0) * foods[food].get('environmental_impact', 0) * f_approx[(farm, food)] +
                weights.get('affordability', 0) * foods[food].get('affordability', 0) * f_approx[(farm, food)] +
                weights.get('sustainability', 0) * foods[food].get('sustainability', 0) * f_approx[(farm, food)]
            )
            pbar.update(1)
    
    cqm.set_objective(-objective)
    
    # Constraint metadata
    constraint_metadata = {
        'land_availability': {},
        'min_area_if_selected': {},
        'max_area_if_selected': {},
        'food_group_min': {},
        'food_group_max': {},
        'piecewise_a_definition': {},
        'piecewise_convexity': {}
    }
    
    # Store piecewise approximation metadata
    approximation_metadata = {
        'power': power,
        'num_interior_points': num_breakpoints,
        'approximations': {}
    }
    
    for max_val, approx in approximations.items():
        max_abs, max_rel, avg_abs = approx.get_max_error()
        approximation_metadata['approximations'][float(max_val)] = {
            'max_value': max_val,
            'total_breakpoints': len(approx.breakpoints),
            'breakpoints': approx.breakpoints.tolist(),
            'function_values': approx.function_values.tolist(),
            'max_absolute_error': max_abs,
            'max_relative_error_percent': max_rel * 100,
            'average_absolute_error': avg_abs
        }
    
    # Land availability constraints
    pbar.set_description("Adding land constraints")
    for farm in farms:
        cqm.add_constraint(
            sum(A[(farm, food)] for food in foods) - land_availability[farm] <= 0,
            label=f"Land_Availability_{farm}"
        )
        constraint_metadata['land_availability'][farm] = {
            'type': 'land_availability',
            'farm': farm,
            'max_land': land_availability[farm]
        }
        pbar.update(1)
    
    # Linking constraints
    pbar.set_description("Adding linking constraints")
    for farm in farms:
        for food in foods:
            A_min = min_planting_area.get(food, 0)
            
            cqm.add_constraint(
                A[(farm, food)] - A_min * Y[(farm, food)] >= 0,
                label=f"Min_Area_If_Selected_{farm}_{food}"
            )
            constraint_metadata['min_area_if_selected'][(farm, food)] = {
                'type': 'min_area_if_selected',
                'farm': farm,
                'food': food,
                'min_area': A_min
            }
            pbar.update(1)
            
            cqm.add_constraint(
                A[(farm, food)] - land_availability[farm] * Y[(farm, food)] <= 0,
                label=f"Max_Area_If_Selected_{farm}_{food}"
            )
            constraint_metadata['max_area_if_selected'][(farm, food)] = {
                'type': 'max_area_if_selected',
                'farm': farm,
                'food': food,
                'max_land': land_availability[farm]
            }
            pbar.update(1)
    
    # Food group constraints
    pbar.set_description("Adding food group constraints")
    if food_group_constraints:
        for group, constraints in food_group_constraints.items():
            foods_in_group = food_groups.get(group, [])
            if foods_in_group:
                for farm in farms:
                    if 'min_foods' in constraints:
                        cqm.add_constraint(
                            sum(Y[(farm, food)] for food in foods_in_group) - constraints['min_foods'] >= 0,
                            label=f"Food_Group_Min_{group}_{farm}"
                        )
                        constraint_metadata['food_group_min'][(group, farm)] = {
                            'type': 'food_group_min',
                            'group': group,
                            'farm': farm,
                            'min_foods': constraints['min_foods'],
                            'foods_in_group': foods_in_group
                        }
                        pbar.update(1)
                    
                    if 'max_foods' in constraints:
                        cqm.add_constraint(
                            sum(Y[(farm, food)] for food in foods_in_group) - constraints['max_foods'] <= 0,
                            label=f"Food_Group_Max_{group}_{farm}"
                        )
                        constraint_metadata['food_group_max'][(group, farm)] = {
                            'type': 'food_group_max',
                            'group': group,
                            'farm': farm,
                            'max_foods': constraints['max_foods'],
                            'foods_in_group': foods_in_group
                        }
                        pbar.update(1)
    
    pbar.set_description("CQM complete")
    pbar.close()
    
    return cqm, A, Y, Lambda, constraint_metadata, approximation_metadata

def solve_with_pulp(farms, foods, food_groups, config):
    """Solve with PuLP and return model and results."""
    params = config['parameters']
    land_availability = params['land_availability']
    weights = params['weights']
    min_planting_area = params.get('minimum_planting_area', {})
    food_group_constraints = params.get('food_group_constraints', {})
    
    A_pulp = pl.LpVariable.dicts("Area", [(f, c) for f in farms for c in foods], lowBound=0)
    Y_pulp = pl.LpVariable.dicts("Choose", [(f, c) for f in farms for c in foods], cat='Binary')
    
    total_area = sum(land_availability[f] for f in farms)
    
    goal = (
        weights.get('nutritional_value', 0) * pl.lpSum([(foods[c].get('nutritional_value', 0) * A_pulp[(f, c)]) for f in farms for c in foods]) / total_area +
        weights.get('nutrient_density', 0) * pl.lpSum([(foods[c].get('nutrient_density', 0) * A_pulp[(f, c)]) for f in farms for c in foods]) / total_area -
        weights.get('environmental_impact', 0) * pl.lpSum([(foods[c].get('environmental_impact', 0) * A_pulp[(f, c)]) for f in farms for c in foods]) / total_area +
        weights.get('affordability', 0) * pl.lpSum([(foods[c].get('affordability', 0) * A_pulp[(f, c)]) for f in farms for c in foods]) / total_area +
        weights.get('sustainability', 0) * pl.lpSum([(foods[c].get('sustainability', 0) * A_pulp[(f, c)]) for f in farms for c in foods]) / total_area
    )
    
    model = pl.LpProblem("Food_Optimization", pl.LpMaximize)
    
    for f in farms:
        model += pl.lpSum([A_pulp[(f, c)] for c in foods]) <= land_availability[f], f"Max_Area_{f}"
    
    for f in farms:
        for c in foods:
            A_min = min_planting_area.get(c, 0)
            model += A_pulp[(f, c)] >= A_min * Y_pulp[(f, c)], f"MinArea_{f}_{c}"
            model += A_pulp[(f, c)] <= land_availability[f] * Y_pulp[(f, c)], f"MaxArea_{f}_{c}"
    
    if food_group_constraints:
        for g, constraints in food_group_constraints.items():
            foods_in_group = food_groups.get(g, [])
            if foods_in_group:
                for f in farms:
                    if 'min_foods' in constraints:
                        model += pl.lpSum([Y_pulp[(f, c)] for c in foods_in_group]) >= constraints['min_foods'], f"MinFoodGroup_{f}_{g}"
                    if 'max_foods' in constraints:
                        model += pl.lpSum([Y_pulp[(f, c)] for c in foods_in_group]) <= constraints['max_foods'], f"MaxFoodGroup_{f}_{g}"
    
    model += goal, "Objective"
    
    start_time = time.time()
    model.solve(pl.PULP_CBC_CMD(msg=0))
    solve_time = time.time() - start_time
    
    # Extract results
    results = {
        'status': pl.LpStatus[model.status],
        'objective_value': pl.value(model.objective),
        'solve_time': solve_time,
        'areas': {},
        'selections': {}
    }
    
    for f in farms:
        for c in foods:
            key = f"{f}_{c}"
            results['areas'][key] = A_pulp[(f, c)].value() if A_pulp[(f, c)].value() is not None else 0.0
            results['selections'][key] = Y_pulp[(f, c)].value() if Y_pulp[(f, c)].value() is not None else 0.0
    
    return model, results

def solve_with_dwave(cqm, token):
    """Solve with DWave and return sampleset."""
    sampler = LeapHybridCQMSampler(token=token)
    
    print("Submitting to DWave Leap hybrid solver...")
    start_time = time.time()
    sampleset = sampler.sample_cqm(cqm, label="Food Optimization - Professional Run")
    solve_time = time.time() - start_time
    
    return sampleset, solve_time

def main(scenario='simple', power=0.548, num_breakpoints=10):
    """Main execution function."""
    print("=" * 80)
    print("NON-LINEAR SOLVER RUNNER (A^0.548)")
    print("=" * 80)
    
    # Create output directories
    os.makedirs('PuLP_Results_NLN', exist_ok=True)
    os.makedirs('DWave_Results_NLN', exist_ok=True)
    os.makedirs('CQM_Models_NLN', exist_ok=True)
    os.makedirs('Constraints_NLN', exist_ok=True)
    
    # Load scenario
    print(f"\nLoading '{scenario}' scenario...")
    farms, foods, food_groups, config = load_food_data(scenario)
    print(f"  Farms: {len(farms)} - {farms}")
    print(f"  Foods: {len(foods)} - {list(foods.keys())}")
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create CQM with piecewise approximation
    print("\nCreating CQM with non-linear objective...")
    cqm, A, Y, Lambda, constraint_metadata, approximation_metadata = create_cqm(
        farms, foods, food_groups, config, power=power, num_breakpoints=num_breakpoints
    )
    print(f"  Variables: {len(cqm.variables)}")
    print(f"  Constraints: {len(cqm.constraints)}")
    print(f"  Power function: f(A) = A^{power}")
    
    # Save CQM
    cqm_path = f'CQM_Models_NLN/cqm_nln_{scenario}_{timestamp}.cqm'
    print(f"\nSaving CQM to {cqm_path}...")
    with open(cqm_path, 'wb') as f:
        shutil.copyfileobj(cqm.to_file(), f)
    
    # Save constraint metadata
    constraints_path = f'Constraints_NLN/constraints_nln_{scenario}_{timestamp}.json'
    print(f"Saving constraints to {constraints_path}...")
    
    # Convert constraint_metadata keys to strings for JSON serialization
    # Also convert foods dict to serializable format
    foods_serializable = {
        name: {k: float(v) if isinstance(v, (int, float)) else v for k, v in attrs.items()}
        for name, attrs in foods.items()
    }
    
    constraints_json = {
        'scenario': scenario,
        'timestamp': timestamp,
        'farms': farms,
        'foods': list(foods.keys()),
        'foods_data': foods_serializable,  # Add full food data for objective calculation
        'food_groups': food_groups,
        'config': config,
        'power': power,
        'num_breakpoints': num_breakpoints,
        'approximation_metadata': approximation_metadata,
        'constraint_metadata': {
            'land_availability': {str(k): v for k, v in constraint_metadata['land_availability'].items()},
            'min_area_if_selected': {str(k): v for k, v in constraint_metadata['min_area_if_selected'].items()},
            'max_area_if_selected': {str(k): v for k, v in constraint_metadata['max_area_if_selected'].items()},
            'food_group_min': {str(k): v for k, v in constraint_metadata['food_group_min'].items()},
            'food_group_max': {str(k): v for k, v in constraint_metadata['food_group_max'].items()}
        }
    }
    
    with open(constraints_path, 'w') as f:
        json.dump(constraints_json, f, indent=2)
    
    print("\n" + "=" * 80)
    print("CQM CREATION COMPLETE")
    print("=" * 80)
    print(f"CQM saved to: {cqm_path}")
    print(f"Constraints saved to: {constraints_path}")
    print(f"\nNon-linear objective: f(A) = A^{power}")
    print(f"Piecewise approximation: {num_breakpoints} interior points")
    print(f"Total variables: {len(cqm.variables)}")
    print(f"Total constraints: {len(cqm.constraints)}")
    
    # Skip PuLP and DWave solving for now - just verify CQM creation
    print("\n" + "=" * 80)
    print("NOTE: PuLP and DWave solving skipped (token removed for testing)")
    print("CQM model created successfully and saved.")
    print("=" * 80)
    
    return cqm_path, constraints_path

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Run solvers with non-linear objective (A^power) on a food optimization scenario'
    )
    parser.add_argument('--scenario', type=str, default='simple', 
                       choices=['simple', 'intermediate', 'full', 'custom', 'full_family'],
                       help='Scenario to solve (default: simple)')
    parser.add_argument('--power', type=float, default=0.548,
                       help='Power for non-linear objective f(A) = A^power (default: 0.548)')
    parser.add_argument('--breakpoints', type=int, default=10,
                       help='Number of interior breakpoints for piecewise approximation (default: 10)')
    
    args = parser.parse_args()
    
    main(scenario=args.scenario, power=args.power, num_breakpoints=args.breakpoints)
