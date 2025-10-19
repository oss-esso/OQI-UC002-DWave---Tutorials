#!/usr/bin/env python3
"""
CQM (Constrained Quadratic Model) solver for the custom food optimization scenario.
Converts the problem to CQM format and solves with D-Wave.
"""

import sys
import os
import time
import numpy as np
from typing import Dict, List, Tuple, Any

try:
    from dimod import ConstrainedQuadraticModel, Real, Binary
    from dwave.system import LeapHybridCQMSampler
    DWAVE_AVAILABLE = True
except ImportError:
    print("Warning: D-Wave Ocean SDK not available. Install with: pip install dwave-ocean-sdk")
    DWAVE_AVAILABLE = False

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from scenarios import load_food_data

def solve_food_optimization_cqm(farms: List[str], foods: Dict[str, Dict[str, float]], 
                               food_groups: Dict[str, List[str]], config: Dict, 
                               dwave_token: str = None) -> Dict[str, Any]:
    """
    Solve the food optimization problem using D-Wave CQM.
    
    Args:
        farms: List of farm names
        foods: Dictionary of food data with nutritional values
        food_groups: Dictionary mapping food groups to foods
        config: Configuration parameters
        dwave_token: D-Wave API token (use dummy if None)
        
    Returns:
        Dictionary containing solution results
    """
    if not DWAVE_AVAILABLE:
        return {
            'status': 'ERROR',
            'error': 'D-Wave Ocean SDK not available',
            'objective_value': None,
            'solution': None,
            'binary_solution': None,
            'solve_time': 0,
            'solver': 'CQM-DWave'
        }
    
    print("Setting up CQM optimization problem...")
    
    # Extract parameters
    params = config['parameters']
    weights = params['weights']
    land_availability = params['land_availability']
    min_planting_area = params['minimum_planting_area']
    max_percentage_per_crop = params['max_percentage_per_crop']
    social_benefit = params['social_benefit']
    food_group_constraints = params['food_group_constraints']
    
    # Create the CQM
    cqm = ConstrainedQuadraticModel()
    
    # Decision variables: x[farm, food] = area allocated to food on farm
    x = {}
    y = {}
    
    for farm in farms:
        for food in foods.keys():
            # Continuous variable for area
            max_area = land_availability[farm]  # Upper bound is farm's total land
            x[(farm, food)] = Real(f"area_{farm}_{food}", lower_bound=0, upper_bound=max_area)
            
            # Binary variable for planting decision
            y[(farm, food)] = Binary(f"planted_{farm}_{food}")
    
    print(f"Created {len(x)} continuous variables and {len(y)} binary variables")
    
    # Objective function: weighted sum of objectives (maximize)
    objective = 0
    
    for farm in farms:
        for food in foods.keys():
            food_data = foods[food]
            
            # Calculate weighted objective for this farm-food combination
            weighted_score = (
                weights['nutritional_value'] * food_data['nutritional_value'] +
                weights['nutrient_density'] * food_data['nutrient_density'] +
                weights['affordability'] * food_data['affordability'] +
                weights['sustainability'] * food_data['sustainability'] -
                weights['environmental_impact'] * food_data['environmental_impact']  # Minimize environmental impact
            )
            
            objective += weighted_score * x[(farm, food)]
    
    # Set objective (CQM minimizes, so negate for maximization)
    cqm.set_objective(-objective)
    
    print("Added objective function")
    
    # Constraints
    constraint_count = 0
    
    # 1. Land availability constraints
    for farm in farms:
        land_constraint = sum(x[(farm, food)] for food in foods.keys()) <= land_availability[farm]
        cqm.add_constraint(land_constraint, label=f"Land_Availability_{farm}")
        constraint_count += 1
    
    # 2. Social benefit constraints (minimum land utilization)
    for farm in farms:
        min_land = social_benefit[farm] * land_availability[farm]
        social_constraint = sum(x[(farm, food)] for food in foods.keys()) >= min_land
        cqm.add_constraint(social_constraint, label=f"Social_Benefit_{farm}")
        constraint_count += 1
    
    # 3. Planting area constraints
    for farm in farms:
        for food in foods.keys():
            # If not planted (y=0), then area must be 0
            # If planted (y=1), then area can be > 0
            # This is modeled as: x <= M * y where M is a large constant
            M = land_availability[farm]
            max_constraint = x[(farm, food)] <= M * y[(farm, food)]
            cqm.add_constraint(max_constraint, label=f"Max_Area_{farm}_{food}")
            
            # If planted (y=1), then area must be at least minimum
            # This is modeled as: x >= min_area * y
            min_constraint = x[(farm, food)] >= min_planting_area[food] * y[(farm, food)]
            cqm.add_constraint(min_constraint, label=f"Min_Area_{farm}_{food}")
            
            # Maximum percentage per crop constraint
            max_area = max_percentage_per_crop[food] * land_availability[farm]
            percentage_constraint = x[(farm, food)] <= max_area
            cqm.add_constraint(percentage_constraint, label=f"Max_Percentage_{farm}_{food}")
            constraint_count += 3
    
    # 4. Food group constraints
    for farm in farms:
        for group, foods_in_group in food_groups.items():
            group_constraints = food_group_constraints[group]
            
            # At least min_foods from each group must be selected
            min_group_constraint = sum(y[(farm, food)] for food in foods_in_group) >= group_constraints['min_foods']
            cqm.add_constraint(min_group_constraint, label=f"Min_Group_{farm}_{group}")
            
            # At most max_foods from each group can be selected
            max_group_constraint = sum(y[(farm, food)] for food in foods_in_group) <= group_constraints['max_foods']
            cqm.add_constraint(max_group_constraint, label=f"Max_Group_{farm}_{group}")
            constraint_count += 2
    
    print(f"Added {constraint_count} constraints")
    print(f"CQM has {len(cqm.variables)} variables and {len(cqm.constraints)} constraints")
    
    # Check if we have a dummy token
    if dwave_token is None or dwave_token.strip().lower() in ['dummy', 'test', '']:
        print("Using dummy D-Wave token - simulating solve...")
        
        # Simulate a solution for testing
        simulated_solution = {}
        simulated_binary = {}
        
        for farm in farms:
            simulated_solution[farm] = {}
            simulated_binary[farm] = {}
            
            # Simple heuristic: allocate based on weighted scores
            food_scores = []
            for food in foods.keys():
                food_data = foods[food]
                weighted_score = (
                    weights['nutritional_value'] * food_data['nutritional_value'] +
                    weights['nutrient_density'] * food_data['nutrient_density'] +
                    weights['affordability'] * food_data['affordability'] +
                    weights['sustainability'] * food_data['sustainability'] -
                    weights['environmental_impact'] * food_data['environmental_impact']
                )
                food_scores.append((food, weighted_score))
            
            # Sort by score and allocate
            food_scores.sort(key=lambda x: x[1], reverse=True)
            remaining_land = land_availability[farm]
            min_land_required = social_benefit[farm] * land_availability[farm]
            
            # Ensure at least one food from each group
            selected_foods = set()
            for group, foods_in_group in food_groups.items():
                if not any(food in selected_foods for food in foods_in_group):
                    # Select best food from this group
                    best_food = max(foods_in_group, key=lambda f: dict(food_scores)[f])
                    selected_foods.add(best_food)
            
            # Allocate areas
            for food in foods.keys():
                if food in selected_foods and remaining_land >= min_planting_area[food]:
                    # Allocate minimum area
                    area = min(min_planting_area[food] * 2, remaining_land * 0.3)  # Simple allocation
                    simulated_solution[farm][food] = area
                    simulated_binary[farm][food] = 1
                    remaining_land -= area
                else:
                    simulated_solution[farm][food] = 0
                    simulated_binary[farm][food] = 0
        
        # Calculate simulated objective
        simulated_objective = 0
        for farm in farms:
            for food in foods.keys():
                food_data = foods[food]
                weighted_score = (
                    weights['nutritional_value'] * food_data['nutritional_value'] +
                    weights['nutrient_density'] * food_data['nutrient_density'] +
                    weights['affordability'] * food_data['affordability'] +
                    weights['sustainability'] * food_data['sustainability'] -
                    weights['environmental_impact'] * food_data['environmental_impact']
                )
                simulated_objective += weighted_score * simulated_solution[farm][food]
        
        return {
            'status': 'SIMULATED',
            'objective_value': simulated_objective,
            'solution': simulated_solution,
            'binary_solution': simulated_binary,
            'solve_time': 0.1,
            'solver': 'CQM-DWave-Simulated',
            'note': 'Simulated solution - dummy D-Wave token used'
        }
    
    # Real D-Wave solve
    print("Solving with D-Wave CQM sampler...")
    start_time = time.time()
    
    try:
        # Set up the sampler
        sampler = LeapHybridCQMSampler(token=dwave_token)
        
        # Solve
        sampleset = sampler.sample_cqm(cqm, label="Food_Optimization_CQM")
        solve_time = time.time() - start_time
        
        # Get the best solution
        best_sample = sampleset.first
        
        print(f"Solve time: {solve_time:.2f} seconds")
        print(f"Best energy: {best_sample.energy}")
        print(f"Is feasible: {sampleset.data_vectors['is_feasible'][0]}")
        
        # Extract solution
        solution = {}
        binary_solution = {}
        
        for farm in farms:
            solution[farm] = {}
            binary_solution[farm] = {}
            
            for food in foods.keys():
                area_var = f"area_{farm}_{food}"
                planted_var = f"planted_{farm}_{food}"
                
                solution[farm][food] = best_sample.sample.get(area_var, 0)
                binary_solution[farm][food] = int(best_sample.sample.get(planted_var, 0))
        
        # Display results
        for farm in farms:
            farm_total = sum(solution[farm].values())
            print(f"{farm}: Total area = {farm_total:.2f}/{land_availability[farm]} (utilization: {farm_total/land_availability[farm]*100:.1f}%)")
            
            for food in foods.keys():
                if solution[farm][food] > 0.01:
                    planted_status = "Planted" if binary_solution[farm][food] else "Not planted"
                    print(f"  {food}: {solution[farm][food]:.2f} hectares ({planted_status})")
        
        return {
            'status': 'OPTIMAL' if sampleset.data_vectors['is_feasible'][0] else 'INFEASIBLE',
            'objective_value': -best_sample.energy,  # Negate back since we minimized
            'solution': solution,
            'binary_solution': binary_solution,
            'solve_time': solve_time,
            'solver': 'CQM-DWave',
            'energy': best_sample.energy,
            'is_feasible': sampleset.data_vectors['is_feasible'][0]
        }
        
    except Exception as e:
        solve_time = time.time() - start_time
        print(f"Error solving with D-Wave: {e}")
        return {
            'status': 'ERROR',
            'error': str(e),
            'objective_value': None,
            'solution': None,
            'binary_solution': None,
            'solve_time': solve_time,
            'solver': 'CQM-DWave'
        }

def main():
    """Main function to run the CQM solver."""
    print("Loading custom food optimization scenario...")
    
    # Load the custom scenario
    farms, foods, food_groups, config = load_food_data('custom')
    
    print(f"Loaded scenario with {len(farms)} farms, {len(foods)} foods, {len(food_groups)} food groups")
    print("Farms:", farms)
    print("Foods:", list(foods.keys()))
    print("Food groups:", {k: v for k, v in food_groups.items()})
    
    # Use dummy token (replace with real token for actual D-Wave access)
    dwave_token = "dummy"
    
    # Solve the problem
    result = solve_food_optimization_cqm(farms, foods, food_groups, config, dwave_token)
    
    print(f"\nCQM Solver Result:")
    print(f"Status: {result['status']}")
    print(f"Objective Value: {result['objective_value']}")
    print(f"Solve Time: {result['solve_time']:.2f} seconds")
    
    return result

if __name__ == "__main__":
    result = main()