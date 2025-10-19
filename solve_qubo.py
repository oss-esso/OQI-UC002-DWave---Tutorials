#!/usr/bin/env python3
"""
QUBO (Quadratic Unconstrained Binary Optimization) solver for the custom food optimization scenario.
Converts the problem to QUBO format and solves with D-Wave.
"""

import sys
import os
import time
import numpy as np
from typing import Dict, List, Tuple, Any
from itertools import product

try:
    import dimod
    from dwave.system import DWaveSampler, EmbeddingComposite
    from dwave.system import LeapHybridSampler
    DWAVE_AVAILABLE = True
except ImportError:
    print("Warning: D-Wave Ocean SDK not available. Install with: pip install dwave-ocean-sdk")
    DWAVE_AVAILABLE = False

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from scenarios import load_food_data

def discretize_areas(farms: List[str], foods: Dict[str, Dict[str, float]], 
                    config: Dict, num_levels: int = 5) -> Dict:
    """
    Discretize the continuous area variables for QUBO formulation.
    
    Args:
        farms: List of farm names
        foods: Dictionary of food data
        config: Configuration parameters
        num_levels: Number of discrete levels for each area variable
        
    Returns:
        Dictionary with discretization information
    """
    params = config['parameters']
    land_availability = params['land_availability']
    min_planting_area = params['minimum_planting_area']
    max_percentage_per_crop = params['max_percentage_per_crop']
    
    discretization = {}
    
    for farm in farms:
        discretization[farm] = {}
        for food in foods.keys():
            min_area = min_planting_area[food]
            max_area = min(max_percentage_per_crop[food] * land_availability[farm], 
                          land_availability[farm])
            
            # Create discrete levels including 0 (not planted)
            if num_levels > 1:
                levels = [0] + list(np.linspace(min_area, max_area, num_levels - 1))
            else:
                levels = [0, max_area]
            
            discretization[farm][food] = {
                'levels': levels,
                'num_levels': len(levels),
                'min_area': min_area,
                'max_area': max_area
            }
    
    return discretization

def create_qubo_variables(farms: List[str], foods: Dict[str, Dict[str, float]], 
                         discretization: Dict) -> Dict:
    """
    Create binary variables for QUBO formulation.
    
    For each farm-food pair, create binary variables x[farm,food,level] 
    where exactly one level is selected.
    """
    variables = {}
    var_to_key = {}
    key_to_var = {}
    var_count = 0
    
    for farm in farms:
        variables[farm] = {}
        for food in foods.keys():
            variables[farm][food] = {}
            levels = discretization[farm][food]['levels']
            
            for i, level in enumerate(levels):
                var_name = f"x_{farm}_{food}_{i}"
                variables[farm][food][i] = var_name
                var_to_key[var_name] = (farm, food, i, level)
                key_to_var[(farm, food, i)] = var_name
                var_count += 1
    
    return variables, var_to_key, key_to_var, var_count

def solve_food_optimization_qubo(farms: List[str], foods: Dict[str, Dict[str, float]], 
                                food_groups: Dict[str, List[str]], config: Dict, 
                                dwave_token: str = None, num_levels: int = 3) -> Dict[str, Any]:
    """
    Solve the food optimization problem using D-Wave QUBO.
    
    Args:
        farms: List of farm names
        foods: Dictionary of food data with nutritional values
        food_groups: Dictionary mapping food groups to foods
        config: Configuration parameters
        dwave_token: D-Wave API token (use dummy if None)
        num_levels: Number of discrete levels for area variables
        
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
            'solver': 'QUBO-DWave'
        }
    
    print(f"Setting up QUBO optimization problem with {num_levels} discrete levels...")
    
    # Extract parameters
    params = config['parameters']
    weights = params['weights']
    land_availability = params['land_availability']
    min_planting_area = params['minimum_planting_area']
    max_percentage_per_crop = params['max_percentage_per_crop']
    social_benefit = params['social_benefit']
    food_group_constraints = params['food_group_constraints']
    
    # Discretize continuous variables
    discretization = discretize_areas(farms, foods, config, num_levels)
    
    # Create binary variables
    variables, var_to_key, key_to_var, var_count = create_qubo_variables(farms, foods, discretization)
    
    print(f"Created {var_count} binary variables")
    
    # Build QUBO matrix
    Q = {}
    
    # Penalty weights for constraints (should be larger than objective coefficients)
    penalty_land = 1000
    penalty_social = 1000
    penalty_selection = 1000
    penalty_group = 1000
    
    # 1. Objective function (maximize, so negate for QUBO minimization)
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
            
            levels = discretization[farm][food]['levels']
            for i, level in enumerate(levels):
                var_name = variables[farm][food][i]
                # Add to diagonal (negate for maximization)
                Q[(var_name, var_name)] = Q.get((var_name, var_name), 0) - weighted_score * level
    
    print("Added objective function")
    
    # 2. Selection constraints: exactly one level per farm-food pair
    for farm in farms:
        for food in foods.keys():
            levels = discretization[farm][food]['levels']
            vars_for_food = [variables[farm][food][i] for i in range(len(levels))]
            
            # Constraint: sum(x_i) = 1
            # Penalty form: penalty * (sum(x_i) - 1)^2
            # = penalty * (sum(x_i^2) + sum_i sum_j≠i (x_i * x_j) - 2*sum(x_i) + 1)
            
            # Linear terms: -2 * penalty
            for var in vars_for_food:
                Q[(var, var)] = Q.get((var, var), 0) + penalty_selection * (1 - 2)
            
            # Quadratic terms: 2 * penalty
            for i, var1 in enumerate(vars_for_food):
                for j, var2 in enumerate(vars_for_food):
                    if i < j:  # Upper triangular
                        key = (var1, var2)
                        Q[key] = Q.get(key, 0) + 2 * penalty_selection
    
    # 3. Land availability constraints
    for farm in farms:
        max_land = land_availability[farm]
        
        # Collect all area variables for this farm
        farm_vars = []
        for food in foods.keys():
            levels = discretization[farm][food]['levels']
            for i, level in enumerate(levels):
                var_name = variables[farm][food][i]
                farm_vars.append((var_name, level))
        
        # Constraint: sum(level * x_var) <= max_land
        # Penalty form: penalty * max(0, sum(level * x_var) - max_land)^2
        # For simplicity, we'll use a quadratic penalty: penalty * (sum(level * x_var) - max_land)^2
        
        # This is complex to implement exactly, so we'll use a simplified approach
        # We'll add penalties for combinations that exceed the limit
        for i, (var1, level1) in enumerate(farm_vars):
            for j, (var2, level2) in enumerate(farm_vars):
                if i <= j:  # Include diagonal and upper triangular
                    total_area = level1 + level2
                    if total_area > max_land:
                        excess = total_area - max_land
                        penalty = penalty_land * excess / max_land  # Normalized penalty
                        
                        if i == j:
                            Q[(var1, var1)] = Q.get((var1, var1), 0) + penalty
                        else:
                            Q[(var1, var2)] = Q.get((var1, var2), 0) + penalty
    
    # 4. Social benefit constraints (minimum land utilization)
    for farm in farms:
        min_land = social_benefit[farm] * land_availability[farm]
        
        # Collect all area variables for this farm
        farm_vars = []
        for food in foods.keys():
            levels = discretization[farm][food]['levels']
            for i, level in enumerate(levels):
                var_name = variables[farm][food][i]
                farm_vars.append((var_name, level))
        
        # Constraint: sum(level * x_var) >= min_land
        # Penalty form: penalty * max(0, min_land - sum(level * x_var))^2
        # We'll use a linear penalty for undershoot
        for var_name, level in farm_vars:
            if level > 0:  # Reward using land
                Q[(var_name, var_name)] = Q.get((var_name, var_name), 0) - penalty_social * level / min_land
    
    # 5. Food group constraints
    for farm in farms:
        for group, foods_in_group in food_groups.items():
            constraints = food_group_constraints[group]
            min_foods = constraints['min_foods']
            max_foods = constraints['max_foods']
            
            # Binary variables indicating if a food from this group is selected (level > 0)
            group_vars = []
            for food in foods_in_group:
                levels = discretization[farm][food]['levels']
                for i, level in enumerate(levels):
                    if level > 0:  # Only non-zero levels count as "selected"
                        var_name = variables[farm][food][i]
                        group_vars.append(var_name)
            
            # This is a simplified constraint - in a full implementation,
            # we'd need additional binary variables to properly model group selection
            # For now, we'll add a soft penalty
            if len(group_vars) > 0:
                # Encourage having at least min_foods
                for var in group_vars[:min_foods]:
                    Q[(var, var)] = Q.get((var, var), 0) - penalty_group * 0.1
    
    print(f"Built QUBO matrix with {len(Q)} non-zero elements")
    
    # Check if we have a dummy token
    if dwave_token is None or dwave_token.strip().lower() in ['dummy', 'test', '']:
        print("Using dummy D-Wave token - simulating solve...")
        
        # Simple heuristic solution
        simulated_sample = {}
        
        # For each farm-food pair, select the level with best objective value
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
                
                levels = discretization[farm][food]['levels']
                best_level_idx = 0
                best_score = 0
                
                # Find best level considering objective and feasibility
                for i, level in enumerate(levels):
                    if level == 0:
                        score = 0  # No contribution if not planted
                    else:
                        score = weighted_score * level
                    
                    if score > best_score and level >= min_planting_area[food]:
                        best_score = score
                        best_level_idx = i
                
                # Set the selected level
                for i in range(len(levels)):
                    var_name = variables[farm][food][i]
                    simulated_sample[var_name] = 1 if i == best_level_idx else 0
        
        # Convert to solution format
        solution = {}
        binary_solution = {}
        
        for farm in farms:
            solution[farm] = {}
            binary_solution[farm] = {}
            
            for food in foods.keys():
                levels = discretization[farm][food]['levels']
                selected_area = 0
                
                for i, level in enumerate(levels):
                    var_name = variables[farm][food][i]
                    if simulated_sample.get(var_name, 0) == 1:
                        selected_area = level
                        break
                
                solution[farm][food] = selected_area
                binary_solution[farm][food] = 1 if selected_area > 0 else 0
        
        # Calculate objective
        objective_value = 0
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
                objective_value += weighted_score * solution[farm][food]
        
        return {
            'status': 'SIMULATED',
            'objective_value': objective_value,
            'solution': solution,
            'binary_solution': binary_solution,
            'solve_time': 0.1,
            'solver': 'QUBO-DWave-Simulated',
            'note': 'Simulated solution - dummy D-Wave token used',
            'discretization': discretization
        }
    
    # Real D-Wave solve
    print("Solving with D-Wave QUBO sampler...")
    start_time = time.time()
    
    try:
        # Create BQM from QUBO
        bqm = dimod.BinaryQuadraticModel.from_qubo(Q)
        
        print(f"BQM has {len(bqm.variables)} variables and {len(bqm.quadratic)} quadratic terms")
        
        # Use Hybrid sampler for larger problems
        sampler = LeapHybridSampler(token=dwave_token)
        
        # Solve
        sampleset = sampler.sample(bqm, label="Food_Optimization_QUBO")
        solve_time = time.time() - start_time
        
        # Get the best solution
        best_sample = sampleset.first
        
        print(f"Solve time: {solve_time:.2f} seconds")
        print(f"Best energy: {best_sample.energy}")
        
        # Convert to solution format
        solution = {}
        binary_solution = {}
        
        for farm in farms:
            solution[farm] = {}
            binary_solution[farm] = {}
            
            for food in foods.keys():
                levels = discretization[farm][food]['levels']
                selected_area = 0
                
                for i, level in enumerate(levels):
                    var_name = variables[farm][food][i]
                    if best_sample.sample.get(var_name, 0) == 1:
                        selected_area = level
                        break
                
                solution[farm][food] = selected_area
                binary_solution[farm][food] = 1 if selected_area > 0 else 0
        
        # Display results
        for farm in farms:
            farm_total = sum(solution[farm].values())
            print(f"{farm}: Total area = {farm_total:.2f}/{land_availability[farm]} (utilization: {farm_total/land_availability[farm]*100:.1f}%)")
            
            for food in foods.keys():
                if solution[farm][food] > 0.01:
                    planted_status = "Planted" if binary_solution[farm][food] else "Not planted"
                    print(f"  {food}: {solution[farm][food]:.2f} hectares ({planted_status})")
        
        # Calculate objective value
        objective_value = 0
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
                objective_value += weighted_score * solution[farm][food]
        
        return {
            'status': 'OPTIMAL',
            'objective_value': objective_value,
            'solution': solution,
            'binary_solution': binary_solution,
            'solve_time': solve_time,
            'solver': 'QUBO-DWave',
            'energy': best_sample.energy,
            'discretization': discretization
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
            'solver': 'QUBO-DWave'
        }

def main():
    """Main function to run the QUBO solver."""
    print("Loading custom food optimization scenario...")
    
    # Load the custom scenario
    farms, foods, food_groups, config = load_food_data('custom')
    
    print(f"Loaded scenario with {len(farms)} farms, {len(foods)} foods, {len(food_groups)} food groups")
    print("Farms:", farms)
    print("Foods:", list(foods.keys()))
    print("Food groups:", {k: v for k, v in food_groups.items()})
    
    # Use dummy token (replace with real token for actual D-Wave access)
    dwave_token = "dummy"
    
    # Solve the problem with 3 discrete levels for area variables
    result = solve_food_optimization_qubo(farms, foods, food_groups, config, dwave_token, num_levels=3)
    
    print(f"\nQUBO Solver Result:")
    print(f"Status: {result['status']}")
    print(f"Objective Value: {result['objective_value']}")
    print(f"Solve Time: {result['solve_time']:.2f} seconds")
    
    if 'discretization' in result:
        print(f"\nDiscretization levels used:")
        for farm in farms:
            for food in foods.keys():
                levels = result['discretization'][farm][food]['levels']
                print(f"  {farm}-{food}: {[f'{l:.1f}' for l in levels]}")
    
    return result

if __name__ == "__main__":
    result = main()