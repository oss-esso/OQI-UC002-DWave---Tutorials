#!/usr/bin/env python3
"""
PuLP solver for the custom food optimization scenario.
Solves the multi-objective optimization problem with all constraints.
"""

import sys
import os
import time
import pulp
import numpy as np
from typing import Dict, List, Tuple, Any

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from scenarios import load_food_data

def solve_food_optimization_pulp(farms: List[str], foods: Dict[str, Dict[str, float]], 
                                food_groups: Dict[str, List[str]], config: Dict) -> Dict[str, Any]:
    """
    Solve the food optimization problem using PuLP.
    This implementation follows the pulp_sim.py formulation closely with global coordination.
    
    Args:
        farms: List of farm names
        foods: Dictionary of food data with nutritional values
        food_groups: Dictionary mapping food groups to foods
        config: Configuration parameters
        
    Returns:
        Dictionary containing solution results
    """
    print("Setting up PuLP optimization problem (following pulp_sim.py formulation)...")
    
    # Extract parameters
    params = config['parameters']
    weights = params['weights']
    land_availability = params['land_availability']
    min_planting_area = params['minimum_planting_area']
    max_percentage_per_crop = params['max_percentage_per_crop']
    social_benefit = params['social_benefit']
    food_group_constraints = params['food_group_constraints']
    
    # Additional global parameters from pulp_sim.py
    global_min_different_foods = params.get('global_min_different_foods', 5)
    min_foods_per_farm = params.get('min_foods_per_farm', 1)
    max_foods_per_farm = params.get('max_foods_per_farm', 8)
    min_total_land_usage_percentage = params.get('min_total_land_usage_percentage', 0.5)
    
    # Create the problem
    prob = pulp.LpProblem("Food_Production_Optimization", pulp.LpMaximize)
    
    # Decision variables: x[farm, food] = area allocated to food on farm
    x = pulp.LpVariable.dicts("area", 
                             [(farm, food) for farm in farms for food in foods.keys()],
                             lowBound=0, cat='Continuous')
    
    # Binary variables: y[farm, food] = 1 if food is planted on farm, 0 otherwise
    y = pulp.LpVariable.dicts("planted", 
                             [(farm, food) for farm in farms for food in foods.keys()],
                             cat='Binary')
    
    print(f"Created {len(x)} continuous variables and {len(y)} binary variables")
    
    # Objective function: weighted sum of objectives (same as pulp_sim.py)
    objective_terms = []
    
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
            
            objective_terms.append(weighted_score * x[(farm, food)])
    
    # Add objective
    prob += pulp.lpSum(objective_terms), "Total_Weighted_Objective"
    
    print("Added objective function")
    
    # Constraints (following pulp_sim.py structure)
    constraint_count = 0
    
    # 1. Land availability constraints
    for farm in farms:
        prob += pulp.lpSum([x[(farm, food)] for food in foods.keys()]) <= land_availability[farm], f"Land_Availability_{farm}"
        constraint_count += 1
    
    print("Added land availability constraints")
    
    # 2. Global food selection constraint (key difference from original solve_pulp.py)
    # Create binary indicator variables for whether each food is selected at all
    food_selected = {}
    for food in foods.keys():
        food_selected[food] = pulp.LpVariable(f"food_selected_{food}", cat='Binary')
        
        # Link food_selected to individual farm selections
        for farm in farms:
            prob += food_selected[food] >= y[(farm, food)], f"Food_Selected_Lower_{food}_{farm}"
        
        # Ensure food_selected is 0 if no farm selects the food
        prob += food_selected[food] * len(farms) <= pulp.lpSum([y[(farm, food)] for farm in farms]), f"Food_Selected_Upper_{food}"
        constraint_count += len(farms) + 1
    
    # Global minimum different foods constraint
    prob += pulp.lpSum([food_selected[food] for food in foods.keys()]) >= global_min_different_foods, "Global_Min_Different_Foods"
    constraint_count += 1
    
    print(f"Added global food selection constraint (min {global_min_different_foods} different foods)")
    
    # 3. Linking constraints - x and y (following pulp_sim.py exactly)
    for farm in farms:
        for food in foods.keys():
            min_area = max(min_planting_area.get(food, 0.0001), 0.0001)
            max_percentage = max_percentage_per_crop.get(food, 0.3)
            
            # If y=0, then x=0; if y=1, then x <= max_percentage * land_availability
            prob += x[(farm, food)] <= land_availability[farm] * max_percentage * y[(farm, food)], f"Upper_Link_{farm}_{food}"
            
            # Apply minimum planting area constraint when selected
            prob += x[(farm, food)] >= min_area * y[(farm, food)], f"Lower_Link_{farm}_{food}"
            constraint_count += 2
    
    print("Added linking constraints")
    
    # 4. Farm utilization - social benefit constraints (using social_benefit parameter)
    for farm in farms:
        min_util = social_benefit.get(farm, 0.2)
        prob += pulp.lpSum([x[(farm, food)] for food in foods.keys()]) >= min_util * land_availability[farm], f"Min_Land_Use_{farm}"
        constraint_count += 1
    
    print("Added farm utilization constraints")
    
    # 5. Food variety constraints per farm (key addition from pulp_sim.py)
    for farm in farms:
        prob += pulp.lpSum([y[(farm, food)] for food in foods.keys()]) >= min_foods_per_farm, f"Min_Foods_{farm}"
        prob += pulp.lpSum([y[(farm, food)] for food in foods.keys()]) <= max_foods_per_farm, f"Max_Foods_{farm}"
        constraint_count += 2
    
    print(f"Added food variety constraints per farm (min: {min_foods_per_farm}, max: {max_foods_per_farm})")
    
    # 6. Stronger food group constraints (following pulp_sim.py)
    total_land = sum(land_availability[farm] for farm in farms)
    
    for group, group_foods in food_groups.items():
        if group_foods:
            # Ensure at least 10% of total land for each food group
            min_group_area = total_land * 0.10
            prob += pulp.lpSum([x[(farm, food)] for farm in farms for food in group_foods]) >= min_group_area, f"Min_Area_Group_{group}"
            constraint_count += 1
            
            # Require at least 2 different food types from each group with area > 1 hectare
            # Create binary indicators for foods with significant area
            significant_food = {}
            for food in group_foods:
                significant_food[food] = pulp.LpVariable(f"significant_{food}", cat='Binary')
                
                # Food is significant if total area across all farms is > 1 hectare
                prob += pulp.lpSum([x[(farm, food)] for farm in farms]) >= 1.0 * significant_food[food], f"Significant_Lower_{food}"
                prob += pulp.lpSum([x[(farm, food)] for farm in farms]) <= total_land * significant_food[food], f"Significant_Upper_{food}"
                constraint_count += 2
            
            # Require at least 2 significant foods per group (or all foods if less than 2)
            min_significant = min(2, len(group_foods))
            prob += pulp.lpSum([significant_food[food] for food in group_foods]) >= min_significant, f"Min_Significant_Foods_{group}"
            constraint_count += 1
    
    print("Added stronger food group constraints")
    
    # 7. Global minimum total land utilization constraint
    if min_total_land_usage_percentage > 0:
        min_total_usage = min_total_land_usage_percentage * total_land
        prob += pulp.lpSum([x[(farm, food)] for farm in farms for food in foods.keys()]) >= min_total_usage, "Min_Total_Land"
        constraint_count += 1
        print(f"Added global land utilization constraint (min {min_total_land_usage_percentage*100:.0f}% = {min_total_usage:.2f} hectares)")
    
    print(f"Total constraints added: {constraint_count}")
    
    # Solve the problem
    print("Solving with PuLP...")
    start_time = time.time()
    
    # Use CBC solver with increased time limit and relaxed gap (following pulp_sim.py)
    solver = pulp.PULP_CBC_CMD(timeLimit=config.get('pulp_time_limit', 120), msg=1, options=['allowableGap=0.05'])
    prob.solve(solver)
    
    solve_time = time.time() - start_time
    
    # Extract results
    status = pulp.LpStatus[prob.status]
    print(f"Status: {status}")
    print(f"Solve time: {solve_time:.2f} seconds")
    
    if prob.status == pulp.LpStatusOptimal:
        objective_value = pulp.value(prob.objective)
        print(f"Optimal objective value: {objective_value:.6f}")
        
        # Extract solution
        solution = {}
        binary_solution = {}
        
        for farm in farms:
            solution[farm] = {}
            binary_solution[farm] = {}
            farm_total = 0
            
            for food in foods.keys():
                area = pulp.value(x[(farm, food)])
                planted = pulp.value(y[(farm, food)])
                
                solution[farm][food] = area if area is not None else 0
                binary_solution[farm][food] = int(planted) if planted is not None else 0
                farm_total += solution[farm][food]
            
            print(f"{farm}: Total area = {farm_total:.2f}/{land_availability[farm]} (utilization: {farm_total/land_availability[farm]*100:.1f}%)")
            
            for food in foods.keys():
                if solution[farm][food] > 0.01:  # Only show non-zero allocations
                    planted_status = "Planted" if binary_solution[farm][food] else "Not planted"
                    print(f"  {food}: {solution[farm][food]:.2f} hectares ({planted_status})")
        
        # Show global food selection summary
        print("\nGlobal Food Selection Summary:")
        total_foods_selected = 0
        for food in foods.keys():
            food_selected_value = pulp.value(food_selected[food]) if food in food_selected else 0
            total_area = sum(solution[farm][food] for farm in farms)
            if food_selected_value and total_area > 0.01:
                print(f"  {food}: {total_area:.2f} hectares total across all farms")
                total_foods_selected += 1
        
        print(f"Total different foods selected: {total_foods_selected} (required: {global_min_different_foods})")
        
        return {
            'status': status,
            'objective_value': objective_value,
            'solution': solution,
            'binary_solution': binary_solution,
            'solve_time': solve_time,
            'solver': 'PuLP-CBC-GlobalCoordinated',
            'global_foods_selected': total_foods_selected
        }
    
    else:
        print(f"Problem could not be solved optimally. Status: {status}")
        return {
            'status': status,
            'objective_value': None,
            'solution': None,
            'binary_solution': None,
            'solve_time': solve_time,
            'solver': 'PuLP-CBC-GlobalCoordinated'
        }

def validate_solution(solution: Dict, binary_solution: Dict, farms: List[str], 
                     foods: Dict[str, Dict[str, float]], food_groups: Dict[str, List[str]], 
                     config: Dict) -> Dict[str, bool]:
    """
    Validate the solution against all constraints.
    Updated to include validation for global constraints from pulp_sim.py formulation.
    
    Returns:
        Dictionary with constraint validation results
    """
    if solution is None or binary_solution is None:
        return {'valid': False, 'reason': 'No solution to validate'}
    
    params = config['parameters']
    land_availability = params['land_availability']
    min_planting_area = params['minimum_planting_area']
    max_percentage_per_crop = params['max_percentage_per_crop']
    social_benefit = params['social_benefit']
    food_group_constraints = params['food_group_constraints']
    
    # Additional global parameters
    global_min_different_foods = params.get('global_min_different_foods', 5)
    min_foods_per_farm = params.get('min_foods_per_farm', 1)
    max_foods_per_farm = params.get('max_foods_per_farm', 8)
    min_total_land_usage_percentage = params.get('min_total_land_usage_percentage', 0.5)
    
    validation_results = {'valid': True, 'violations': []}
    
    # Calculate global metrics
    total_land = sum(land_availability[farm] for farm in farms)
    total_used_land = sum(sum(solution[farm].values()) for farm in farms)
    
    # Count globally selected foods
    globally_selected_foods = set()
    for farm in farms:
        for food in foods.keys():
            if binary_solution[farm][food] == 1:
                globally_selected_foods.add(food)
    
    for farm in farms:
        farm_total = sum(solution[farm].values())
        
        # Check land availability
        if farm_total > land_availability[farm] + 1e-6:
            validation_results['violations'].append(f"{farm}: Land availability violated ({farm_total:.2f} > {land_availability[farm]})")
            validation_results['valid'] = False
        
        # Check social benefit (minimum land utilization)
        min_land = social_benefit[farm] * land_availability[farm]
        if farm_total < min_land - 1e-6:
            validation_results['violations'].append(f"{farm}: Social benefit violated ({farm_total:.2f} < {min_land:.2f})")
            validation_results['valid'] = False
        
        # Check food variety constraints per farm
        farm_foods_selected = sum(binary_solution[farm][food] for food in foods.keys())
        if farm_foods_selected < min_foods_per_farm:
            validation_results['violations'].append(f"{farm}: Minimum foods per farm violated ({farm_foods_selected} < {min_foods_per_farm})")
            validation_results['valid'] = False
        
        if farm_foods_selected > max_foods_per_farm:
            validation_results['violations'].append(f"{farm}: Maximum foods per farm violated ({farm_foods_selected} > {max_foods_per_farm})")
            validation_results['valid'] = False
        
        # Check planting area constraints
        for food in foods.keys():
            area = solution[farm][food]
            planted = binary_solution[farm][food]
            
            if planted == 1:
                # If planted, check minimum area
                min_area = max(min_planting_area.get(food, 0.0001), 0.0001)
                if area < min_area - 1e-6:
                    validation_results['violations'].append(f"{farm}-{food}: Minimum planting area violated ({area:.2f} < {min_area})")
                    validation_results['valid'] = False
                
                # Check maximum percentage
                max_area = max_percentage_per_crop[food] * land_availability[farm]
                if area > max_area + 1e-6:
                    validation_results['violations'].append(f"{farm}-{food}: Maximum percentage violated ({area:.2f} > {max_area:.2f})")
                    validation_results['valid'] = False
            else:
                # If not planted, area should be 0
                if area > 1e-6:
                    validation_results['violations'].append(f"{farm}-{food}: Area should be 0 when not planted ({area:.2f})")
                    validation_results['valid'] = False
        
        # Check food group constraints (per farm)
        for group, foods_in_group in food_groups.items():
            group_count = sum(binary_solution[farm][food] for food in foods_in_group)
            constraints = food_group_constraints[group]
            
            if group_count < constraints['min_foods']:
                validation_results['violations'].append(f"{farm}-{group}: Minimum foods violated ({group_count} < {constraints['min_foods']})")
                validation_results['valid'] = False
            
            if group_count > constraints['max_foods']:
                validation_results['violations'].append(f"{farm}-{group}: Maximum foods violated ({group_count} > {constraints['max_foods']})")
                validation_results['valid'] = False
    
    # Global constraint validations
    
    # 1. Global minimum different foods
    if len(globally_selected_foods) < global_min_different_foods:
        validation_results['violations'].append(f"Global: Minimum different foods violated ({len(globally_selected_foods)} < {global_min_different_foods})")
        validation_results['valid'] = False
    
    # 2. Global minimum total land usage
    if min_total_land_usage_percentage > 0:
        min_total_usage = min_total_land_usage_percentage * total_land
        if total_used_land < min_total_usage - 1e-6:
            validation_results['violations'].append(f"Global: Minimum total land usage violated ({total_used_land:.2f} < {min_total_usage:.2f})")
            validation_results['valid'] = False
    
    # 3. Food group global constraints (at least 10% of total land per group)
    for group, foods_in_group in food_groups.items():
        group_total_area = sum(sum(solution[farm][food] for farm in farms) for food in foods_in_group)
        min_group_area = total_land * 0.10
        if group_total_area < min_group_area - 1e-6:
            validation_results['violations'].append(f"Global-{group}: Minimum group area violated ({group_total_area:.2f} < {min_group_area:.2f})")
            validation_results['valid'] = False
        
        # Check significant foods per group (foods with > 1 hectare total)
        significant_foods_in_group = 0
        for food in foods_in_group:
            food_total_area = sum(solution[farm][food] for farm in farms)
            if food_total_area > 1.0:
                significant_foods_in_group += 1
        
        min_significant = min(2, len(foods_in_group))
        if significant_foods_in_group < min_significant:
            validation_results['violations'].append(f"Global-{group}: Minimum significant foods violated ({significant_foods_in_group} < {min_significant})")
            validation_results['valid'] = False
    
    return validation_results

def main():
    """Main function to run the PuLP solver."""
    print("Loading custom food optimization scenario...")
    
    # Load the custom scenario
    farms, foods, food_groups, config = load_food_data('custom')
    
    print(f"Loaded scenario with {len(farms)} farms, {len(foods)} foods, {len(food_groups)} food groups")
    print("Farms:", farms)
    print("Foods:", list(foods.keys()))
    print("Food groups:", {k: v for k, v in food_groups.items()})
    
    # Solve the problem
    result = solve_food_optimization_pulp(farms, foods, food_groups, config)
    
    # Validate the solution
    if result['solution'] is not None:
        print("\n" + "="*60)
        print("SOLUTION VALIDATION")
        print("="*60)
        validation = validate_solution(result['solution'], result['binary_solution'], 
                                     farms, foods, food_groups, config)
        
        if validation['valid']:
            print("✓ Solution is valid and satisfies all constraints")
        else:
            print("✗ Solution violates constraints:")
            for violation in validation['violations']:
                print(f"  - {violation}")
    
    return result

if __name__ == "__main__":
    result = main()