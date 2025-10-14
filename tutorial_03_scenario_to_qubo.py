"""
Tutorial 3: Scenario to QUBO Conversion

This tutorial demonstrates how to convert real-world scenario data (food production
optimization) into QUBO formulation for D-Wave solvers. You'll learn:

1. How to load and understand scenario data
2. How to formulate the optimization objective as QUBO
3. How to encode constraints as penalty terms
4. How to balance objective and constraint penalties
5. How to interpret solutions in the context of the original problem

This bridges the gap between abstract QUBO theory and practical applications.
"""

import sys
import os
import numpy as np
import dimod
from typing import Dict, List, Tuple

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from src.scenarios import load_food_data


def example_1_understand_scenario_data():
    """
    Example 1: Understanding the scenario data structure
    
    Load and explore the simple scenario data to understand what we're optimizing.
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Understanding Scenario Data")
    print("="*70)
    
    # Load simple scenario
    farms, foods, food_groups, config = load_food_data('simple')
    
    print("\nScenario Components:")
    print(f"  Farms: {farms}")
    print(f"  Foods: {list(foods.keys())}")
    print(f"  Food Groups: {list(food_groups.keys())}")
    
    print("\nFood Attributes (for each food):")
    for food, attributes in foods.items():
        print(f"\n  {food}:")
        for attr, value in attributes.items():
            print(f"    {attr}: {value}")
    
    print("\nOptimization Weights:")
    weights = config.get('weights', config.get('parameters', {}).get('weights', {}))
    for criterion, weight in weights.items():
        if weight > 0:
            print(f"  {criterion}: {weight}")
    
    print("\nLand Availability (per farm):")
    land_avail = config.get('land_availability', config.get('parameters', {}).get('land_availability', {}))
    for farm, land in land_avail.items():
        print(f"  {farm}: {land} units")
    
    print("\nDecision Variables:")
    print("  We need to decide how much of each food to grow on each farm")
    print(f"  Total variables: {len(farms)} farms x {len(foods)} foods = {len(farms) * len(foods)}")
    
    return farms, foods, food_groups, config


def example_2_simple_objective_only():
    """
    Example 2: Formulate QUBO with objective only (no constraints)
    
    Simplest case: Maximize weighted sum of food attributes.
    We'll use binary variables: grow food on farm (1) or not (0).
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Objective-Only QUBO Formulation")
    print("="*70)
    
    farms, foods, food_groups, config = load_food_data('simple')
    weights = config.get('weights', config.get('parameters', {}).get('weights', {}))
    
    print("\nProblem: Select which foods to grow on which farms")
    print("Objective: Maximize weighted sum of food attributes")
    print(f"Active weights: {[(k, v) for k, v in weights.items() if v > 0]}")
    
    # Create binary variables: farm_food (e.g., Farm1_Wheat)
    variables = []
    variable_map = {}
    idx = 0
    
    for farm in farms:
        for food in foods.keys():
            var_name = f"{farm}_{food}"
            variables.append(var_name)
            variable_map[var_name] = idx
            idx += 1
    
    print(f"\nTotal variables: {len(variables)}")
    print(f"Example variables: {variables[:3]}...")
    
    # Build QUBO - only objective terms
    Q = {}
    
    for farm in farms:
        for food, attributes in foods.items():
            var_name = f"{farm}_{food}"
            var_idx = variable_map[var_name]
            
            # Calculate weighted score for this farm-food combination
            score = 0
            for criterion, weight in weights.items():
                if weight > 0 and criterion in attributes:
                    score += weight * attributes[criterion]
            
            # We want to MAXIMIZE, so negate for minimization
            Q[(var_idx, var_idx)] = -score
    
    print(f"\nQUBO matrix size: {len(variables)} x {len(variables)}")
    print(f"Non-zero entries: {len(Q)}")
    
    # Show a few entries
    print("\nSample QUBO entries (objective coefficients):")
    for i, (var_name, var_idx) in enumerate(variable_map.items()):
        if i >= 3:
            break
        coeff = Q.get((var_idx, var_idx), 0)
        print(f"  {var_name}: {coeff:.3f}")
    
    # Convert to BQM and solve
    bqm = dimod.BinaryQuadraticModel.from_qubo(Q)
    
    print("\nSolving with Simulated Annealing...")
    sampler = dimod.SimulatedAnnealingSampler()
    sampleset = sampler.sample(bqm, num_reads=100, seed=42)
    
    # Interpret solution
    best_sample = sampleset.first.sample
    best_energy = sampleset.first.energy
    
    print(f"\nBest solution found:")
    print(f"  Energy: {best_energy:.3f}")
    print(f"  Objective value (negated): {-best_energy:.3f}")
    
    print("\n  Selected farm-food combinations:")
    for var_name, var_idx in variable_map.items():
        if best_sample[var_idx] == 1:
            print(f"    {var_name}")
    
    return bqm, sampleset, variable_map


def example_3_add_land_constraints():
    """
    Example 3: Add land availability constraints
    
    Each farm has limited land. We need to ensure that selected foods
    don't exceed the available land.
    
    Constraint: For each farm, sum of selected foods <= available land
    
    We'll use a simplified version where each food-farm combination uses 1 unit of land.
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Adding Land Availability Constraints")
    print("="*70)
    
    farms, foods, food_groups, config = load_food_data('simple')
    weights = config.get('weights', config.get('parameters', {}).get('weights', {}))
    land_availability = config.get('land_availability', config.get('parameters', {}).get('land_availability', {}))
    
    print("\nConstraints: Each farm has limited land capacity")
    for farm, capacity in land_availability.items():
        print(f"  {farm}: max {capacity} units")
    
    # For simplicity, assume each food takes 20 units of land
    land_per_food = 20
    
    print(f"\nAssumption: Each food uses {land_per_food} units of land")
    
    # Create variables
    variables = []
    variable_map = {}
    idx = 0
    
    for farm in farms:
        for food in foods.keys():
            var_name = f"{farm}_{food}"
            variables.append(var_name)
            variable_map[var_name] = idx
            idx += 1
    
    # Build QUBO with objective
    Q = {}
    
    # Objective terms
    for farm in farms:
        for food, attributes in foods.items():
            var_name = f"{farm}_{food}"
            var_idx = variable_map[var_name]
            
            score = 0
            for criterion, weight in weights.items():
                if weight > 0 and criterion in attributes:
                    score += weight * attributes[criterion]
            
            Q[(var_idx, var_idx)] = -score
    
    # Add land constraints as penalties
    # For each farm: sum(xi) <= capacity/land_per_food
    penalty_weight = 5.0  # Weight for constraint violations
    
    print(f"\nConstraint penalty weight: {penalty_weight}")
    
    for farm in farms:
        farm_capacity = land_availability[farm]
        max_foods = farm_capacity // land_per_food
        
        print(f"  {farm}: can grow max {max_foods} foods")
        
        # Get all variables for this farm
        farm_vars = []
        for food in foods.keys():
            var_name = f"{farm}_{food}"
            farm_vars.append(variable_map[var_name])
        
        # Add penalty: P * (sum(xi) - max_foods)^2 for sum(xi) > max_foods
        # We'll use a simpler soft constraint: P * sum(xi*xj) to discourage selecting too many
        for i, var_i in enumerate(farm_vars):
            for j in range(i+1, len(farm_vars)):
                var_j = farm_vars[j]
                # Penalize pairs (discourages selecting multiple foods on same farm)
                if (var_i, var_j) in Q:
                    Q[(var_i, var_j)] += penalty_weight * 0.5
                else:
                    Q[(var_i, var_j)] = penalty_weight * 0.5
    
    print(f"\nTotal QUBO entries: {len(Q)}")
    
    # Solve
    bqm = dimod.BinaryQuadraticModel.from_qubo(Q)
    
    print("\nSolving with Simulated Annealing...")
    sampler = dimod.SimulatedAnnealingSampler()
    sampleset = sampler.sample(bqm, num_reads=200, seed=42)
    
    # Interpret top solutions
    print("\nTop 3 solutions:")
    for sol_idx, sample_data in enumerate(list(sampleset.data(['sample', 'energy']))[:3]):
        sample = sample_data.sample
        energy = sample_data.energy
        
        print(f"\n  Solution {sol_idx + 1}:")
        print(f"    Energy: {energy:.3f}")
        
        # Check selections per farm
        for farm in farms:
            selected = []
            for food in foods.keys():
                var_name = f"{farm}_{food}"
                var_idx = variable_map[var_name]
                if sample[var_idx] == 1:
                    selected.append(food)
            
            land_used = len(selected) * land_per_food
            capacity = land_availability[farm]
            
            print(f"    {farm}: {selected} (land used: {land_used}/{capacity})")
    
    return bqm, sampleset, variable_map


def example_4_complete_formulation():
    """
    Example 4: Complete QUBO formulation with multiple constraint types
    
    Realistic formulation including:
    - Objective: Maximize weighted food attributes
    - Constraint 1: Land availability per farm
    - Constraint 2: Diversity (select from different food groups)
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Complete QUBO Formulation")
    print("="*70)
    
    farms, foods, food_groups, config = load_food_data('simple')
    weights = config.get('weights', config.get('parameters', {}).get('weights', {}))
    land_availability = config.get('land_availability', config.get('parameters', {}).get('land_availability', {}))
    
    print("\nComplete optimization problem:")
    print("  Objective: Maximize weighted food attributes")
    print("  Constraint 1: Respect land availability")
    print("  Constraint 2: Select diverse foods (from different groups)")
    
    # Variables
    variables = []
    variable_map = {}
    idx = 0
    
    for farm in farms:
        for food in foods.keys():
            var_name = f"{farm}_{food}"
            variables.append(var_name)
            variable_map[var_name] = idx
            idx += 1
    
    print(f"\nTotal variables: {len(variables)}")
    
    # Build QUBO
    Q = {}
    
    # 1. Objective terms
    print("\n1. Adding objective terms...")
    for farm in farms:
        for food, attributes in foods.items():
            var_name = f"{farm}_{food}"
            var_idx = variable_map[var_name]
            
            score = 0
            for criterion, weight in weights.items():
                if weight > 0 and criterion in attributes:
                    score += weight * attributes[criterion]
            
            Q[(var_idx, var_idx)] = -score * 2.0  # Scale up objective
    
    # 2. Land constraints
    print("2. Adding land constraints...")
    land_penalty = 3.0
    land_per_food = 20
    
    for farm in farms:
        farm_vars = [variable_map[f"{farm}_{food}"] for food in foods.keys()]
        
        # Discourage selecting too many foods on one farm
        for i, var_i in enumerate(farm_vars):
            for j in range(i+1, len(farm_vars)):
                var_j = farm_vars[j]
                Q[(var_i, var_j)] = Q.get((var_i, var_j), 0) + land_penalty * 0.3
    
    # 3. Diversity constraints
    print("3. Adding diversity constraints...")
    diversity_bonus = -1.0  # Bonus for selecting from different food groups
    
    # Encourage selecting foods from different groups
    for farm in farms:
        for group1, foods1 in food_groups.items():
            for group2, foods2 in food_groups.items():
                if group1 < group2:  # Avoid double counting
                    for food1 in foods1:
                        for food2 in foods2:
                            var1 = variable_map[f"{farm}_{food1}"]
                            var2 = variable_map[f"{farm}_{food2}"]
                            # Bonus for diversity (negative coefficient)
                            if var1 < var2:
                                Q[(var1, var2)] = Q.get((var1, var2), 0) + diversity_bonus
                            else:
                                Q[(var2, var1)] = Q.get((var2, var1), 0) + diversity_bonus
    
    print(f"\nQUBO matrix statistics:")
    print(f"  Total entries: {len(Q)}")
    print(f"  Linear terms: {sum(1 for (i,j) in Q.keys() if i==j)}")
    print(f"  Quadratic terms: {sum(1 for (i,j) in Q.keys() if i!=j)}")
    
    # Solve
    bqm = dimod.BinaryQuadraticModel.from_qubo(Q)
    
    print("\nSolving with Simulated Annealing (500 reads)...")
    sampler = dimod.SimulatedAnnealingSampler()
    sampleset = sampler.sample(bqm, num_reads=500, seed=42)
    
    # Analyze best solution
    print("\nBest solution:")
    best_sample = sampleset.first.sample
    best_energy = sampleset.first.energy
    
    print(f"  Energy: {best_energy:.3f}")
    
    # Detailed analysis
    for farm in farms:
        print(f"\n  {farm}:")
        selected_foods = []
        selected_groups = set()
        total_score = 0
        
        for food in foods.keys():
            var_name = f"{farm}_{food}"
            var_idx = variable_map[var_name]
            if best_sample[var_idx] == 1:
                selected_foods.append(food)
                
                # Find food group
                for group, group_foods in food_groups.items():
                    if food in group_foods:
                        selected_groups.add(group)
                
                # Calculate score
                for criterion, weight in weights.items():
                    if weight > 0 and criterion in foods[food]:
                        total_score += weight * foods[food][criterion]
        
        land_used = len(selected_foods) * land_per_food
        capacity = land_availability[farm]
        
        print(f"    Selected: {selected_foods}")
        print(f"    Food groups: {selected_groups}")
        print(f"    Land used: {land_used}/{capacity}")
        print(f"    Score contribution: {total_score:.3f}")
    
    return bqm, sampleset, variable_map


def example_5_tuning_penalties():
    """
    Example 5: The importance of tuning penalty weights
    
    Shows how different penalty weights affect the solution quality.
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: Tuning Penalty Weights")
    print("="*70)
    
    farms, foods, food_groups, config = load_food_data('simple')
    weights = config.get('weights', config.get('parameters', {}).get('weights', {}))
    
    # Simplified problem: Select 2 foods total
    print("\nSimplified Problem: Select exactly 2 foods across all farms")
    
    # Variables for each food (ignoring farms for simplicity)
    food_list = list(foods.keys())
    variable_map = {food: idx for idx, food in enumerate(food_list)}
    
    # Test different penalty weights
    penalty_weights = [1.0, 5.0, 20.0, 100.0]
    
    for penalty in penalty_weights:
        print(f"\n--- Testing penalty weight: {penalty} ---")
        
        Q = {}
        
        # Objective: maximize weighted scores
        for food, attributes in foods.items():
            var_idx = variable_map[food]
            score = sum(weights.get(c, 0) * attributes.get(c, 0) 
                       for c in attributes.keys())
            Q[(var_idx, var_idx)] = -score
        
        # Constraint: exactly 2 foods (sum(xi) = 2)
        # Penalty: P * (sum(xi) - 2)^2
        # Expanded: P * (-3*sum(xi) + 2*sum(xi*xj) + 4)
        for i in range(len(food_list)):
            Q[(i, i)] += penalty * (-3)
        
        for i in range(len(food_list)):
            for j in range(i+1, len(food_list)):
                Q[(i, j)] = penalty * 2
        
        offset = penalty * 4
        
        # Solve
        bqm = dimod.BinaryQuadraticModel.from_qubo(Q, offset=offset)
        sampler = dimod.SimulatedAnnealingSampler()
        sampleset = sampler.sample(bqm, num_reads=100, seed=42)
        
        best_sample = sampleset.first.sample
        selected = [food_list[i] for i in range(len(food_list)) if best_sample[i] == 1]
        num_selected = len(selected)
        
        print(f"  Selected foods: {selected}")
        print(f"  Number selected: {num_selected}")
        print(f"  Constraint satisfied: {num_selected == 2}")
        print(f"  Energy: {sampleset.first.energy:.3f}")
    
    print("\nObservation:")
    print("  - Low penalty: Solver may violate constraints for better objective")
    print("  - High penalty: Solver prioritizes constraint satisfaction")
    print("  - Sweet spot: Balance between objective and constraints")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("TUTORIAL 3: SCENARIO TO QUBO CONVERSION")
    print("="*70)
    print("\nThis tutorial demonstrates how to convert real-world scenario data")
    print("into QUBO formulation for quantum annealing.")
    
    # Run all examples
    example_1_understand_scenario_data()
    example_2_simple_objective_only()
    example_3_add_land_constraints()
    example_4_complete_formulation()
    example_5_tuning_penalties()
    
    print("\n" + "="*70)
    print("TUTORIAL 3 COMPLETE")
    print("="*70)
    print("\nKey Takeaways:")
    print("1. Start by understanding your scenario data structure")
    print("2. Define clear decision variables (binary for QUBO)")
    print("3. Formulate objective as linear/quadratic terms")
    print("4. Encode constraints as penalty terms")
    print("5. Tune penalty weights to balance objective and constraints")
    print("6. Always interpret solutions in the original problem context")
    print("\nNext: tutorial_04_dwave_integration.py")


if __name__ == "__main__":
    main()
