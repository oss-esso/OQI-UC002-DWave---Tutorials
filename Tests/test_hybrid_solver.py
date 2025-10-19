"""
Test script for solving the food optimization problem with a CQM hybrid solver.

This script demonstrates how to:
1. Load a complex problem scenario.
2. Formulate the problem as a Constrained Quadratic Model (CQM).
3. Use the LeapHybridCQMSampler to solve the CQM.
4. Print and interpret the results.
5. Compare with PuLP solution.
"""

import os
import sys
import logging
import time
import pickle

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from src.scenarios import load_food_data
    from dimod import ConstrainedQuadraticModel, Binary, Real
    from dwave.system import LeapHybridCQMSampler
    import pulp as pl
except ImportError as e:
    logger.error(f"Import error: {e}. Please ensure all required libraries are installed.")
    sys.exit(1)

def create_cqm_for_food_problem(farms, foods, food_groups, config):
    """
    Creates a CQM for the food optimization problem.
    Uses the same formulation as dwave-test.py
    """
    cqm = ConstrainedQuadraticModel()
    
    # Extract parameters from config
    params = config['parameters']
    land_availability = params['land_availability']
    weights = params['weights']
    
    # Get minimum planting area (if exists)
    min_planting_area = params.get('minimum_planting_area', {})
    
    # Get food group constraints (if exists)
    food_group_constraints = params.get('food_group_constraints', {})
    
    # --- Define variables ---
    A = {}  # Continuous area variables
    Y = {}  # Binary selection variables
    
    for farm in farms:
        for food in foods:
            # Area variable: lower_bound=0, upper_bound=land available
            A[(farm, food)] = Real(f"A_{farm}_{food}", lower_bound=0, upper_bound=land_availability[farm])
            # Binary variable: indicates if food is selected on farm
            Y[(farm, food)] = Binary(f"Y_{farm}_{food}")
    
    # --- Objective Function ---
    # Direct objective formulation: maximize the weighted sum
    # Since we're dividing by a constant (total_area), we can maximize the numerator directly
    total_area = sum(land_availability[farm] for farm in farms)
    
    objective = sum(
        weights.get('nutritional_value', 0) * foods[food].get('nutritional_value', 0) * A[(farm, food)] +
        weights.get('nutrient_density', 0) * foods[food].get('nutrient_density', 0) * A[(farm, food)] -
        weights.get('environmental_impact', 0) * foods[food].get('environmental_impact', 0) * A[(farm, food)] +
        weights.get('affordability', 0) * foods[food].get('affordability', 0) * A[(farm, food)] +
        weights.get('sustainability', 0) * foods[food].get('sustainability', 0) * A[(farm, food)]
        for farm in farms for food in foods
    )
    
    # CQM minimizes by default, so negate to maximize
    cqm.set_objective(-objective)
    
    # --- Constraints ---
    
    # 1. Land availability: total area used on a farm cannot exceed available land
    for farm in farms:
        cqm.add_constraint(
            sum(A[(farm, food)] for food in foods) - land_availability[farm] <= 0,
            label=f"Land_Availability_{farm}"
        )
    
    # 2. Linking A and Y variables
    for farm in farms:
        for food in foods:
            # Get minimum area for this food (default to 0 if not specified)
            A_min = min_planting_area.get(food, 0)
            
            # If Y=1, area must be at least A_min
            cqm.add_constraint(
                A[(farm, food)] - A_min * Y[(farm, food)] >= 0,
                label=f"Min_Area_If_Selected_{farm}_{food}"
            )
            # If Y=0, area must be 0
            cqm.add_constraint(
                A[(farm, food)] - land_availability[farm] * Y[(farm, food)] <= 0,
                label=f"Max_Area_If_Selected_{farm}_{food}"
            )
    
    # 3. Food group constraints (if specified)
    if food_group_constraints:
        for group, constraints in food_group_constraints.items():
            foods_in_group = food_groups.get(group, [])
            if foods_in_group:
                for farm in farms:
                    # Minimum foods from group per farm
                    if 'min_foods' in constraints:
                        cqm.add_constraint(
                            sum(Y[(farm, food)] for food in foods_in_group) - constraints['min_foods'] >= 0,
                            label=f"Food_Group_Min_{group}_{farm}"
                        )
                    # Maximum foods from group per farm
                    if 'max_foods' in constraints:
                        cqm.add_constraint(
                            sum(Y[(farm, food)] for food in foods_in_group) - constraints['max_foods'] <= 0,
                            label=f"Food_Group_Max_{group}_{farm}"
                        )
    
    logger.info(f"CQM created with {len(cqm.variables)} variables and {len(cqm.constraints)} constraints.")
    return cqm, A, Y

def solve_with_pulp(farms, foods, food_groups, config):
    """
    Solve the same problem with PuLP for comparison.
    """
    params = config['parameters']
    land_availability = params['land_availability']
    weights = params['weights']
    min_planting_area = params.get('minimum_planting_area', {})
    food_group_constraints = params.get('food_group_constraints', {})
    
    # Create PuLP variables
    A_pulp = pl.LpVariable.dicts("Area", [(f, c) for f in farms for c in foods], lowBound=0)
    Y_pulp = pl.LpVariable.dicts("Choose", [(f, c) for f in farms for c in foods], cat='Binary')
    
    total_area = sum(land_availability[f] for f in farms)
    
    # Objective function
    goal = (
        weights.get('nutritional_value', 0) * pl.lpSum([(foods[c].get('nutritional_value', 0) * A_pulp[(f, c)]) for f in farms for c in foods]) / total_area +
        weights.get('nutrient_density', 0) * pl.lpSum([(foods[c].get('nutrient_density', 0) * A_pulp[(f, c)]) for f in farms for c in foods]) / total_area -
        weights.get('environmental_impact', 0) * pl.lpSum([(foods[c].get('environmental_impact', 0) * A_pulp[(f, c)]) for f in farms for c in foods]) / total_area +
        weights.get('affordability', 0) * pl.lpSum([(foods[c].get('affordability', 0) * A_pulp[(f, c)]) for f in farms for c in foods]) / total_area +
        weights.get('sustainability', 0) * pl.lpSum([(foods[c].get('sustainability', 0) * A_pulp[(f, c)]) for f in farms for c in foods]) / total_area
    )
    
    model = pl.LpProblem("Food_Optimization", pl.LpMaximize)
    
    # Land availability constraints
    for f in farms:
        model += pl.lpSum([A_pulp[(f, c)] for c in foods]) <= land_availability[f], f"Max_Area_{f}"
    
    # Linking constraints
    for f in farms:
        for c in foods:
            A_min = min_planting_area.get(c, 0)
            model += A_pulp[(f, c)] >= A_min * Y_pulp[(f, c)], f"MinArea_{f}_{c}"
            model += A_pulp[(f, c)] <= land_availability[f] * Y_pulp[(f, c)], f"MaxArea_{f}_{c}"
    
    # Food group constraints
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
    
    # Solve
    model.solve(pl.PULP_CBC_CMD(msg=0))
    
    return model, A_pulp, Y_pulp, pl.value(model.objective)

def main():
    """
    Main function to run the CQM hybrid solver test.
    """
    print("=" * 80)
    print("CQM HYBRID SOLVER TEST SCRIPT")
    print("=" * 80)

    # --- 1. Load Problem Data ---
    print("\nLoading 'simple' complexity food data...")
    farms, foods, food_groups, config = load_food_data('simple')
    
    print(f"Loaded {len(farms)} farms: {farms}")
    print(f"Loaded {len(foods)} foods: {list(foods.keys())}")
    print(f"Food groups: {list(food_groups.keys())}")

    # --- 2. Formulate the CQM ---
    print("\nFormulating the problem as a Constrained Quadratic Model (CQM)...")
    cqm, A, Y = create_cqm_for_food_problem(farms, foods, food_groups, config)

    # --- 3. Configure and Run the Solver ---
    print("\nConfiguring the D-Wave CQM hybrid solver...")
    
    # Get token from environment or use the one from dwave-test.py
    token = os.getenv('DWAVE_API_TOKEN', '45FS-23cfb48dca2296ed24550846d2e7356eb6c19551')

    sampler = LeapHybridCQMSampler(token=token)

    print("Submitting the CQM to the D-Wave Leap hybrid solver...")
    start_time = time.time()
    sampleset = sampler.sample_cqm(cqm, label="Food Optimization Test - Simple Scenario")
    solve_time = time.time() - start_time

    # Save the sampleset to a file
    output_dir = os.path.join(project_root, 'Tests')
    os.makedirs(output_dir, exist_ok=True)
    pickle_path = os.path.join(output_dir, '../DWave_Results/test_simple_sampleset.pickle')
    
    with open(pickle_path, 'wb') as f:
        pickle.dump(sampleset, f)
    
    print(f"Sampleset saved to {pickle_path}")

    # --- 4. Process and Display Results ---
    print("\n" + "=" * 80)
    print("DWAVE RESULTS")
    print("=" * 80)
    print(f"Solved in {solve_time:.2f} seconds.")
    
    feasible_sampleset = sampleset.filter(lambda d: d.is_feasible)
    print(f"{len(feasible_sampleset)} feasible solutions of {len(sampleset)}.")
    
    if not feasible_sampleset:
        print("\nNo feasible solution found.")
        return

    best_solution = feasible_sampleset.first
    
    # Calculate efficiency
    params = config['parameters']
    land_availability = params['land_availability']
    weights = params['weights']
    total_area = sum(land_availability[farm] for farm in farms)
    
    numerator = 0
    for farm in farms:
        for food in foods:
            a_name = f"A_{farm}_{food}"
            if a_name in best_solution.sample:
                a_val = best_solution.sample[a_name]
                numerator += (
                    weights.get('nutritional_value', 0) * foods[food].get('nutritional_value', 0) * a_val +
                    weights.get('nutrient_density', 0) * foods[food].get('nutrient_density', 0) * a_val -
                    weights.get('environmental_impact', 0) * foods[food].get('environmental_impact', 0) * a_val +
                    weights.get('affordability', 0) * foods[food].get('affordability', 0) * a_val +
                    weights.get('sustainability', 0) * foods[food].get('sustainability', 0) * a_val
                )
    
    dwave_efficiency = numerator / total_area
    print(f"\nDWave Optimal Efficiency: {dwave_efficiency:.6f}")
    print(f"CQM Energy (negative objective): {best_solution.energy:.6f}")
    
    print("\nDWave Solution:")
    for farm in farms:
        print(f"\n{farm}:")
        for food in foods:
            y_name = f"Y_{farm}_{food}"
            a_name = f"A_{farm}_{food}"
            
            if y_name in best_solution.sample and a_name in best_solution.sample:
                y_val = best_solution.sample[y_name]
                a_val = best_solution.sample[a_name]
                
                if y_val > 0.5:
                    print(f"  {food}: {a_val:.2f} ha")

    # --- 5. Solve with PuLP for Comparison ---
    print("\n" + "=" * 80)
    print("SOLVING WITH PULP FOR COMPARISON")
    print("=" * 80)
    
    pulp_model, A_pulp, Y_pulp, pulp_objective = solve_with_pulp(farms, foods, food_groups, config)
    
    print(f"PuLP Status: {pl.LpStatus[pulp_model.status]}")
    print(f"PuLP Objective Value: {pulp_objective:.6f}")

    # --- 6. Create Comparison Table ---
    print("\n" + "=" * 80)
    print("FINAL COMPARISON: DWave vs PuLP")
    print("=" * 80)

    print("\nObjective Value:")
    print(f"  DWave:  {dwave_efficiency:.6f}")
    print(f"  PuLP:   {pulp_objective:.6f}")
    if abs(dwave_efficiency - pulp_objective) < 0.0001:
        print(f"  Status: ✅ IDENTICAL")
    else:
        print(f"  Status: ❌ DIFFERENT (diff: {abs(dwave_efficiency - pulp_objective):.6f})")

    for farm in farms:
        print(f"\n{farm} ({land_availability[farm]} ha):")
        print(f"  {'Food':<15} | {'DWave':<8} | {'PuLP':<8} | {'Match':<6}")
        print(f"  {'-'*15}-+-{'-'*8}-+-{'-'*8}-+-{'-'*6}")
        
        for food in foods:
            # Get DWave values
            a_name = f"A_{farm}_{food}"
            dwave_val = 0.0
            if a_name in best_solution.sample:
                dwave_val = best_solution.sample[a_name]
            
            # Get PuLP values
            pulp_val = 0.0
            if A_pulp[(farm, food)].value() is not None:
                pulp_val = A_pulp[(farm, food)].value()
            
            # Check match
            match = "✅" if abs(dwave_val - pulp_val) < 0.01 else "❌"
            
            print(f"  {food:<15} | {dwave_val:<8.2f} | {pulp_val:<8.2f} | {match:<6}")

    print("\n" + "=" * 80)
    if abs(dwave_efficiency - pulp_objective) < 0.0001:
        print("Both solvers found the SAME solution!")
    else:
        print("WARNING: Solutions differ!")
    print("=" * 80)

if __name__ == "__main__":
    main()