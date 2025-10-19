"""
Reader script for test_hybrid_solver results.
Loads the saved sampleset and compares with PuLP solution.
"""

import pickle
import os
import sys
import pulp as pl

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.scenarios import load_food_data

# --- Load the same scenario used in test ---
print("Loading 'simple' complexity food data...")
farms, foods, food_groups, config = load_food_data('simple')

params = config['parameters']
land_availability = params['land_availability']
weights = params['weights']
min_planting_area = params.get('minimum_planting_area', {})
food_group_constraints = params.get('food_group_constraints', {})

# --- Load Sampleset ---
file_path = os.path.join(project_root, 'Tests', 'test_simple_sampleset.pickle')
if not os.path.exists(file_path):
    print(f"Error: Sampleset file not found at {file_path}")
    print("Please run Tests/test_hybrid_solver.py first to generate the sampleset.")
    exit()

with open(file_path, 'rb') as f:
    sampleset = pickle.load(f)

# --- Analysis Code ---
feasible_sampleset = sampleset.filter(lambda d: d.is_feasible)

print(f"\n{len(feasible_sampleset)} feasible solutions of {len(sampleset)}.")

print("\nBest feasible solution:")
if len(feasible_sampleset) > 0:
    best_sample = feasible_sampleset.first
    
    # Calculate the efficiency from the solution
    total_area = sum(land_availability[farm] for farm in farms)
    
    numerator = 0
    for farm in farms:
        for food in foods:
            a_name = f"A_{farm}_{food}"
            if a_name in best_sample.sample:
                a_val = best_sample.sample[a_name]
                numerator += (
                    weights.get('nutritional_value', 0) * foods[food].get('nutritional_value', 0) * a_val +
                    weights.get('nutrient_density', 0) * foods[food].get('nutrient_density', 0) * a_val -
                    weights.get('environmental_impact', 0) * foods[food].get('environmental_impact', 0) * a_val +
                    weights.get('affordability', 0) * foods[food].get('affordability', 0) * a_val +
                    weights.get('sustainability', 0) * foods[food].get('sustainability', 0) * a_val
                )
    
    efficiency = numerator / total_area
    print(f"Optimal Efficiency: {efficiency:.6f}")
    print(f"CQM Energy (negative objective): {best_sample.energy:.6f}\n")

    for farm in farms:
        print(f"--- {farm} ---")
        for food in foods:
            y_name = f"Y_{farm}_{food}"
            a_name = f"A_{farm}_{food}"
            
            if y_name in best_sample.sample and a_name in best_sample.sample:
                y_val = best_sample.sample[y_name]
                a_val = best_sample.sample[a_name]
                
                if y_val > 0.5:
                    A_min = min_planting_area.get(food, 0)
                    required_min = A_min * y_val
                    status = "GOOD" if a_val >= required_min - 0.001 else "VIOLATED"
                    print(f"  {food}: A={a_val:.2f} ha")

        # Check food group constraints
        if food_group_constraints:
            print(f"\n  Food Group Constraints:")
            for g, constraints in food_group_constraints.items():
                foods_in_group = food_groups.get(g, [])
                if foods_in_group:
                    y_names_in_group = [f"Y_{farm}_{food}" for food in foods_in_group]
                    count = sum(1 for y_name in y_names_in_group if y_name in best_sample.sample and best_sample.sample[y_name] > 0.5)
                    
                    selected_foods = [food for food in foods_in_group if f"Y_{farm}_{food}" in best_sample.sample and best_sample.sample[f"Y_{farm}_{food}"] > 0.5]

                    min_req = constraints.get('min_foods', 0)
                    max_req = constraints.get('max_foods', float('inf'))
                    status = "GOOD" if min_req <= count <= max_req else "VIOLATED"
                    print(f"    {g}: {count} selected (range: {min_req}-{max_req}) {status}")
                    if selected_foods:
                        print(f"      Selected: {selected_foods}")

# --- DWave Timing Information ---
times = sampleset.info

print("\nDWave Timing Information:")
if isinstance(times, dict):
    for key, value in times.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value*1e-3:.2f} ms")
        else:
            print(f"  {key}: {value}")
else:
    print("No timing information available.")

# --- Solve with PuLP for Comparison ---
print("\n" + "="*80)
print("SOLVING WITH PULP FOR COMPARISON")
print("="*80)

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

model.solve(pl.PULP_CBC_CMD(msg=0))

pulp_objective = pl.value(model.objective)
pulp_status = pl.LpStatus[model.status]

print(f"PuLP Status: {pulp_status}")
print(f"PuLP Objective Value: {pulp_objective:.6f}")

# --- Create Comparison Table ---
print("\n" + "="*80)
print("FINAL COMPARISON: DWave vs PuLP")
print("="*80)

print("\nObjective Value:")
print(f"  DWave:  {efficiency:.6f}")
print(f"  PuLP:   {pulp_objective:.6f}")
if abs(efficiency - pulp_objective) < 0.0001:
    print(f"  Status: ✅ IDENTICAL")
else:
    print(f"  Status: ❌ DIFFERENT (diff: {abs(efficiency - pulp_objective):.6f})")

for f in farms:
    print(f"\n{f} ({land_availability[f]} ha):")
    print(f"  {'Food':<15} | {'DWave':<8} | {'PuLP':<8} | {'Match':<6}")
    print(f"  {'-'*15}-+-{'-'*8}-+-{'-'*8}-+-{'-'*6}")
    
    for c in foods:
        # Get DWave values
        a_name = f"A_{f}_{c}"
        dwave_val = 0.0
        if a_name in best_sample.sample:
            dwave_val = best_sample.sample[a_name]
        
        # Get PuLP values
        pulp_val = 0.0
        if A_pulp[(f, c)].value() is not None:
            pulp_val = A_pulp[(f, c)].value()
        
        # Check match
        match = "✅" if abs(dwave_val - pulp_val) < 0.01 else "❌"
        
        print(f"  {c:<15} | {dwave_val:<8.2f} | {pulp_val:<8.2f} | {match:<6}")

print("\n" + "="*80)
if abs(efficiency - pulp_objective) < 0.0001:
    print("Both solvers found the SAME solution!")
else:
    print("WARNING: Solutions differ!")
print("="*80)
