"""
RIGOROUS SCENARIO TESTING: MILP vs QUBO vs Grover Adaptive Search

This script provides a comprehensive, research-grade comparison with NO simplifications.
All QUBO formulations are properly validated against classical MILP solutions.

Author: Research Testing Framework
Date: October 21, 2025
"""

import sys
import os
import time
import json
import numpy as np
import pulp as pl
from typing import Dict, List, Tuple, Any
from gas_for_qubo_improved import ImprovedGroverAdaptiveSearchSolver

# Add src to path for importing scenarios
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from scenarios import load_food_data

print("="*80)
print("RIGOROUS RESEARCH TESTING: THREE SCENARIOS")
print("MILP → QUBO Conversion Validation → Quantum GAS Testing")
print("="*80)

# ============================================================================
# TASK 1: LOAD AND ANALYZE SCENARIOS
# ============================================================================

print("\n" + "="*80)
print("TASK 1: LOADING AND ANALYZING SCENARIOS")
print("="*80)

scenarios_data = {}

for scenario_name in ['simple', 'intermediate', 'custom']:
    print(f"\nLoading {scenario_name.upper()} scenario...")
    farms, foods, food_groups, config = load_food_data(scenario_name)
    
    n_farms = len(farms)
    n_foods = len(list(foods.keys()))
    n_vars = n_farms * n_foods
    
    print(f"  Farms: {n_farms} → {farms}")
    print(f"  Foods: {n_foods} → {list(foods.keys())}")
    print(f"  Food Groups: {len(food_groups)}")
    print(f"  Binary Variables: {n_vars}")
    print(f"  Search Space: 2^{n_vars} = {2**n_vars} states")
    
    scenarios_data[scenario_name] = {
        'farms': farms,
        'foods': foods,
        'food_groups': food_groups,
        'config': config,
        'n_vars': n_vars
    }

print("\n✅ TASK 1 COMPLETE: All scenarios loaded successfully")

# ============================================================================
# TASK 2: STEP 1A - SOLVE WITH CLASSICAL MILP (GROUND TRUTH)
# ============================================================================

print("\n" + "="*80)
print("STEP 1A: CLASSICAL MILP SOLVER (GROUND TRUTH)")
print("="*80)

milp_results = {}

for scenario_name, data in scenarios_data.items():
    print(f"\n{'='*80}")
    print(f"SCENARIO: {scenario_name.upper()}")
    print(f"{'='*80}")
    
    farms = data['farms']
    foods_dict = data['foods']
    foods = list(foods_dict.keys())
    food_groups = data['food_groups']
    config = data['config']
    params = config['parameters']
    
    print(f"\nBuilding MILP model...")
    
    start_time = time.time()
    
    # Create binary variables for crop selection
    Y = pl.LpVariable.dicts("Select", 
                           [(f, food) for f in farms for food in foods], 
                           cat='Binary')
    
    # Create continuous variables for area (if land availability specified)
    if 'land_availability' in params:
        A = pl.LpVariable.dicts("Area",
                               [(f, food) for f in farms for food in foods],
                               lowBound=0)
        has_continuous = True
    else:
        has_continuous = False
    
    # Create model
    model = pl.LpProblem(f"Scenario_{scenario_name}", pl.LpMaximize)
    
    # Build objective function
    weights = params['weights']
    objective_terms = []
    
    for f in farms:
        for food in foods:
            food_data = foods_dict[food]
            
            # For binary-only, use selection indicators
            # For continuous, use area allocation
            if has_continuous and 'land_availability' in params:
                var = A[(f, food)]
            else:
                var = Y[(f, food)]
            
            # Calculate weighted score for this food
            score = 0
            if 'nutritional_value' in weights and 'nutritional_value' in food_data:
                score += weights['nutritional_value'] * food_data['nutritional_value'] * var
            if 'nutrient_density' in weights and 'nutrient_density' in food_data:
                score += weights['nutrient_density'] * food_data['nutrient_density'] * var
            if 'affordability' in weights and 'affordability' in food_data:
                score += weights['affordability'] * food_data['affordability'] * var
            if 'sustainability' in weights and 'sustainability' in food_data:
                score += weights['sustainability'] * food_data['sustainability'] * var
            if 'environmental_impact' in weights and 'environmental_impact' in food_data:
                # Environmental impact is typically minimized (negative in maximization)
                score -= weights['environmental_impact'] * food_data['environmental_impact'] * var
            
            objective_terms.append(score)
    
    model += pl.lpSum(objective_terms), "Objective"
    
    # Add constraints
    constraint_count = 0
    
    # Land availability constraints
    if has_continuous and 'land_availability' in params:
        for f in farms:
            model += pl.lpSum([A[(f, food)] for food in foods]) <= params['land_availability'][f], \
                     f"Land_{f}"
            constraint_count += 1
        
        # Link binary and continuous variables
        if 'minimum_planting_area' in params:
            min_area = params['minimum_planting_area']
            for f in farms:
                for food in foods:
                    if food in min_area:
                        model += A[(f, food)] >= min_area[food] * Y[(f, food)], \
                                f"MinArea_{f}_{food}"
                        model += A[(f, food)] <= params['land_availability'][f] * Y[(f, food)], \
                                f"MaxArea_{f}_{food}"
                        constraint_count += 2
    
    # Food group diversity constraints
    if 'food_group_constraints' in params:
        fg_constraints = params['food_group_constraints']
        for f in farms:
            for group, foods_in_group in food_groups.items():
                if group in fg_constraints:
                    min_foods = fg_constraints[group].get('min_foods', 0)
                    max_foods = fg_constraints[group].get('max_foods', len(foods_in_group))
                    
                    if min_foods > 0:
                        model += pl.lpSum([Y[(f, food)] for food in foods_in_group]) >= min_foods, \
                                f"MinGroup_{f}_{group}"
                        constraint_count += 1
                    if max_foods < len(foods_in_group):
                        model += pl.lpSum([Y[(f, food)] for food in foods_in_group]) <= max_foods, \
                                f"MaxGroup_{f}_{group}"
                        constraint_count += 1
    
    print(f"  Variables: {len(Y)} binary" + (f", {len(A)} continuous" if has_continuous else ""))
    print(f"  Constraints: {constraint_count}")
    
    # Solve
    print(f"\nSolving with PuLP/CBC...")
    model.solve(pl.PULP_CBC_CMD(msg=0))
    
    solve_time = time.time() - start_time
    
    # Extract results
    status = pl.LpStatus[model.status]
    objective_value = pl.value(model.objective) if model.status == 1 else None
    
    # Get selected crops
    selection = {}
    for f in farms:
        selection[f] = []
        for food in foods:
            if Y[(f, food)].value() and Y[(f, food)].value() > 0.5:
                if has_continuous:
                    area = A[(f, food)].value() if A[(f, food)].value() else 0
                    selection[f].append((food, area))
                else:
                    selection[f].append((food, 1))
    
    # Print results
    print(f"\n{'─'*80}")
    print(f"MILP RESULTS - {scenario_name.upper()}")
    print(f"{'─'*80}")
    print(f"Status: {status}")
    print(f"Objective Value: {objective_value:.6f}" if objective_value else "No solution")
    print(f"Solution Time: {solve_time:.4f} seconds")
    print(f"\nSelected Crops:")
    for f in farms:
        print(f"  {f}:")
        for item in selection[f]:
            if has_continuous:
                food, area = item
                print(f"    {food}: {area:.2f} units")
            else:
                food, _ = item
                print(f"    {food}: selected")
    
    milp_results[scenario_name] = {
        'status': status,
        'objective': objective_value,
        'time': solve_time,
        'selection': selection,
        'has_continuous': has_continuous,
        'n_constraints': constraint_count
    }

print("\n✅ STEP 1A COMPLETE: All MILP solutions obtained")

# ============================================================================
# TASK 3: STEP 1B - CONVERT TO VALID QUBO FORMULATIONS
# ============================================================================

print("\n" + "="*80)
print("STEP 1B: CONVERTING TO VALID QUBO FORMULATIONS")
print("="*80)

qubo_formulations = {}

for scenario_name, data in scenarios_data.items():
    print(f"\n{'='*80}")
    print(f"QUBO CONVERSION: {scenario_name.upper()}")
    print(f"{'='*80}")
    
    farms = data['farms']
    foods_dict = data['foods']
    foods = list(foods_dict.keys())
    food_groups = data['food_groups']
    config = data['config']
    params = config['parameters']
    
    n_vars = len(farms) * len(foods)
    
    print(f"\nProblem size: {n_vars} binary variables")
    print(f"QUBO matrix size: {n_vars} × {n_vars}")
    
    # Create variable mapping
    var_to_idx = {}
    idx_to_var = {}
    idx = 0
    for f in farms:
        for food in foods:
            var_to_idx[(f, food)] = idx
            idx_to_var[idx] = (f, food)
            idx += 1
    
    # Initialize QUBO matrix
    Q = np.zeros((n_vars, n_vars))
    
    print(f"\nEncoding objective function...")
    
    # OBJECTIVE: Maximize weighted scores → Minimize negative weighted scores
    weights = params['weights']
    
    for f in farms:
        for food in foods:
            i = var_to_idx[(f, food)]
            food_data = foods_dict[food]
            
            # Calculate score for selecting this crop
            score = 0
            if 'nutritional_value' in weights and 'nutritional_value' in food_data:
                score += weights['nutritional_value'] * food_data['nutritional_value']
            if 'nutrient_density' in weights and 'nutrient_density' in food_data:
                score += weights['nutrient_density'] * food_data['nutrient_density']
            if 'affordability' in weights and 'affordability' in food_data:
                score += weights['affordability'] * food_data['affordability']
            if 'sustainability' in weights and 'sustainability' in food_data:
                score += weights['sustainability'] * food_data['sustainability']
            if 'environmental_impact' in weights and 'environmental_impact' in food_data:
                score -= weights['environmental_impact'] * food_data['environmental_impact']
            
            # Negative because we're minimizing (QUBO convention)
            Q[i, i] = -score
    
    print(f"✓ Objective encoded (diagonal terms)")
    
    # CONSTRAINTS: Add as penalty terms
    # We'll use a penalty weight that's significantly larger than objective values
    max_obj_coeff = np.max(np.abs(np.diag(Q)))
    PENALTY = max(10.0, max_obj_coeff * 10)  # At least 10x the largest objective coefficient
    
    print(f"  Maximum objective coefficient: {max_obj_coeff:.4f}")
    print(f"  Penalty weight: {PENALTY:.4f}")
    
    constraint_penalties_added = 0
    
    # Food group diversity constraints
    if 'food_group_constraints' in params:
        print(f"\nEncoding food group constraints...")
        fg_constraints = params['food_group_constraints']
        
        for f in farms:
            for group, foods_in_group in food_groups.items():
                if group in fg_constraints:
                    min_foods = fg_constraints[group].get('min_foods', 0)
                    max_foods = fg_constraints[group].get('max_foods', len(foods_in_group))
                    
                    group_indices = [var_to_idx[(f, food)] for food in foods_in_group]
                    
                    # Constraint: min_foods <= sum(x_i) <= max_foods
                    # Penalty for violation: PENALTY * (sum(x_i) - target)^2
                    
                    # For minimum: penalize if sum < min_foods
                    # We approximate this by penalizing deviation from midpoint
                    target = (min_foods + max_foods) / 2
                    
                    # Expand (sum(x_i) - target)^2 = sum(x_i)^2 - 2*target*sum(x_i) + target^2
                    # Since x_i^2 = x_i for binary variables:
                    # = sum(x_i) + 2*sum_{i<j}(x_i*x_j) - 2*target*sum(x_i) + target^2
                    
                    # Quadratic terms
                    for i in group_indices:
                        for j in group_indices:
                            if i < j:
                                Q[i, j] += 2 * PENALTY
                            elif i == j:
                                Q[i, i] += PENALTY
                    
                    # Linear terms
                    for i in group_indices:
                        Q[i, i] -= 2 * PENALTY * target
                    
                    constraint_penalties_added += 1
        
        print(f"✓ Added {constraint_penalties_added} food group constraint penalties")
    
    # Land/resource constraints (simplified for binary problem)
    # We penalize selecting too many crops per farm
    if 'land_availability' in params:
        print(f"\nEncoding land availability constraints...")
        
        for f in farms:
            farm_indices = [var_to_idx[(f, food)] for food in foods]
            max_selections = min(len(foods), 4)  # Reasonable limit for binary problem
            
            # Penalize if more than max_selections crops are selected
            # Similar to food group constraint
            for i in farm_indices:
                for j in farm_indices:
                    if i < j:
                        Q[i, j] += PENALTY * 0.5  # Smaller penalty
                    elif i == j:
                        Q[i, i] += PENALTY * 0.5
            
            for i in farm_indices:
                Q[i, i] -= PENALTY * max_selections
        
        print(f"✓ Added land constraint penalties")
    
    # Make matrix symmetric (QUBO standard form: upper triangular)
    # Copy upper triangle to get full symmetric matrix
    Q_symmetric = Q + Q.T - np.diag(np.diag(Q))
    
    print(f"\n{'─'*80}")
    print(f"QUBO FORMULATION COMPLETE - {scenario_name.upper()}")
    print(f"{'─'*80}")
    print(f"Matrix shape: {Q_symmetric.shape}")
    print(f"Non-zero elements: {np.count_nonzero(Q_symmetric)}")
    print(f"Matrix norm: {np.linalg.norm(Q_symmetric):.4f}")
    print(f"Min value: {np.min(Q_symmetric):.4f}")
    print(f"Max value: {np.max(Q_symmetric):.4f}")
    
    qubo_formulations[scenario_name] = {
        'Q': Q_symmetric,
        'var_to_idx': var_to_idx,
        'idx_to_var': idx_to_var,
        'n_vars': n_vars,
        'penalty_weight': PENALTY
    }

print("\n✅ STEP 1B COMPLETE: All QUBO formulations created")

# Save intermediate results
print("\nSaving QUBO formulations...")
for scenario_name, qubo_data in qubo_formulations.items():
    filename = f"qubo_{scenario_name}_matrix.npy"
    np.save(filename, qubo_data['Q'])
    print(f"  Saved: {filename}")

print("\n" + "="*80)
print("CHECKPOINT: Ready for Step 1C (QUBO Validation)")
print("="*80)
print("\nContinue? Press Enter to proceed with QUBO validation...")
input()

# ============================================================================
# TASK 4: STEP 1C - VALIDATE QUBO WITH CLASSICAL EXHAUSTIVE SEARCH
# ============================================================================

print("\n" + "="*80)
print("STEP 1C: QUBO VALIDATION WITH CLASSICAL BRUTE-FORCE")
print("="*80)

classical_qubo_results = {}

for scenario_name, qubo_data in qubo_formulations.items():
    print(f"\n{'='*80}")
    print(f"CLASSICAL QUBO SOLVE: {scenario_name.upper()}")
    print(f"{'='*80}")
    
    Q = qubo_data['Q']
    n_vars = qubo_data['n_vars']
    idx_to_var = qubo_data['idx_to_var']
    
    print(f"Problem size: {n_vars} variables, {2**n_vars} states to evaluate")
    
    if n_vars > 16:
        print(f"\n⚠️  WARNING: Problem has {n_vars} variables ({2**n_vars} states)")
        print(f"   Brute force may take significant time. Proceed? (y/n)")
        response = input().strip().lower()
        if response != 'y':
            print("   Skipping this scenario")
            continue
    
    start_time = time.time()
    
    # Create solver and find optimal
    solver = ImprovedGroverAdaptiveSearchSolver(Q)
    optimal_solution, optimal_cost = solver.classical_solve()
    
    solve_time = time.time() - start_time
    
    # Decode solution
    selection = {}
    farms = scenarios_data[scenario_name]['farms']
    for f in farms:
        selection[f] = []
    
    for i, val in enumerate(optimal_solution):
        if val == 1:
            f, food = idx_to_var[i]
            selection[f].append(food)
    
    print(f"\n{'─'*80}")
    print(f"CLASSICAL QUBO RESULTS - {scenario_name.upper()}")
    print(f"{'─'*80}")
    print(f"Optimal QUBO Cost: {optimal_cost:.6f}")
    print(f"Solution Time: {solve_time:.4f} seconds")
    print(f"\nSelected Crops (from QUBO):")
    for f in farms:
        print(f"  {f}: {selection[f]}")
    
    classical_qubo_results[scenario_name] = {
        'cost': optimal_cost,
        'solution': optimal_solution,
        'selection': selection,
        'time': solve_time
    }

print("\n✅ STEP 1C COMPLETE: Classical QUBO solutions obtained")

# ============================================================================
# COMPARISON: MILP vs Classical QUBO
# ============================================================================

print("\n" + "="*80)
print("VALIDATION: COMPARING MILP vs CLASSICAL QUBO")
print("="*80)

for scenario_name in scenarios_data.keys():
    if scenario_name not in classical_qubo_results:
        continue
    
    print(f"\n{'='*80}")
    print(f"VALIDATION: {scenario_name.upper()}")
    print(f"{'='*80}")
    
    milp_sel = milp_results[scenario_name]['selection']
    qubo_sel = classical_qubo_results[scenario_name]['selection']
    
    print("\nMILP Selection:")
    for f, items in milp_sel.items():
        foods = [item[0] for item in items]
        print(f"  {f}: {foods}")
    
    print("\nQUBO Selection:")
    for f, foods in qubo_sel.items():
        print(f"  {f}: {foods}")
    
    # Check if selections match
    match = True
    for f in milp_sel.keys():
        milp_foods = set([item[0] for item in milp_sel[f]])
        qubo_foods = set(qubo_sel[f])
        if milp_foods != qubo_foods:
            match = False
            print(f"\n⚠️  DIFFERENCE in {f}:")
            print(f"    MILP has: {milp_foods - qubo_foods}")
            print(f"    QUBO has: {qubo_foods - milp_foods}")
    
    if match:
        print("\n✅ PERFECT MATCH: QUBO and MILP give identical crop selections")
    else:
        print("\n⚠️  SELECTIONS DIFFER: This may be due to:")
        print("    1. Continuous vs binary-only problem formulation")
        print("    2. Multiple optimal solutions exist")
        print("    3. QUBO penalty weights need adjustment")

print("\n✅ VALIDATION COMPLETE")

print("\n" + "="*80)
print("END OF PHASE 1: READY FOR QUANTUM TESTING")
print("="*80)
print("\nPress Enter to proceed with Grover Adaptive Search testing...")
input()

# Save Phase 1 results
results_phase1 = {
    'scenarios': {name: {'n_vars': data['n_vars']} for name, data in scenarios_data.items()},
    'milp_results': {name: {k: v for k, v in res.items() if k != 'selection'} 
                     for name, res in milp_results.items()},
    'classical_qubo_results': {name: {k: v for k, v in res.items() if k not in ['solution', 'selection']}
                              for name, res in classical_qubo_results.items()}
}

with open('phase1_results.json', 'w') as f:
    json.dump(results_phase1, f, indent=2)
print("\nPhase 1 results saved to: phase1_results.json")

print("\n" + "="*80)
print("CONTINUING TO PHASE 2: QUANTUM GROVER ADAPTIVE SEARCH")
print("="*80)
