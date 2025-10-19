"""
Tutorial 6: Converting Scenarios to Constrained Quadratic Models (CQM)

This tutorial demonstrates how to convert real-world scenario data into a CQM formulation
for D-Wave solvers. Unlike Tutorial 3 which used BQM with penalty-based soft constraints,
this tutorial shows how to use CQM with hard constraints that are ALWAYS satisfied.

You'll learn:
1. The differences between BQM (with penalties) and CQM (with hard constraints)
2. How to define CQM variables (Binary, Integer, etc.)
3. How to formulate the optimization objective for CQM
4. How to add hard constraints (equality, inequality)
5. How to solve with LeapHybridCQMSampler
6. How to interpret and validate CQM solutions
7. When to use CQM vs BQM

Key Differences from Tutorial 3 (BQM/QUBO):
- BQM: Constraints are soft (penalties), may be violated, works with QPU
- CQM: Constraints are hard (always satisfied), cleaner formulation, requires Hybrid solver

This approach is better when:
- Constraints MUST be satisfied (feasibility required)
- Complex constraints (inequalities, multiple constraint types)
- Don't want to tune penalty weights
"""

import sys
import os
import numpy as np
from dimod import ConstrainedQuadraticModel, Binary, Integer, Real
from typing import Dict, List, Tuple
import time

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from src.scenarios import load_food_data


def example_1_understand_cqm_basics():
    """
    Example 1: Understanding CQM Basics
    
    Before working with scenarios, let's understand CQM fundamentals:
    - How to create a CQM
    - Variable types (Binary, Integer, Real)
    - How objectives and constraints differ from BQM
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Understanding CQM Basics")
    print("="*70)
    
    print("\nCQM vs BQM Comparison:")
    print("  BQM (Binary Quadratic Model):")
    print("    - Variables: Binary only (0 or 1)")
    print("    - Constraints: Soft (added as penalties)")
    print("    - Solvers: SimulatedAnnealing, QPU, Hybrid")
    print("    - Feasibility: May violate constraints")
    
    print("\n  CQM (Constrained Quadratic Model):")
    print("    - Variables: Binary, Integer, Real")
    print("    - Constraints: Hard (always satisfied)")
    print("    - Solvers: LeapHybridCQMSampler only")
    print("    - Feasibility: Guaranteed or declared infeasible")
    
    # Create a simple CQM
    cqm = ConstrainedQuadraticModel()
    
    print("\nCreating a simple CQM with different variable types:")
    
    # Binary variable (0 or 1)
    x = Binary('x')
    print(f"  Binary variable: x (values: 0 or 1)")
    
    # Integer variable with bounds
    y = Integer('y', lower_bound=0, upper_bound=10)
    print(f"  Integer variable: y (values: 0 to 10)")
    
    # Another binary variable
    z = Binary('z')
    print(f"  Binary variable: z (values: 0 or 1)")
    
    # Set objective: minimize x + 2*y - 3*z
    objective = x + 2*y - 3*z
    cqm.set_objective(objective)
    print(f"\nObjective: minimize x + 2*y - 3*z")
    
    # Add hard constraints (these WILL be satisfied)
    # Constraint 1: x + z >= 1 (at least one must be selected)
    cqm.add_constraint(x + z >= 1, label='at_least_one')
    print(f"  Constraint 1: x + z >= 1 (at least one binary variable must be 1)")
    
    # Constraint 2: y <= 5 (integer variable limited)
    cqm.add_constraint(y <= 5, label='y_limit')
    print(f"  Constraint 2: y <= 5 (integer variable upper limit)")
    
    # Constraint 3: x + y >= 2 (combined constraint)
    cqm.add_constraint(x + y >= 2, label='combined')
    print(f"  Constraint 3: x + y >= 2 (combined constraint)")
    
    print(f"\nCQM Summary:")
    print(f"  Variables: {len(cqm.variables)} (x, y, z)")
    print(f"  Constraints: {len(cqm.constraints)} hard constraints")
    print(f"  Note: ALL constraints will be satisfied in the solution!")
    
    return cqm


def example_2_food_scenario_to_cqm():
    """
    Example 2: Convert Food Production Scenario to CQM
    
    Load the intermediate food scenario and formulate it as a CQM.
    We'll use Real (continuous) variables representing the amount of land allocated
    to each food on each farm. This is the KEY difference from BQM!
    
    KEY INSIGHT: Unlike Tutorial 3 which uses binary (0/1) for "plant or not",
    CQM uses continuous area variables that can be 0 OR any value up to farm capacity.
    The objective maximizes benefit per hectare Ã— area, not just binary selection.
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Food Production Scenario to CQM")
    print("="*70)
    
    # Load intermediate scenario
    farms, foods, food_groups, config = load_food_data('intermediate')
    
    print("\nScenario Overview:")
    print(f"  Farms: {farms}")
    print(f"  Foods: {list(foods.keys())}")
    print(f"  Food Groups: {list(food_groups.keys())}")
    
    weights = config.get('parameters', {}).get('weights', {})
    land_availability = config.get('parameters', {}).get('land_availability', {})
    
    print("\nOptimization Weights (what we're trying to maximize/minimize):")
    for criterion, weight in weights.items():
        if weight > 0:
            print(f"  {criterion}: {weight}")
    
    print("\nLand Availability:")
    for farm, land in land_availability.items():
        print(f"  {farm}: {land} hectares")
    
    # Create CQM
    cqm = ConstrainedQuadraticModel()
    
    print("\n" + "-"*70)
    print("Step 1: Define Variables (CONTINUOUS AREA)")
    print("-"*70)
    
    print("\nKEY DIFFERENCE FROM TUTORIAL 3 (BQM):")
    print("  Tutorial 3: Binary variables (0 = don't plant, 1 = plant)")
    print("              Benefit per farm-food is fixed")
    print("  Tutorial 6: Real/Integer variables (0 to max hectares)")
    print("              Benefit = (benefit per hectare) Ã— (area planted)")
    print("              This allows OPTIMAL AREA allocation, not just yes/no!")
    
    # Create Real variables for land allocation (continuous area)
    # Each variable represents hectares of land allocated to a food on a farm
    land_vars = {}
    
    for farm in farms:
        for food in foods.keys():
            var_name = f"{farm}_{food}"
            # Real variable: 0 to full farm capacity (continuous)
            max_land = land_availability.get(farm, 100)
            land_vars[var_name] = Real(var_name, lower_bound=0, upper_bound=max_land)
            
    print(f"\n  Created {len(land_vars)} Real (continuous) variables")
    print(f"  Example: Farm1_Wheat can be 0 or any value up to {land_availability['Farm1']} hectares")
    print(f"  Each variable = AREA (hectares) of land for that farm-food combination")
    
    print("\n" + "-"*70)
    print("Step 2: Define Objective Function")
    print("-"*70)
    
    print("\nObjective: Maximize total weighted benefit across all allocations")
    print("  Benefit = sum over all (farm, food) of:")
    print("            (benefit per hectare) Ã— (area allocated)")
    
    # Build objective: maximize weighted sum of food attributes Ã— area
    # CQM minimizes, so we'll negate values we want to maximize
    objective = 0
    
    for farm in farms:
        for food, attributes in foods.items():
            var_name = f"{farm}_{food}"
            var = land_vars[var_name]
            
            # Calculate weighted score per hectare for this food
            score_per_hectare = 0
            for criterion, weight in weights.items():
                if weight > 0 and criterion in attributes:
                    score_per_hectare += weight * attributes[criterion]
            
            # Add to objective: benefit = score_per_hectare Ã— area
            # Negate because we want to maximize but CQM minimizes
            objective += -score_per_hectare * var
    
    cqm.set_objective(objective)
    print(f"  Objective formula: minimize -sum(benefit_per_ha * area)")
    print(f"  This maximizes total benefit across all farm-food allocations")
    
    print("\n" + "-"*70)
    print("Step 3: Add Hard Constraints")
    print("-"*70)
    
    # Constraint 1: Land availability per farm
    print("\n  Constraint Type 1: Land Availability (per farm)")
    for farm in farms:
        capacity = land_availability.get(farm, 100)
        
        # Sum of all food allocations on this farm <= capacity
        farm_allocation = sum(land_vars[f"{farm}_{food}"] for food in foods.keys())
        cqm.add_constraint(farm_allocation <= capacity, label=f'land_{farm}')
        
        print(f"    {farm}: sum(allocations) <= {capacity} hectares")
    
    # Constraint 2: Minimum planting area per food (if planted)
    print("\n  Constraint Type 2: Minimum Planting Area")
    min_planting = config.get('parameters', {}).get('minimum_planting_area', {})
    
    if min_planting:
        print("    If a food is planted, it must use minimum area:")
        for food, min_area in list(min_planting.items())[:3]:  # Show first 3
            print(f"      {food}: {min_area} hectares minimum")
        print("    (Note: This requires binary indicator variables - see Example 4)")
    
    # Constraint 3: Maximum percentage per crop across all farms
    print("\n  Constraint Type 3: Maximum Percentage Per Crop")
    max_percentage = config.get('parameters', {}).get('max_percentage_per_crop', {})
    total_land = sum(land_availability.values())
    
    if max_percentage:
        for food, max_pct in max_percentage.items():
            max_land_for_food = total_land * max_pct
            
            # Sum across all farms for this food
            total_food_allocation = sum(land_vars[f"{farm}_{food}"] for farm in farms)
            cqm.add_constraint(total_food_allocation <= max_land_for_food, 
                             label=f'max_pct_{food}')
        
        print(f"    Each crop limited to {max_pct*100}% of total land")
        print(f"    Total land: {total_land} hectares")
        print(f"    Example: Max {max_land_for_food:.1f} hectares for any single crop")
    
    print(f"\n  Total Constraints: {len(cqm.constraints)}")
    print(f"  All constraints are HARD - they WILL be satisfied!")
    
    print("\n" + "-"*70)
    print("CQM Model Summary")
    print("-"*70)
    print(f"  Variables: {len(land_vars)} Real (continuous) variables")
    print(f"  Constraints: {len(cqm.constraints)} hard constraints")
    print(f"  Objective: Minimize (negated weighted benefits Ã— area)")
    
    return cqm, land_vars, farms, foods, food_groups, config


def example_3_solve_cqm_locally():
    """
    Example 3: Solve CQM Locally (No D-Wave Access Required!)
    
    Use a classical solver to solve small CQM problems.
    This demonstrates that the formulation is correct and shows expected results.
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Solving CQM Locally (Classical Solver)")
    print("="*70)
    
    print("\nNOTE: Since CQM requires D-Wave Hybrid solver for production,")
    print("      we'll solve a simplified version using classical methods.")
    print("      This demonstrates the formulation and expected behavior.")
    
    # Build a simpler version with the 'simple' scenario
    farms, foods, food_groups, config = load_food_data('simple')
    
    # Use only 2 farms and 3 foods to keep problem small
    farms = farms[:2]  # Farm1, Farm2
    foods_subset = {k: v for i, (k, v) in enumerate(foods.items()) if i < 3}  # First 3 foods
    
    print(f"\nSimplified Problem:")
    print(f"  Farms: {farms}")
    print(f"  Foods: {list(foods_subset.keys())}")
    
    weights = config.get('parameters', {}).get('weights', {})
    land_availability = config.get('parameters', {}).get('land_availability', {})
    
    # Build CQM
    cqm = ConstrainedQuadraticModel()
    land_vars = {}
    
    # Variables - Use INTEGER for ExactCQMSolver compatibility
    for farm in farms:
        for food in foods_subset.keys():
            var_name = f"{farm}_{food}"
            max_land = land_availability.get(farm, 100)
            # Use Integer instead of Real for compatibility with ExactCQMSolver
            land_vars[var_name] = Integer(var_name, lower_bound=0, upper_bound=max_land)
    
    # Objective
    objective = 0
    for farm in farms:
        for food, attributes in foods_subset.items():
            var_name = f"{farm}_{food}"
            var = land_vars[var_name]
            score_per_hectare = sum(weights.get(criterion, 0) * attributes.get(criterion, 0)
                                   for criterion in weights.keys())
            objective += -score_per_hectare * var
    
    cqm.set_objective(objective)
    
    # Constraints
    for farm in farms:
        capacity = land_availability.get(farm, 100)
        farm_allocation = sum(land_vars[f"{farm}_{food}"] for food in foods_subset.keys())
        cqm.add_constraint(farm_allocation <= capacity, label=f'land_{farm}')
    
    print(f"\nCQM Model:")
    print(f"  Variables: {len(land_vars)} Integer variables (for solver compatibility)")
    print(f"  Constraints: {len(cqm.constraints)} hard constraints")
    print(f"  Note: Using Integer instead of Real for ExactCQMSolver compatibility")
    
    # Try to solve with ExactCQMSolver (available in newer dimod versions)
    print("\nAttempting to solve with classical methods...")
    
    # NOTE: ExactCQMSolver requires enumerating all possibilities
    # For continuous/integer variables, this is infeasible for even small problems
    # So we'll use a greedy heuristic instead
    
    print("  Note: ExactCQMSolver requires enumerating all combinations")
    print("  For this problem with integers 0-75: 76^6 = 208 billion combinations!")
    print("  Using greedy heuristic instead (simple classical optimization)...")
    
    # Simple greedy solution
    solution = {}
    print(f"\n  Greedy Heuristic Solution:")
    print(f"  Strategy: Allocate capacity to foods with highest benefit per hectare")
    
    # Calculate benefit per hectare for each food
    food_benefits = {}
    for food, attributes in foods_subset.items():
        benefit = sum(weights.get(criterion, 0) * attributes.get(criterion, 0)
                     for criterion in weights.keys())
        food_benefits[food] = benefit
    
    print(f"\n  Benefit per hectare for each food:")
    for food, benefit in sorted(food_benefits.items(), key=lambda x: x[1], reverse=True):
        print(f"    {food}: {benefit:.4f}")
    
    # Sort foods by benefit
    sorted_foods = sorted(food_benefits.items(), key=lambda x: x[1], reverse=True)
    
    total_benefit = 0
    for farm in farms:
        print(f"\n  {farm}:")
        capacity = land_availability.get(farm, 100)
        remaining_capacity = capacity
        
        # Allocate to top 2 foods to ensure diversity
        for i, (food, benefit) in enumerate(sorted_foods[:2]):
            var_name = f"{farm}_{food}"
            if i == 0:
                # Give 60% to best food
                allocation = min(capacity * 0.6, remaining_capacity)
            else:
                # Give remaining (up to 40%) to second food
                allocation = min(capacity * 0.4, remaining_capacity)
            
            solution[var_name] = allocation
            remaining_capacity -= allocation
            total_benefit += benefit * allocation
            
            print(f"    {food}: {allocation:.2f} hectares (benefit/ha: {benefit:.4f})")
        
        print(f"    Total allocated: {capacity - remaining_capacity:.2f}/{capacity} hectares")
    
    print(f"\n  Total weighted benefit: {total_benefit:.2f}")
    print(f"  Objective value (negated): {-total_benefit:.2f}")
    
    print("\n  Key Observations:")
    print(f"  1. Each farm gets the SAME allocation pattern")
    print(f"     (60% best food, 40% second best)")
    print(f"  2. ALL CAPACITY is used (greedy strategy)")
    print(f"  3. Constraints are satisfied (sum <= capacity)")
    print(f"  4. This is a heuristic solution, not necessarily optimal")
    
    # Try solving with PuLP for exact solution
    print("\n" + "-"*70)
    print("Solving with PuLP (Exact Linear Programming Solver)")
    print("-"*70)
    
    try:
        import pulp
        
        print("  PuLP is available! Solving with linear programming...")
        print("  This will find the OPTIMAL solution (not just heuristic)")
        
        # Create PuLP problem (maximization)
        prob = pulp.LpProblem("Food_Production_CQM", pulp.LpMaximize)
        
        # Create PuLP variables (continuous or integer)
        pulp_vars = {}
        for farm in farms:
            for food in foods_subset.keys():
                var_name = f"{farm}_{food}"
                max_land = land_availability.get(farm, 100)
                # Use continuous variables (can be 0.0 to max_land)
                pulp_vars[var_name] = pulp.LpVariable(var_name, 
                                                       lowBound=0, 
                                                       upBound=max_land, 
                                                       cat='Continuous')
        
        # Objective: Maximize total weighted benefit
        objective_expr = pulp.lpSum([
            food_benefits[food] * pulp_vars[f"{farm}_{food}"]
            for farm in farms 
            for food in foods_subset.keys()
        ])
        prob += objective_expr, "Total_Weighted_Benefit"
        
        # Constraints: Land availability per farm
        for farm in farms:
            capacity = land_availability.get(farm, 100)
            prob += (
                pulp.lpSum([pulp_vars[f"{farm}_{food}"] for food in foods_subset.keys()]) 
                <= capacity,
                f"Land_Capacity_{farm}"
            )
        
        # Solve
        solver = pulp.PULP_CBC_CMD(msg=0)  # Use CBC solver, suppress output
        prob.solve(solver)
        
        print(f"\n  Solver Status: {pulp.LpStatus[prob.status]}")
        
        if prob.status == pulp.LpStatusOptimal:
            print(f"  Optimal objective value: {pulp.value(prob.objective):.4f}")
            
            print(f"\n  Optimal Solution:")
            for farm in farms:
                print(f"\n    {farm}:")
                farm_total = 0
                for food in foods_subset.keys():
                    var_name = f"{farm}_{food}"
                    area = pulp_vars[var_name].varValue
                    if area > 0.001:  # Only show non-zero allocations
                        print(f"      {food}: {area:.2f} hectares (benefit/ha: {food_benefits[food]:.4f})")
                        farm_total += area
                capacity = land_availability.get(farm, 100)
                print(f"      Total allocated: {farm_total:.2f}/{capacity} hectares")
            
            print(f"\n  Comparison:")
            print(f"    Greedy solution benefit:  {total_benefit:.2f}")
            print(f"    PuLP optimal benefit:     {pulp.value(prob.objective):.2f}")
            improvement = ((pulp.value(prob.objective) - total_benefit) / total_benefit * 100)
            print(f"    Improvement:              {improvement:+.2f}%")
            
            print(f"\n  Key Insights:")
            print(f"    1. PuLP finds the EXACT optimal solution")
            print(f"    2. Can handle continuous variables (not just integer)")
            print(f"    3. Scales well to medium-sized problems (1000s of variables)")
            print(f"    4. Uses proven LP solvers (CBC, GLPK, Gurobi, etc.)")
            print(f"    5. Perfect for validating CQM formulations!")
            
        else:
            print(f"  WARNING: Solver did not find optimal solution")
            print(f"  Status: {pulp.LpStatus[prob.status]}")
    
    except ImportError:
        print("  PuLP is not installed.")
        print("  Install with: pip install pulp")
        print("  PuLP provides exact solutions using linear programming!")
    except Exception as e:
        print(f"  Error solving with PuLP: {e}")
    
    print(f"\n  With D-Wave CQM Hybrid solver:")
    print(f"     - Would find better allocation considering ALL constraints")
    print(f"     - Could handle diversity requirements more sophisticatedly")
    print(f"     - Scales to much larger problems (100s of variables)")
    print(f"     - Can handle quadratic objectives (PuLP is linear only)")
    
    return cqm, land_vars


def example_3b_solve_cqm_with_hybrid():
    """
    Example 3b: Solve CQM with Hybrid Solver (D-Wave Access Required)
    
    Use LeapHybridCQMSampler to solve the food production CQM.
    This requires D-Wave API access.
    """
    print("\n" + "="*70)
    print("EXAMPLE 3b: Solving CQM with LeapHybridCQMSampler")
    print("="*70)
    
    # Build the CQM from previous example
    cqm, variables, farms, foods, food_groups, config = example_2_food_scenario_to_cqm()
    
    print("\nPreparing to solve with D-Wave Hybrid CQM Solver...")
    print("  Note: This requires D-Wave API access (DWAVE_API_TOKEN)")
    print("  The solver will find a solution that satisfies ALL constraints")
    
    # Note: Actual solving requires D-Wave credentials
    print("\nSolver Configuration:")
    print("  Solver: LeapHybridCQMSampler")
    print("  Time limit: Will use solver's minimum time limit")
    print("  Label: Food Production CQM Tutorial")
    
    # This is where you would solve with actual D-Wave access:
    """
    from dwave.system import LeapHybridCQMSampler
    
    sampler = LeapHybridCQMSampler()
    
    # Get minimum time limit for this problem
    min_time_limit = sampler.min_time_limit(cqm)
    print(f"  Minimum time limit: {min_time_limit} seconds")
    
    # Solve
    print("\nSending problem to D-Wave Hybrid CQM Solver...")
    start_time = time.time()
    sampleset = sampler.sample_cqm(cqm, 
                                   time_limit=min_time_limit,
                                   label="Food Production CQM Tutorial")
    solve_time = time.time() - start_time
    
    print(f"  Solve completed in {solve_time:.2f} seconds")
    print(f"  Total samples returned: {len(sampleset)}")
    
    # Filter for feasible solutions
    feasible_sampleset = sampleset.filter(lambda d: d.is_feasible)
    num_feasible = len(feasible_sampleset)
    
    print(f"  Feasible solutions: {num_feasible}")
    
    if num_feasible > 0:
        best_sample = feasible_sampleset.first
        print(f"\nBest Feasible Solution:")
        print(f"  Energy: {best_sample.energy}")
        print(f"  Is feasible: {best_sample.is_feasible}")
        
        # Interpret solution
        print("\n  Land Allocations:")
        for farm in farms:
            print(f"\n    {farm}:")
            total_used = 0
            for food in foods.keys():
                var_name = f"{farm}_{food}"
                allocated = best_sample.sample[var_name]
                if allocated > 0:
                    print(f"      {food}: {allocated} hectares")
                    total_used += allocated
            
            capacity = config.get('parameters', {}).get('land_availability', {}).get(farm, 0)
            print(f"      Total used: {total_used}/{capacity} hectares")
    else:
        print("\nWARNING: No feasible solutions found!")
        print("  This means the constraints cannot all be satisfied simultaneously")
        print("  You may need to relax some constraints")
    """
    
    print("\n" + "-"*70)
    print("Expected Output Structure:")
    print("-"*70)
    print("  When solved, you would see:")
    print("    - Energy value (objective function value)")
    print("    - Feasibility status (should be True)")
    print("    - Land allocations for each farm-food combination")
    print("    - Constraint satisfaction verification")
    print("    - Total land usage per farm")
    
    return cqm, variables


def example_4_cqm_with_binary_indicators():
    """
    Example 4: CQM with Binary Indicator Variables (Semi-Continuous)
    
    A more advanced example showing how to use binary variables
    alongside continuous variables to enforce conditional constraints.
    This is the "semi-continuous" pattern: area is either 0 OR >= minimum.
    
    Use case: "If a food is planted (any amount), it must use at least X hectares"
    
    This solves the issue from Tutorial 3 where everything gets the same value!
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: CQM with Semi-Continuous Variables")
    print("="*70)
    
    # Load scenario
    farms, foods, food_groups, config = load_food_data('simple')
    
    print("\nProblem: Semi-Continuous Decision Variables")
    print("  For each farm-food: area is either:")
    print("    - 0 (not planted), OR")
    print("    - Between MIN and MAX hectares (planted)")
    print("\n  This requires TWO types of variables:")
    print("    - Real variable: area (continuous 0 to MAX)")
    print("    - Binary variable: is_planted (0 or 1)")
    print("    - Linking constraints: if is_planted=0, then area=0")
    print("                          if is_planted=1, then area >= MIN")
    
    cqm = ConstrainedQuadraticModel()
    
    # Variables
    land_vars = {}      # Real: hectares allocated (continuous)
    planted_vars = {}   # Binary: is this food planted?
    
    land_availability = config.get('parameters', {}).get('land_availability', {})
    weights = config.get('parameters', {}).get('weights', {})
    
    MIN_PLANTING = 10  # Minimum hectares if planted
    
    print(f"\nMinimum planting area: {MIN_PLANTING} hectares")
    print("\nVariable Creation:")
    
    for farm in farms:
        for food in foods.keys():
            base_name = f"{farm}_{food}"
            
            # Real variable for land area (continuous)
            max_land = land_availability.get(farm, 100)
            land_vars[base_name] = Real(f"land_{base_name}", 
                                        lower_bound=0, 
                                        upper_bound=max_land)
            
            # Binary indicator for whether food is planted
            planted_vars[base_name] = Binary(f"planted_{base_name}")
    
    print(f"  Created {len(land_vars)} Real variables (continuous area)")
    print(f"  Created {len(planted_vars)} Binary variables (indicators)")
    
    # Objective: maximize weighted benefits per hectare Ã— area
    objective = 0
    for farm in farms:
        for food, attributes in foods.items():
            base_name = f"{farm}_{food}"
            score_per_hectare = sum(weights.get(criterion, 0) * attributes.get(criterion, 0)
                                   for criterion in weights.keys())
            # Benefit = score_per_hectare Ã— area
            objective += -score_per_hectare * land_vars[base_name]
    
    cqm.set_objective(objective)
    print("\nObjective: Maximize (benefit per hectare) Ã— (area planted)")
    
    # Constraints
    print("\nConstraints:")
    
    # 1. Land availability per farm
    print("  1. Land availability per farm")
    for farm in farms:
        capacity = land_availability.get(farm, 100)
        farm_total = sum(land_vars[f"{farm}_{food}"] for food in foods.keys())
        cqm.add_constraint(farm_total <= capacity, label=f'land_{farm}')
    
    # 2. Link binary indicators to land allocation (SEMI-CONTINUOUS!)
    print(f"  2. Semi-continuous linking (area=0 OR area>={MIN_PLANTING})")
    
    for farm in farms:
        for food in foods.keys():
            base_name = f"{farm}_{food}"
            land_var = land_vars[base_name]
            planted_var = planted_vars[base_name]
            
            max_land = land_availability.get(farm, 100)
            
            # Constraint 1: if planted=1, then area >= MIN_PLANTING
            # area - planted * MIN_PLANTING >= 0
            # When planted=0: area >= 0 (always true)
            # When planted=1: area >= MIN_PLANTING (enforced!)
            cqm.add_constraint(land_var - planted_var * MIN_PLANTING >= 0,
                             label=f'min_planting_{base_name}')
            
            # Constraint 2: if planted=0, then area = 0 (area <= planted * MAX)
            # area - planted * MAX_LAND <= 0
            # When planted=0: area <= 0, so area must be 0
            # When planted=1: area <= MAX_LAND (normal bound)
            cqm.add_constraint(land_var - planted_var * max_land <= 0,
                             label=f'link_planted_{base_name}')
    
    print(f"    Added {len(farms) * len(foods) * 2} semi-continuous constraints")
    print(f"    This ensures: area = 0 OR area >= {MIN_PLANTING}")
    
    # 3. Diversity: plant at least 2 different foods per farm
    print("  3. Diversity: at least 2 foods per farm")
    for farm in farms:
        planted_count = sum(planted_vars[f"{farm}_{food}"] for food in foods.keys())
        cqm.add_constraint(planted_count >= 2, label=f'diversity_{farm}')
    
    print(f"\nTotal CQM Summary:")
    print(f"  Variables: {len(land_vars) + len(planted_vars)}")
    print(f"    - {len(land_vars)} Real (area variables)")
    print(f"    - {len(planted_vars)} Binary (indicator variables)")
    print(f"  Constraints: {len(cqm.constraints)}")
    
    print("\nKEY INSIGHT - Why this solves the Tutorial 3 problem:")
    print("  Tutorial 3 (BQM): Binary selection â†’ all farms pick same best food")
    print("  Tutorial 6 (CQM): Continuous area â†’ optimizer balances:")
    print("    - Which foods to plant (binary indicators)")
    print("    - How much area to allocate (continuous variables)")
    print("    - Diversity constraints (at least 2 foods)")
    print("    - Land capacity limits (hard constraints)")
    print("  Result: More realistic allocation with area optimization!")
    
    return cqm, land_vars, planted_vars


def example_5_compare_bqm_vs_cqm():
    """
    Example 5: BQM vs CQM Comparison
    
    Side-by-side comparison showing when to use each approach.
    Highlights the KEY difference in how variables are formulated.
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: BQM vs CQM Comparison")
    print("="*70)
    
    print("\n" + "="*70)
    print("KEY DIFFERENCE: Variable Formulation")
    print("="*70)
    
    print("\nTUTORIAL 3 (BQM with Binary Variables):")
    print("  Variables: x[farm, food] in {0, 1}")
    print("             0 = don't plant, 1 = plant")
    print("  Objective: maximize sum (benefit * x[farm, food])")
    print("             Fixed benefit per selection")
    print("\n  PROBLEM: All farms select the same 'best' food!")
    print("           Example: Everyone picks Rice because it has highest score")
    print("           No optimization of HOW MUCH to plant")
    
    print("\nTUTORIAL 6 (CQM with Continuous Variables):")
    print("  Variables: area[farm, food] in [0, farm_capacity]")
    print("             Continuous hectares of land")
    print("  Objective: maximize sum (benefit_per_ha * area[farm, food])")
    print("             Benefit scales with area!")
    print("\n  SOLUTION: Optimizer balances area allocation")
    print("            Can plant different amounts of different foods")
    print("            Respects capacity, minimums, diversity")
    
    print("\n" + "="*70)
    print("When to Use BQM (with Penalty-Based Constraints):")
    print("="*70)
    print("  âœ“ Need to use QPU directly")
    print("  âœ“ Developing/testing with simulators")
    print("  âœ“ Soft constraints are acceptable (may be violated)")
    print("  âœ“ Need flexibility in constraint trade-offs")
    print("  âœ“ All variables are binary (yes/no decisions)")
    print("  âœ“ Problem size fits on QPU")
    print("  âœ“ Decision is selection, not allocation")
    
    print("\n  Example from Tutorial 3:")
    print("    Problem: Which foods to grow on which farms (yes/no)")
    print("    Variables: Binary (grow food or not)")
    print("    Constraints: Land limits (soft penalties)")
    print("    Result: May slightly violate constraints if beneficial")
    print("    Issue: All farms pick same foods (no area optimization)")
    
    print("\n" + "="*70)
    print("When to Use CQM (with Hard Constraints):")
    print("="*70)
    print("  âœ“ Constraints MUST be satisfied (feasibility critical)")
    print("  âœ“ Complex constraints (inequalities, conditionals)")
    print("  âœ“ Need Integer or Real variables")
    print("  âœ“ Don't want to tune penalty weights")
    print("  âœ“ Have access to Hybrid CQM solver")
    print("  âœ“ Problem is too large for QPU")
    print("  âœ“ Need continuous allocation, not just selection")
    
    print("\n  Example from This Tutorial:")
    print("    Problem: How much area to allocate to each food")
    print("    Variables: Real/Integer (hectares of land)")
    print("    Constraints: Land limits (hard, always satisfied)")
    print("    Result: Guaranteed feasible or declared infeasible")
    print("    Advantage: Optimizes BOTH selection AND allocation!")
    
    print("\n" + "-"*70)
    print("Constraint Handling Comparison:")
    print("-"*70)
    
    print("\nBQM Approach (Tutorial 3):")
    print("  # Add land constraint as penalty")
    print("  penalty_weight = 5.0")
    print("  for farm in farms:")
    print("      violation = sum(farm_vars) - capacity")
    print("      Q += penalty_weight * violation^2  # Soft penalty")
    print("\n  Result: May violate if penalty too small")
    
    print("\nCQM Approach (This Tutorial):")
    print("  # Add land constraint (hard)")
    print("  for farm in farms:")
    print("      farm_total = sum(land_vars)")
    print("      cqm.add_constraint(farm_total <= capacity)")
    print("\n  Result: ALWAYS satisfied or problem declared infeasible")
    
    print("\n" + "-"*70)
    print("Performance Characteristics:")
    print("-"*70)
    
    comparison_table = [
        ["Aspect", "BQM + Penalties", "CQM + Hard Constraints"],
        ["-" * 20, "-" * 25, "-" * 25],
        ["Constraint Type", "Soft (may violate)", "Hard (always satisfied)"],
        ["Variable Types", "Binary only", "Binary, Integer, Real"],
        ["Penalty Tuning", "Required", "Not needed"],
        ["Solver Options", "SA, QPU, Hybrid", "Hybrid CQM only"],
        ["Problem Size", "Limited by QPU", "Larger (Hybrid)"],
        ["Feasibility", "Always finds solution", "May be infeasible"],
        ["Formulation", "Complex (penalties)", "Cleaner (direct)"],
        ["Decision Type", "Selection (yes/no)", "Allocation (how much)"],
    ]
    
    for row in comparison_table:
        print(f"  {row[0]:20} | {row[1]:25} | {row[2]:25}")
    
    print("\n" + "-"*70)
    print("Summary:")
    print("-"*70)
    print("  Use BQM when:")
    print("    - Making yes/no decisions (binary selection)")
    print("    - Flexibility > Strict Feasibility")
    print("    - Need QPU access")
    
    print("\n  Use CQM when:")
    print("    - Optimizing continuous/integer amounts (allocation)")
    print("    - Strict Feasibility > Flexibility")
    print("    - Need complex constraints")
    
    print("\n  Often, start with BQM for prototyping,")
    print("  then move to CQM for production if:")
    print("    1. Feasibility is critical, OR")
    print("    2. You need area/amount optimization, not just selection")


def main():
    """
    Main function to run all examples.
    """
    print("\n" + "="*70)
    print("TUTORIAL 6: CONVERTING SCENARIOS TO CQM")
    print("="*70)
    print("\nThis tutorial demonstrates Constrained Quadratic Models (CQM)")
    print("as an alternative to BQM with penalty-based constraints.")
    print("\nKey Advantage: Hard constraints that are ALWAYS satisfied!")
    
    # Run examples
    example_1_understand_cqm_basics()
    
    example_2_food_scenario_to_cqm()
    
    example_3_solve_cqm_locally()  # NEW: Solve locally without D-Wave access
    
    example_3b_solve_cqm_with_hybrid()  # D-Wave hybrid solver (requires access)
    
    example_4_cqm_with_binary_indicators()
    
    example_5_compare_bqm_vs_cqm()
    
    print("\n" + "="*70)
    print("TUTORIAL COMPLETE!")
    print("="*70)
    print("\nKey Takeaways:")
    print("  1. CQM provides hard constraints (always satisfied)")
    print("  2. Supports Binary, Integer, and Real variables")
    print("  3. No need to tune penalty weights")
    print("  4. Requires LeapHybridCQMSampler (not QPU)")
    print("  5. Better for problems where feasibility is critical")
    print("\nNext Steps:")
    print("  - Compare this approach with Tutorial 3 (BQM)")
    print("  - Try solving with your own D-Wave API credentials")
    print("  - Experiment with different constraint types")
    print("  - See untitled:Untitled-1 for a complete Job Shop Scheduling CQM example")


if __name__ == "__main__":
    main()

