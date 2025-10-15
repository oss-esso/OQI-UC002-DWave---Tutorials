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
    We'll use Integer variables representing the amount of land allocated
    to each food on each farm.
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
    print("Step 1: Define Variables")
    print("-"*70)
    
    # Create Integer variables for land allocation
    # Each variable represents hectares of land allocated to a food on a farm
    variables = {}
    
    for farm in farms:
        for food in foods.keys():
            var_name = f"{farm}_{food}"
            # Integer variable: 0 to full farm capacity
            max_land = land_availability.get(farm, 100)
            variables[var_name] = Integer(var_name, lower_bound=0, upper_bound=max_land)
            
    print(f"  Created {len(variables)} integer variables")
    print(f"  Example: Farm1_Wheat (0 to {land_availability['Farm1']} hectares)")
    print(f"  Each variable = hectares of land for that farm-food combination")
    
    print("\n" + "-"*70)
    print("Step 2: Define Objective Function")
    print("-"*70)
    
    # Build objective: maximize weighted sum of food attributes
    # CQM minimizes, so we'll negate values we want to maximize
    objective = 0
    
    for farm in farms:
        for food, attributes in foods.items():
            var_name = f"{farm}_{food}"
            var = variables[var_name]
            
            # Calculate weighted score for this food
            score = 0
            for criterion, weight in weights.items():
                if weight > 0 and criterion in attributes:
                    # Multiply by land amount to get total benefit
                    score += weight * attributes[criterion]
            
            # Add to objective (negate because we want to maximize but CQM minimizes)
            objective += -score * var
    
    cqm.set_objective(objective)
    print(f"  Objective: Maximize weighted benefits across all farm-food allocations")
    print(f"  (Negated for minimization in CQM)")
    
    print("\n" + "-"*70)
    print("Step 3: Add Hard Constraints")
    print("-"*70)
    
    # Constraint 1: Land availability per farm
    print("\n  Constraint Type 1: Land Availability (per farm)")
    for farm in farms:
        capacity = land_availability.get(farm, 100)
        
        # Sum of all food allocations on this farm <= capacity
        farm_allocation = sum(variables[f"{farm}_{food}"] for food in foods.keys())
        cqm.add_constraint(farm_allocation <= capacity, label=f'land_{farm}')
        
        print(f"    {farm}: sum(allocations) <= {capacity} hectares")
    
    # Constraint 2: Minimum planting area per food (if planted)
    print("\n  Constraint Type 2: Minimum Planting Area")
    min_planting = config.get('parameters', {}).get('minimum_planting_area', {})
    
    if min_planting:
        print("    If a food is planted, it must use minimum area:")
        for food, min_area in list(min_planting.items())[:3]:  # Show first 3
            print(f"      {food}: {min_area} hectares minimum")
        print("    (Implementation note: This requires binary indicator variables)")
    
    # Constraint 3: Maximum percentage per crop across all farms
    print("\n  Constraint Type 3: Maximum Percentage Per Crop")
    max_percentage = config.get('parameters', {}).get('max_percentage_per_crop', {})
    total_land = sum(land_availability.values())
    
    if max_percentage:
        for food, max_pct in max_percentage.items():
            max_land_for_food = total_land * max_pct
            
            # Sum across all farms for this food
            total_food_allocation = sum(variables[f"{farm}_{food}"] for farm in farms)
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
    print(f"  Variables: {len(variables)} Integer variables")
    print(f"  Constraints: {len(cqm.constraints)} hard constraints")
    print(f"  Objective: Minimize (negated weighted benefits)")
    
    return cqm, variables, farms, foods, food_groups, config


def example_3_solve_cqm_with_hybrid():
    """
    Example 3: Solve CQM with Hybrid Solver
    
    Use LeapHybridCQMSampler to solve the food production CQM.
    This requires D-Wave API access.
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Solving CQM with LeapHybridCQMSampler")
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
    Example 4: CQM with Binary Indicator Variables
    
    A more advanced example showing how to use binary variables
    alongside integer variables to enforce conditional constraints.
    
    Use case: "If a food is planted (any amount), it must use at least X hectares"
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: CQM with Binary Indicators")
    print("="*70)
    
    # Load scenario
    farms, foods, food_groups, config = load_food_data('simple')
    
    print("\nProblem: Conditional Constraints")
    print("  If we plant a food on a farm, we need at least MIN hectares")
    print("  This requires binary indicator variables:")
    print("    - Binary var: is_planted (0 = not planted, 1 = planted)")
    print("    - Integer var: hectares (0 to max)")
    print("    - Constraint: if is_planted=1, then hectares >= MIN")
    
    cqm = ConstrainedQuadraticModel()
    
    # Variables
    land_vars = {}      # Integer: hectares allocated
    planted_vars = {}   # Binary: is this food planted?
    
    land_availability = config.get('parameters', {}).get('land_availability', {})
    weights = config.get('parameters', {}).get('weights', {})
    
    print("\nVariable Creation:")
    for farm in farms:
        for food in foods.keys():
            base_name = f"{farm}_{food}"
            
            # Integer variable for land
            max_land = land_availability.get(farm, 100)
            land_vars[base_name] = Integer(f"land_{base_name}", 
                                          lower_bound=0, 
                                          upper_bound=max_land)
            
            # Binary indicator for whether food is planted
            planted_vars[base_name] = Binary(f"planted_{base_name}")
    
    print(f"  Created {len(land_vars)} integer variables (land)")
    print(f"  Created {len(planted_vars)} binary variables (indicators)")
    
    # Objective: maximize weighted benefits (negate for minimization)
    objective = 0
    for farm in farms:
        for food, attributes in foods.items():
            base_name = f"{farm}_{food}"
            score = sum(weights.get(criterion, 0) * attributes.get(criterion, 0)
                       for criterion in weights.keys())
            objective += -score * land_vars[base_name]
    
    cqm.set_objective(objective)
    print("\nObjective: Maximize weighted benefits")
    
    # Constraints
    print("\nConstraints:")
    
    # 1. Land availability per farm
    print("  1. Land availability per farm")
    for farm in farms:
        capacity = land_availability.get(farm, 100)
        farm_total = sum(land_vars[f"{farm}_{food}"] for food in foods.keys())
        cqm.add_constraint(farm_total <= capacity, label=f'land_{farm}')
    
    # 2. Link binary indicators to land allocation
    print("  2. Link binary indicators: planted=1 if land>0")
    MIN_PLANTING = 10  # Minimum hectares if planted
    
    for farm in farms:
        for food in foods.keys():
            base_name = f"{farm}_{food}"
            land_var = land_vars[base_name]
            planted_var = planted_vars[base_name]
            
            # If planted=1, then land >= MIN_PLANTING
            # If planted=0, then land = 0
            # This can be encoded as:
            #   land <= planted * MAX_LAND (if not planted, land must be 0)
            #   land >= planted * MIN_PLANTING (if planted, land >= MIN)
            
            max_land = land_availability.get(farm, 100)
            
            # Constraint: if planted, land must be >= MIN_PLANTING
            # land - planted * MIN_PLANTING >= 0
            cqm.add_constraint(land_var - planted_var * MIN_PLANTING >= 0,
                             label=f'min_planting_{base_name}')
            
            # Constraint: if not planted, land must be 0 (land <= planted * MAX)
            # land - planted * MAX_LAND <= 0
            cqm.add_constraint(land_var - planted_var * max_land <= 0,
                             label=f'link_planted_{base_name}')
    
    print(f"    Minimum planting area: {MIN_PLANTING} hectares")
    print(f"    Added {len(farms) * len(foods) * 2} linking constraints")
    
    # 3. Diversity: plant at least 2 different foods per farm
    print("  3. Diversity: at least 2 foods per farm")
    for farm in farms:
        planted_count = sum(planted_vars[f"{farm}_{food}"] for food in foods.keys())
        cqm.add_constraint(planted_count >= 2, label=f'diversity_{farm}')
    
    print(f"\nTotal CQM Summary:")
    print(f"  Variables: {len(land_vars) + len(planted_vars)}")
    print(f"  Constraints: {len(cqm.constraints)}")
    
    return cqm, land_vars, planted_vars


def example_5_compare_bqm_vs_cqm():
    """
    Example 5: BQM vs CQM Comparison
    
    Side-by-side comparison showing when to use each approach.
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: BQM vs CQM Comparison")
    print("="*70)
    
    print("\nWhen to Use BQM (with Penalty-Based Constraints):")
    print("  ✓ Need to use QPU directly")
    print("  ✓ Developing/testing with simulators")
    print("  ✓ Soft constraints are acceptable (may be violated)")
    print("  ✓ Need flexibility in constraint trade-offs")
    print("  ✓ All variables are binary")
    print("  ✓ Problem size fits on QPU")
    
    print("\n  Example from Tutorial 3:")
    print("    Problem: Food production with land constraints")
    print("    Variables: Binary (grow food or not)")
    print("    Constraints: Land limits (soft penalties)")
    print("    Result: May slightly violate constraints if beneficial")
    
    print("\nWhen to Use CQM (with Hard Constraints):")
    print("  ✓ Constraints MUST be satisfied (feasibility critical)")
    print("  ✓ Complex constraints (inequalities, conditionals)")
    print("  ✓ Need Integer or Real variables")
    print("  ✓ Don't want to tune penalty weights")
    print("  ✓ Have access to Hybrid CQM solver")
    print("  ✓ Problem is too large for QPU")
    
    print("\n  Example from This Tutorial:")
    print("    Problem: Food production with land constraints")
    print("    Variables: Integer (hectares of land)")
    print("    Constraints: Land limits (hard, always satisfied)")
    print("    Result: Guaranteed feasible or declared infeasible")
    
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
    ]
    
    for row in comparison_table:
        print(f"  {row[0]:20} | {row[1]:25} | {row[2]:25}")
    
    print("\n" + "-"*70)
    print("Summary:")
    print("-"*70)
    print("  Use BQM when: Flexibility > Strict Feasibility")
    print("  Use CQM when: Strict Feasibility > Flexibility")
    print("\n  Often, start with BQM for prototyping,")
    print("  then move to CQM for production if feasibility is critical.")


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
    
    example_3_solve_cqm_with_hybrid()
    
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
