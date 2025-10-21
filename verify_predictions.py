"""
Verification script to validate the predicted solve times.

This script tests the exact farm counts predicted to achieve 5s and 6.5s solve times.
"""

import sys
import os
import time

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from farm_sampler import generate_farms
import pulp as pl
import pandas as pd


def load_full_family_with_n_farms(n_farms, seed=42):
    """Load full_family scenario with specified number of farms."""
    # Generate farms
    L = generate_farms(n_farms=n_farms, seed=seed)
    farms = list(L.keys())
    
    # Load food data from Excel
    excel_path = os.path.join(project_root, "Inputs", "Combined_Food_Data.xlsx")
    
    if os.path.exists(excel_path):
        df = pd.read_excel(excel_path)
        
        # Sample 2 per group
        sampled = df.groupby('food_group').apply(
            lambda x: x.sample(n=min(len(x), 2), random_state=seed)
        ).reset_index(drop=True)
        foods_list = sampled['Food_Name'].tolist()
        
        filt = df[df['Food_Name'].isin(foods_list)][['Food_Name', 'food_group', 
                                                       'nutritional_value', 'nutrient_density',
                                                       'environmental_impact', 'affordability', 
                                                       'sustainability']].copy()
        filt.rename(columns={'Food_Name': 'Food_Name', 'food_group': 'Food_Group'}, inplace=True)
        
        objectives = ['nutritional_value', 'nutrient_density', 'environmental_impact', 'affordability', 'sustainability']
        for obj in objectives:
            filt[obj] = filt[obj].fillna(0.5).clip(0, 1)
        
        # Build foods dict
        foods = {}
        for _, row in filt.iterrows():
            fname = row['Food_Name']
            foods[fname] = {
                'nutritional_value': float(row['nutritional_value']),
                'nutrient_density': float(row['nutrient_density']),
                'environmental_impact': float(row['environmental_impact']),
                'affordability': float(row['affordability']),
                'sustainability': float(row['sustainability'])
            }
        
        # Build food groups
        food_groups = {}
        for _, row in filt.iterrows():
            g = row['Food_Group']
            fname = row['Food_Name']
            if g not in food_groups:
                food_groups[g] = []
            food_groups[g].append(fname)
    else:
        print("Excel file not found, using default foods")
        foods = {f'Food{i}': {'nutritional_value': 0.5, 'nutrient_density': 0.5,
                              'environmental_impact': 0.5, 'affordability': 0.5,
                              'sustainability': 0.5} for i in range(1, 11)}
        food_groups = {'Group1': list(foods.keys())}
    
    # Config
    min_areas = {food: 0.01 for food in foods.keys()}
    
    parameters = {
        'land_availability': L,
        'minimum_planting_area': min_areas,
        'max_percentage_per_crop': {food: 0.4 for food in foods},
        'social_benefit': {farm: 0.2 for farm in farms},
        'food_group_constraints': {
            g: {'min_foods': 2, 'max_foods': len(lst)}
            for g, lst in food_groups.items()
        },
        'weights': {
            'nutritional_value': 0.25,
            'nutrient_density': 0.2,
            'environmental_impact': 0.25,
            'affordability': 0.15,
            'sustainability': 0.15
        }
    }
    
    config = {'parameters': parameters}
    
    return farms, foods, food_groups, config


def solve_and_time(farms, foods, food_groups, config):
    """Solve the problem and return timing info."""
    params = config['parameters']
    land_availability = params['land_availability']
    weights = params['weights']
    min_planting_area = params.get('minimum_planting_area', {})
    food_group_constraints = params.get('food_group_constraints', {})
    
    # Create variables
    A_pulp = pl.LpVariable.dicts("Area", [(f, c) for f in farms for c in foods], lowBound=0)
    Y_pulp = pl.LpVariable.dicts("Choose", [(f, c) for f in farms for c in foods], cat='Binary')
    
    total_area = sum(land_availability[f] for f in farms)
    
    # Objective
    goal = (
        weights.get('nutritional_value', 0) * pl.lpSum([(foods[c].get('nutritional_value', 0) * A_pulp[(f, c)]) for f in farms for c in foods]) / total_area +
        weights.get('nutrient_density', 0) * pl.lpSum([(foods[c].get('nutrient_density', 0) * A_pulp[(f, c)]) for f in farms for c in foods]) / total_area -
        weights.get('environmental_impact', 0) * pl.lpSum([(foods[c].get('environmental_impact', 0) * A_pulp[(f, c)]) for f in farms for c in foods]) / total_area +
        weights.get('affordability', 0) * pl.lpSum([(foods[c].get('affordability', 0) * A_pulp[(f, c)]) for f in farms for c in foods]) / total_area +
        weights.get('sustainability', 0) * pl.lpSum([(foods[c].get('sustainability', 0) * A_pulp[(f, c)]) for f in farms for c in foods]) / total_area
    )
    
    model = pl.LpProblem("Verification_Test", pl.LpMaximize)
    
    # Constraints
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
    
    # Solve
    start_time = time.time()
    model.solve(pl.PULP_CBC_CMD(msg=0))
    solve_time = time.time() - start_time
    
    return {
        'solve_time': solve_time,
        'status': pl.LpStatus[model.status],
        'objective': pl.value(model.objective) if model.status == 1 else None,
        'num_variables': len(A_pulp) + len(Y_pulp),
        'num_constraints': len(model.constraints)
    }


def main():
    """Run verification tests."""
    print("=" * 80)
    print("VERIFICATION: PREDICTED SOLVE TIMES")
    print("=" * 80)
    
    test_cases = [
        {'target_time': 5.0, 'n_farms': 2988, 'description': '5 second target'},
        {'target_time': 6.5, 'n_farms': 3438, 'description': '6.5 second target'}
    ]
    
    results = []
    
    for test in test_cases:
        print(f"\n{'-' * 80}")
        print(f"Test: {test['description']}")
        print(f"Predicted farms: {test['n_farms']}")
        print(f"Target time: {test['target_time']} seconds")
        print(f"{'-' * 80}")
        
        # Load scenario
        print("Loading scenario...")
        farms, foods, food_groups, config = load_full_family_with_n_farms(test['n_farms'], seed=42)
        
        num_foods = len(foods)
        n = test['n_farms'] * num_foods
        
        print(f"  Farms: {len(farms)}")
        print(f"  Foods: {num_foods}")
        print(f"  n = {test['n_farms']} × {num_foods} = {n}")
        
        # Solve
        print("\nSolving...")
        result = solve_and_time(farms, foods, food_groups, config)
        
        print(f"\nResults:")
        print(f"  Status: {result['status']}")
        print(f"  Solve time: {result['solve_time']:.3f} seconds")
        print(f"  Variables: {result['num_variables']:,}")
        print(f"  Constraints: {result['num_constraints']:,}")
        
        # Compare to target
        error = abs(result['solve_time'] - test['target_time'])
        percent_error = (error / test['target_time']) * 100
        
        print(f"\nAccuracy:")
        print(f"  Target: {test['target_time']:.3f} seconds")
        print(f"  Actual: {result['solve_time']:.3f} seconds")
        print(f"  Error: {error:.3f} seconds ({percent_error:.1f}%)")
        
        if percent_error < 10:
            print(f"  ✅ PASSED: Within 10% of target")
        elif percent_error < 20:
            print(f"  ⚠️  WARNING: Within 20% of target")
        else:
            print(f"  ❌ FAILED: More than 20% error")
        
        results.append({
            'test': test['description'],
            'n_farms': test['n_farms'],
            'target_time': test['target_time'],
            'actual_time': result['solve_time'],
            'error': error,
            'percent_error': percent_error
        })
    
    # Summary
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    
    print(f"\n{'Test':<20} {'Farms':<10} {'Target (s)':<12} {'Actual (s)':<12} {'Error %':<10}")
    print("-" * 80)
    for r in results:
        print(f"{r['test']:<20} {r['n_farms']:<10} {r['target_time']:<12.3f} {r['actual_time']:<12.3f} {r['percent_error']:<10.1f}")
    
    avg_error = sum(r['percent_error'] for r in results) / len(results)
    print(f"\nAverage error: {avg_error:.1f}%")
    
    if avg_error < 10:
        print("\n✅ VERIFICATION PASSED: Model predictions are highly accurate!")
    elif avg_error < 20:
        print("\n⚠️  VERIFICATION WARNING: Model predictions are reasonably accurate")
    else:
        print("\n❌ VERIFICATION FAILED: Model predictions need refinement")


if __name__ == "__main__":
    main()
