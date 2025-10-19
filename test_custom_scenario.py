#!/usr/bin/env python3
"""
Test script to validate the custom scenario and show its key properties.
"""

import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from scenarios import load_food_data

def test_custom_scenario():
    """Test and display the custom scenario details."""
    print("Testing Custom Food Optimization Scenario")
    print("=" * 50)
    
    # Load the custom scenario
    farms, foods, food_groups, config = load_food_data('custom')
    
    # Display basic information
    print(f"Number of Farms: {len(farms)}")
    print(f"Farms: {farms}")
    print()
    
    print(f"Number of Foods: {len(foods)}")
    print(f"Foods: {list(foods.keys())}")
    print()
    
    print(f"Number of Food Groups: {len(food_groups)}")
    for group, foods_in_group in food_groups.items():
        print(f"  {group}: {foods_in_group} ({len(foods_in_group)} foods)")
    print()
    
    # Display key parameters
    params = config['parameters']
    
    print("Land Availability:")
    for farm, land in params['land_availability'].items():
        print(f"  {farm}: {land} hectares")
    print()
    
    print("Social Benefit (Minimum Land Utilization):")
    for farm, benefit in params['social_benefit'].items():
        min_land = benefit * params['land_availability'][farm]
        print(f"  {farm}: {benefit:.1%} ({min_land:.1f} hectares minimum)")
    print()
    
    print("Minimum Planting Areas:")
    for food, min_area in params['minimum_planting_area'].items():
        print(f"  {food}: {min_area} hectares")
    print()
    
    print("Maximum Percentage per Crop:")
    for food, max_pct in params['max_percentage_per_crop'].items():
        print(f"  {food}: {max_pct:.1%}")
    print()
    
    print("Food Group Constraints:")
    for group, constraints in params['food_group_constraints'].items():
        print(f"  {group}: {constraints['min_foods']}-{constraints['max_foods']} foods required")
    print()
    
    print("Objective Weights:")
    for objective, weight in params['weights'].items():
        print(f"  {objective}: {weight:.1%}")
    print()
    
    # Display food attributes
    print("Food Attributes:")
    print(f"{'Food':<12} {'Nutrition':<10} {'Density':<8} {'Environ':<8} {'Afford':<8} {'Sustain':<8}")
    print("-" * 60)
    for food, attributes in foods.items():
        print(f"{food:<12} {attributes['nutritional_value']:<10.2f} {attributes['nutrient_density']:<8.2f} "
              f"{attributes['environmental_impact']:<8.2f} {attributes['affordability']:<8.2f} "
              f"{attributes['sustainability']:<8.2f}")
    print()
    
    # Calculate problem size
    num_continuous_vars = len(farms) * len(foods)
    num_binary_vars = len(farms) * len(foods)
    num_constraints = (
        len(farms) +  # Land availability
        len(farms) +  # Social benefit
        len(farms) * len(foods) * 3 +  # Planting area constraints (3 per farm-food pair)
        len(farms) * len(food_groups) * 2  # Food group constraints (2 per farm-group pair)
    )
    
    print("Problem Size:")
    print(f"  Continuous Variables: {num_continuous_vars}")
    print(f"  Binary Variables: {num_binary_vars}")
    print(f"  Total Variables: {num_continuous_vars + num_binary_vars}")
    print(f"  Estimated Constraints: {num_constraints}")
    print()
    
    # Validation checks
    print("Scenario Validation:")
    
    # Check if all foods are in food groups
    all_foods_in_groups = set()
    for foods_in_group in food_groups.values():
        all_foods_in_groups.update(foods_in_group)
    
    missing_foods = set(foods.keys()) - all_foods_in_groups
    extra_foods = all_foods_in_groups - set(foods.keys())
    
    if not missing_foods and not extra_foods:
        print("  ✓ All foods are properly assigned to food groups")
    else:
        if missing_foods:
            print(f"  ✗ Foods not in any group: {missing_foods}")
        if extra_foods:
            print(f"  ✗ Foods in groups but not defined: {extra_foods}")
    
    # Check if we have exactly 2 foods per group
    foods_per_group_correct = all(len(foods_list) == 2 for foods_list in food_groups.values())
    if foods_per_group_correct:
        print("  ✓ Each food group has exactly 2 foods")
    else:
        for group, foods_list in food_groups.items():
            print(f"  - {group}: {len(foods_list)} foods")
    
    # Check if we have exactly 2 farms
    if len(farms) == 2:
        print("  ✓ Scenario has exactly 2 farms")
    else:
        print(f"  ✗ Scenario has {len(farms)} farms instead of 2")
    
    # Check if we have exactly 3 food groups
    if len(food_groups) == 3:
        print("  ✓ Scenario has exactly 3 food groups")
    else:
        print(f"  ✗ Scenario has {len(food_groups)} food groups instead of 3")
    
    print()
    print("Custom scenario test completed successfully!")
    return farms, foods, food_groups, config

if __name__ == "__main__":
    test_custom_scenario()