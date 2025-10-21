"""
Diagnostic script to analyze why full_family scenario is infeasible
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from scenarios import load_food_data

# Load the scenario
print("="*80)
print("FULL_FAMILY SCENARIO DIAGNOSTIC")
print("="*80)

farms, foods, food_groups, config = load_food_data('full_family')

print(f"\nNumber of farms: {len(farms)}")
print(f"Number of foods: {len(foods)}")
print(f"Number of food groups: {len(food_groups)}")

params = config['parameters']
land_avail = params['land_availability']
min_areas = params['minimum_planting_area']
fg_constraints = params['food_group_constraints']

print(f"\n{'='*80}")
print("FARM SIZES")
print(f"{'='*80}")
for farm in farms:
    print(f"{farm}: {land_avail[farm]:.2f} ha")

print(f"\nTotal land: {sum(land_avail.values()):.2f} ha")
print(f"Average farm size: {sum(land_avail.values())/len(farms):.2f} ha")
print(f"Min farm size: {min(land_avail.values()):.2f} ha")
print(f"Max farm size: {max(land_avail.values()):.2f} ha")

print(f"\n{'='*80}")
print("MINIMUM AREAS PER CROP")
print(f"{'='*80}")
for food in sorted(foods.keys()):
    print(f"{food}: {min_areas.get(food, 0):.2f} ha")

print(f"\n{'='*80}")
print("FOOD GROUP CONSTRAINTS")
print(f"{'='*80}")
for group, crops in food_groups.items():
    constraint = fg_constraints.get(group, {})
    min_req = constraint.get('min_foods', 0)
    max_req = constraint.get('max_foods', len(crops))
    print(f"{group} ({len(crops)} crops): min={min_req}, max={max_req}")
    print(f"  Crops: {crops}")

print(f"\n{'='*80}")
print("FEASIBILITY ANALYSIS")
print(f"{'='*80}")

# Check if smallest farm can meet requirements
smallest_farm = min(land_avail.values())
print(f"\nSmallest farm: {smallest_farm:.2f} ha")

# Calculate minimum land needed per farm
min_land_per_group = {}
for group, crops in food_groups.items():
    constraint = fg_constraints.get(group, {})
    min_req = constraint.get('min_foods', 0)
    # Find the smallest min_areas for crops in this group
    crop_min_areas = sorted([min_areas.get(crop, 0) for crop in crops])
    # Take the smallest min_req crops
    min_land_per_group[group] = sum(crop_min_areas[:min_req])
    print(f"\n{group}: requires min {min_req} crops")
    print(f"  Smallest crops in group: {crop_min_areas[:min_req]}")
    print(f"  Minimum land needed: {min_land_per_group[group]:.2f} ha")

total_min_land = sum(min_land_per_group.values())
print(f"\nTotal minimum land per farm: {total_min_land:.2f} ha")
print(f"Smallest farm has: {smallest_farm:.2f} ha")

if smallest_farm < total_min_land:
    print(f"\n❌ INFEASIBLE: Smallest farm ({smallest_farm:.2f} ha) < minimum required ({total_min_land:.2f} ha)")
    print(f"   Shortfall: {total_min_land - smallest_farm:.2f} ha")
    print(f"\nSOLUTION: Reduce minimum areas to at most {smallest_farm / len(food_groups):.2f} ha per group")
else:
    print(f"\n✅ FEASIBLE: Smallest farm ({smallest_farm:.2f} ha) >= minimum required ({total_min_land:.2f} ha)")

# Count how many farms are too small
farms_too_small = sum(1 for f in farms if land_avail[f] < total_min_land)
print(f"\nFarms too small: {farms_too_small}/{len(farms)}")
if farms_too_small > 0:
    print(f"These farms cannot satisfy constraints:")
    for farm in farms:
        if land_avail[farm] < total_min_land:
            print(f"  {farm}: {land_avail[farm]:.2f} ha (needs {total_min_land:.2f} ha)")
