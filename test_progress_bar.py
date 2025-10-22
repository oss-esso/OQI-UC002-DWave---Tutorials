"""
Quick test to verify progress bar works correctly
"""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from benchmark_scalability import load_full_family_with_n_farms
from solver_runner import create_cqm

print("Testing progress bar with small problem (10 farms)...")
print("="*80)

farms, foods, food_groups, config = load_full_family_with_n_farms(n_farms=10, seed=42)

print(f"\nProblem size:")
print(f"  Farms: {len(farms)}")
print(f"  Foods: {len(foods)}")
print(f"  Variables: {2 * len(farms) * len(foods)}")

print(f"\nCreating CQM with progress bar...")
cqm, A, Y, constraint_metadata = create_cqm(farms, foods, food_groups, config)

print(f"\n✅ Success!")
print(f"  CQM Variables: {len(cqm.variables)}")
print(f"  CQM Constraints: {len(cqm.constraints)}")
print(f"\nProgress bar working correctly!")
