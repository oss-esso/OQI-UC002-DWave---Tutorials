import sys
import os
import time
import pulp as pl
import numpy as np
from typing import Dict, List, Tuple, Any

farms = ['Farm1', 'Farm2']
crops = ['Wheat', 'Corn', 'Soy', 'Tomato']

food_groups = {
    'Grains': ['Wheat', 'Corn'],
    'Legumes': ['Soy'],
    'Vegetables': ['Tomato']
}

N = {'Wheat': 0.7, 'Corn': 0.9, 'Soy': 0.5, 'Tomato': 0.8}
D = {'Wheat': 0.6, 'Corn': 0.85, 'Soy': 0.55, 'Tomato': 0.9}
E = {'Wheat': 0.4, 'Corn': 0.3, 'Soy': 0.5, 'Tomato': 0.2}
P = {'Wheat': 0.7, 'Corn': 0.5, 'Soy': 0.6, 'Tomato': 0.9}

L = {'Farm1': 100, 'Farm2': 150}
A_min = {'Wheat': 5, 'Corn': 4, 'Soy': 3, 'Tomato': 2}

FG_min = {'Grains': 1, 'Legumes': 1, 'Vegetables': 1}
FG_max = {'Grains': 2, 'Legumes': 1, 'Vegetables': 1}

weights = {'w_1': 0.25, 'w_2': 0.25, 'w_3': 0.25, 'w_4': 0.25}


A = pl.LpVariable.dicts("Area", [(f, c) for f in farms for c in crops], lowBound=0)
Y = pl.LpVariable.dicts("Choose", [(f, c) for f in farms for c in crops], cat='Binary')

total_area = pl.lpSum(L[f] for f in farms)

goal = (
    weights['w_1'] * pl.lpSum([(N[c] * A[(f, c)]) for f in farms for c in crops]) / total_area +
    weights['w_2'] * pl.lpSum([(D[c] * A[(f, c)]) for f in farms for c in crops]) / total_area -
    weights['w_3'] * pl.lpSum([(E[c] * A[(f, c)]) for f in farms for c in crops]) / total_area +
    weights['w_4'] * pl.lpSum([(P[c] * A[(f, c)]) for f in farms for c in crops]) / total_area
)


# Constraints

model = pl.LpProblem("Crop_Allocation_Optimization", pl.LpMaximize)

for f in farms:
    model += pl.lpSum([A[(f, c)] for c in crops]) <= L[f], f"Max_Area_{f}"

for g, crops_in_group in food_groups.items():
    for f in farms:
        model += pl.lpSum([Y[(f, c)] for c in crops_in_group]) >= FG_min[g], f"MinFoodGroup_{f}_{g}"
        model += pl.lpSum([Y[(f, c)] for c in crops_in_group]) <= FG_max[g], f"MaxFoodGroup_{f}_{g}"


model += goal, "Objective"

print("=" * 80)
print("MODEL CONSTRAINTS:")
print("=" * 80)
for name, constraint in model.constraints.items():
    print(f"{name}: {constraint}")

print("\n" + "=" * 80)
print("SOLVING...")
print("=" * 80)

start_time = time.time()
model.solve(pl.PULP_CBC_CMD(msg=1))
end_time = time.time()

print("\n" + "=" * 80)
print("SOLUTION:")
print("=" * 80)
print(f"Status: {pl.LpStatus[model.status]}")
print(f"\nArea Variables (A):")
for f in farms:
    print(f"\n{f}:")
    for c in crops:
        if A[(f,c)].value() is not None:
            print(f"  {c}: Area = {A[(f,c)].value():.2f}")

print(f"\nBinary Selection Variables (Y):")
for f in farms:
    print(f"\n{f}:")
    for c in crops:
        if Y[(f,c)].value() is not None:
            print(f"  {c}: Selected = {Y[(f,c)].value()}")

print("\n" + "=" * 80)
print("FOOD GROUP VERIFICATION:")
print("=" * 80)
for f in farms:
    print(f"\n{f}:")
    for g, crops_in_group in food_groups.items():
        selected_count = sum(Y[(f, c)].value() or 0 for c in crops_in_group)
        print(f"  {g}: {selected_count} selected (min={FG_min[g]}, max={FG_max[g]})")
        for c in crops_in_group:
            if Y[(f, c)].value():
                print(f"    - {c} (Y={Y[(f,c)].value()})")

print(f"\nTotal Time: {end_time - start_time:.2f} seconds")
print(f"Objective Value: {pl.value(model.objective):.4f}")
