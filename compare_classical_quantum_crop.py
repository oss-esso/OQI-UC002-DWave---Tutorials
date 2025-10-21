"""
Crop Allocation Problem: Comparing Classical vs Quantum Solvers

This script compares:
1. PuLP (Classical MILP solver)
2. Simplified QUBO version with Grover Adaptive Search
3. Performance analysis

The original problem is a Mixed Integer Linear Programming problem with:
- Binary variables Y (crop selection)
- Continuous variables A (area allocation)

For quantum comparison, we simplify to a pure binary optimization:
- Focus on the binary crop selection problem
- Approximate the continuous area allocation
"""

import sys
import time
import pulp as pl
import numpy as np
from typing import Dict, List, Tuple
from gas_for_qubo_improved import ImprovedGroverAdaptiveSearchSolver

print("="*80)
print("CROP ALLOCATION: CLASSICAL vs QUANTUM COMPARISON")
print("="*80)

# Problem data
farms = ['Farm1', 'Farm2']
crops = ['Wheat', 'Corn', 'Soy', 'Tomato']

food_groups = {
    'Grains': ['Wheat', 'Corn'],
    'Legumes': ['Soy'],
    'Vegetables': ['Tomato']
}

# Nutritional and environmental scores
N = {'Wheat': 0.7, 'Corn': 0.9, 'Soy': 0.5, 'Tomato': 0.8}
D = {'Wheat': 0.6, 'Corn': 0.85, 'Soy': 0.55, 'Tomato': 0.9}
E = {'Wheat': 0.4, 'Corn': 0.3, 'Soy': 0.5, 'Tomato': 0.2}
P = {'Wheat': 0.7, 'Corn': 0.5, 'Soy': 0.6, 'Tomato': 0.9}

L = {'Farm1': 100, 'Farm2': 150}  # Land availability
A_min = {'Wheat': 5, 'Corn': 4, 'Soy': 3, 'Tomato': 2}  # Minimum area if selected

FG_min = {'Grains': 1, 'Legumes': 1, 'Vegetables': 1}  # Min crops per food group
FG_max = {'Grains': 2, 'Legumes': 1, 'Vegetables': 1}  # Max crops per food group

weights = {'w_1': 0.25, 'w_2': 0.25, 'w_3': 0.25, 'w_4': 0.25}

# ============================================================================
# PART 1: CLASSICAL MILP SOLVER (PuLP with CBC)
# ============================================================================

print("\n" + "="*80)
print("PART 1: CLASSICAL MILP SOLVER (PuLP)")
print("="*80)

start_classical = time.time()

# Define variables
A = pl.LpVariable.dicts("Area", [(f, c) for f in farms for c in crops], lowBound=0)
Y = pl.LpVariable.dicts("Choose", [(f, c) for f in farms for c in crops], cat='Binary')

total_area = pl.lpSum(L[f] for f in farms)

# Objective function
goal = (
    weights['w_1'] * pl.lpSum([(N[c] * A[(f, c)]) for f in farms for c in crops]) / total_area +
    weights['w_2'] * pl.lpSum([(D[c] * A[(f, c)]) for f in farms for c in crops]) / total_area -
    weights['w_3'] * pl.lpSum([(E[c] * A[(f, c)]) for f in farms for c in crops]) / total_area +
    weights['w_4'] * pl.lpSum([(P[c] * A[(f, c)]) for f in farms for c in crops]) / total_area
)

model = pl.LpProblem("Crop_Allocation_Optimization", pl.LpMaximize)

# Constraints
for f in farms:
    model += pl.lpSum([A[(f, c)] for c in crops]) <= L[f], f"Max_Area_{f}"

for f in farms:
    for c in crops:
        model += A[(f, c)] >= A_min[c] * Y[(f, c)], f"MinArea_{f}_{c}"
        model += A[(f, c)] <= L[f] * Y[(f, c)], f"MaxArea_{f}_{c}"

for g, crops_group in food_groups.items():
    for f in farms:
        model += pl.lpSum([Y[(f, c)] for c in crops_group]) >= FG_min[g], f"MinFoodGroup_{f}_{g}"
        model += pl.lpSum([Y[(f, c)] for c in crops_group]) <= FG_max[g], f"MaxFoodGroup_{f}_{g}"

model += goal, "Objective"

# Solve
model.solve(pl.PULP_CBC_CMD(msg=0))

time_classical = time.time() - start_classical

print(f"\nStatus: {pl.LpStatus[model.status]}")
print(f"Objective Value: {pl.value(model.objective):.6f}")
print(f"Solution Time: {time_classical:.4f} seconds")

print("\nCrop Selection (Binary Y):")
classical_selection = {}
for f in farms:
    print(f"\n{f}:")
    for c in crops:
        y_val = Y[(f, c)].value() if Y[(f, c)].value() is not None else 0.0
        a_val = A[(f, c)].value() if A[(f, c)].value() is not None else 0.0
        classical_selection[(f, c)] = (int(y_val > 0.5), a_val)
        if y_val > 0.5:
            print(f"  {c}: Selected (Area = {a_val:.2f})")

classical_objective = pl.value(model.objective)

# ============================================================================
# PART 2: SIMPLIFIED QUBO FORMULATION
# ============================================================================

print("\n" + "="*80)
print("PART 2: QUANTUM SOLVER (Grover Adaptive Search)")
print("="*80)

print("\nCreating QUBO formulation...")
print("Note: This is a simplified version focusing on crop selection")
print("      Area allocation is approximated using A_min values")

# Create binary variable mapping
# We have 2 farms × 4 crops = 8 binary variables
n_vars = len(farms) * len(crops)
var_to_idx = {}
idx_to_var = {}
idx = 0
for f in farms:
    for c in crops:
        var_to_idx[(f, c)] = idx
        idx_to_var[idx] = (f, c)
        idx += 1

print(f"\nNumber of binary variables: {n_vars}")
print(f"Total possible states: 2^{n_vars} = {2**n_vars}")

# Build QUBO matrix
# For maximization, we need to convert to minimization: min(-objective)
Q = np.zeros((n_vars, n_vars))

# Calculate total area as a number (not PuLP expression)
total_area_value = sum(L[f] for f in farms)

# Simplified objective: Use A_min as proxy for area when crop is selected
# Maximize: w1*N + w2*D - w3*E + w4*P (per selected crop)
for i in range(n_vars):
    f, c = idx_to_var[i]
    # Reward for selecting this crop (negative for minimization)
    score = (weights['w_1'] * N[c] * A_min[c] +
             weights['w_2'] * D[c] * A_min[c] -
             weights['w_3'] * E[c] * A_min[c] +
             weights['w_4'] * P[c] * A_min[c]) / total_area_value
    Q[i, i] = -score  # Negative because we minimize

# Add penalties for constraint violations
PENALTY = 100.0  # Large penalty for constraint violations (increased from 10)

# Constraint 1: Food group constraints
# Each farm must select crops within min/max bounds per food group
for f in farms:
    for g, crops_group in food_groups.items():
        group_indices = [var_to_idx[(f, c)] for c in crops_group]
        
        # Penalty for too few crops: (sum - min)^2 when sum < min
        # This becomes: sum^2 - 2*min*sum + min^2
        # Quadratic terms
        for i in group_indices:
            for j in group_indices:
                if i <= j:
                    Q[i, j] += PENALTY  # Coefficient of x_i*x_j in sum^2
        
        # Linear terms (diagonal adjustment)
        for i in group_indices:
            Q[i, i] -= 2 * PENALTY * FG_min[g]
        
        # Penalty for too many crops
        # Similar formulation but with max bound
        for i in group_indices:
            for j in group_indices:
                if i <= j:
                    Q[i, j] += PENALTY
        
        for i in group_indices:
            Q[i, i] += 2 * PENALTY * FG_max[g]

# Constraint 2: Land availability (simplified)
# Sum of A_min for selected crops should not exceed land
for f in farms:
    farm_indices = [var_to_idx[(f, c)] for c in crops]
    
    # If sum of A_min > L, add penalty
    for i in farm_indices:
        for j in farm_indices:
            if i <= j:
                _, ci = idx_to_var[i]
                _, cj = idx_to_var[j]
                Q[i, j] += PENALTY * A_min[ci] * A_min[cj] / (L[f] ** 2)

print("\nQUBO Matrix shape:", Q.shape)
print("QUBO Matrix (first 4x4 block):")
print(Q[:4, :4])

# ============================================================================
# PART 3: SOLVE WITH GROVER ADAPTIVE SEARCH
# ============================================================================

print("\n" + "-"*80)
print("Solving with Grover Adaptive Search...")
print("-"*80)

start_quantum = time.time()

solver = ImprovedGroverAdaptiveSearchSolver(Q)
quantum_solution, quantum_cost = solver.solve(
    max_iterations=15,
    num_restarts=3,
    repetitions=2000,
    verbose=False  # Set to True to see detailed quantum algorithm progress
)

time_quantum = time.time() - start_quantum

# Convert quantum solution back to crop selection
quantum_selection = {}
print("\nQuantum Solution - Crop Selection:")
for i, val in enumerate(quantum_solution):
    f, c = idx_to_var[i]
    quantum_selection[(f, c)] = int(val)

for f in farms:
    print(f"\n{f}:")
    for c in crops:
        if quantum_selection[(f, c)] == 1:
            # Approximate area using A_min
            approx_area = A_min[c]
            print(f"  {c}: Selected (Approx Area = {approx_area:.2f})")

# Calculate approximate objective for quantum solution
quantum_objective_approx = 0
for f in farms:
    for c in crops:
        if quantum_selection[(f, c)] == 1:
            score = (weights['w_1'] * N[c] * A_min[c] +
                    weights['w_2'] * D[c] * A_min[c] -
                    weights['w_3'] * E[c] * A_min[c] +
                    weights['w_4'] * P[c] * A_min[c]) / total_area_value
            quantum_objective_approx += score

print(f"\nQuantum QUBO Cost: {quantum_cost:.6f}")
print(f"Approximate Objective: {quantum_objective_approx:.6f}")
print(f"Solution Time: {time_quantum:.4f} seconds")

# ============================================================================
# PART 4: VERIFY CONSTRAINTS FOR QUANTUM SOLUTION
# ============================================================================

print("\n" + "="*80)
print("CONSTRAINT VERIFICATION (Quantum Solution)")
print("="*80)

constraint_violations = 0

# Check food group constraints
print("\nFood Group Constraints:")
for f in farms:
    print(f"\n{f}:")
    for g, crops_group in food_groups.items():
        count = sum(quantum_selection[(f, c)] for c in crops_group)
        selected = [c for c in crops_group if quantum_selection[(f, c)] == 1]
        
        if FG_min[g] <= count <= FG_max[g]:
            status = "✅ SATISFIED"
        else:
            status = "❌ VIOLATED"
            constraint_violations += 1
        
        print(f"  {g}: {count} selected (range: {FG_min[g]}-{FG_max[g]}) {status}")
        print(f"    Selected: {selected}")

# Check land constraints (approximate)
print("\nLand Constraints (Approximate using A_min):")
for f in farms:
    total_land_used = sum(A_min[c] * quantum_selection[(f, c)] for c in crops)
    
    if total_land_used <= L[f]:
        status = "✅ SATISFIED"
    else:
        status = "❌ VIOLATED"
        constraint_violations += 1
    
    print(f"  {f}: {total_land_used:.2f} / {L[f]} hectares {status}")

# ============================================================================
# PART 5: COMPARISON SUMMARY
# ============================================================================

print("\n" + "="*80)
print("COMPARISON SUMMARY")
print("="*80)

print(f"\n{'Metric':<30} {'Classical (PuLP)':<20} {'Quantum (GAS)':<20}")
print("-" * 70)
print(f"{'Solution Time':<30} {time_classical:>18.4f}s {time_quantum:>18.4f}s")
print(f"{'Objective Value':<30} {classical_objective:>18.6f} {quantum_objective_approx:>18.6f}")
print(f"{'Constraint Violations':<30} {'0':<20} {constraint_violations:<20}")
print(f"{'Solution Status':<30} {'Optimal':<20} {'Feasible' if constraint_violations == 0 else 'Infeasible':<20}")

# Calculate speedup/slowdown
if time_classical > 0:
    speedup = time_classical / time_quantum
    if speedup > 1:
        print(f"\n⏱️  Quantum was {speedup:.2f}x FASTER than classical")
    else:
        print(f"\n⏱️  Classical was {1/speedup:.2f}x FASTER than quantum")

# Calculate objective gap
obj_gap = abs(classical_objective - quantum_objective_approx) / abs(classical_objective) * 100
print(f"📊 Objective gap: {obj_gap:.2f}%")

print("\n" + "="*80)
print("ANALYSIS")
print("="*80)

print("""
This comparison shows:

1. **Problem Complexity**: 
   - Classical: Solves full MILP with continuous + binary variables
   - Quantum: Solves simplified binary-only QUBO approximation

2. **Solution Quality**:
   - Classical: Finds true optimal with continuous area optimization
   - Quantum: Finds good binary selection, but area allocation is approximate

3. **Constraint Handling**:
   - Classical: Exact constraint satisfaction guaranteed
   - Quantum: Constraint violations penalized in objective (soft constraints)

4. **Scalability**:
   - Classical: Efficient for small-medium MILP problems
   - Quantum: Current implementation limited by oracle enumeration

5. **Best Use Cases**:
   - Classical: When you need exact MILP solutions with continuous variables
   - Quantum: When pure binary decisions dominate and approximate solutions acceptable
""")

# ============================================================================
# PART 6: CLASSICAL SOLVER FOR PURE BINARY PROBLEM (FAIR COMPARISON)
# ============================================================================

print("\n" + "="*80)
print("BONUS: CLASSICAL SOLVER ON SAME BINARY QUBO")
print("="*80)

print("\nSolving the QUBO problem with classical brute force...")

start_classical_qubo = time.time()
classical_qubo_solution, classical_qubo_cost = solver.classical_solve()
time_classical_qubo = time.time() - start_classical_qubo

print(f"Classical QUBO Solution Cost: {classical_qubo_cost:.6f}")
print(f"Classical QUBO Solution Time: {time_classical_qubo:.4f} seconds")
print(f"Quantum QUBO Solution Cost: {quantum_cost:.6f}")
print(f"Quantum QUBO Solution Time: {time_quantum:.4f} seconds")

if quantum_cost == classical_qubo_cost:
    print("\n✅ Quantum found the OPTIMAL solution to the QUBO!")
else:
    gap = abs(quantum_cost - classical_qubo_cost) / abs(classical_qubo_cost) * 100
    print(f"\n⚠️  Quantum solution is {gap:.2f}% from QUBO optimal")

print("\n" + "="*80)
print("COMPARISON COMPLETE")
print("="*80)
