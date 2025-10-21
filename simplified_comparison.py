"""
Simplified Crop Allocation - Classical vs Quantum 
A cleaner comparison focusing on a simpler binary selection problem
"""

import time
import pulp as pl
import numpy as np
from gas_for_qubo_improved import ImprovedGroverAdaptiveSearchSolver

print("="*80)
print("SIMPLIFIED CROP SELECTION: CLASSICAL vs QUANTUM")
print("="*80)

# Simplified problem: Select best crops to maximize score
# Each crop has a score, and we want to select exactly K crops
crops = ['Wheat', 'Corn', 'Soy', 'Tomato', 'Rice', 'Beans']
scores = {
    'Wheat': 5.0,
    'Corn': 7.0,
    'Soy': 4.0,
    'Tomato': 8.0,
    'Rice': 6.0,
    'Beans': 3.0
}

K = 3  # Must select exactly 3 crops

print(f"\nProblem: Select exactly {K} crops to maximize total score")
print("\nCrop Scores:")
for crop in crops:
    print(f"  {crop}: {scores[crop]}")

# ============================================================================
# PART 1: CLASSICAL SOLVER (PuLP)
# ============================================================================

print("\n" + "="*80)
print("CLASSICAL SOLVER (PuLP)")
print("="*80)

start_classical = time.time()

# Variables
x = pl.LpVariable.dicts("select", crops, cat='Binary')

# Model
model = pl.LpProblem("Crop_Selection", pl.LpMaximize)

# Objective: Maximize total score
model += pl.lpSum([scores[c] * x[c] for c in crops])

# Constraint: Select exactly K crops
model += pl.lpSum([x[c] for c in crops]) == K

# Solve
model.solve(pl.PULP_CBC_CMD(msg=0))

time_classical = time.time() - start_classical

print(f"\nStatus: {pl.LpStatus[model.status]}")
print(f"Objective Value: {pl.value(model.objective):.2f}")
print(f"Solution Time: {time_classical:.6f} seconds")

print("\nSelected Crops:")
classical_selection = []
for c in crops:
    if x[c].value() and x[c].value() > 0.5:
        classical_selection.append(c)
        print(f"  ✓ {c} (score: {scores[c]})")

classical_objective = pl.value(model.objective)

# ============================================================================
# PART 2: QUANTUM SOLVER (QUBO + Grover Adaptive Search)
# ============================================================================

print("\n" + "="*80)
print("QUANTUM SOLVER (Grover Adaptive Search)")
print("="*80)

# Create QUBO formulation
n = len(crops)
Q = np.zeros((n, n))

# Map crops to indices
crop_to_idx = {crop: i for i, crop in enumerate(crops)}

# Objective: maximize sum of scores = minimize -sum of scores
for i, crop in enumerate(crops):
    Q[i, i] = -scores[crop]  # Negative for minimization

# Constraint: sum of x_i should equal K
# Penalty for (sum x_i - K)^2
# Expanding: sum(x_i)^2 - 2K*sum(x_i) + K^2
# sum(x_i)^2 = sum(x_i*x_j) for all i,j

PENALTY = 20.0  # Penalty weight

# Quadratic penalty terms
for i in range(n):
    for j in range(n):
        if i < j:
            Q[i, j] += 2 * PENALTY  # Coefficient for x_i * x_j
        elif i == j:
            Q[i, i] += PENALTY  # Coefficient for x_i^2 = x_i (since binary)

# Linear penalty terms
for i in range(n):
    Q[i, i] -= 2 * PENALTY * K

print(f"\nProblem size: {n} variables, 2^{n} = {2**n} possible states")
print(f"QUBO matrix shape: {Q.shape}")

print("\nSolving with Grover Adaptive Search...")

start_quantum = time.time()

solver = ImprovedGroverAdaptiveSearchSolver(Q)
solution, cost = solver.solve(
    max_iterations=20,
    num_restarts=5,
    repetitions=3000,
    verbose=False
)

time_quantum = time.time() - start_quantum

print(f"\nQUBO Cost: {cost:.6f}")
print(f"Solution Time: {time_quantum:.6f} seconds")

# Decode solution
quantum_selection = []
quantum_score = 0
print("\nSelected Crops:")
for i, crop in enumerate(crops):
    if solution[i] == 1:
        quantum_selection.append(crop)
        quantum_score += scores[crop]
        print(f"  ✓ {crop} (score: {scores[crop]})")

num_selected = sum(solution)
print(f"\nNumber of crops selected: {num_selected}")
print(f"Total score: {quantum_score:.2f}")

# ============================================================================
# PART 3: VERIFY AND COMPARE
# ============================================================================

print("\n" + "="*80)
print("VERIFICATION")
print("="*80)

# Check constraint satisfaction
constraint_satisfied = (num_selected == K)
print(f"\nConstraint (select exactly {K}): {num_selected} {'✅ SATISFIED' if constraint_satisfied else '❌ VIOLATED'}")

# ============================================================================
# PART 4: COMPARISON SUMMARY
# ============================================================================

print("\n" + "="*80)
print("COMPARISON SUMMARY")
print("="*80)

print(f"\n{'Metric':<30} {'Classical':<20} {'Quantum':<20}")
print("-" * 70)
print(f"{'Solution Time (seconds)':<30} {time_classical:>18.6f}  {time_quantum:>18.6f}")
print(f"{'Objective Value':<30} {classical_objective:>18.2f}  {quantum_score:>18.2f}")
print(f"{'Number of Crops Selected':<30} {len(classical_selection):>18}  {num_selected:>18}")
print(f"{'Constraint Satisfied':<30} {'Yes':<20} {'Yes' if constraint_satisfied else 'No':<20}")

if time_classical > 0 and time_quantum > 0:
    if time_classical < time_quantum:
        speedup = time_quantum / time_classical
        print(f"\n⏱️  Classical was {speedup:.2f}x faster")
    else:
        speedup = time_classical / time_quantum
        print(f"\n⏱️  Quantum was {speedup:.2f}x faster")

if classical_objective == quantum_score:
    print("🎯 Quantum found the OPTIMAL solution!")
else:
    gap = abs(classical_objective - quantum_score) / classical_objective * 100
    print(f"📊 Solution quality gap: {gap:.2f}%")

print("\nClassical solution:", sorted(classical_selection))
print("Quantum solution:  ", sorted(quantum_selection))

# ============================================================================
# PART 5: CLASSICAL BRUTE FORCE ON SAME QUBO
# ============================================================================

print("\n" + "="*80)
print("CLASSICAL BRUTE FORCE ON SAME QUBO")
print("="*80)

start_brute = time.time()
classical_qubo_solution, classical_qubo_cost = solver.classical_solve()
time_brute = time.time() - start_brute

classical_qubo_score = 0
classical_qubo_selection = []
for i, crop in enumerate(crops):
    if classical_qubo_solution[i] == 1:
        classical_qubo_selection.append(crop)
        classical_qubo_score += scores[crop]

print(f"\nClassical QUBO solution cost: {classical_qubo_cost:.6f}")
print(f"Classical QUBO solution time: {time_brute:.6f} seconds")
print(f"Classical QUBO selected: {sorted(classical_qubo_selection)}")
print(f"Classical QUBO score: {classical_qubo_score:.2f}")

print(f"\nQuantum QUBO cost: {cost:.6f}")
print(f"Quantum solution: {sorted(quantum_selection)}")
print(f"Quantum score: {quantum_score:.2f}")

if cost == classical_qubo_cost:
    print("\n✅ Quantum found the OPTIMAL QUBO solution!")
else:
    print(f"\n⚠️  Quantum solution differs from optimal QUBO")
    print(f"   Cost gap: {abs(cost - classical_qubo_cost):.6f}")

# ============================================================================
# ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("ANALYSIS")
print("="*80)

print(f"""
Problem Characteristics:
- Problem type: Constrained binary optimization
- Variables: {n} binary decision variables
- Search space: {2**n} possible combinations
- Valid solutions: {np.math.comb(n, K)} (combinations of {n} choose {K})

Classical Solver (PuLP + CBC):
- Method: Branch-and-bound MILP solver
- Time: {time_classical:.6f} seconds
- Guarantee: Finds provably optimal solution
- Constraint handling: Exact

Quantum Solver (Grover Adaptive Search):
- Method: Quantum amplitude amplification
- Time: {time_quantum:.6f} seconds
- Guarantee: Probabilistic, may require multiple runs
- Constraint handling: Penalty-based (soft constraints)

Key Observations:
1. For this small problem ({n} variables), classical is faster
2. Quantum advantage expected for larger problems (n > 20)
3. Both methods found {'the same' if sorted(classical_selection) == sorted(quantum_selection) else 'different'} solutions
4. Constraint satisfaction: {'Both perfect' if constraint_satisfied and len(classical_selection) == K else 'Classical perfect, Quantum ' + ('perfect' if constraint_satisfied else 'violated')}
""")

print("="*80)
print("TEST COMPLETE")
print("="*80)
