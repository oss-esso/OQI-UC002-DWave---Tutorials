# Understanding BQM Constraints and BQM vs CQM

## Why Constraints ARE Added (But It's Subtle)

In Tutorial 1, Example 3, constraints **are being added** to the BQM. The confusion arises because constraints in BQM are added as **penalty terms** that modify the existing objective coefficients, not as separate entities.

### How BQM Constraints Work

When you add a constraint penalty to a BQM, you're actually modifying the energy function by adding terms:

```python
# Original objective:
Energy = 2.0*t1_p1 + 3.0*t1_p2 + ... (task costs)

# Constraint: t1_p1 + t1_p2 = 1
# Penalty form: P * (t1_p1 + t1_p2 - 1)^2
# Expanded: P * (-t1_p1 - t1_p2 + 2*t1_p1*t1_p2 + 1)

# Combined energy:
Energy = (2.0 - 10.0)*t1_p1 + (3.0 - 10.0)*t1_p2 + 20.0*t1_p1*t1_p2 + 10.0
       = -8.0*t1_p1 + -7.0*t1_p2 + 20.0*t1_p1*t1_p2 + 10.0
```

### Evidence from Tutorial Output

```
BQM BEFORE adding constraints:
  Linear terms sample: t1_p1=2.0, t1_p2=3.0
  Quadratic terms: 6
  Offset: 0.0

BQM AFTER adding constraints:
  Linear terms sample: t1_p1=-8.0, t1_p2=-7.0  ← Changed!
  Quadratic terms: 9 (3 new constraint interactions)  ← Added!
  Offset: 30.0 (3 tasks x penalty = 30)  ← Changed!
```

**The constraints are definitely being added** - you can see:
1. Linear coefficients changed (objective + penalty)
2. New quadratic terms added (constraint interactions)
3. Offset updated (constant penalty term)

## BQM vs CQM: Key Differences

### BQM with Penalties (Soft Constraints)

**What it is:**
- Constraints encoded as penalty terms in the objective function
- Violations are penalized but not prevented

**How it works:**
```python
# Constraint: x + y = 1
# Becomes: objective + penalty_weight * (x + y - 1)^2

bqm.add_variable('x', objective_cost - penalty_weight)
bqm.add_interaction('x', 'y', 2 * penalty_weight)
```

**Pros:**
- ✓ Works with all solvers (SimulatedAnnealing, QPU, Hybrid)
- ✓ Simple to implement
- ✓ Flexible (can trade off constraints vs objective)
- ✓ Always finds some solution

**Cons:**
- ✗ Constraints can be violated if penalty too small
- ✗ Must tune penalty weights carefully
- ✗ Adds extra terms (larger problem)
- ✗ No guarantee of feasibility

**When to use:**
- Development/testing with simulator
- Using QPU directly
- Soft constraints acceptable
- Need flexibility in constraint satisfaction

### CQM (Hard Constraints)

**What it is:**
- Constrained Quadratic Model with built-in constraint support
- Constraints are guaranteed to be satisfied

**How it works:**
```python
from dimod import ConstrainedQuadraticModel, Binary

cqm = ConstrainedQuadraticModel()
x = Binary('x')
y = Binary('y')

# Objective (no penalties needed)
cqm.set_objective(2.0*x + 3.0*y)

# Constraint (hard, always satisfied)
cqm.add_constraint(x + y == 1, label='assignment')
```

**Pros:**
- ✓ Constraints ALWAYS satisfied (hard constraints)
- ✓ No penalty tuning needed
- ✓ Cleaner formulation (objective separate from constraints)
- ✓ Better for complex constraints (inequalities, etc.)

**Cons:**
- ✗ Requires CQM-capable solver (Hybrid CQM only)
- ✗ Not available on all D-Wave systems
- ✗ May return "infeasible" if constraints impossible
- ✗ More complex to set up

**When to use:**
- Constraints MUST be satisfied
- Complex constraints (inequalities, multiple types)
- Production systems with Hybrid solver
- Don't want to tune penalty weights

## Comparison Table

```
+---------------------+-----------------------+-----------------------+
| Aspect              | BQM (Penalties)       | CQM (Hard Constraints)|
+---------------------+-----------------------+-----------------------+
| Constraint Type     | Soft (may violate)    | Hard (always met)     |
| Penalty Tuning      | Required              | Not needed            |
| Solver Availability | All (SA, QPU, Hybrid) | Hybrid CQM only       |
| Feasibility         | Always finds solution | May be infeasible     |
| Model Size          | Larger (added terms)  | Smaller (cleaner)     |
| Use Case            | Flexible, trade-offs  | Must satisfy rules    |
+---------------------+-----------------------+-----------------------+
```

## Penalty Weight Tuning

The penalty weight is critical in BQM formulations:

### Too Small (P = 1):
```
Energy = 2.0*t1_p1 + 3.0*t1_p2 - 1.0*(t1_p1 + t1_p2 - 1)^2
```
- Constraint violations only cost 1-4 energy units
- Solver might violate to reduce objective
- Result: May assign task to 0 or 2 processors

### Balanced (P = 10):
```
Energy = 2.0*t1_p1 + 3.0*t1_p2 - 10.0*(t1_p1 + t1_p2 - 1)^2
```
- Constraint violations cost 10-40 energy units
- Balances objective and constraints
- Result: Usually satisfies constraints

### Too Large (P = 1000):
```
Energy = 2.0*t1_p1 + 3.0*t1_p2 - 1000.0*(t1_p1 + t1_p2 - 1)^2
```
- Constraint violations cost 1000-4000 energy units
- Solver ignores objective, just satisfies constraints
- Result: Any feasible solution, possibly high cost

## Mathematical Details

### Constraint to Penalty Conversion

For constraint: $\sum_i x_i = k$

Penalty form: $P \cdot (\sum_i x_i - k)^2$

Expanding:
$$P \cdot (\sum_i x_i - k)^2 = P \cdot [\sum_i x_i^2 + 2\sum_{i<j} x_i x_j - 2k\sum_i x_i + k^2]$$

For binary variables ($x_i^2 = x_i$):
$$= P \cdot [\sum_i x_i + 2\sum_{i<j} x_i x_j - 2k\sum_i x_i + k^2]$$
$$= P \cdot [(1-2k)\sum_i x_i + 2\sum_{i<j} x_i x_j + k^2]$$

This adds:
- **Linear terms**: $P(1-2k)$ to each $x_i$
- **Quadratic terms**: $2P$ for each pair $(x_i, x_j)$
- **Offset**: $Pk^2$

## Example: Task Assignment

For 3 tasks, 2 processors, constraint that each task goes to exactly one processor:

**BQM Formulation:**
```python
# Objective
for task, processor in assignments:
    bqm.add_variable(f"{task}_{processor}", cost[task][processor])

# Constraint: t1_p1 + t1_p2 = 1 (penalty = 10)
bqm.add_variable('t1_p1', -10)     # Adds to existing 2.0 → -8.0
bqm.add_variable('t1_p2', -10)     # Adds to existing 3.0 → -7.0
bqm.add_interaction('t1_p1', 't1_p2', 20)  # New term
bqm.offset += 10
```

**CQM Formulation:**
```python
# Objective
cqm.set_objective(
    sum(cost[t][p] * var[f"{t}_{p}"] 
        for t in tasks for p in processors)
)

# Constraint
cqm.add_constraint(
    var['t1_p1'] + var['t1_p2'] == 1,
    label='t1_assignment'
)
```

## Summary

- **BQM constraints ARE added** - they modify the energy function through penalty terms
- **BQM = Flexible** - soft constraints, works with all solvers, needs tuning
- **CQM = Strict** - hard constraints, cleaner formulation, limited solver support
- **Choose BQM** for development, QPU access, or soft constraint problems
- **Choose CQM** for production systems where constraints must be satisfied

See Tutorial 1, Examples 3 and 5 for working code examples.
