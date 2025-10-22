# MINLP Fractional Programming Implementation

## Overview

This project implements three different approaches to solve a **Mixed-Integer Nonlinear Program (MINLP)** with a fractional objective function for food crop optimization.

### Problem Transformation

**Original MILP (Linear):**
```
max (weighted_sum_of_objectives * A_{f,c}) / total_area
```
Where `total_area` is a **constant** (sum of all available land).

**New MINLP (Nonlinear):**
```
max (weighted_sum_of_objectives * A_{f,c}) / sum(A_{f,c})
```
Where `sum(A_{f,c})` are **decision variables** (allocated areas).

This creates a **fractional programming problem** where the objective is a ratio of two functions.

---

## Three Solution Approaches

### 1. PuLP with Dinkelbach's Algorithm ✓

**Method:** Iterative linearization  
**Solver:** Classical LP solver (CBC)  
**Convergence:** Proven for concave fractional programs

#### Algorithm Steps:
```python
1. Initialize: λ_k = 0
2. Solve LP: max f(x) - λ_k * g(x)
3. Update: λ_k = f(x*) / g(x*)
4. Check convergence: |f(x*) - λ_k * g(x*)| < ε
5. Repeat until converged
```

#### Advantages:
- ✓ Uses existing PuLP infrastructure
- ✓ Fast convergence (typically 5-15 iterations)
- ✓ Guaranteed convergence for concave fractional programs
- ✓ No additional solvers required

#### Usage:
```python
from solver_runner_NLD import solve_with_pulp

farms, foods, food_groups, config = load_food_data('simple')
model, results = solve_with_pulp(farms, foods, food_groups, config)

print(f"Objective: {results['objective_value']}")
print(f"Iterations: {results['iterations']}")
print(f"Convergence history: {results['convergence_history']}")
```

---

### 2. Pyomo with Direct MINLP Solvers ✓

**Method:** Direct optimization  
**Solvers:** Ipopt, BARON, Couenne, SCIP  
**Convergence:** Depends on solver and problem structure

#### Supported Solvers:
- **Ipopt**: Interior Point Optimizer (free, good for convex problems)
- **BARON**: Global solver (commercial, handles non-convex)
- **Couenne**: Open-source global solver
- **SCIP**: Open-source solver (LP/MILP/MINLP)

#### Advantages:
- ✓ Direct modeling of fractional objective
- ✓ Handles general nonlinear constraints
- ✓ Can find global optima (with BARON/Couenne)
- ✓ Standard MINLP framework

#### Installation:
```bash
# Install Pyomo
conda install -c conda-forge pyomo

# Install Ipopt (recommended)
conda install -c conda-forge ipopt

# Or install BARON/Couenne for global optimization
conda install -c conda-forge couenne
```

#### Usage:
```python
from solver_runner_NLD import solve_with_pyomo

farms, foods, food_groups, config = load_food_data('simple')
model, results = solve_with_pyomo(farms, foods, food_groups, config)

print(f"Objective: {results['objective_value']}")
print(f"Solver used: {results['solver']}")
```

---

### 3. D-Wave with Charnes-Cooper Transformation ⚠

**Method:** Variable substitution + normalization  
**Solver:** D-Wave Hybrid CQM Sampler  
**Convergence:** Heuristic (no guarantee)

#### Transformation:
For fractional program `max f(x)/g(x)`:
```
Introduce: z = x/g(x), t = 1/g(x)
Reformulate: max t*f(z/t) s.t. t*g(z/t) = 1, t > 0
```

For linear f,g, this becomes:
```
max f(z) s.t. g(z) = 1, t > 0, z = t*x
```

#### Implementation Notes:
- Uses normalization constraint: `sum(A) = constant`
- Approximates fractional objective for CQM compatibility
- Requires D-Wave API access

#### Advantages:
- ✓ Leverages quantum annealing
- ✓ Can handle large-scale problems
- ⚠ Approximate solution (not exact)

#### Usage:
```python
from solver_runner_NLD import solve_with_dwave_charnes_cooper

token = os.getenv('DWAVE_API_TOKEN')
farms, foods, food_groups, config = load_food_data('simple')
sampleset, results = solve_with_dwave_charnes_cooper(
    farms, foods, food_groups, config, token
)

print(f"Objective: {results['objective_value']}")
print(f"Feasible solutions: {results['feasible_count']}")
```

---

## Quick Start

### 1. Test All Solvers

```bash
# Test on simple scenario (3 farms, 6 foods)
python test_minlp.py

# Enable D-Wave testing (requires API token)
DWAVE_TEST=1 python test_minlp.py
```

### 2. Run Professional Solver

```bash
# Run with all solvers on simple scenario
python solver_runner_NLD.py --scenario simple

# Run on full_family scenario
python solver_runner_NLD.py --scenario full_family
```

### 3. Run Benchmark

```bash
# Benchmark all three solvers on varying problem sizes
python benchmark_scalability_NLD.py
```

---

## Comparison: MILP vs MINLP

| Aspect | MILP (Old) | MINLP (New) |
|--------|-----------|-------------|
| **Objective** | `f(x) / constant` | `f(x) / g(x)` |
| **Problem Type** | Linear | Nonlinear Fractional |
| **Solvers** | PuLP/CBC | Pyomo/Ipopt, Dinkelbach |
| **Complexity** | Polynomial | NP-hard (generally) |
| **Solution Quality** | Exact optimal | Depends on solver |
| **Convergence** | Single solve | Iterative (Dinkelbach) |
| **Interpretation** | Normalized by total land | Normalized by allocated land |

---

## Mathematical Background

### Fractional Programming

A fractional program has the form:
```
max f(x) / g(x)
s.t. x ∈ X
```

Where:
- `f(x)`: Numerator function (linear combination of objectives)
- `g(x)`: Denominator function (sum of allocated areas)
- `X`: Feasible region (land constraints, linking constraints, etc.)

### Dinkelbach's Algorithm

For concave fractional programs, Dinkelbach's algorithm transforms the problem into a sequence of linear programs:

```
max f(x) - λ*g(x)
s.t. x ∈ X
```

**Convergence Property:**
- If `f(x*) - λ*g(x*) = 0`, then `λ = f(x*)/g(x*)` is optimal
- Converges in finite iterations for concave fractional programs
- Each iteration solves a linear program (efficient!)

### Charnes-Cooper Transformation

Transforms fractional program to equivalent problem:

```
Original: max f(x)/g(x)

Substitute: z = x*t, t = 1/g(x)

Equivalent: max f(z)/t
           s.t. g(z) = t
                t > 0
                original constraints on z/t
```

For **linear** f and g, this becomes a linear program!

---

## Results Format

All solvers return results in consistent format:

```python
{
    'status': 'Optimal',           # Solution status
    'objective_value': 0.756432,   # Final fractional objective
    'solve_time': 2.345,           # Time in seconds
    'iterations': 8,               # Number of iterations (Dinkelbach)
    'solver': 'ipopt',             # Solver used (Pyomo)
    'areas': {                     # Area allocations
        'Farm1_Wheat': 45.2,
        'Farm1_Corn': 23.8,
        ...
    },
    'selections': {                # Binary selections
        'Farm1_Wheat': 1,
        'Farm1_Corn': 1,
        ...
    }
}
```

---

## Performance Considerations

### PuLP (Dinkelbach)
- **Speed:** Fast (each iteration is LP solve)
- **Scalability:** Good (same as original MILP)
- **Accuracy:** Exact (within tolerance)
- **Best for:** Medium-sized problems, guaranteed convergence

### Pyomo (Direct MINLP)
- **Speed:** Slower than Dinkelbach
- **Scalability:** Depends on solver (Ipopt: moderate, BARON: excellent)
- **Accuracy:** High (depends on solver settings)
- **Best for:** Complex nonlinear problems, global optimization

### D-Wave (Charnes-Cooper)
- **Speed:** Fast for large problems
- **Scalability:** Excellent (quantum annealing)
- **Accuracy:** Approximate (heuristic)
- **Best for:** Very large problems, exploration

---

## Troubleshooting

### Pyomo Import Errors
```bash
# Install Pyomo and Ipopt
conda install -c conda-forge pyomo ipopt
```

### Dinkelbach Not Converging
- Increase `max_iterations` in `solve_with_pulp()`
- Check if problem is convex (concavity required)
- Adjust `epsilon` tolerance

### D-Wave API Errors
```bash
# Set your API token
export DWAVE_API_TOKEN='your-token-here'

# Or in code
token = os.getenv('DWAVE_API_TOKEN', 'default-token')
```

### Infeasible Solutions
- Check constraint feasibility with original MILP
- Verify food group constraints aren't too restrictive
- Ensure minimum planting areas are reasonable

---

## File Structure

```
OQI-UC002-DWave - Tutorials/
├── solver_runner_NLD.py          # Main MINLP solver implementation
├── test_minlp.py                 # Test script for all three solvers
├── benchmark_scalability_NLD.py  # Benchmark script
├── MINLP_README.md              # This file
├── solver_runner.py              # Original MILP solver (for comparison)
└── src/
    └── scenarios.py              # Scenario definitions
```

---

## References

1. **Dinkelbach's Algorithm:**
   - Dinkelbach, W. (1967). "On Nonlinear Fractional Programming". Management Science.

2. **Charnes-Cooper Transformation:**
   - Charnes, A., & Cooper, W. W. (1962). "Programming with linear fractional functionals". Naval Research Logistics Quarterly.

3. **Fractional Programming:**
   - Schaible, S. (1983). "Fractional programming". Zeitschrift für Operations Research.

4. **Pyomo:**
   - Hart, W. E., et al. (2017). "Pyomo–optimization modeling in python". Springer.

5. **D-Wave Quantum Annealing:**
   - McGeoch, C. C. (2014). "Adiabatic quantum computation and quantum annealing: Theory and practice". Morgan & Claypool.

---

## Contact

For questions or issues, please refer to the main project documentation or create an issue in the repository.

---

## License

This project is part of the OQI-UC002-DWave Tutorials repository.
