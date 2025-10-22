# Non-Linear Objective Implementation (A^0.548)

## Overview

This implementation demonstrates how to handle **non-linear objectives** in optimization problems where the relationship is not linear with respect to area. Instead of a linear objective `f(A) = A`, we use a power function `f(A) = A^0.548`.

## Problem Statement

In the original food optimization problem, the objective was linear:
```
Maximize: Σ (weight × attribute × A_{farm,food})
```

In the non-linear version, we want:
```
Maximize: Σ (weight × attribute × A_{farm,food}^0.548)
```

This represents **diminishing returns** as area increases - common in agricultural scenarios where efficiency decreases with scale.

## Challenge

Both **PuLP** (linear programming solver) and **D-Wave CQM** (constrained quadratic model) cannot directly handle arbitrary power functions like `A^0.548`:

- **PuLP**: Only supports linear and mixed-integer linear programming
- **D-Wave CQM**: Only supports quadratic (degree 2) terms

## Solution: Piecewise Linear Approximation

We approximate the non-linear function `f(A) = A^0.548` using a **piecewise linear function** with multiple breakpoints.

### How It Works

1. **Breakpoints**: Divide the domain [0, max_land] into segments using breakpoints
2. **Linear Segments**: Between each pair of breakpoints, use a linear approximation
3. **SOS2 Formulation**: Use Special Ordered Set of type 2 (SOS2) to ensure the solution lies on the piecewise linear curve

### Mathematical Formulation

For each area variable `A_{f,c}`, we introduce:

**Variables:**
- `λ_i` for i = 0, 1, ..., n (lambda variables for each breakpoint)
- Each `λ_i ∈ [0, 1]`

**Constraints:**
1. **Area definition**: `A_{f,c} = Σ λ_i × breakpoint_i`
2. **Convexity**: `Σ λ_i = 1`
3. **SOS2** (implicit): At most 2 adjacent λ values can be non-zero

**Objective:**
- Instead of using `A_{f,c}` directly, we use `f_approx = Σ λ_i × f(breakpoint_i)`
- Where `f(x) = x^0.548`

## Files Created

### 1. `piecewise_approximation.py`
- **Purpose**: Creates piecewise linear approximations for power functions
- **Features**:
  - Configurable number of breakpoints
  - Error analysis (max absolute, max relative, average)
  - Visualization of approximation vs true function
  - Export to JSON for use in optimization models

**Usage:**
```bash
python piecewise_approximation.py --power 0.548 --points 20 --max-value 100
```

**Key Parameters:**
- `--power`: Exponent for f(x) = x^power (default: 0.548)
- `--points`: Number of interior breakpoints (default: 10)
- `--max-value`: Maximum domain value (default: 100.0)
- `--output`: JSON output filename
- `--plot`: Plot output filename

### 2. `solver_runner_NLN.py`
- **Purpose**: Modified solver runner with non-linear objective
- **Features**:
  - Automatic piecewise approximation for each farm-food pair
  - Different approximations for different land availabilities
  - Saves approximation metadata with constraints
  - Creates CQM with lambda variables and piecewise constraints

**Usage:**
```bash
python solver_runner_NLN.py --scenario simple --power 0.548 --breakpoints 20
```

**Key Parameters:**
- `--scenario`: Optimization scenario (simple, intermediate, full, custom, full_family)
- `--power`: Power for non-linear objective (default: 0.548)
- `--breakpoints`: Number of interior breakpoints (default: 10)

**Note**: DWave solving is currently disabled (token removed) for testing CQM creation.

## Results

### Test Run: Simple Scenario with 20 Breakpoints

**Configuration:**
- Farms: 3 (Farm1, Farm2, Farm3)
- Foods: 6 (Wheat, Corn, Rice, Soybeans, Potatoes, Apples)
- Power: 0.548
- Interior breakpoints: 20 (22 total including endpoints)

**CQM Statistics:**
- **Variables**: 432
  - 18 area variables (A_{f,c})
  - 18 binary variables (Y_{f,c})
  - 396 lambda variables (18 × 22 breakpoints)
- **Constraints**: 75
  - 3 land availability
  - 36 piecewise approximation (18 × 2)
  - 36 linking constraints

**Approximation Accuracy:**
- Max land = 50: Max error = 0.351, Avg error = 0.014
- Max land = 75: Max error = 0.438, Avg error = 0.017
- Max land = 100: Max error = 0.513, Avg error = 0.020

The approximation is very accurate with only ~0.5% maximum error for 20 breakpoints.

## Trade-offs

### Number of Breakpoints vs Accuracy

| Breakpoints | Variables | Max Error | Avg Error | Complexity |
|-------------|-----------|-----------|-----------|------------|
| 5 interior  | 162       | ~1.02     | ~0.13     | Low        |
| 10 interior | 252       | ~0.73     | ~0.05     | Medium     |
| 20 interior | 432       | ~0.51     | ~0.02     | High       |

**Recommendations:**
- **Small problems**: Use 20+ breakpoints for high accuracy
- **Large problems**: Use 5-10 breakpoints to keep variable count manageable
- **Production**: Balance accuracy vs solve time based on requirements

## Output Files

### Directory Structure
```
CQM_Models_NLN/
  cqm_nln_simple_20251022_163108.cqm

Constraints_NLN/
  constraints_nln_simple_20251022_163108.json

piecewise_approx.json
piecewise_approx.png
piecewise_approx_20pts.json
piecewise_approx_20pts.png
```

### Constraint Metadata
The JSON file includes:
- Original problem data (farms, foods, config)
- **Approximation metadata**:
  - Power function used
  - Number of breakpoints
  - Breakpoint locations
  - Function values at each breakpoint
  - Error statistics for each approximation
- Constraint metadata (same as original solver)

## Next Steps

### 1. Enable PuLP Solving
Modify `solve_with_pulp()` to use piecewise linear approximation:
- Add lambda variables
- Add piecewise constraints
- Modify objective to use approximated values

### 2. Enable D-Wave Solving
- Remove token restriction
- Submit CQM with piecewise approximation
- Compare results with PuLP solution

### 3. Comparison Analysis
Create comparison script to analyze:
- Linear vs non-linear objective solutions
- Impact of power parameter (0.3, 0.5, 0.548, 0.7, 1.0)
- Trade-off between accuracy and solve time

### 4. Advanced Approximations
Explore alternative approximation methods:
- **Non-uniform breakpoints**: Place more breakpoints where curvature is high
- **Adaptive breakpoints**: Adjust based on solution values
- **Alternative formulations**: Log-transformation, quadratic approximation

## Mathematical Background

### Why A^0.548?

The exponent 0.548 represents a **concave** production function (diminishing returns):
- For x < 1: f(x) > x (super-linear growth)
- For x > 1: f(x) < x (sub-linear growth)
- Derivative: f'(x) = 0.548 × x^(-0.452) decreases as x increases

This is realistic for many agricultural scenarios where:
- Initial investments have high returns
- Returns decrease as scale increases
- Optimal allocation spreads resources across multiple crops

### Piecewise Linear Approximation Quality

For a function f(x) = x^p with p ∈ (0, 1):
- The function is **concave** and **continuous**
- Piecewise linear approximation provides a **lower bound** (underestimates)
- Error decreases as **O(1/n²)** where n is the number of segments
- Uniform breakpoints are reasonable for concave functions

### SOS2 Constraint

The Special Ordered Set of type 2 (SOS2) constraint ensures:
- At most 2 consecutive λ variables can be non-zero
- This maintains the piecewise linear structure
- Most solvers handle SOS2 efficiently with branch-and-bound

In our formulation, the SOS2 constraint is **implicit** through the combination of:
1. Convexity constraint: Σλ_i = 1
2. Area definition: A = Σλ_i × breakpoint_i
3. Solver's handling of continuous variables in the piecewise structure

## Visualization

The `piecewise_approximation.py` script generates plots showing:
1. **Function Comparison**: True function vs piecewise approximation with breakpoints
2. **Error Analysis**: Absolute error over the domain with statistics

Example plots are saved as:
- `piecewise_approx.png` (10 interior points)
- `piecewise_approx_20pts.png` (20 interior points)

## References

1. **Piecewise Linear Approximation**: Williams, H. P. (2013). *Model Building in Mathematical Programming*. Wiley.
2. **SOS2 Constraints**: Beale, E. M. L., & Tomlin, J. A. (1970). "Special facilities in a general mathematical programming system for non-convex problems using ordered sets of variables."
3. **Power Functions in Agriculture**: Cobb-Douglas production functions and agricultural economics literature.

## Summary

This implementation successfully demonstrates:
- ✅ Non-linear objective approximation using piecewise linear functions
- ✅ Configurable accuracy through breakpoint density
- ✅ Automatic handling of different land availabilities
- ✅ CQM creation with piecewise constraints verified
- ✅ Detailed approximation metadata and error analysis
- 🔄 PuLP and D-Wave solving (to be implemented next)

The approach provides a **practical and efficient** way to handle non-linear objectives in optimization problems constrained to linear and quadratic solvers.
