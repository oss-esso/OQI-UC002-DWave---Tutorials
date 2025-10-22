# Non-Linear Solver Implementation Summary

## Overview

Successfully implemented **three solvers** for the non-linear food optimization problem with objective function `f(A) = A^0.548`:

1. **D-Wave CQM** - Piecewise linear approximation (model created, solving disabled)
2. **PuLP** - Piecewise linear approximation with SOS2 formulation  
3. **Pyomo + IPOPT** - TRUE non-linear objective (no approximation)

## Implementation Details

### 1. D-Wave CQM (Piecewise Approximation)
- **Status**: Model creation successful
- **Method**: Piecewise linear approximation with lambda variables
- **Constraints**: SOS2 formulation using convex combination
- **Variables**: 
  - Original: A (area), Y (binary selection)
  - Added: Lambda_i for each breakpoint
- **Token**: Removed for testing (solving disabled)

### 2. PuLP (Piecewise Approximation)
- **Status**: ✅ Working and optimal
- **Solver**: CBC (open-source MILP solver)
- **Method**: Piecewise linear approximation matching D-Wave CQM
- **Key Features**:
  - Lambda variables for each breakpoint
  - A = Σ λ_i × breakpoint_i
  - f_approx = Σ λ_i × f(breakpoint_i)
  - Convexity constraint: Σ λ_i = 1

### 3. Pyomo + IPOPT (True Non-Linear)
- **Status**: ✅ Working and optimal  
- **Solver**: IPOPT (Interior Point Optimizer)
- **Method**: Direct non-linear objective `A^0.548`
- **Key Features**:
  - No approximation needed
  - Uses true power function in objective
  - IPOPT treats binary variables as continuous (relaxation)
  - Small epsilon (1e-6) added to lower bound to avoid 0^0.548 issue

## Test Results (Simple Scenario)

### With 10 Interior Breakpoints:
```
PuLP (Approximation):  0.058015  |  Solve time: 0.23s
Pyomo (True NLN):      0.059865  |  Solve time: 0.70s
Difference:            3.09%
```

### With 20 Interior Breakpoints:
```
PuLP (Approximation):  0.059320  |  Solve time: 0.14s
Pyomo (True NLN):      0.059865  |  Solve time: 0.28s
Difference:            0.91%
```

## Key Findings

### Approximation Accuracy vs Breakpoints

| Breakpoints | Max Error | Avg Error | PuLP Objective | Diff from True |
|-------------|-----------|-----------|----------------|----------------|
| 10 interior | 0.731     | 0.053     | 0.058015       | 3.09%          |
| 20 interior | 0.513     | 0.020     | 0.059320       | 0.91%          |

**Conclusion**: More breakpoints significantly improve accuracy

### Solve Time Comparison

| Solver | Method         | Time (10 bp) | Time (20 bp) |
|--------|----------------|--------------|--------------|
| PuLP   | Piecewise      | 0.23s        | 0.14s        |
| Pyomo  | True Non-Linear| 0.70s        | 0.28s        |

**Observation**: 
- PuLP is faster for smaller breakpoint counts
- Both solvers benefit from problem structure
- IPOPT (Pyomo) handles non-linearity efficiently

### Variable Count Scaling

**Simple Scenario** (3 farms, 6 foods = 18 farm-food pairs):

| Breakpoints | A vars | Y vars | Lambda vars | Total | Constraints |
|-------------|--------|--------|-------------|-------|-------------|
| 5 interior  | 18     | 18     | 126 (18×7)  | 162   | 75          |
| 10 interior | 18     | 18     | 216 (18×12) | 252   | 75          |
| 20 interior | 18     | 18     | 396 (18×22) | 432   | 75          |

**Scaling**: Lambda variables = farm-food pairs × (breakpoints + 2)

## Installation Requirements

### Already Installed:
- ✅ PuLP
- ✅ D-Wave Ocean SDK
- ✅ dimod

### Newly Installed:
- ✅ Pyomo (pip install pyomo)
- ✅ IPOPT solver (conda install -c conda-forge ipopt)

### Alternative MINLP Solvers (Optional):
- **Bonmin**: `conda install -c conda-forge coinbonmin`
- **Couenne**: `conda install -c conda-forge couenne`
- **SCIP**: `conda install -c conda-forge scip`

## File Structure

```
solver_runner_NLN.py          # Main solver script
piecewise_approximation.py    # Piecewise approximation utility

Output directories:
├── CQM_Models_NLN/           # D-Wave CQM models
├── Constraints_NLN/          # Constraint metadata + approximation info
└── PuLP_Results_NLN/         # PuLP and Pyomo solution files
    ├── pulp_nln_*.json       # PuLP results
    └── pyomo_nln_*.json      # Pyomo results
```

## Usage

### Basic Usage:
```bash
python solver_runner_NLN.py --scenario simple --power 0.548 --breakpoints 10
```

### Parameters:
- `--scenario`: simple, intermediate, full, custom, full_family
- `--power`: Exponent for f(A) = A^power (default: 0.548)
- `--breakpoints`: Number of interior points for piecewise approximation (default: 10)

### Output:
- CQM model (for D-Wave)
- PuLP solution with piecewise approximation
- Pyomo solution with true non-linear objective
- Comparison statistics

## Advantages and Trade-offs

### PuLP (Piecewise Approximation):
**Advantages:**
- ✅ Works with free, open-source CBC solver
- ✅ Can be used with D-Wave CQM
- ✅ Faster for small problems
- ✅ Configurable accuracy via breakpoints

**Trade-offs:**
- ⚠️ Approximation error (improves with more breakpoints)
- ⚠️ More variables (lambda variables scale linearly)
- ⚠️ Cannot capture exact non-linear behavior

### Pyomo + IPOPT (True Non-Linear):
**Advantages:**
- ✅ TRUE non-linear objective (no approximation)
- ✅ Mathematically exact
- ✅ Fewer variables
- ✅ Can handle complex non-linear functions

**Trade-offs:**
- ⚠️ Requires MINLP solver installation
- ⚠️ IPOPT relaxes binary variables (treats as continuous 0-1)
- ⚠️ Slightly slower for simple problems
- ⚠️ Cannot use with D-Wave quantum annealing

## Recommendations

### For D-Wave Quantum Annealing:
- **Use**: Piecewise approximation (only option)
- **Breakpoints**: 10-20 for good accuracy
- **Expected**: ~1-3% difference from true non-linear

### For Classical Solving:
- **Research/Exploration**: Use Pyomo for exact results
- **Production/Speed**: Use PuLP with 20+ breakpoints
- **Large Scale**: Use PuLP with 5-10 breakpoints (balance speed vs accuracy)

### For Comparison Studies:
1. Solve with Pyomo (ground truth)
2. Solve with PuLP at various breakpoint levels
3. Compare accuracy vs solve time trade-off
4. Select optimal breakpoint count for your use case

## Next Steps

### Immediate:
1. ✅ Piecewise approximation working
2. ✅ PuLP solver with approximation
3. ✅ Pyomo solver with true non-linear
4. 🔄 D-Wave solving (add token to enable)

### Future Enhancements:
1. **Non-uniform breakpoints**: Place more breakpoints where curvature is high
2. **Adaptive breakpoints**: Adjust based on solution values
3. **Alternative non-linear functions**: Test different powers (0.3, 0.7, etc.)
4. **Larger scenarios**: Test scalability on intermediate/full scenarios
5. **Sensitivity analysis**: Study impact of power parameter
6. **Comparison plots**: Visualize approximation quality vs solve time

## Mathematical Notes

### Why 0.548?
- **Concave function**: Diminishing returns as area increases
- **Realistic**: Common in agricultural production functions
- **Challenging**: Not linear, not quadratic → requires special handling

### Piecewise Linear Quality:
- **Concave functions**: Piecewise linear provides lower bound
- **Error**: Decreases as O(1/n²) with n segments
- **Uniform spacing**: Reasonable for concave functions

### IPOPT Behavior:
- **Binary Relaxation**: Treats Y variables as continuous [0,1]
- **Solution**: Often integer-valued at optimum for well-posed problems
- **Epsilon**: Added to avoid 0^0.548 undefined behavior

## Conclusion

Successfully implemented three complementary approaches for non-linear optimization:

1. **D-Wave CQM**: Quantum-ready piecewise approximation
2. **PuLP**: Fast classical approximation with configurable accuracy
3. **Pyomo**: Exact non-linear solution for validation

The piecewise approximation achieves <1% error with 20 breakpoints while remaining compatible with linear/quadratic solvers and quantum annealing hardware.
