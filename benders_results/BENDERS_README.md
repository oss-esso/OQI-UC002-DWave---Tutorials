# Benders Decomposition for Crop Allocation MILP

## Overview

This implementation provides a **hybrid optimization approach** that combines:
- **Benders Decomposition** for Mixed Integer Linear Programming (MILP)
- **Simulated Annealing** (Classical or Quantum) for the master problem
- **PuLP** for the subproblem

## Problem Description

The crop allocation problem optimizes food production across multiple farms by selecting which crops to plant (binary decisions) and how much land to allocate to each crop (continuous decisions).

### Decision Variables

- **Y[farm, crop]**: Binary variable (0/1) indicating whether crop is planted on farm
- **A[farm, crop]**: Continuous variable representing hectares allocated to crop on farm

### Objective Function

Maximize weighted sum of:
- Nutritional value
- Nutrient density
- Sustainability
- Environmental impact (minimized)

### Constraints

1. **Land Availability**: Total area per farm ≤ available land
2. **Minimum Planting Area**: If crop selected (Y=1), must plant at least minimum area
3. **Linking Constraints**: Area allocation tied to binary selection
4. **Food Group Requirements**: Each farm must select minimum/maximum crops from each food group

## Benders Decomposition Strategy

### Master Problem (Binary Selection)

**Solved by**: Simulated Annealing (Classical or Quantum)

**Variables**: Y[farm, crop] (binary)

**Objective**: Minimize energy = -estimated_objective + penalties + cuts

The master problem uses annealing to search for binary crop selections that:
- Satisfy food group constraints (via penalties)
- Respect Benders cuts from previous iterations
- Approximate the full MILP objective

### Subproblem (Area Allocation)

**Solved by**: PuLP (CBC solver)

**Variables**: A[farm, crop] (continuous)

**Given**: Fixed binary selections Y from master problem

**Objective**: Maximize actual objective given fixed Y

**Purpose**: 
- Compute optimal area allocation for given crop selections
- Generate dual variables for Benders cuts
- Provide upper bound on objective value

### Benders Cuts

**Optimality Cuts**: Generated from dual variables of feasible subproblems
- Guide master problem toward better solutions
- Provide lower bound on objective value

**Feasibility Cuts**: Generated when subproblem is infeasible
- Force master problem to try different binary combinations

## Algorithm Flow

```
1. Initialize: empty cut set, bounds
2. For each iteration:
   a. Solve Master Problem (Annealing)
      - Find binary Y minimizing energy function
      - Energy includes penalties + all cuts
   b. Solve Subproblem (PuLP)
      - Fix Y values from master
      - Optimize continuous A variables
      - Extract dual variables
   c. Generate Cut
      - If feasible: optimality cut from duals
      - If infeasible: feasibility cut
   d. Update Bounds
      - Upper bound: best subproblem objective found
      - Lower bound: master problem bound from cuts
   e. Check Convergence
      - Gap = |UB - LB| / |UB|
      - Stop if gap < tolerance
3. Return best solution found
```

## Installation

### Prerequisites

```bash
# Python 3.8+
pip install numpy pulp
```

### Files

- `benders_decomposition.py`: Main implementation
- `my_functions/simulated_annealing.py`: Classical simulated annealing
- `my_functions/simulated_Qannealing.py`: Quantum simulated annealing
- `src/scenarios.py`: Problem scenarios and data loading
- `compare_benders.py`: Comparison with standard MILP solver

## Usage

### Basic Usage

```bash
# Run with simple scenario and classical annealing
python benders_decomposition.py --scenario simple

# Run with quantum annealing
python benders_decomposition.py --scenario simple --quantum

# Specify parameters
python benders_decomposition.py \
    --scenario intermediate \
    --quantum \
    --max-iter 50 \
    --tolerance 0.001 \
    --output results.json
```

### Command Line Arguments

- `--scenario`: Scenario complexity (`simple`, `intermediate`, `custom`, `full`, `full_family`)
- `--quantum`: Use quantum annealing for master problem (default: classical)
- `--max-iter`: Maximum Benders iterations (default: 50)
- `--tolerance`: Convergence tolerance (default: 0.001)
- `--output`: Output JSON file (default: `benders_solution.json`)

### Comparison with Standard MILP

```bash
# Compare Benders vs standard PuLP solver
python compare_benders.py --scenario simple --benders-iter 20
```

## Output

### Solution File (JSON)

```json
{
  "status": "Optimal" | "SubOptimal" | "NoSolution",
  "objective_value": 0.289444,
  "lower_bound": -0.023333,
  "upper_bound": 0.289444,
  "gap": 0.108,
  "iterations": 15,
  "total_time": 3.09,
  "binary_variables": {
    "Farm1_Wheat": 1,
    "Farm1_Corn": 0,
    ...
  },
  "area_variables": {
    "Farm1_Wheat": 5.0,
    "Farm1_Corn": 0.0,
    ...
  },
  "iteration_history": [...]
}
```

### Console Output

```
================================================================================
BENDERS DECOMPOSITION SOLUTION
================================================================================
Status: SubOptimal
Objective Value: 0.289444
Gap: 108.061420%
Iterations: 15
Total Time: 3.09s

================================================================================
CROP SELECTION (Y variables)
================================================================================
Farm1: Wheat, Soybeans, Potatoes, Apples
Farm2: Corn, Soybeans, Potatoes, Apples
Farm3: Wheat, Soybeans, Potatoes, Apples

================================================================================
AREA ALLOCATION (A variables)
================================================================================

Farm1:
Crop            Selected   Area (ha)
----------------------------------------
Wheat           1          5.0000
Soybeans        1          60.0000
Potatoes        1          5.0000
Apples          1          5.0000
```

## Scenarios

### Simple
- **Farms**: 3
- **Crops**: 6 (Wheat, Corn, Rice, Soybeans, Potatoes, Apples)
- **Binary Variables**: 18
- **Use Case**: Quick testing and validation

### Intermediate
- **Farms**: 3
- **Crops**: 6
- **Additional Constraints**: Food group requirements, min planting areas
- **Use Case**: Realistic small-scale problem

### Custom
- **Farms**: 2
- **Crops**: 6 (2 per food group)
- **Focus**: Balanced food group representation
- **Use Case**: Targeted scenario testing

### Full
- **Farms**: 5
- **Crops**: Variable (loaded from Excel data)
- **Data Source**: `Inputs/Combined_Food_Data.xlsx`
- **Use Case**: Real-world scale testing

### Full Family
- **Farms**: 125 (generated via farm sampler)
- **Crops**: Variable (from Excel)
- **Use Case**: Large-scale optimization

## Performance Considerations

### Convergence Challenges

**Issue**: Annealing-based master problem may not converge as tightly as exact MILP solvers

**Reasons**:
1. Heuristic search vs exact optimization
2. Energy function approximates true objective
3. Cuts may not fully guide annealing search

**Mitigation Strategies**:
1. Increase annealing iterations
2. Adjust temperature schedule (T0, alpha)
3. Run multiple times with different seeds
4. Use as feasible solution generator for exact solver

### Computational Trade-offs

**Classical Annealing**:
- Faster per iteration (~0.1-0.3s)
- Good for rapid prototyping
- May explore solution space better

**Quantum Annealing**:
- Slower per iteration (~7-9s)
- More replicas = more exploration
- Potentially better global search

**Standard MILP**:
- Exact optimal solution
- May be slower for large problems
- Baseline for comparison

## Advanced Configuration

### Annealing Parameters

Edit in code or pass as dictionary:

```python
annealing_params = {
    'T0': 100.0,        # Initial temperature
    'alpha': 0.95,      # Cooling rate
    'max_iter': 5000,   # Iterations per solve
    'seed': 42          # Random seed
}

# Quantum-specific
annealing_params.update({
    'num_replicas': 10,  # Quantum replicas
    'gamma0': 50.0,      # Initial quantum strength
    'beta': 0.1          # Inverse temperature scale
})
```

### Benders Parameters

In `config` dictionary:

```python
config = {
    'benders_tolerance': 1e-3,        # Convergence tolerance
    'benders_max_iterations': 100,     # Max iterations
    'pulp_time_limit': 120,            # Subproblem time limit
    'use_multi_cut': True,             # Multi-cut strategy
    'use_trust_region': True,          # Trust region
    'use_anticycling': True,           # Anti-cycling
    'use_norm_cuts': True              # Normalized cuts
}
```

## Implementation Details

### Master Problem Energy Function

```python
energy = -estimated_objective + constraint_penalties + cut_violations
```

**Components**:
1. **Estimated Objective**: Heuristic approximation using minimum areas
2. **Constraint Penalties**: Large penalties for food group violations
3. **Cut Violations**: Penalties for violating Benders cuts

### Optimality Cut Generation

From subproblem duals:

```python
cut_coeffs[(farm, crop)] = dual_min * A_min - dual_max * L_farm
rhs = subproblem_obj - sum(cut_coeffs * Y_current)
```

### Feasibility Cut Generation

Forces binary changes:

```python
# Require at least one Y to flip
sum(Y_ones) - sum(Y_zeros) <= |Y_ones| - 1
```

## Limitations and Future Work

### Current Limitations

1. **Convergence**: May not reach exact optimality due to heuristic master problem
2. **Scalability**: Annealing time increases with problem size
3. **Cut Quality**: Simple dual-based cuts may not be strongest possible

### Future Enhancements

1. **Improved Energy Function**: Better objective approximation in master
2. **Adaptive Annealing**: Dynamic parameter adjustment
3. **Cut Strengthening**: More sophisticated cut generation
4. **Warm Starting**: Use previous solutions as starting points
5. **Parallel Solving**: Multiple annealing runs in parallel
6. **Hybrid Approach**: Use annealing to seed exact MILP solver

## References

### Benders Decomposition

- Benders, J.F. (1962). "Partitioning procedures for solving mixed-variables programming problems"

### Simulated Annealing

- Kirkpatrick, S., Gelatt, C.D., Vecchi, M.P. (1983). "Optimization by Simulated Annealing"

### Quantum Annealing

- Kadowaki, T., Nishimori, H. (1998). "Quantum annealing in the transverse Ising model"

## Contributing

Feel free to extend this implementation with:
- Better cut generation strategies
- Alternative heuristic solvers for master problem
- Enhanced convergence criteria
- Warm-start mechanisms
- Parallelization

## License

This implementation is provided for research and educational purposes.

## Contact

For questions or issues, please refer to the project repository.

---

**Author**: Autonomous Agent  
**Date**: October 21, 2025  
**Version**: 1.0
