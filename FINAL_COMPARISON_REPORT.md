# Final Solution Comparison: DWave vs PuLP

## ✅ SUCCESS - Solutions Now Match!

After fixing both issues in the DWave formulation, both solvers now produce **identical optimal solutions**.

## Fixes Applied

### Fix 1: Variable Bounds (Line 48)
**Before:**
```python
A[(f, c)] = Real(f"A_{f}_{c}", lower_bound=A_min[c], upper_bound=L[f])
```

**After:**
```python
A[(f, c)] = Real(f"A_{f}_{c}", lower_bound=0, upper_bound=L[f])
```

**Reason:** Area variables must be able to be 0 when a crop is not selected (Y=0).

### Fix 2: Objective Function (Lines 54-66, 86)
**Before:**
```python
Z = Real("Z", lower_bound=-1e6, upper_bound=1e6)
numerator = sum(...)
cqm.add_constraint(numerator - Z * total_area >= 0, label="Efficiency_Constraint")
cqm.set_objective(Z)
```

**After:**
```python
objective = sum(
    weights['w_1'] * N[c] * A[(f, c)] +
    weights['w_2'] * D[c] * A[(f, c)] -
    weights['w_3'] * E[c] * A[(f, c)] +
    weights['w_4'] * P[c] * A[(f, c)]
    for f in farms for c in crops
)
cqm.set_objective(-objective)  # Negative because CQM minimizes by default
```

**Reason:** The auxiliary variable Z approach was flawed. Direct optimization of the numerator (weighted sum) is equivalent to maximizing the ratio since the denominator (total_area) is constant.

## Final Results

### Objective Value
- **PuLP**: 0.588900
- **DWave**: 0.588900 ✅ **MATCH!**

### Farm1 Solution (100 ha)
| Crop   | Selected | Area (ha) |
|--------|----------|-----------|
| Wheat  | No       | 0         |
| Corn   | Yes      | 4         |
| Soy    | Yes      | 3         |
| Tomato | Yes      | 93        |
| **Total** | **3 crops** | **100** |

### Farm2 Solution (150 ha)
| Crop   | Selected | Area (ha) |
|--------|----------|-----------|
| Wheat  | No       | 0         |
| Corn   | Yes      | 4         |
| Soy    | Yes      | 3         |
| Tomato | Yes      | 143       |
| **Total** | **3 crops** | **150** |

### Solution Characteristics
- **Total land used**: 250 ha (100% utilization)
- **Crops selected**: Corn, Soy, Tomato (same on both farms)
- **Primary crop**: Tomato (236 ha = 94.4% of total)
- **Minimum allocations**: Corn (4 ha min), Soy (3 ha min), Tomato (2 ha min)

## Constraint Verification

Both solutions satisfy ALL constraints:

✅ **Land Availability**
- Farm1: 100 ha used ≤ 100 ha available
- Farm2: 150 ha used ≤ 150 ha available

✅ **Minimum Area When Selected**
- All selected crops meet minimum area requirements
- Corn: 4 ha ≥ 4 ha min
- Soy: 3 ha ≥ 3 ha min  
- Tomato: 93/143 ha ≥ 2 ha min

✅ **Food Group Constraints** (per farm)
- Grains: 1 selected (range: 1-2) ✓
- Legumes: 1 selected (range: 1-1) ✓
- Vegetables: 1 selected (range: 1-1) ✓

✅ **Linking Constraints**
- When Y=0: A=0 (wheat not selected, 0 area)
- When Y=1: A≥A_min (all selected crops meet minimum)

## Performance Metrics

### DWave Hybrid Solver
- **QPU Access Time**: 69.55 ms
- **Total Run Time**: 5,296.77 ms (~5.3 seconds)
- **Charge Time**: 5,000.00 ms (5 seconds)
- **Feasible Solutions**: 116 out of 125 (92.8%)

### PuLP CBC Solver
- **Solver**: COIN-OR CBC (open source)
- **Runtime**: < 1 second
- **Status**: Optimal

## Analysis

### Why This Solution is Optimal

The optimal solution allocates maximum area to **Tomato** because:

**Tomato has the highest weighted score:**
- Nutrition (N): 0.8 × 0.25 = 0.20
- Diversity (D): 0.9 × 0.25 = 0.225
- Environment (E): 0.2 × 0.25 = -0.05 (negative weight)
- Preference (P): 0.9 × 0.25 = 0.225
- **Total weighted score**: 0.20 + 0.225 - 0.05 + 0.225 = **0.60**

**For comparison:**
- Corn: 0.175 + 0.2125 - 0.075 + 0.125 = **0.4375**
- Soy: 0.125 + 0.1375 - 0.125 + 0.15 = **0.2875**
- Wheat: 0.175 + 0.15 - 0.10 + 0.175 = **0.40**

The solver maximally allocates to Tomato (236 ha) while respecting:
1. Food group requirements (must select at least 1 grain and 1 legume)
2. Minimum area requirements for selected crops
3. Land availability on each farm

## Conclusion

Both DWave and PuLP now produce **identical optimal solutions** with:
- Same objective value (0.588900)
- Same crop selections
- Same area allocations
- All constraints satisfied

The DWave CQM formulation is now correct and equivalent to the PuLP LP formulation.

### Key Learnings

1. **Variable bounds matter**: Incorrect bounds can create fundamental conflicts with constraints
2. **Objective formulation for ratios**: When the denominator is constant, maximizing the numerator is equivalent to maximizing the ratio
3. **Auxiliary variables**: The Z-based formulation was mathematically incorrect due to unbounded negative values
4. **Direct optimization**: For CQMs, directly optimizing the linear/quadratic objective is often simpler and more reliable

### Verification
Both solvers independently found the same optimal solution, confirming correctness.
