# Custom Food Optimization Scenario - Solver Comparison

This project implements a custom food optimization scenario and compares three different solving approaches:
1. **PuLP** (Classical Mixed-Integer Linear Programming with Global Coordination)
2. **CQM** (D-Wave Constrained Quadratic Model)
3. **QUBO** (D-Wave Quadratic Unconstrained Binary Optimization)

## ✅ **Updated to Follow pulp_sim.py Formulation**

The PuLP solver (`solve_pulp.py`) has been **completely rewritten** to match the original `pulp_sim.py` formulation exactly, ensuring both solvers solve the **same globally coordinated optimization problem**.

## Custom Scenario Specifications

The custom scenario meets your exact requirements:
- **2 Farms**: Farm1 (75 hectares), Farm2 (100 hectares)
- **3 Food Groups** with **2 foods each**:
  - **Grains**: Wheat, Rice
  - **Legumes**: Soybeans, Potatoes
  - **Fruits**: Apples, Tomatoes
- **Same parameters as intermediate scenario**: weights, constraints, and optimization criteria
- **Additional global coordination parameters** from pulp_sim.py

## Key Formulation Features (Following pulp_sim.py)

### **Global Coordination Constraints**
1. **Global Food Selection**: Minimum 5 different food types selected across all farms
2. **Farm-Level Food Variety**: Min 1, Max 6 different foods per farm
3. **Global Land Utilization**: Minimum 50% total land usage across all farms
4. **Food Group Global Constraints**: Each food group requires at least 10% of total land
5. **Significant Foods per Group**: At least 2 foods with >1 hectare per group

### **Enhanced Linking Constraints**
- More sophisticated area-binary variable relationships
- Percentage-based maximum area constraints
- Minimum area enforcement when planted

## Files Created

### 1. Core Scenario (`src/scenarios.py`)
- Updated `_load_custom_food_data()` function with global parameters
- Added: `global_min_different_foods`, `min_foods_per_farm`, `max_foods_per_farm`, `min_total_land_usage_percentage`
- Accessible via `load_food_data('custom')`

### 2. PuLP Solver (`solve_pulp.py`) - **UPDATED**
- **Now follows pulp_sim.py formulation exactly**
- Implements global coordination constraints
- Enhanced constraint validation
- 70 total constraints (vs. 52 in simple per-farm version)
- Same objective function and variable structure as pulp_sim.py

### 3. CQM D-Wave Solver (`solve_cqm.py`)
- Converts problem to Constrained Quadratic Model format
- Uses D-Wave LeapHybridCQMSampler
- Includes simulation mode for dummy tokens

### 4. QUBO D-Wave Solver (`solve_qubo.py`)
- Converts problem to Quadratic Unconstrained Binary Optimization
- Discretizes continuous area variables into multiple levels
- Uses D-Wave LeapHybridSampler

### 5. Comparison Script (`compare_solutions.py`)
- Runs all three solvers automatically
- Comprehensive constraint validation (updated for global constraints)
- Detailed solution comparison and analysis

### 6. Test Script (`test_custom_scenario.py`)
- Validates scenario specifications
- Shows problem size and structure

## Current Results (Updated PuLP Solver)

**Optimal Solution Found:**
- **Objective Value**: 81.595000 (changed from 84.7 due to global constraints)
- **Solve Time**: ~0.07 seconds
- **Status**: All constraints satisfied ✓
- **Total Constraints**: 70 (vs. 52 in simple version)

**Farm Allocations (Global Coordination):**
- **Farm1** (75 hectares, 100% utilization):
  - Wheat: 10.00 hectares (minimum area)
  - Rice: 12.00 hectares (minimum area)
  - Soybeans: 25.00 hectares
  - Potatoes: 5.00 hectares (minimum area)
  - Apples: 15.00 hectares (minimum area)
  - Tomatoes: 8.00 hectares (minimum area)
- **Farm2** (100 hectares, 100% utilization):
  - Wheat: 20.00 hectares
  - Rice: 12.00 hectares (minimum area)
  - Soybeans: 40.00 hectares
  - Potatoes: 5.00 hectares (minimum area)
  - Apples: 15.00 hectares (minimum area)
  - Tomatoes: 8.00 hectares (minimum area)

**Global Metrics:**
- **All 6 foods selected** (required: minimum 5) ✓
- **175 hectares total usage** (required: 87.5 minimum) ✓
- **Each food group >10% land allocation** ✓
- **All farms have 6 different foods** (within 1-6 range) ✓

## Key Differences from Original solve_pulp.py

| Feature | Original solve_pulp.py | Updated solve_pulp.py (following pulp_sim.py) |
|---------|------------------------|-----------------------------------------------|
| **Food Group Constraints** | Per-farm only | Global + Per-farm coordination |
| **Food Selection** | Per-farm independent | Global minimum 5 foods across all farms |
| **Farm Food Variety** | Not constrained | Min 1, Max 6 foods per farm |
| **Land Utilization** | Per-farm social benefit only | Global 50% minimum + per-farm constraints |
| **Food Group Land Allocation** | Basic constraints | 10% minimum land per group globally |
| **Significant Foods** | Not considered | 2+ foods with >1 hectare per group |
| **Total Constraints** | 52 | 70 |
| **Objective Value** | 84.7 | 81.595 (more constrained problem) |

## Constraint Implementation Summary

### **All pulp_sim.py Constraints Implemented:**
1. ✅ Land availability constraints
2. ✅ Global food selection constraint (food_selected variables)
3. ✅ Linking constraints (enhanced x-y relationships)  
4. ✅ Farm utilization (social benefit)
5. ✅ Food variety per farm (min/max foods)
6. ✅ Stronger food group constraints (10% land minimum)
7. ✅ Global total land utilization (50% minimum)
8. ✅ Significant foods per group constraint
9. ✅ All food groups represented meaningfully

## Usage Instructions

### Run Individual Solvers
```bash
# PuLP solver (now globally coordinated, following pulp_sim.py)
python solve_pulp.py

# CQM solver (requires D-Wave token, uses simulation with dummy)
python solve_cqm.py

# QUBO solver (requires D-Wave token, uses simulation with dummy)  
python solve_qubo.py
```

### Run Comparison
```bash
# Runs all solvers and generates comparison report
python compare_solutions.py
```

### Test Scenario
```bash
# Validate scenario specifications
python test_custom_scenario.py
```

## Problem Complexity (Updated)

- **Variables**: 36 total (12 continuous + 12 binary + 12 auxiliary binary)
- **Constraints**: 70 total (significant increase from global coordination)
- **Decision Points**: 12 farm-food allocation decisions + global coordination
- **Optimization Objectives**: 5 weighted criteria
- **Global Coordination**: Cross-farm food selection and land utilization

## Validation Features (Enhanced)

All solutions are validated against:
- **Mathematical constraint satisfaction** (all 70 constraints)
- **Global food selection** (minimum 5 different foods)
- **Global land utilization** (minimum 50% total usage)
- **Food group global requirements** (10% land per group)
- **Farm-level food variety** (1-6 foods per farm)
- **Significant foods per group** (2+ foods with >1 hectare)
- **Original constraints** (land availability, social benefit, etc.)

## ✅ **Key Achievement**

The updated `solve_pulp.py` now solves **exactly the same problem** as `pulp_sim.py`:
- ✅ **Identical objective function**
- ✅ **Identical variable structure** 
- ✅ **Identical constraint formulation**
- ✅ **Global coordination enforced**
- ✅ **Same complexity and problem scope**

Both solvers now implement a **globally coordinated planting strategy** rather than independent per-farm optimization, ensuring the comparison is meaningful and the formulations are equivalent.

## Usage Instructions

### Run Individual Solvers
```bash
# PuLP solver (working)
python solve_pulp.py

# CQM solver (requires D-Wave token, uses simulation with dummy)
python solve_cqm.py

# QUBO solver (requires D-Wave token, uses simulation with dummy)
python solve_qubo.py
```

### Run Comparison
```bash
# Runs all solvers and generates comparison report
python compare_solutions.py
```

### Test Scenario
```bash
# Validate scenario specifications
python test_custom_scenario.py
```

## D-Wave Integration

The CQM and QUBO solvers are ready for real D-Wave execution:
- Replace `dwave_token = "dummy"` with your actual D-Wave API token
- Install D-Wave Ocean SDK: `pip install dwave-ocean-sdk`
- The solvers will automatically use real D-Wave quantum/hybrid solvers

## Output Files

After running `compare_solutions.py`:
- `comparison_report.txt`: Human-readable comparison report
- `solver_results.json`: Machine-readable results data

## Problem Complexity

- **Variables**: 24 total (12 continuous + 12 binary)
- **Constraints**: 52 total
- **Decision Points**: 12 farm-food allocation decisions
- **Optimization Objectives**: 5 weighted criteria (nutrition, density, affordability, sustainability, environmental impact)

## Validation Features

All solutions are validated against:
- Mathematical constraint satisfaction
- Logical consistency (planting decisions vs. areas)
- Food group requirements
- Land utilization bounds
- Minimum/maximum area constraints

The system provides detailed violation reports if any constraints are not satisfied.

## Next Steps

1. **Add Real D-Wave Token**: Replace dummy tokens for actual quantum solving
2. **Install Ocean SDK**: `pip install dwave-ocean-sdk` for D-Wave functionality
3. **Compare Results**: Run comparison script to see all three approaches
4. **Tune Parameters**: Adjust discretization levels for QUBO solver if needed
5. **Scale Up**: Modify scenario for larger problems as needed

All scripts are ready to run with PuLP working immediately and D-Wave solvers ready for token integration.