---
applyTo: '**'
---

# User Memory

## Project Context
- Current project type: Quantum/Classical Optimization Research
- Tech stack: Python, PuLP, NumPy, Simulated Annealing
- Architecture patterns: Benders Decomposition, MILP optimization
- Key requirements: Hybrid solver combining annealing (master) + PuLP (subproblem)

## Current Task
- ✅ COMPLETED: PuLP Solver Scaling Analysis for full_family scenario
- ✅ Analyzed solve time scaling with varying farm counts (log scale)
- ✅ Fitted polynomial, power law, and exponential models
- ✅ Determined exact n values for 5s and 6.5s solve times
- ✅ Verified predictions with actual runs

## Latest Analysis: PuLP Scaling (October 21, 2025)
- **Objective**: Find n (farms × foods) for 5s and 6.5s solve times
- **Method**: Varied farms 1-5000 on log scale, fitted scaling models
- **Best model**: Polynomial (R² = 0.9965)
- **Results**:
  - For 5.0s: n = 29,882 (2,988 farms) - Verified: 4.678s (6.4% error) ✓
  - For 6.5s: n = 34,375 (3,438 farms) - Verified: 6.252s (3.8% error) ✓
- **Scaling**: Nearly quadratic O(n²) - typical of LP interior-point methods

## Previous Task
- ✅ COMPLETED: Implemented Benders Decomposition for crop allocation MILP
- ✅ Master problem: Binary Y variables (crop selection) solved via simulated annealing
- ✅ Subproblem: Continuous A variables (area allocation) solved via PuLP
- ✅ Works with all scenarios in scenarios.py

## Implementation Notes
- MILP formulation in pulp_2.py shows:
  - Binary Y variables: crop selection per farm
  - Continuous A variables: area allocation per farm-crop pair
  - Constraints: land limits, min area, food group requirements, linking constraints
- Two annealing algorithms available:
  - simulated_annealing.py: Classical SA
  - simulated_Qannealing.py: Simulated Quantum Annealing (SQA)
- Both expect: objective_function(binary_array) -> energy (lower is better)

## Implementation Complete
- Created benders_decomposition.py with full Benders algorithm
- Supports both classical and quantum annealing for master problem
- PuLP solves subproblem and generates cuts
- Includes comprehensive logging and solution reporting
- JSON output for results and convergence history
- Created compare_benders.py for benchmarking vs standard MILP
- Created BENDERS_README.md with full documentation

## Key Files Created
1. **Scaling Analysis** (October 21, 2025):
   - analyze_pulp_scaling.py - Main scaling analysis script
   - verify_predictions.py - Verification script for predictions
   - PULP_SCALING_SUMMARY.md - Executive summary
   - DETAILED_SCALING_ANALYSIS.md - Detailed results and data
   - Scaling_Analysis/scaling_results_20251021_195409.json - Raw data
   - Scaling_Analysis/scaling_plot_20251021_195409.png - Visualization
   - Scaling_Analysis/scaling_report_20251021_195409.md - Full report

2. **Benders Decomposition**:
   - benders_decomposition.py - Main implementation (800+ lines)
   - compare_benders.py - Comparison script
   - BENDERS_README.md - Comprehensive documentation
   - Test outputs: benders_test_simple.json, benders_improved.json, benders_quantum.json

## Testing Results
- Simple scenario: 3 farms, 6 crops, 18 binary variables
- Classical annealing: ~3s total time, 15 iterations
- Quantum annealing: ~80s total time, 10 iterations  
- Both find feasible solutions with objective ~0.29
- Convergence challenge: gap remains ~108% (expected with heuristic master)

## Known Limitations
- Heuristic master problem may not achieve tight convergence
- Energy function approximates true objective
- Best used as feasible solution generator or for comparison studies
