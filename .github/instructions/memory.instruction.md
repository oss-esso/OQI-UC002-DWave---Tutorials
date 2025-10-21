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
1. benders_decomposition.py - Main implementation (800+ lines)
2. compare_benders.py - Comparison script
3. BENDERS_README.md - Comprehensive documentation
4. Test outputs: benders_test_simple.json, benders_improved.json, benders_quantum.json

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
