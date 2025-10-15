# D-Wave QUBO/DIMOD Tutorials

Educational Python scripts demonstrating how to build optimization scenarios into DIMOD and QUBO models for D-Wave quantum annealers.

## Overview

These tutorials teach you how to:
1. Build Binary Quadratic Models (BQM) using DIMOD
2. Formulate optimization problems as QUBO
3. Convert real-world scenarios into QUBO formulations
4. Integrate with D-Wave solvers (simulator and QPU)
5. Implement complete end-to-end workflows
6. Use Constrained Quadratic Models (CQM) with hard constraints

## Prerequisites

```bash
# Required packages
pip install dimod numpy

# Optional (for QPU access)
pip install dwave-ocean-sdk
```

## Tutorial Structure

### Tutorial 1: Basic DIMOD BQM Construction
**File:** `tutorial_01_basic_dimod.py`

Learn the fundamentals of Binary Quadratic Models:
- Creating BQMs with linear and quadratic terms
- Different construction methods
- Solving with SimulatedAnnealingSampler
- Understanding energy landscapes
- Practical resource allocation example

**Run:**
```bash
python tutorial_01_basic_dimod.py
```

**Key Concepts:**
- BQM components (linear, quadratic, offset)
- Energy minimization
- Variable interactions
- Constraint penalties

### Tutorial 2: QUBO Formulation Basics
**File:** `tutorial_02_qubo_basics.py`

Master QUBO (Quadratic Unconstrained Binary Optimization):
- QUBO matrix representation
- Encoding constraints as penalties
- Number partitioning problem
- Comparing different samplers
- QUBO ↔ Ising conversions

**Run:**
```bash
python tutorial_02_qubo_basics.py
```

**Key Concepts:**
- QUBO matrix structure
- Penalty-based constraints
- Sampler comparison (Exact, SimulatedAnnealing, Random)
- Formulation equivalence

### Tutorial 3: Scenario to QUBO Conversion
**File:** `tutorial_03_scenario_to_qubo.py`

Bridge theory to practice with real scenarios:
- Loading food production scenario data
- Building objective functions
- Adding land availability constraints
- Implementing diversity bonuses
- Tuning penalty weights

**Run:**
```bash
python tutorial_03_scenario_to_qubo.py
```

**Key Concepts:**
- Real-world problem formulation
- Multi-objective optimization
- Constraint encoding strategies
- Penalty weight tuning

### Tutorial 4: D-Wave Solver Integration
**File:** `tutorial_04_dwave_integration.py`

Implement plug-and-play solver selection:
- Using SimulatedAnnealingSampler (local, free)
- QPU workflow structure
- Hybrid solver benefits
- Configuration management
- Token handling

**Run:**
```bash
python tutorial_04_dwave_integration.py
```

**Key Concepts:**
- Solver types and selection
- API token management
- Cost considerations
- Development vs production

### Tutorial 5: Complete Workflow
**File:** `tutorial_05_complete_workflow.py`

End-to-end implementation:
- Step-by-step workflow execution
- QUBO builder class design
- Solution interpretation
- Multi-solver comparison
- Production code structure

**Run:**
```bash
python tutorial_05_complete_workflow.py
```

**Key Concepts:**
- Complete workflow integration
- Modular code design
- Solution validation
- Best practices

### Tutorial 6: Scenario to CQM (Constrained Quadratic Model)
**File:** `tutorial_06_scenario_to_cqm.py`

Learn CQM formulation with hard constraints:
- Understanding CQM vs BQM differences
- Using Integer and Binary variables together
- Adding hard constraints (always satisfied)
- No penalty weight tuning needed
- When to use CQM over BQM

**Run:**
```bash
python tutorial_06_scenario_to_cqm.py
```

**Key Concepts:**
- CQM hard constraints vs BQM soft penalties
- Multi-type variables (Binary, Integer, Real)
- LeapHybridCQMSampler usage
- Feasibility guarantees
- Advanced constraint formulation

## Running the Tutorials

### Individual Tutorials
```bash
# Run any tutorial directly
python tutorial_01_basic_dimod.py
python tutorial_02_qubo_basics.py
python tutorial_03_scenario_to_qubo.py
python tutorial_04_dwave_integration.py
python tutorial_05_complete_workflow.py
python tutorial_06_scenario_to_cqm.py
```

### Run All Tests
```bash
# Verify all tutorials work correctly
python test_tutorials.py
```

**Test Results:**
- 26 tests covering all tutorials
- All tests use simulator (no token required)
- Verifies functionality and integration

## Features

### Plug-and-Play Solver Selection
Switch between simulator and real quantum hardware with minimal code changes:

```python
# Development (free, local)
sampler = dimod.SimulatedAnnealingSampler()

# Production (requires token)
from dwave.system import DWaveSampler, EmbeddingComposite
sampler = EmbeddingComposite(DWaveSampler())
```

### Configuration Management
```python
# Set API token as environment variable (Windows PowerShell)
$env:DWAVE_API_TOKEN = 'your-token-here'

# Or use dwave CLI
dwave config create
dwave ping
```

## Problem Sizes

**Solver Recommendations:**
- **< 20 variables**: ExactSolver (guaranteed optimal)
- **20-100 variables**: SimulatedAnnealing or QPU
- **100-1000 variables**: QPU with embedding
- **> 1000 variables**: Hybrid Solver

## Example Output

```
======================================================================
TUTORIAL 1: BASIC DIMOD BQM CONSTRUCTION
======================================================================

EXAMPLE 1: Simple BQM Construction
BQM Structure:
  Number of variables: 3
  Variables: ['x0', 'x1', 'x2']
  Linear terms: {'x0': 1.0, 'x1': -2.0, 'x2': 3.0}
  Quadratic terms: {('x0', 'x1'): -1.0, ('x1', 'x2'): 2.0}

Solving with Simulated Annealing...
Best solution found:
  Variables: {'x0': 1, 'x1': 1, 'x2': 0}
  Energy: -2.0
```

## Key Takeaways

### Tutorial 1
- BQMs consist of linear terms, quadratic terms, and an offset
- Variables are binary (0 or 1)
- The goal is to minimize the energy function
- Constraints can be added as penalty terms

### Tutorial 2
- QUBO is a matrix formulation: minimize x^T Q x
- Constraints encode as penalty terms (violation^2 × weight)
- ExactSolver finds optimal but is slow for large problems
- SimulatedAnnealing is a fast heuristic sampler

### Tutorial 3
- Start by understanding your scenario data structure
- Define clear decision variables (binary for QUBO)
- Formulate objective as linear/quadratic terms
- Tune penalty weights to balance objective and constraints

### Tutorial 4
- SimulatedAnnealing is perfect for development (free, local)
- QPU requires API token and has usage costs
- Hybrid solvers handle large problems (1000+ variables)
- Use configuration objects for flexible solver selection

### Tutorial 5
- Start with clear data loading and validation
- Use builder classes for complex QUBO formulations
- Test with multiple samplers to verify results
- Always interpret solutions in the original problem context

### Tutorial 6
- CQM provides hard constraints (always satisfied)
- No penalty weight tuning required
- Supports Integer and Real variables alongside Binary
- Use LeapHybridCQMSampler (requires D-Wave API)
- Better when feasibility is critical (constraints MUST be met)

## Troubleshooting

### Import Errors
```bash
# Install required packages
pip install dimod numpy

# For QPU access
pip install dwave-ocean-sdk
```

### No D-Wave Token
All tutorials work with the free simulator. For QPU access:
```bash
# Set environment variable
$env:DWAVE_API_TOKEN = 'DEV-...'

# Or configure via CLI
dwave config create
```

### Test Failures
```bash
# Run tests to identify issues
python test_tutorials.py

# Check individual tutorial
python tutorial_01_basic_dimod.py
```

## Additional Resources

- **D-Wave Ocean Documentation**: https://docs.ocean.dwavesys.com/
- **DIMOD Documentation**: https://docs.ocean.dwavesys.com/projects/dimod/
- **D-Wave Leap**: https://cloud.dwavesys.com/leap/ (Free account)
- **Project Documentation**: WORKSPACE_TECHNICAL_REPORT.md

## License

Educational materials for OQI-UC002-DWave project.

## Author

Created for teaching QUBO/DIMOD concepts with D-Wave quantum annealers.
