# Professional Solver Workflow

This document describes the professional solver workflow for food optimization problems.

## Overview

The workflow consists of two main scripts:
1. **solver_runner.py** - Runs both solvers and saves all results
2. **verifier.py** - Verifies and compares the solutions

## Directory Structure

```
Project Root/
├── solver_runner.py          # Main solver runner script
├── verifier.py                # Solution verifier script
├── CQM_Models/               # Saved CQM models (.cqm files)
├── Constraints/              # Constraint metadata (.json files)
├── PuLP_Results/             # PuLP solution results (.json files)
├── DWave_Results/            # DWave solution results (.pickle files)
├── run_manifest_*.json       # Run manifests linking all components
└── verification_report_*.json # Verification reports
```

## Usage

### Step 1: Run the Solver

```bash
python solver_runner.py --scenario simple
```

**Options:**
- `--scenario`: Choose from `simple`, `intermediate`, or `custom` (default: `simple`)

**What it does:**
1. Loads the specified scenario
2. Creates a CQM formulation
3. Saves the CQM model to `CQM_Models/`
4. Saves constraint metadata to `Constraints/`
5. Solves with PuLP and saves results to `PuLP_Results/`
6. Solves with DWave and saves results to `DWave_Results/`
7. Creates a run manifest file: `run_manifest_<scenario>_<timestamp>.json`

**Output:**
```
================================================================================
PROFESSIONAL SOLVER RUNNER
================================================================================

Loading 'simple' scenario...
  Farms: 3 - ['Farm1', 'Farm2', 'Farm3']
  Foods: 6 - ['Wheat', 'Corn', 'Rice', 'Soybeans', 'Potatoes', 'Apples']

Creating CQM...
  Variables: 36
  Constraints: 39

Saving CQM to CQM_Models/cqm_simple_20251019_214925.cqm...
Saving constraints to Constraints/constraints_simple_20251019_214925.json...

================================================================================
SOLVING WITH PULP
================================================================================
  Status: Optimal
  Objective: 0.325000
  Solve time: 0.04 seconds

Saving PuLP results to PuLP_Results/pulp_simple_20251019_214925.json...

================================================================================
SOLVING WITH DWAVE
================================================================================
Submitting to DWave Leap hybrid solver...
  Feasible solutions: 124 of 125
  Solve time: 3.85 seconds
  Best energy: -73.125000

Saving DWave results to DWave_Results/dwave_simple_20251019_214925.pickle...

Saving run manifest to run_manifest_simple_20251019_214925.json...

================================================================================
SOLVER RUN COMPLETE
================================================================================
Manifest file: run_manifest_simple_20251019_214925.json

Run the verifier script with this manifest to check results:
  python verifier.py run_manifest_simple_20251019_214925.json
```

### Step 2: Verify the Results

```bash
python verifier.py run_manifest_simple_20251019_214925.json
```

**What it does:**
1. Loads all components from the manifest
2. Verifies PuLP solution against constraints
3. Verifies DWave solution against constraints
4. Compares the two solutions
5. Generates a verification report

**Output:**
```
================================================================================
PROFESSIONAL SOLUTION VERIFIER
================================================================================

Loading manifest: run_manifest_simple_20251019_214925.json
  Scenario: simple
  Timestamp: 20251019_214925

Loading components...
  CQM: CQM_Models/cqm_simple_20251019_214925.cqm
  Constraints: Constraints/constraints_simple_20251019_214925.json
  PuLP results: PuLP_Results/pulp_simple_20251019_214925.json
  DWave results: DWave_Results/dwave_simple_20251019_214925.pickle

================================================================================
VERIFYING PULP SOLUTION
================================================================================

  Land Availability: PASS
  Linking Constraints: PASS
  Food Group Constraints: PASS
  STATUS: ALL CONSTRAINTS SATISFIED

================================================================================
VERIFYING DWAVE SOLUTION
================================================================================

  Land Availability: PASS
  Linking Constraints: PASS
  Food Group Constraints: PASS
  STATUS: ALL CONSTRAINTS SATISFIED

================================================================================
SOLUTION COMPARISON
================================================================================

Farm1:
  Food            | PuLP Area    | DWave Area   | Match
  ----------------+--------------+--------------+-------
  Wheat           | 0.00         | 0.00         | YES
  Corn            | 0.00         | 0.00         | YES
  Rice            | 0.00         | 0.00         | YES
  Soybeans        | 75.00        | 75.00        | YES
  Potatoes        | 0.00         | 0.00         | YES
  Apples          | 0.00         | 0.00         | YES

...

================================================================================
OBJECTIVE COMPARISON
================================================================================

  PuLP Objective:  0.325000
  DWave Objective: 0.325000
  Difference:      0.000000
  Status: IDENTICAL

================================================================================
Saving verification report to: verification_report_simple_20251019_214925.json
================================================================================

VERIFICATION SUMMARY:
  PuLP: PASS (0 violations)
  DWave: PASS (0 violations)
  Solutions match: YES (0 differences)
  Objectives match: YES

  OVERALL: PERFECT MATCH - Both solvers found the same valid solution!
```

## File Formats

### Run Manifest (`run_manifest_<scenario>_<timestamp>.json`)

Links all components of a solver run:

```json
{
  "scenario": "simple",
  "timestamp": "20251019_214925",
  "cqm_path": "CQM_Models/cqm_simple_20251019_214925.cqm",
  "constraints_path": "Constraints/constraints_simple_20251019_214925.json",
  "pulp_path": "PuLP_Results/pulp_simple_20251019_214925.json",
  "dwave_path": "DWave_Results/dwave_simple_20251019_214925.pickle",
  "farms": ["Farm1", "Farm2", "Farm3"],
  "foods": ["Wheat", "Corn", "Rice", "Soybeans", "Potatoes", "Apples"],
  "pulp_status": "Optimal",
  "pulp_objective": 0.325,
  "dwave_feasible_count": 124,
  "dwave_total_count": 125
}
```

### Constraints File (`Constraints/constraints_<scenario>_<timestamp>.json`)

Contains all constraint metadata for verification:

```json
{
  "scenario": "simple",
  "timestamp": "20251019_214925",
  "farms": ["Farm1", "Farm2", "Farm3"],
  "foods": ["Wheat", "Corn", "Rice", "Soybeans", "Potatoes", "Apples"],
  "food_groups": {...},
  "config": {...},
  "constraint_metadata": {
    "land_availability": {...},
    "min_area_if_selected": {...},
    "max_area_if_selected": {...},
    "food_group_min": {...},
    "food_group_max": {...}
  }
}
```

### PuLP Results (`PuLP_Results/pulp_<scenario>_<timestamp>.json`)

```json
{
  "status": "Optimal",
  "objective_value": 0.325,
  "solve_time": 0.04,
  "areas": {
    "Farm1_Wheat": 0.0,
    "Farm1_Soybeans": 75.0,
    ...
  },
  "selections": {
    "Farm1_Wheat": 0.0,
    "Farm1_Soybeans": 1.0,
    ...
  }
}
```

### DWave Results (`DWave_Results/dwave_<scenario>_<timestamp>.pickle`)

Binary pickle file containing the complete DWave sampleset object with:
- All solutions (feasible and infeasible)
- Variable values
- Energies
- Timing information
- Problem metadata

### Verification Report (`verification_report_<scenario>_<timestamp>.json`)

```json
{
  "manifest": {...},
  "verification_timestamp": "2025-10-19T21:49:30.123456",
  "pulp_verification": {
    "violations": [],
    "passed": true,
    "objective": 0.325
  },
  "dwave_verification": {
    "violations": [],
    "passed": true,
    "objective": 0.325
  },
  "comparison": {
    "differences": [],
    "solutions_match": true,
    "objectives_match": true
  }
}
```

## Constraint Verification

The verifier checks:

1. **Land Availability**: Total area per farm <= available land
2. **Linking Constraints**:
   - If food is selected (Y=1), area >= minimum planting area
   - If food is not selected (Y=0), area = 0
3. **Food Group Constraints**: Number of foods selected from each group is within min/max bounds

## Integration with Existing Code

This workflow is compatible with:
- `dwave-test.py` - Uses the same CQM formulation
- `pulp_2.py` - Uses the same PuLP formulation
- `src/scenarios.py` - Loads scenarios from the same source
- `Tests/test_hybrid_solver.py` - Similar structure but with integrated comparison

## Benefits

1. **Reproducibility**: All components saved with timestamps
2. **Traceability**: Run manifests link all related files
3. **Verification**: Automatic constraint checking and solution comparison
4. **Professional**: Clean separation of concerns
5. **Flexible**: Easy to run different scenarios
6. **Comprehensive**: Saves CQM models, constraints, and both solutions

## Example Workflow

```bash
# Run solver on simple scenario
python solver_runner.py --scenario simple

# Verify the results (use the manifest file name from previous output)
python verifier.py run_manifest_simple_20251019_214925.json

# Run on different scenario
python solver_runner.py --scenario intermediate

# Verify
python verifier.py run_manifest_intermediate_20251019_215130.json
```

## Notes

- CQM files are saved in binary format using `cqm.to_file()`
- DWave results are pickled sampleset objects
- PuLP results are JSON for easy inspection
- All files include timestamps to avoid overwriting
- Manifest files provide complete audit trail
