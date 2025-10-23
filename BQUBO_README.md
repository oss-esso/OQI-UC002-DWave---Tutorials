# BQUBO Solver Implementation

## Overview

This implementation enhances the DWave solving approach by converting the Constrained Quadratic Model (CQM) to a Binary Quadratic Model (BQM) and using the HybridBQM solver. This approach enables **more QPU usage** and **better scaling advantages** compared to the CQM solver.

## Key Advantages of BQUBO Approach

1. **Increased QPU Utilization**: The BQM formulation allows the HybridBQM solver to leverage more quantum processing unit (QPU) time
2. **Better Scaling**: Binary formulations often scale better on quantum annealers than mixed-integer formulations
3. **Quadratic Structure**: Pure quadratic problems are more suitable for quantum annealing hardware
4. **Performance Metrics**: Detailed tracking of QPU access time and BQM conversion overhead
5. **Linear Objective**: Uses the same linear objective as the standard solver for direct comparison

## Files Modified/Created

### 1. `solver_runner_BQUBO.py` (Created)

**Key Features:**
- Based on `solver_runner.py` with linear objective
- Added import for `cqm_to_bqm` from dimod
- Added import for `LeapHybridBQMSampler` from dwave.system
- Modified `solve_with_dwave()` function to:
  - Convert CQM to BQM using `dimod.cqm_to_bqm()`
  - Use `LeapHybridBQMSampler` instead of `LeapHybridCQMSampler`
  - Track BQM conversion time separately
  - Track QPU access time from sampleset info
  - Return detailed timing metrics
- Updated main function to:
  - Handle the new return values from `solve_with_dwave()`
  - Display BQM conversion and QPU access times
  - Save detailed DWave results including timing metrics (JSON + pickle)
  - Provide clear output distinguishing BQUBO approach
- **Uses LINEAR objective** (same as solver_runner.py) for direct comparison

### 2. `scalability_benchmark_BQUBO.py` (Created)

A comprehensive benchmarking script for BQUBO approach with **linear objective**:

**Features:**
- Tests 6 problem sizes: [5, 19, 72, 279, 1096, 1535] farms
- Multiple runs per configuration (default: 5 runs) for statistical analysis
- Tracks two solvers: PuLP (classical) and DWave BQUBO (quantum-hybrid)
- Records detailed metrics:
  - Total solve time
  - QPU access time
  - BQM conversion time
  - Problem size characteristics
- **Uses LINEAR objective** for fair comparison
  
**Outputs:**
- Aggregated statistics with mean and standard deviation
- Four visualization plots:
  1. Solve time comparison (PuLP vs DWave on log-log scale)
  2. QPU utilization over problem sizes
  3. Solution quality placeholder
  4. Speedup analysis (DWave vs classical)
- Professional summary table with all metrics
- JSON files with raw and aggregated data

## How to Use

### Running the BQUBO Solver

```powershell
# Basic usage
python solver_runner_BQUBO.py --scenario simple

# With custom parameters
python solver_runner_BQUBO.py --scenario full_family --power 0.548 --breakpoints 10
```

**Arguments:**
- `--scenario`: Scenario to solve (simple, intermediate, full, custom, full_family)
- `--power`: Power for non-linear objective f(A) = A^power (default: 0.548)
- `--breakpoints`: Number of interior breakpoints for piecewise approximation (default: 10)

### Running the Scalability Benchmark

```powershell
# Make sure to set your DWave API token
$env:DWAVE_API_TOKEN = "your-token-here"

# Run the benchmark
python scalability_benchmark_BQUBO.py
```

**Requirements:**
- DWave API token (set as environment variable)
- Adequate Leap quota for multiple solver calls
- ~2-4 hours runtime depending on problem sizes and runs

## Technical Details

### CQM to BQM Conversion

The `dimod.cqm_to_bqm()` function:
1. Takes a CQM with both binary and continuous variables
2. Discretizes continuous variables into multiple binary variables
3. Returns a pure BQM suitable for quantum annealing
4. Provides an inversion function to map BQM solutions back to CQM space

**Trade-offs:**
- **Pro**: Better QPU utilization and potentially faster solving
- **Pro**: More suitable for quantum annealing hardware
- **Con**: Increased number of binary variables (discretization overhead)
- **Con**: Additional conversion time (typically < 1 second)

### HybridBQM vs HybridCQM Solver

| Aspect | HybridCQM | HybridBQM |
|--------|-----------|-----------|
| Input | Mixed binary/continuous | Pure binary |
| QPU Usage | Lower | Higher |
| Scalability | Good | Better |
| Variable Count | Lower | Higher (due to discretization) |
| Best For | Mixed-integer problems | Pure quadratic problems |

## Performance Expectations

Based on the BQUBO approach, you can expect:

1. **QPU Access Time**: Measurable QPU time (microseconds to milliseconds)
2. **Conversion Overhead**: < 1 second for typical problem sizes
3. **Total Solve Time**: Competitive with or better than CQM solver
4. **Scaling**: Better performance on larger problems compared to CQM

## Output Files

### Solver Runner Outputs

- `CQM_Models_NLN/cqm_nln_{scenario}_{timestamp}.cqm` - CQM model file
- `Constraints_NLN/constraints_nln_{scenario}_{timestamp}.json` - Constraint metadata
- `PuLP_Results_NLN/pulp_nln_{scenario}_{timestamp}.json` - PuLP results
- `PuLP_Results_NLN/pyomo_nln_{scenario}_{timestamp}.json` - Pyomo results
- `DWave_Results_NLN/dwave_bqubo_{scenario}_{timestamp}.json` - DWave results (JSON)
- `DWave_Results_NLN/dwave_bqubo_{scenario}_{timestamp}.pickle` - DWave sampleset (pickle)

### Benchmark Outputs

- `benchmark_bqubo_all_runs_{timestamp}.json` - All individual benchmark runs
- `benchmark_bqubo_aggregated_{timestamp}.json` - Aggregated statistics
- `scalability_benchmark_bqubo_{timestamp}.png` - Main visualization (4 subplots)
- `scalability_table_bqubo.png` - Summary table

## Comparison: BQUBO vs Standard CQM

### Standard CQM Approach (solver_runner_NLN.py)
- Uses `LeapHybridCQMSampler`
- Mixed binary and continuous variables
- Direct CQM submission to Leap
- Lower QPU utilization
- Good for prototyping

### BQUBO Approach (solver_runner_BQUBO.py)
- Uses `LeapHybridBQMSampler` after CQM→BQM conversion
- Pure binary formulation
- Discretizes continuous variables
- Higher QPU utilization
- Better for performance and scaling studies

## Next Steps

1. **Run Initial Test**: Start with simple scenario to verify setup
   ```powershell
   python solver_runner_BQUBO.py --scenario simple
   ```

2. **Full Benchmark**: Run complete scalability analysis
   ```powershell
   python scalability_benchmark_BQUBO.py
   ```

3. **Analyze Results**: Review plots and JSON files for insights
   - Compare QPU utilization vs problem size
   - Analyze speedup factors
   - Evaluate approximation quality

4. **Optimize**: Adjust parameters based on results
   - Increase/decrease number of breakpoints
   - Modify problem sizes in benchmark
   - Tune solver parameters

## Troubleshooting

### Issue: "No DWave API token"
**Solution**: Set environment variable
```powershell
$env:DWAVE_API_TOKEN = "your-token-here"
```

### Issue: "Insufficient Leap quota"
**Solution**: 
- Wait for quota to refresh
- Reduce `NUM_RUNS` in benchmark script
- Use fewer problem sizes

### Issue: "BQM conversion fails"
**Solution**:
- Check CQM formulation for issues
- Ensure all variables are properly bounded
- Verify constraint formulation

### Issue: "No feasible solutions"
**Solution**:
- Review constraint feasibility
- Check problem formulation
- Increase solver time limit
- Adjust discretization parameters

## References

- [DWave Ocean SDK Documentation](https://docs.ocean.dwavesys.com/)
- [dimod Reference](https://docs.ocean.dwavesys.com/en/stable/docs_dimod/)
- [Hybrid Solvers Guide](https://docs.ocean.dwavesys.com/en/stable/docs_system/)

## Support

For issues or questions:
1. Check this README
2. Review DWave documentation
3. Check solver logs and error messages
4. Verify API token and quota

---

**Last Updated**: 2025-10-23
**Version**: 1.0
**Approach**: BQUBO (CQM→BQM + HybridBQM)
