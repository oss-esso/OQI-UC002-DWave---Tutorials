# Benchmark Scalability NLN - Quick Reference

## Overview
This benchmark tests the scalability of the **Non-Linear (NLN)** optimization solvers with objective function f(A) = A^0.548.

## Configuration

### Test Points: 6 configurations
- 5, 19, 72, 279, 1096, 1535 farms
- Logarithmically spaced from small to large problems

### Multiple Runs: 5 runs per configuration
- Statistical analysis with mean and standard deviation
- Error bars on all plots
- Variance tracking for reliability analysis

### Solvers Tested:
1. **PuLP** - Piecewise linear approximation (10 interior breakpoints)
2. **Pyomo + IPOPT** - True non-linear objective (no approximation)
3. **DWave** - Disabled (no token for testing)

### Non-Linear Parameters:
- **Power**: 0.548 (diminishing returns)
- **Breakpoints**: 10 interior points (12 total with endpoints)

## Running the Benchmark

### Basic Usage:
```bash
python benchmark_scalability_NLN.py
```

### What It Does:
1. Tests 6 farm configurations (5 to 1535 farms)
2. Runs each configuration 5 times
3. Total: 30 benchmark runs
4. Calculates statistics: mean, std, min, max
5. Generates plots and tables

### Expected Runtime:
- Small problems (5-72 farms): ~1-5 seconds per run
- Medium problems (279 farms): ~10-30 seconds per run
- Large problems (1096-1535 farms): ~60-300 seconds per run
- **Total estimated time**: 30-60 minutes for all 30 runs

## Output Files

### JSON Results:
1. `benchmark_nln_all_runs_TIMESTAMP.json`
   - All 30 individual runs with full details
   - Raw timing data for each run

2. `benchmark_nln_aggregated_TIMESTAMP.json`
   - Aggregated statistics per configuration
   - Mean, std, min, max for each metric

### Plots:
1. `scalability_benchmark_nln_TIMESTAMP.png`
   - Two-panel plot:
     - Left: Solve time vs problem size (with error bars)
     - Right: Approximation error vs problem size (with error bars)

2. `scalability_table_nln.png`
   - Summary table with all statistics

## Key Metrics Tracked

### For Each Configuration:
- **Problem Size**: n = farms × foods
- **Variables**: Base (A, Y) + Lambda variables
- **Constraints**: Total constraint count
- **CQM Creation Time**: Time to build the CQM model
- **PuLP Solve Time**: Mean ± std across 5 runs
- **Pyomo Solve Time**: Mean ± std across 5 runs
- **Approximation Error**: PuLP vs Pyomo (%) with std

## Understanding the Results

### Solve Time Plot:
- **X-axis**: Problem size (n = farms × foods)
- **Y-axis**: Solve time (seconds, log scale)
- **Error bars**: Standard deviation across 5 runs
- **PuLP**: Blue line (piecewise approximation)
- **Pyomo**: Purple line (true non-linear)

### Approximation Error Plot:
- **X-axis**: Problem size (n = farms × foods)
- **Y-axis**: Approximation error (%)
- **Error bars**: Standard deviation across 5 runs
- **Reference lines**: 
  - Green dashed: 1% error threshold
  - Orange dashed: 5% error threshold

### Winner Determination:
- 🏆 **PuLP**: Faster solve time
- 🏆 **Pyomo**: More accurate (ground truth)
- Shows speed vs accuracy tradeoff

## Expected Results

### Approximation Quality:
- **Small problems**: 1-3% error typical
- **Large problems**: May increase slightly with problem size
- **Consistency**: Error should be relatively stable across runs

### Solve Time Scaling:
- **PuLP**: Generally faster, especially for larger problems
- **Pyomo**: More accurate but potentially slower
- **Scaling**: Both should show polynomial growth

### Statistical Reliability:
- **Low std**: Consistent performance across runs
- **High std**: Variability might indicate solver sensitivity

## Customization

### Change Number of Runs:
```python
NUM_RUNS = 10  # Change from 5 to 10 for more data
```

### Change Test Points:
```python
BENCHMARK_CONFIGS = [5, 50, 100, 500]  # Custom points
```

### Change Approximation Quality:
```python
NUM_BREAKPOINTS = 20  # More accurate (default: 10)
POWER = 0.7  # Different power function (default: 0.548)
```

## Troubleshooting

### Pyomo Fails:
- Check if IPOPT is installed: `conda install -c conda-forge ipopt`
- Error will be logged but benchmark continues with PuLP only

### Memory Issues:
- Reduce `BENCHMARK_CONFIGS` to smaller problems
- Reduce `NUM_RUNS` from 5 to 3

### Timeout on Large Problems:
- Normal for 1535 farms configuration
- May take 5-10 minutes per run
- Consider reducing to 1096 max farms

## Comparison with Linear Benchmark

### Linear (`benchmark_scalability.py`):
- Objective: f(A) = A (linear)
- Solvers: PuLP, DWave
- Focus: Classical vs Quantum speedup

### Non-Linear (`benchmark_scalability_NLN.py`):
- Objective: f(A) = A^0.548 (non-linear)
- Solvers: PuLP (approx), Pyomo (exact)
- Focus: Approximation accuracy vs speed

## Notes

- **DWave disabled**: No token for testing (can be enabled later)
- **Pyomo uses IPOPT**: Relaxes binary variables to continuous [0,1]
- **PuLP uses CBC**: True MILP solver with binary variables
- **Multiple runs**: Provides statistical confidence in results
- **Error bars**: Show consistency and reliability of solvers

## Summary

This benchmark provides comprehensive analysis of:
1. ✅ Scalability of piecewise approximation approach
2. ✅ Quality vs speed tradeoffs
3. ✅ Statistical reliability across multiple runs
4. ✅ Approximation error characterization
5. ✅ Production-ready performance metrics

Perfect for presentations and research papers! 📊
