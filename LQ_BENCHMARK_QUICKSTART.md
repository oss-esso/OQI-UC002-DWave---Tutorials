# Quick Start Guide: LQ Benchmark

## Run the Benchmark

```bash
# Activate your conda environment
conda activate oqi_project

# Run the full benchmark suite
python benchmark_scalability_LQ.py
```

## What It Does

1. **Tests 6 problem sizes:** 5, 19, 72, 279, 1096, 1535 farms
2. **Runs each 5 times** for statistical analysis
3. **Total: 30 benchmarks** (~5-10 minutes)
4. **Generates:**
   - JSON results with all data
   - Performance plots (solve time comparison)
   - Accuracy plots (solution validation)
   - Summary table

## Expected Output Files

```
benchmark_lq_all_runs_20251023_HHMMSS.json
benchmark_lq_aggregated_20251023_HHMMSS.json
scalability_benchmark_lq_20251023_HHMMSS.png
scalability_table_lq.png
```

## Quick Test (Single Run)

If you want to test just one configuration quickly:

```python
# In Python console
from benchmark_scalability_LQ import run_benchmark

# Test with 5 farms
result = run_benchmark(n_farms=5, run_number=1, total_runs=1)
print(result)
```

## Key Metrics to Watch

1. **Variable Count:**
   - PuLP adds Z variables for linearization
   - CQM/Pyomo use fewer variables (native quadratic)

2. **Solve Time:**
   - PuLP (linearized) vs Pyomo (native)
   - Should scale sub-linearly with problem size

3. **Solution Accuracy:**
   - PuLP vs Pyomo difference
   - Should be ~0% (exact linearization)

4. **Synergy Pairs:**
   - More pairs → more Z variables in PuLP
   - Affects linearization overhead

## Compare with NLN

Run both benchmarks and compare:

```bash
# Run NLN benchmark
python benchmark_scalability_NLN.py

# Run LQ benchmark
python benchmark_scalability_LQ.py

# Results:
# - LQ uses 50-70% fewer variables
# - LQ has 0% approximation error
# - Solve times comparable or faster
```

## Troubleshooting

**If Pyomo fails:**
- Check solver availability: `conda install -c conda-forge glpk`
- Script will continue with PuLP results

**If out of memory:**
- Reduce BENCHMARK_CONFIGS to smaller values
- Reduce NUM_RUNS to 3 or 1

**If too slow:**
- Remove largest configurations (1096, 1535)
- Use only: [5, 19, 72, 279]

## Understanding Results

### Solve Time Plot
- X-axis: Problem size (n = farms × foods)
- Y-axis: Solve time (log scale)
- Lines: PuLP vs Pyomo with error bars

### Accuracy Plot
- Shows PuLP-Pyomo difference
- Should be near 0% (green line)
- Validates exact linearization

### Summary Table
- Compare solvers side-by-side
- Winner column shows fastest
- Stats show consistency across runs

🎉 **Ready to benchmark!**
