# 🚀 Grover Adaptive Search - Quick Start Guide

Get started with Grover Adaptive Search for QUBO problems in under 5 minutes!

## 📦 Installation

### Step 1: Install Cirq

```bash
# Using pip
pip install cirq numpy

# Or using conda
conda install -c conda-forge cirq
conda install numpy
```

### Step 2: Download the Files

- `gas_for_qubo.py` - Original implementation
- `gas_for_qubo_improved.py` - Enhanced version ⭐ **Recommended**

## 🎯 Quick Examples

### Example 1: Simplest Usage

```python
import numpy as np
from gas_for_qubo_improved import ImprovedGroverAdaptiveSearchSolver

# Define your QUBO matrix
Q = np.array([
    [-1, 2],
    [0, -1]
])

# Solve it!
solver = ImprovedGroverAdaptiveSearchSolver(Q)
solution, cost = solver.solve()

print(f"Solution: {solution}")
print(f"Cost: {cost}")
```

**Output:**
```
Solution: [1 0]
Cost: -1
```

---

### Example 2: Silent Mode (Production)

```python
# Run without verbose output
solution, cost = solver.solve(
    max_iterations=15,
    num_restarts=3,
    repetitions=2000,
    verbose=False  # 🔇 Quiet mode
)
```

---

### Example 3: Compare with Classical

```python
# Get quantum solution
quantum_solution, quantum_cost = solver.solve(verbose=False)

# Get classical optimal solution
classical_solution, classical_cost = solver.classical_solve()

# Compare
print(f"Quantum found: {quantum_cost}")
print(f"Classical optimal: {classical_cost}")
print(f"Match: {quantum_cost == classical_cost}")
```

---

### Example 4: MAX-CUT Problem

```python
# Define a graph's MAX-CUT as QUBO
# Graph: 0--1--2--3--0 (square)
Q_maxcut = np.array([
    [-2,  2,  0,  2],
    [ 0, -2,  2,  0],
    [ 0,  0, -2,  2],
    [ 0,  0,  0, -2]
])

solver = ImprovedGroverAdaptiveSearchSolver(Q_maxcut)
solution, cost = solver.solve(
    max_iterations=15,
    num_restarts=3
)

print(f"Partition: {solution}")
print(f"Cut value: {-cost}")  # Negate for maximization
```

---

### Example 5: Custom Parameters

```python
solution, cost = solver.solve(
    max_iterations=20,    # More iterations = more exploration
    num_restarts=5,       # More restarts = better global search
    repetitions=3000,     # More samples = better accuracy
    verbose=True          # See what's happening
)
```

---

## 🎨 Creating Your Own QUBO

### From Scratch

```python
# 3-variable problem
n = 3
Q = np.array([
    [-5,  2,  1],   # Diagonal: individual costs
    [ 0, -3,  2],   # Off-diagonal: interaction costs
    [ 0,  0, -4]
])

# Note: Upper triangular form
# Q[i][j] for i < j represents interaction between variables i and j
```

### From Optimization Problem

```python
# Example: Minimize x₁ + 2x₂ - x₁x₂
# QUBO form: x^T Q x where Q is:
Q = np.array([
    [ 1, -0.5],  # x₁ coefficient and half of interaction
    [ 0,  2  ]   # x₂ coefficient
])
```

---

## 📊 Understanding Results

### Solution Format

```python
solution = [1, 0, 1]  # Binary vector
# Means: variable 0 = 1, variable 1 = 0, variable 2 = 1
```

### Cost Interpretation

```python
cost = -8  # Negative = good (for minimization)
# The algorithm tries to find the most negative cost
```

### Checking Optimality

```python
# Always verify against classical solution
optimal_sol, optimal_cost = solver.classical_solve()

if cost == optimal_cost:
    print("✅ Found optimal solution!")
else:
    print(f"⚠️ Suboptimal: {cost} vs {optimal_cost}")
```

---

## ⚙️ Parameter Tuning Guide

### When to Increase `max_iterations`

```python
# Small problems (n ≤ 3): max_iterations = 10
# Medium problems (n = 4-5): max_iterations = 15
# Larger problems (n ≥ 6): max_iterations = 20+
```

### When to Increase `num_restarts`

```python
# Simple cost landscape: num_restarts = 1-2
# Complex/symmetric problems: num_restarts = 3-5
# Very hard problems: num_restarts = 5-10
```

### When to Increase `repetitions`

```python
# Quick testing: repetitions = 1000
# Normal use: repetitions = 2000
# High accuracy needed: repetitions = 5000+
```

---

## 🐛 Troubleshooting

### Problem: "ModuleNotFoundError: No module named 'cirq'"

**Solution:**
```bash
pip install cirq
```

### Problem: "ValueError: QUBO matrix must be a square 2D array"

**Solution:**
```python
# Make sure your matrix is square
Q = np.array([
    [1, 2],
    [0, 3]
])  # 2x2 ✅

Q = np.array([[1, 2, 3]])  # 1x3 ❌
```

### Problem: Solution seems wrong

**Solution:**
```python
# 1. Increase restarts
solver.solve(num_restarts=5)

# 2. Increase iterations
solver.solve(max_iterations=20)

# 3. Check your QUBO formulation
optimal_sol, optimal_cost = solver.classical_solve()
print(f"Optimal should be: {optimal_cost}")
```

### Problem: Too slow

**Solution:**
```python
# 1. Reduce repetitions
solver.solve(repetitions=1000)

# 2. Reduce restarts
solver.solve(num_restarts=1)

# 3. For n > 8, consider classical methods
if n > 8:
    solution, cost = solver.classical_solve()  # Faster for small n
```

---

## 📈 Performance Tips

### ⚡ Fast Mode (Testing)

```python
solution, cost = solver.solve(
    max_iterations=5,
    num_restarts=1,
    repetitions=1000,
    verbose=False
)
```

### 🎯 Accurate Mode (Production)

```python
solution, cost = solver.solve(
    max_iterations=20,
    num_restarts=5,
    repetitions=3000,
    verbose=False
)
```

### 🔍 Debug Mode (Development)

```python
solution, cost = solver.solve(
    max_iterations=10,
    num_restarts=2,
    repetitions=2000,
    verbose=True  # See all details
)
```

---

## 🔗 Integration Examples

### With Existing Code

```python
def my_optimization_problem(params):
    """Your existing optimization function"""
    # Convert your problem to QUBO
    Q = problem_to_qubo(params)
    
    # Solve with quantum
    solver = ImprovedGroverAdaptiveSearchSolver(Q)
    solution, cost = solver.solve(verbose=False)
    
    return solution, cost
```

### In a Loop

```python
results = []
for problem_instance in problem_list:
    Q = instance_to_qubo(problem_instance)
    solver = ImprovedGroverAdaptiveSearchSolver(Q)
    sol, cost = solver.solve(verbose=False)
    results.append((sol, cost))
```

### Parallel Processing

```python
from concurrent.futures import ProcessPoolExecutor

def solve_instance(Q):
    solver = ImprovedGroverAdaptiveSearchSolver(Q)
    return solver.solve(verbose=False)

# Solve multiple problems in parallel
with ProcessPoolExecutor() as executor:
    results = executor.map(solve_instance, qubo_matrices)
```

---

## 📚 Next Steps

1. **Try the examples**: Run `gas_for_qubo_improved.py`
2. **Read the docs**: Check `GAS_README.md`
3. **See test results**: Review `GAS_FINAL_SUMMARY.md`
4. **Modify for your problem**: Adapt the code to your QUBO

---

## 🆘 Getting Help

### Check the Documentation

- `GAS_README.md` - Comprehensive guide
- `GAS_TEST_REPORT.md` - Detailed test analysis
- `GAS_FINAL_SUMMARY.md` - Complete results

### Common Issues

| Issue | File to Check |
|-------|--------------|
| Installation | This file (Quick Start) |
| Usage examples | This file (Quick Start) |
| Algorithm details | GAS_README.md |
| Performance tuning | GAS_FINAL_SUMMARY.md |
| Test results | GAS_TEST_REPORT.md |

---

## ✅ Checklist for Your First Run

- [ ] Cirq installed (`pip install cirq`)
- [ ] NumPy installed (`pip install numpy`)
- [ ] Downloaded `gas_for_qubo_improved.py`
- [ ] Created a QUBO matrix
- [ ] Ran `solver.solve()`
- [ ] Compared with `solver.classical_solve()`
- [ ] Verified results make sense

---

## 🎉 Success!

If you got this far, you're ready to use Grover Adaptive Search for your QUBO problems!

```python
# Your first working example:
import numpy as np
from gas_for_qubo_improved import ImprovedGroverAdaptiveSearchSolver

Q = np.array([[-1, 2], [0, -1]])
solver = ImprovedGroverAdaptiveSearchSolver(Q)
solution, cost = solver.solve()
print(f"✅ Solution: {solution}, Cost: {cost}")
```

**Happy Quantum Computing! 🚀**

---

*Last Updated: October 21, 2025*  
*Version: 1.0*  
*Cirq Version: 1.6.1*
