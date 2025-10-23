# Technical Report: Solver Comparison

## File Location
`tasks/technical_report_solver_comparison.txt`

## Description
Comprehensive LaTeX technical report comparing four food optimization solvers:
1. **solver_runner.py** - Linear objective
2. **solver_runner_NLN.py** - Non-linear with piecewise approximation
3. **solver_runner_NLD.py** - Non-linear with Dinkelbach algorithm
4. **solver_runner_LQ.py** - Linear-quadratic with synergy effects

## Report Contents

### Main Sections
1. **Introduction** - Problem context and overview
2. **Solver Implementations** - Detailed analysis of each solver:
   - Mathematical formulation
   - Problem classification
   - Solution methods
   - Computational complexity
   - Advantages and limitations
3. **Comparative Analysis** - Side-by-side comparison:
   - Variable counts
   - Constraint counts
   - Performance metrics
   - Approximation errors
   - Model realism
4. **Recommendations** - When to use each solver
5. **Implementation Details** - Code structure and usage

### Key Comparisons

| Solver | Variables | Solve Time | Error | Best For |
|--------|-----------|------------|-------|----------|
| Linear | 2n | 0.15s | 0% | Large scale, speed |
| NLN | 14n | 0.45s | 0.1-0.5% | Diminishing returns |
| NLD | 2n | 1.2s | 0% | Efficiency ratios |
| LQ | 2.3n | 0.18s | 0% | Synergy, balance |

## How to Compile

### Option 1: Online (Overleaf)
1. Go to https://www.overleaf.com
2. Create new project → Upload Project
3. Upload `technical_report_solver_comparison.txt`
4. Rename to `.tex` extension
5. Click "Recompile"

### Option 2: Local LaTeX
```bash
# Rename file
cp technical_report_solver_comparison.txt technical_report_solver_comparison.tex

# Compile with pdflatex
pdflatex technical_report_solver_comparison.tex
pdflatex technical_report_solver_comparison.tex  # Run twice for references

# Or use latexmk for automatic compilation
latexmk -pdf technical_report_solver_comparison.tex
```

### Required LaTeX Packages
All standard packages included in most LaTeX distributions:
- amsmath, amssymb, amsthm (math)
- geometry (margins)
- graphicx (figures)
- hyperref (links)
- algorithm, algpseudocode (algorithms)
- booktabs (tables)
- enumitem (lists)

## Report Highlights

### Mathematical Formulations

**Linear:**
```
max Σ [weights × attributes] × A
```

**NLN (Non-Linear):**
```
max Σ [weights × attributes] × A^0.548
```

**NLD (Fractional):**
```
max Σ benefit × A / (Σ cost × A + ε)
```

**LQ (Linear-Quadratic):**
```
max Σ [weights × attributes] × A + 
    w_synergy × Σ boost × Y[c1] × Y[c2]
```

### Key Results

1. **Variable Reduction:** LQ uses 75% fewer variables than NLN
2. **Exactness:** All solvers provide exact solutions except NLN (PuLP) with 0.1-0.5% error
3. **Speed:** Linear and LQ are fastest, NLN is 2-3× slower
4. **Realism:** NLN best models diminishing returns, LQ models synergy effects

### Algorithms Covered

1. **Branch-and-Cut** (CBC for MILP)
2. **Piecewise Linear Approximation** (SOS2 constraints)
3. **Dinkelbach's Algorithm** (Fractional programming)
4. **McCormick Linearization** (Exact for binary products)

## Output Format

The compiled PDF will be approximately 20-25 pages including:
- Detailed mathematical derivations
- Algorithm pseudocode
- Comparative tables
- Complexity analysis
- Usage examples
- Complete references

## For Presentation

Key slides to extract:
1. Table 1: Variable count comparison
2. Table 3: Performance comparison
3. Table 5: Model realism comparison
4. Equations showing each objective function
5. Recommendation decision tree

## Citation

```bibtex
@techreport{solver_comparison_2025,
  title={Technical Report: Comparison of Objective Functions and 
         Solution Methods in Food Optimization Solvers},
  author={EPFL Quantum Optimization Initiative},
  institution={EPFL},
  year={2025},
  type={Technical Report}
}
```

## Contact

For questions or corrections, please refer to the project repository:
https://github.com/oss-esso/OQI-UC002-DWave---Tutorials
