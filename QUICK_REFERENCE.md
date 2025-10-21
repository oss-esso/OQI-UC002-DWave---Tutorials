# Quick Reference: PuLP Scaling Results

## 🎯 Direct Answer to Your Question

| Target Time | n (farms × foods) | Number of Farms | Verified Time | Error |
|-------------|-------------------|-----------------|---------------|-------|
| **5.0 sec** | **29,882** | **2,988** | 4.678 sec | 6.4% |
| **6.5 sec** | **34,375** | **3,438** | 6.252 sec | 3.8% |

**Formula**: Polynomial model `T = 5.13e-09 × n² + 1.09e-05 × n + 0.10` (R² = 0.9965)

---

## 📊 Scaling Quick Reference

| n | Farms | Approx Time | Use Case |
|---|-------|-------------|----------|
| 100 | 10 | 0.04s | Dev/Testing |
| 1,000 | 100 | 0.10s | Interactive |
| 5,000 | 500 | 0.56s | Real-time |
| 10,000 | 1,000 | 1.26s | Batch OK |
| **29,882** | **2,988** | **~5.0s** | **Target 1** ✅ |
| **34,375** | **3,438** | **~6.5s** | **Target 2** ✅ |
| 50,000 | 5,000 | 13.8s | Large-scale |

**Scaling**: O(n²) - Quadratic

---

## 🚀 How to Use

```python
# For 5-second solve time
from farm_sampler import generate_farms
L = generate_farms(n_farms=2988, seed=42)

# For 6.5-second solve time
L = generate_farms(n_farms=3438, seed=42)
```

---

## 📁 Key Files

1. **Main Analysis**: `analyze_pulp_scaling.py`
2. **Verification**: `verify_predictions.py`
3. **Summary**: `PULP_SCALING_SUMMARY.md`
4. **Detailed**: `DETAILED_SCALING_ANALYSIS.md`
5. **Complete**: `README_SCALING_COMPLETE.md`

---

**Date**: October 21, 2025  
**Status**: ✅ VERIFIED  
**Accuracy**: 5.1% average error
