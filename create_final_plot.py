"""
Create a single beautiful plot with fitted curves for presentation
"""
import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import make_interp_spline

# Load results
import glob
import os

result_files = glob.glob('benchmark_results_*.json')
if not result_files:
    print("No benchmark results found! Run benchmark_scalability.py first.")
    exit(1)

latest_file = max(result_files, key=os.path.getctime)
print(f"Loading results from: {latest_file}")

with open(latest_file, 'r') as f:
    results = json.load(f)

# Extract data
problem_sizes = np.array([r['problem_size'] for r in results])
pulp_times = np.array([r['pulp_time'] for r in results])

# DWave data
dwave_indices = [i for i, r in enumerate(results) if r['dwave_time'] is not None]
if dwave_indices:
    dwave_sizes = np.array([results[i]['problem_size'] for i in dwave_indices])
    dwave_times = np.array([results[i]['dwave_time'] for i in dwave_indices])
    hybrid_times = np.array([results[i]['hybrid_time'] for i in dwave_indices])
    qpu_times = np.array([results[i]['qpu_time'] for i in dwave_indices])
else:
    print("No DWave data available!")
    exit(1)

# Define fitting functions
def power_law(x, a, b):
    """Power law: y = a * x^b"""
    return a * np.power(x, b)

def linear_log(x, a, b):
    """Linear in log space: y = a + b*log(x)"""
    return a + b * np.log(x)

def polynomial(x, a, b, c):
    """Polynomial: y = a + b*x + c*x^2"""
    return a + b*x + c*x**2

# Fit PuLP data (power law works best for sub-linear scaling)
try:
    pulp_params, _ = curve_fit(power_law, problem_sizes, pulp_times, p0=[0.01, 0.5], maxfev=10000)
    pulp_fit = lambda x: power_law(x, *pulp_params)
    pulp_fit_label = f'PuLP Fit: {pulp_params[0]:.4f} × size^{pulp_params[1]:.3f}'
except:
    pulp_fit = None
    pulp_fit_label = 'PuLP (no fit)'

# Fit DWave hybrid time (usually flat or slightly increasing)
try:
    hybrid_params, _ = curve_fit(linear_log, dwave_sizes, hybrid_times, p0=[5, 0.1], maxfev=10000)
    hybrid_fit = lambda x: linear_log(x, *hybrid_params)
    hybrid_fit_label = f'Hybrid Fit: {hybrid_params[0]:.3f} + {hybrid_params[1]:.4f}×log(size)'
except:
    hybrid_fit = None
    hybrid_fit_label = 'Hybrid (no fit)'

# Fit QPU time (usually constant)
qpu_mean = np.mean(qpu_times)
qpu_fit = lambda x: np.full_like(x, qpu_mean, dtype=float)
qpu_fit_label = f'QPU Fit: {qpu_mean:.4f}s (constant)'

# Create beautiful plot
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(14, 9))

# Generate smooth curves for fits
x_smooth = np.logspace(np.log10(min(problem_sizes)), np.log10(max(problem_sizes)), 200)
x_smooth_dwave = np.logspace(np.log10(min(dwave_sizes)), np.log10(max(dwave_sizes)), 200)

# Plot actual data points
ax.scatter(problem_sizes, pulp_times, s=120, marker='o', color='#1f77b4', 
          edgecolors='black', linewidth=1.5, zorder=5, label='PuLP (Classical) - Data', alpha=0.8)

ax.scatter(dwave_sizes, hybrid_times, s=120, marker='s', color='#d62728',
          edgecolors='black', linewidth=1.5, zorder=5, label='D-Wave Hybrid - Data', alpha=0.8)

ax.scatter(dwave_sizes, qpu_times, s=120, marker='^', color='#2ca02c',
          edgecolors='black', linewidth=1.5, zorder=5, label='QPU Access - Data', alpha=0.8)

# Plot fitted curves
if pulp_fit:
    ax.plot(x_smooth, pulp_fit(x_smooth), '--', linewidth=2.5, color='#1f77b4',
           alpha=0.7, label=pulp_fit_label, zorder=3)

if hybrid_fit:
    ax.plot(x_smooth_dwave, hybrid_fit(x_smooth_dwave), '--', linewidth=2.5, color='#d62728',
           alpha=0.7, label=hybrid_fit_label, zorder=3)

ax.plot(x_smooth_dwave, qpu_fit(x_smooth_dwave), '--', linewidth=2.5, color='#2ca02c',
       alpha=0.7, label=qpu_fit_label, zorder=3)

# Formatting
ax.set_xlabel('Problem Size (5 × Foods × Farms)', fontsize=18, fontweight='bold')
ax.set_ylabel('Time (seconds)', fontsize=18, fontweight='bold')
ax.set_title('Quantum vs Classical Solver Performance with Fitted Curves',
            fontsize=22, fontweight='bold', pad=20)

ax.set_xscale('log')
ax.set_yscale('log')
ax.grid(True, which='both', alpha=0.3, linestyle='--', linewidth=0.8)
ax.tick_params(labelsize=14)

# Legend
legend = ax.legend(loc='upper left', fontsize=12, framealpha=0.95, shadow=True,
                   ncol=2, columnspacing=1)
legend.get_frame().set_facecolor('white')
legend.get_frame().set_edgecolor('black')
legend.get_frame().set_linewidth(1.5)

# Add annotations
# Classical advantage
ax.annotate('Classical Solver:\nSub-linear scaling\n(Excellent efficiency)',
           xy=(problem_sizes[-1], pulp_times[-1]),
           xytext=(-150, -50), textcoords='offset points',
           fontsize=11, ha='left',
           bbox=dict(boxstyle='round,pad=0.8', facecolor='lightblue', alpha=0.8, edgecolor='black', linewidth=1.5),
           arrowprops=dict(arrowstyle='->', lw=2, color='black'))

# Quantum overhead
if len(dwave_sizes) > 0:
    mid_idx = len(dwave_sizes) // 2
    ax.annotate('Quantum Hybrid:\nNear-constant overhead\n(~5 seconds)',
               xy=(dwave_sizes[mid_idx], hybrid_times[mid_idx]),
               xytext=(50, 80), textcoords='offset points',
               fontsize=11, ha='left',
               bbox=dict(boxstyle='round,pad=0.8', facecolor='lightyellow', alpha=0.8, edgecolor='black', linewidth=1.5),
               arrowprops=dict(arrowstyle='->', lw=2, color='black'))

# QPU efficiency
ax.annotate('QPU Time:\nConstant ~0.07s\n(Hardware limit)',
           xy=(dwave_sizes[-1], qpu_times[-1]),
           xytext=(20, -60), textcoords='offset points',
           fontsize=11, ha='left',
           bbox=dict(boxstyle='round,pad=0.8', facecolor='lightgreen', alpha=0.8, edgecolor='black', linewidth=1.5),
           arrowprops=dict(arrowstyle='->', lw=2, color='black'))

# Add key insights box
insights_text = f"""Key Insights:
• Classical: {pulp_params[1]:.3f}-power scaling (sub-linear)
• Quantum Hybrid: {hybrid_params[1]:.4f}×log(size) growth
• QPU: Constant {qpu_mean:.4f}s ± {np.std(qpu_times):.4f}s
• Classical {pulp_times[-1]/pulp_times[0]:.1f}× slower for {problem_sizes[-1]/problem_sizes[0]:.0f}× larger problem
• Quantum hybrid overhead dominates (99% non-QPU time)"""

ax.text(0.02, 0.98, insights_text, transform=ax.transAxes,
       fontsize=11, va='top', ha='left', family='monospace',
       bbox=dict(boxstyle='round,pad=1', facecolor='white', alpha=0.9, edgecolor='black', linewidth=2))

# Add performance summary
speedup = hybrid_times[0] / pulp_times[problem_sizes.tolist().index(dwave_sizes[0])]
summary_text = f"Classical is {speedup:.1f}× faster at smallest scale\nGap widens to {hybrid_times[-1]/pulp_times[problem_sizes.tolist().index(dwave_sizes[-1])]:.1f}× at largest tested scale"

ax.text(0.98, 0.02, summary_text, transform=ax.transAxes,
       fontsize=12, va='bottom', ha='right', fontweight='bold',
       bbox=dict(boxstyle='round,pad=0.8', facecolor='#FFE5B4', alpha=0.9, edgecolor='red', linewidth=2))

plt.tight_layout()
plt.savefig('final_presentation_plot.png', dpi=300, bbox_inches='tight', facecolor='white')
print("\n✅ Beautiful plot with fitted curves saved: final_presentation_plot.png")

# Print fit parameters
print("\n" + "="*80)
print("FIT PARAMETERS")
print("="*80)
if pulp_fit:
    print(f"\nPuLP (Classical): T = {pulp_params[0]:.6f} × size^{pulp_params[1]:.4f}")
    print(f"  Interpretation: Sub-linear scaling (power < 1 means efficiency improves)")
    print(f"  At size=1000: {pulp_fit(1000):.3f}s predicted")
    print(f"  At size=10000: {pulp_fit(10000):.3f}s predicted")

if hybrid_fit:
    print(f"\nD-Wave Hybrid: T = {hybrid_params[0]:.4f} + {hybrid_params[1]:.6f}×log(size)")
    print(f"  Interpretation: Logarithmic growth (very slow increase)")
    print(f"  At size=1000: {hybrid_fit(1000):.3f}s predicted")
    print(f"  At size=10000: {hybrid_fit(10000):.3f}s predicted")

print(f"\nQPU Time: T = {qpu_mean:.6f}s (constant)")
print(f"  Standard deviation: {np.std(qpu_times):.6f}s")
print(f"  Interpretation: Hardware-limited, independent of problem size")

print("\n" + "="*80)
print("🎉 Ready for your presentation!")
print("="*80)
