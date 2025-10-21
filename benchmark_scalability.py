"""
Scalability Benchmark Script
Tests different combinations of farms and food groups to analyze solver performance
"""
import os
import sys
import json
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from farm_sampler import generate_farms
from src.scenarios import load_food_data
from solver_runner import create_cqm, solve_with_pulp, solve_with_dwave
import pulp as pl

# Benchmark configurations
# Format: (n_food_groups, n_farms)
BENCHMARK_CONFIGS = [
    (1, 1),
    (2, 1),
    (1, 2),
    (2, 2),
    (1, 5),
    (2, 5),
    (2, 10),
    (2, 25),
    (2, 125),
    (2, 625),
    (2, 1250),
]

def create_custom_scenario(n_food_groups, n_farms, seed=42):
    """
    Create a custom scenario with specified number of food groups and farms.
    """
    # Generate farms
    L = generate_farms(n_farms=n_farms, seed=seed)
    farms = list(L.keys())
    
    # Base foods with scores (use simple, well-balanced foods)
    all_foods = {
        'Wheat': {'nutritional_value': 0.7, 'nutrient_density': 0.6, 'environmental_impact': 0.3, 
                  'affordability': 0.8, 'sustainability': 0.7},
        'Corn': {'nutritional_value': 0.6, 'nutrient_density': 0.5, 'environmental_impact': 0.4, 
                 'affordability': 0.9, 'sustainability': 0.6},
        'Rice': {'nutritional_value': 0.8, 'nutrient_density': 0.7, 'environmental_impact': 0.6, 
                 'affordability': 0.7, 'sustainability': 0.5},
        'Soybeans': {'nutritional_value': 0.9, 'nutrient_density': 0.8, 'environmental_impact': 0.2, 
                     'affordability': 0.6, 'sustainability': 0.8},
        'Potatoes': {'nutritional_value': 0.5, 'nutrient_density': 0.4, 'environmental_impact': 0.3, 
                     'affordability': 0.9, 'sustainability': 0.7},
        'Tomatoes': {'nutritional_value': 0.6, 'nutrient_density': 0.5, 'environmental_impact': 0.2, 
                     'affordability': 0.7, 'sustainability': 0.9},
        'Apples': {'nutritional_value': 0.7, 'nutrient_density': 0.6, 'environmental_impact': 0.2, 
                   'affordability': 0.5, 'sustainability': 0.8},
        'Carrots': {'nutritional_value': 0.8, 'nutrient_density': 0.7, 'environmental_impact': 0.2, 
                    'affordability': 0.8, 'sustainability': 0.8},
    }
    
    # Define food groups (2 foods per group)
    all_food_groups = {
        'Grains': ['Wheat', 'Corn'],
        'Legumes': ['Soybeans', 'Rice'],
        'Vegetables': ['Potatoes', 'Tomatoes'],
        'Fruits': ['Apples', 'Carrots'],
    }
    
    # Select food groups
    selected_group_names = list(all_food_groups.keys())[:n_food_groups]
    food_groups = {g: all_food_groups[g] for g in selected_group_names}
    
    # Select foods (only from selected groups)
    selected_food_names = []
    for g in selected_group_names:
        selected_food_names.extend(all_food_groups[g])
    
    foods = {name: all_foods[name] for name in selected_food_names}
    
    # Calculate minimum area based on smallest farm
    smallest_farm = min(L.values())
    min_area_per_crop = (smallest_farm / n_food_groups) * 0.95  # 95% safety margin
    
    min_areas = {food: min_area_per_crop for food in foods.keys()}
    
    # Create config
    parameters = {
        'land_availability': L,
        'minimum_planting_area': min_areas,
        'food_group_constraints': {
            g: {'min_foods': 1, 'max_foods': len(crops)}
            for g, crops in food_groups.items()
        },
        'weights': {
            'nutritional_value': 0.25,
            'nutrient_density': 0.2,
            'environmental_impact': 0.25,
            'affordability': 0.15,
            'sustainability': 0.15
        }
    }
    
    config = {
        'parameters': parameters,
    }
    
    return farms, foods, food_groups, config

def run_benchmark(n_food_groups, n_farms):
    """
    Run a single benchmark test.
    Returns timing results and problem size metrics.
    """
    print(f"\n{'='*80}")
    print(f"BENCHMARK: {n_food_groups} Food Groups, {n_farms} Farms")
    print(f"{'='*80}")
    
    try:
        # Create scenario
        farms, foods, food_groups, config = create_custom_scenario(n_food_groups, n_farms)
        
        n_foods = len(foods)
        n_vars = 2 * n_farms * n_foods  # Binary + continuous
        n_constraints = n_farms + 2*n_farms*n_foods + 2*len(food_groups)*n_farms
        problem_size = 5 * n_foods * n_farms  # Custom metric
        
        print(f"  Foods: {n_foods}")
        print(f"  Variables: {n_vars}")
        print(f"  Constraints: {n_constraints}")
        print(f"  Problem Size: {problem_size}")
        
        # Solve with PuLP
        print(f"\n  Solving with PuLP...")
        pulp_start = time.time()
        pulp_model, pulp_results = solve_with_pulp(farms, foods, food_groups, config)
        pulp_time = time.time() - pulp_start
        
        print(f"    Status: {pulp_results['status']}")
        print(f"    Time: {pulp_time:.3f}s")
        
        # Solve with DWave (can handle up to ~10,000 variables)
        dwave_time = None
        qpu_time = None
        hybrid_time = None
        dwave_feasible = False
        
        print(f"\n  Solving with DWave...")
        try:
            # Create CQM
            cqm, A, Y, constraint_metadata = create_cqm(farms, foods, food_groups, config)
            
            # Solve
            token = os.getenv('DWAVE_API_TOKEN', '45FS-23cfb48dca2296ed24550846d2e7356eb6c19551')
            sampleset, dwave_time = solve_with_dwave(cqm, token)
            
            # Extract timing info
            if hasattr(sampleset, 'info') and isinstance(sampleset.info, dict):
                qpu_time = sampleset.info.get('qpu_access_time', 0) / 1000000.0  # microseconds to seconds
                hybrid_time = sampleset.info.get('run_time', 0) / 1000000.0  # microseconds to seconds
            
            feasible_sampleset = sampleset.filter(lambda d: d.is_feasible)
            dwave_feasible = len(feasible_sampleset) > 0
            
            print(f"    Time: {dwave_time:.3f}s")
            print(f"    QPU Time: {qpu_time:.6f}s" if qpu_time else "    QPU Time: N/A")
            print(f"    Hybrid Time: {hybrid_time:.3f}s" if hybrid_time else "    Hybrid Time: N/A")
            print(f"    Feasible: {dwave_feasible}")
        except Exception as e:
            print(f"    DWave Error: {e}")
        
        result = {
            'n_food_groups': n_food_groups,
            'n_farms': n_farms,
            'n_foods': n_foods,
            'n_vars': n_vars,
            'n_constraints': n_constraints,
            'problem_size': problem_size,
            'pulp_time': pulp_time,
            'pulp_status': pulp_results['status'],
            'dwave_time': dwave_time,
            'qpu_time': qpu_time,
            'hybrid_time': hybrid_time,
            'dwave_feasible': dwave_feasible
        }
        
        return result
        
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

def plot_results(results, output_file='scalability_benchmark.png'):
    """
    Create beautiful plots for presentation.
    """
    # Filter valid results
    valid_results = [r for r in results if r is not None]
    
    if not valid_results:
        print("No valid results to plot!")
        return
    
    # Extract data
    problem_sizes = [r['problem_size'] for r in valid_results]
    pulp_times = [r['pulp_time'] for r in valid_results]
    
    # DWave times (only where available)
    dwave_problem_sizes = [r['problem_size'] for r in valid_results if r['dwave_time'] is not None]
    dwave_times = [r['dwave_time'] for r in valid_results if r['dwave_time'] is not None]
    qpu_times = [r['qpu_time'] for r in valid_results if r['qpu_time'] is not None]
    hybrid_times = [r['hybrid_time'] for r in valid_results if r['hybrid_time'] is not None]
    
    # Create figure with professional styling
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: All times on log scale
    ax1.plot(problem_sizes, pulp_times, 'o-', linewidth=2.5, markersize=8, 
             label='Classical (PuLP/CBC)', color='#2E86AB', alpha=0.9)
    
    if dwave_times:
        ax1.plot(dwave_problem_sizes, dwave_times, 's-', linewidth=2.5, markersize=8,
                 label='Quantum Total (D-Wave)', color='#A23B72', alpha=0.9)
    
    if hybrid_times:
        ax1.plot(dwave_problem_sizes, hybrid_times, '^-', linewidth=2, markersize=7,
                 label='Quantum Hybrid Time', color='#F18F01', alpha=0.8)
    
    if qpu_times:
        ax1.plot(dwave_problem_sizes, qpu_times, 'D-', linewidth=2, markersize=7,
                 label='QPU Access Time', color='#C73E1D', alpha=0.8)
    
    ax1.set_xlabel('Problem Size (5 × Foods × Farms)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Solve Time (seconds)', fontsize=14, fontweight='bold')
    ax1.set_title('Solver Performance vs Problem Size', fontsize=16, fontweight='bold', pad=20)
    ax1.legend(loc='upper left', fontsize=11, framealpha=0.95)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    
    # Add annotations for key points
    if len(problem_sizes) > 0:
        # Annotate largest problem
        max_idx = problem_sizes.index(max(problem_sizes))
        ax1.annotate(f'{problem_sizes[max_idx]} vars\n{pulp_times[max_idx]:.2f}s',
                    xy=(problem_sizes[max_idx], pulp_times[max_idx]),
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=9, bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # Plot 2: Speedup ratio (Classical vs Quantum)
    if dwave_times and len(dwave_times) > 0:
        speedup_ratios = [pulp_times[problem_sizes.index(ps)] / dt 
                         for ps, dt in zip(dwave_problem_sizes, dwave_times)]
        
        ax2.plot(dwave_problem_sizes, speedup_ratios, 'o-', linewidth=2.5, markersize=8,
                color='#6A4C93', alpha=0.9)
        ax2.axhline(y=1, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Parity')
        
        ax2.set_xlabel('Problem Size (5 × Foods × Farms)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Speedup Ratio (Classical / Quantum)', fontsize=14, fontweight='bold')
        ax2.set_title('Classical vs Quantum Speedup', fontsize=16, fontweight='bold', pad=20)
        ax2.legend(loc='best', fontsize=11, framealpha=0.95)
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.set_xscale('log')
        
        # Add annotation for quantum advantage threshold
        ax2.fill_between(dwave_problem_sizes, 0, 1, alpha=0.2, color='red', label='Quantum Advantage Zone')
        ax2.text(min(dwave_problem_sizes), 0.5, 'Quantum\nFaster', fontsize=10, ha='left', va='center',
                bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))
        ax2.text(min(dwave_problem_sizes), 1.5, 'Classical\nFaster', fontsize=10, ha='left', va='center',
                bbox=dict(boxstyle='round', facecolor='blue', alpha=0.3))
    else:
        ax2.text(0.5, 0.5, 'No Quantum Data Available\n(Problems too large)', 
                ha='center', va='center', fontsize=14, transform=ax2.transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✅ Plot saved to: {output_file}")
    
    # Also create a summary table plot
    create_summary_table(valid_results, 'scalability_table.png')

def create_summary_table(results, output_file='scalability_table.png'):
    """
    Create a beautiful summary table.
    """
    fig, ax = plt.subplots(figsize=(14, len(results) * 0.5 + 2))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    headers = ['Food\nGroups', 'Farms', 'Foods', 'Vars', 'Constraints', 'Problem\nSize', 
               'PuLP\nTime (s)', 'D-Wave\nTime (s)', 'QPU\nTime (s)', 'Status']
    
    table_data = []
    for r in results:
        row = [
            r['n_food_groups'],
            r['n_farms'],
            r['n_foods'],
            r['n_vars'],
            r['n_constraints'],
            r['problem_size'],
            f"{r['pulp_time']:.3f}",
            f"{r['dwave_time']:.3f}" if r['dwave_time'] else 'N/A',
            f"{r['qpu_time']:.6f}" if r['qpu_time'] else 'N/A',
            '✅' if r['pulp_status'] == 'Optimal' else '❌'
        ]
        table_data.append(row)
    
    table = ax.table(cellText=table_data, colLabels=headers, cellLoc='center',
                    loc='center', bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style the table
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#2E86AB')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    for i in range(1, len(table_data) + 1):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#E8F4F8')
            else:
                table[(i, j)].set_facecolor('white')
    
    plt.title('Scalability Benchmark Results', fontsize=16, fontweight='bold', pad=20)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Table saved to: {output_file}")

def main():
    """
    Run all benchmarks and create plots.
    """
    print("="*80)
    print("SCALABILITY BENCHMARK")
    print("="*80)
    
    results = []
    
    for n_food_groups, n_farms in BENCHMARK_CONFIGS:
        result = run_benchmark(n_food_groups, n_farms)
        if result:
            results.append(result)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f'benchmark_results_{timestamp}.json'
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"BENCHMARK COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to: {results_file}")
    
    # Create plots
    print(f"\nGenerating plots...")
    plot_results(results, f'scalability_benchmark_{timestamp}.png')
    
    print(f"\n🎉 All done! Ready for your presentation!")

if __name__ == "__main__":
    main()
