"""
Visualization Script for Benders Decomposition Results

Creates plots showing convergence history, bounds progression,
and comparative performance analysis.
"""

import json
import os
import sys
from typing import Dict, List, Any

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Matplotlib not available. Install with: pip install matplotlib")


def plot_convergence(
    solution_file: str,
    output_file: str = None
):
    """
    Plot convergence history from a Benders solution file
    
    Args:
        solution_file: Path to JSON solution file
        output_file: Path to save plot (default: solution_file.png)
    """
    if not HAS_MATPLOTLIB:
        print("Cannot create plot: matplotlib not installed")
        return
    
    # Load solution
    with open(solution_file, 'r') as f:
        solution = json.load(f)
    
    if 'iteration_history' not in solution:
        print("No iteration history found in solution file")
        return
    
    history = solution['iteration_history']
    
    # Extract data
    iterations = [it['iteration'] for it in history]
    lower_bounds = [it['lower_bound'] for it in history]
    upper_bounds = [it['upper_bound'] for it in history]
    gaps = [it['gap'] * 100 for it in history]  # Convert to percentage
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot 1: Bounds progression
    ax1 = axes[0]
    ax1.plot(iterations, upper_bounds, 'b-o', label='Upper Bound', linewidth=2)
    ax1.plot(iterations, lower_bounds, 'r-s', label='Lower Bound', linewidth=2)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Objective Value')
    ax1.set_title(f'Benders Decomposition Convergence\nFinal Objective: {solution["objective_value"]:.6f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Gap progression
    ax2 = axes[1]
    ax2.plot(iterations, gaps, 'g-^', linewidth=2)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Gap (%)')
    ax2.set_title('Optimality Gap Over Iterations')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')  # Log scale for large gaps
    
    plt.tight_layout()
    
    # Save figure
    if output_file is None:
        output_file = solution_file.replace('.json', '_convergence.png')
    
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Convergence plot saved to: {output_file}")
    plt.close()


def plot_comparison(
    comparison_file: str,
    output_file: str = None
):
    """
    Plot comparison of different solution methods
    
    Args:
        comparison_file: Path to JSON comparison file
        output_file: Path to save plot (default: comparison_file.png)
    """
    if not HAS_MATPLOTLIB:
        print("Cannot create plot: matplotlib not installed")
        return
    
    # Load comparison
    with open(comparison_file, 'r') as f:
        comparison = json.load(f)
    
    # Extract data
    methods = ['Standard MILP', 'Benders (Classical)', 'Benders (Quantum)']
    objectives = [
        comparison['standard_milp']['objective'],
        comparison['benders_classical']['objective'],
        comparison['benders_quantum']['objective']
    ]
    times = [
        comparison['standard_milp']['solve_time'],
        comparison['benders_classical']['solve_time'],
        comparison['benders_quantum']['solve_time']
    ]
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Objective values
    ax1 = axes[0]
    bars1 = ax1.bar(methods, objectives, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax1.set_ylabel('Objective Value')
    ax1.set_title('Solution Quality Comparison')
    ax1.tick_params(axis='x', rotation=15)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom')
    
    # Plot 2: Solve times
    ax2 = axes[1]
    bars2 = ax2.bar(methods, times, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax2.set_ylabel('Solve Time (s)')
    ax2.set_title('Computational Time Comparison')
    ax2.tick_params(axis='x', rotation=15)
    ax2.set_yscale('log')  # Log scale for time differences
    
    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}s',
                ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save figure
    if output_file is None:
        output_file = comparison_file.replace('.json', '_comparison.png')
    
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Comparison plot saved to: {output_file}")
    plt.close()


def plot_batch_summary(
    summary_file: str,
    output_file: str = None
):
    """
    Plot batch test summary
    
    Args:
        summary_file: Path to JSON batch summary file
        output_file: Path to save plot (default: summary_file.png)
    """
    if not HAS_MATPLOTLIB:
        print("Cannot create plot: matplotlib not installed")
        return
    
    # Load summary
    with open(summary_file, 'r') as f:
        summary = json.load(f)
    
    results = summary['results']
    successful = [r for r in results if r['success']]
    
    if not successful:
        print("No successful results to plot")
        return
    
    # Group by scenario and mode
    scenarios = list(set(r['scenario'] for r in successful))
    modes = list(set(r['annealing_mode'] for r in successful))
    
    # Create data structure
    data = {scenario: {mode: None for mode in modes} for scenario in scenarios}
    
    for result in successful:
        scenario = result['scenario']
        mode = result['annealing_mode']
        data[scenario][mode] = {
            'objective': result['objective'],
            'time': result['total_time']
        }
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Objectives by scenario and mode
    ax1 = axes[0]
    x = range(len(scenarios))
    width = 0.35
    
    classical_objs = [data[s]['classical']['objective'] if data[s]['classical'] else 0 for s in scenarios]
    quantum_objs = [data[s]['quantum']['objective'] if data[s]['quantum'] else 0 for s in scenarios]
    
    ax1.bar([i - width/2 for i in x], classical_objs, width, label='Classical', color='#ff7f0e')
    ax1.bar([i + width/2 for i in x], quantum_objs, width, label='Quantum', color='#2ca02c')
    
    ax1.set_xlabel('Scenario')
    ax1.set_ylabel('Objective Value')
    ax1.set_title('Solution Quality by Scenario and Mode')
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenarios)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Times by scenario and mode
    ax2 = axes[1]
    classical_times = [data[s]['classical']['time'] if data[s]['classical'] else 0 for s in scenarios]
    quantum_times = [data[s]['quantum']['time'] if data[s]['quantum'] else 0 for s in scenarios]
    
    ax2.bar([i - width/2 for i in x], classical_times, width, label='Classical', color='#ff7f0e')
    ax2.bar([i + width/2 for i in x], quantum_times, width, label='Quantum', color='#2ca02c')
    
    ax2.set_xlabel('Scenario')
    ax2.set_ylabel('Solve Time (s)')
    ax2.set_title('Computational Time by Scenario and Mode')
    ax2.set_xticks(x)
    ax2.set_xticklabels(scenarios)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save figure
    if output_file is None:
        output_file = summary_file.replace('.json', '_plots.png')
    
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Batch summary plot saved to: {output_file}")
    plt.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize Benders Decomposition Results')
    parser.add_argument(
        '--convergence',
        type=str,
        help='Path to solution JSON file for convergence plot'
    )
    parser.add_argument(
        '--comparison',
        type=str,
        help='Path to comparison JSON file'
    )
    parser.add_argument(
        '--batch',
        type=str,
        help='Path to batch summary JSON file'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Generate all available plots from standard locations'
    )
    
    args = parser.parse_args()
    
    if not HAS_MATPLOTLIB:
        print("Error: matplotlib is required for visualization")
        print("Install with: pip install matplotlib")
        return
    
    if args.all:
        # Find and plot all available results
        print("Generating all available plots...")
        
        # Convergence plots from benders_results
        if os.path.exists('benders_results'):
            for filename in os.listdir('benders_results'):
                if filename.startswith('solution_') and filename.endswith('.json'):
                    filepath = os.path.join('benders_results', filename)
                    print(f"\nPlotting convergence for: {filename}")
                    plot_convergence(filepath)
        
        # Comparison plots
        for filename in os.listdir('.'):
            if filename.startswith('comparison_') and filename.endswith('.json'):
                print(f"\nPlotting comparison: {filename}")
                plot_comparison(filename)
        
        # Batch summary
        if os.path.exists('benders_results/batch_test_summary.json'):
            print(f"\nPlotting batch summary")
            plot_batch_summary('benders_results/batch_test_summary.json')
        
        print("\n✓ All plots generated")
    
    else:
        # Individual plots
        if args.convergence:
            plot_convergence(args.convergence)
        
        if args.comparison:
            plot_comparison(args.comparison)
        
        if args.batch:
            plot_batch_summary(args.batch)
        
        if not (args.convergence or args.comparison or args.batch):
            parser.print_help()


if __name__ == "__main__":
    main()
