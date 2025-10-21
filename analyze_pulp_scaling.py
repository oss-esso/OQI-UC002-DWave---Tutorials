"""
Analyze PuLP solver scaling for full_family scenario with varying farm counts.

This script:
1. Varies number of farms on log scale
2. Measures solve time for each configuration
3. Fits scaling relationship (power law, polynomial, exponential)
4. Extrapolates to find n values for 5s and 6.5s solve times
5. Generates visualization plots
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from src.scenarios import load_food_data
import pulp as pl


def solve_with_pulp_timed(farms, foods, food_groups, config):
    """
    Solve with PuLP and return timing information.
    
    Returns:
        dict: Results including solve time, status, objective value
    """
    params = config['parameters']
    land_availability = params['land_availability']
    weights = params['weights']
    min_planting_area = params.get('minimum_planting_area', {})
    food_group_constraints = params.get('food_group_constraints', {})
    
    # Create variables
    A_pulp = pl.LpVariable.dicts("Area", [(f, c) for f in farms for c in foods], lowBound=0)
    Y_pulp = pl.LpVariable.dicts("Choose", [(f, c) for f in farms for c in foods], cat='Binary')
    
    total_area = sum(land_availability[f] for f in farms)
    
    # Objective function
    goal = (
        weights.get('nutritional_value', 0) * pl.lpSum([(foods[c].get('nutritional_value', 0) * A_pulp[(f, c)]) for f in farms for c in foods]) / total_area +
        weights.get('nutrient_density', 0) * pl.lpSum([(foods[c].get('nutrient_density', 0) * A_pulp[(f, c)]) for f in farms for c in foods]) / total_area -
        weights.get('environmental_impact', 0) * pl.lpSum([(foods[c].get('environmental_impact', 0) * A_pulp[(f, c)]) for f in farms for c in foods]) / total_area +
        weights.get('affordability', 0) * pl.lpSum([(foods[c].get('affordability', 0) * A_pulp[(f, c)]) for f in farms for c in foods]) / total_area +
        weights.get('sustainability', 0) * pl.lpSum([(foods[c].get('sustainability', 0) * A_pulp[(f, c)]) for f in farms for c in foods]) / total_area
    )
    
    model = pl.LpProblem("Food_Optimization_Scaling", pl.LpMaximize)
    
    # Constraints
    for f in farms:
        model += pl.lpSum([A_pulp[(f, c)] for c in foods]) <= land_availability[f], f"Max_Area_{f}"
    
    for f in farms:
        for c in foods:
            A_min = min_planting_area.get(c, 0)
            model += A_pulp[(f, c)] >= A_min * Y_pulp[(f, c)], f"MinArea_{f}_{c}"
            model += A_pulp[(f, c)] <= land_availability[f] * Y_pulp[(f, c)], f"MaxArea_{f}_{c}"
    
    if food_group_constraints:
        for g, constraints in food_group_constraints.items():
            foods_in_group = food_groups.get(g, [])
            if foods_in_group:
                for f in farms:
                    if 'min_foods' in constraints:
                        model += pl.lpSum([Y_pulp[(f, c)] for c in foods_in_group]) >= constraints['min_foods'], f"MinFoodGroup_{f}_{g}"
                    if 'max_foods' in constraints:
                        model += pl.lpSum([Y_pulp[(f, c)] for c in foods_in_group]) <= constraints['max_foods'], f"MaxFoodGroup_{f}_{g}"
    
    model += goal, "Objective"
    
    # Count problem size
    num_variables = len(A_pulp) + len(Y_pulp)
    num_constraints = len(model.constraints)
    
    # Solve and time it
    start_time = time.time()
    model.solve(pl.PULP_CBC_CMD(msg=0, timeLimit=300))  # 5 min timeout
    solve_time = time.time() - start_time
    
    results = {
        'status': pl.LpStatus[model.status],
        'objective_value': pl.value(model.objective) if model.status == 1 else None,
        'solve_time': solve_time,
        'num_variables': num_variables,
        'num_constraints': num_constraints,
        'num_farms': len(farms),
        'num_foods': len(foods),
        'n': len(farms) * len(foods)
    }
    
    return results


def generate_farm_counts_log_scale(min_farms=1, max_farms=200, num_points=20):
    """
    Generate farm counts on a log scale.
    
    Args:
        min_farms: Minimum number of farms
        max_farms: Maximum number of farms
        num_points: Number of points to sample
        
    Returns:
        list: Farm counts (integers)
    """
    # Generate log-spaced values
    log_counts = np.logspace(np.log10(min_farms), np.log10(max_farms), num_points)
    
    # Round to integers and remove duplicates
    farm_counts = sorted(list(set([int(round(x)) for x in log_counts])))
    
    return farm_counts


def load_full_family_with_n_farms(n_farms, seed=42):
    """
    Load full_family scenario with specified number of farms.
    
    Args:
        n_farms: Number of farms to generate
        seed: Random seed for reproducibility
        
    Returns:
        tuple: (farms, foods, food_groups, config)
    """
    # Import farm_sampler
    from farm_sampler import generate_farms
    
    # Generate farms
    L = generate_farms(n_farms=n_farms, seed=seed)
    farms = list(L.keys())
    
    # Load food data from Excel or use fallback
    script_dir = os.path.dirname(os.path.abspath(__file__))
    excel_path = os.path.join(script_dir, "Inputs", "Combined_Food_Data.xlsx")
    
    if not os.path.exists(excel_path):
        # Fallback: use intermediate scenario foods
        from src.scenarios import _load_intermediate_food_data
        _, foods, food_groups, _ = _load_intermediate_food_data()
    else:
        # Load from Excel (same logic as in scenarios.py)
        df = pd.read_excel(excel_path)
        
        col_map = {
            'Food_Name': 'Food_Name',
            'food_group': 'Food_Group',
            'nutritional_value': 'nutritional_value',
            'nutrient_density': 'nutrient_density',
            'environmental_impact': 'environmental_impact',
            'affordability': 'affordability',
            'sustainability': 'sustainability'
        }
        
        # Sample 2 per group
        sampled = df.groupby('food_group').apply(
            lambda x: x.sample(n=min(len(x), 2), random_state=seed)
        ).reset_index(drop=True)
        foods_list = sampled['Food_Name'].tolist()
        
        filt = df[df['Food_Name'].isin(foods_list)][list(col_map.keys())].copy()
        filt.rename(columns=col_map, inplace=True)
        
        objectives = ['nutritional_value', 'nutrient_density', 'environmental_impact', 'affordability', 'sustainability']
        for obj in objectives:
            filt[obj] = filt[obj].fillna(0.5).clip(0, 1)
        
        # Build foods dict
        foods = {}
        for _, row in filt.iterrows():
            fname = row['Food_Name']
            foods[fname] = {
                'nutritional_value': float(row['nutritional_value']),
                'nutrient_density': float(row['nutrient_density']),
                'environmental_impact': float(row['environmental_impact']),
                'affordability': float(row['affordability']),
                'sustainability': float(row['sustainability'])
            }
        
        # Build food groups
        food_groups = {}
        for _, row in filt.iterrows():
            g = row['Food_Group']
            fname = row['Food_Name']
            if g not in food_groups:
                food_groups[g] = []
            food_groups[g].append(fname)
    
    # Set minimum planting areas
    min_areas = {food: 0.01 for food in foods.keys()}
    
    # Build config
    parameters = {
        'land_availability': L,
        'minimum_planting_area': min_areas,
        'max_percentage_per_crop': {food: 0.4 for food in foods},
        'social_benefit': {farm: 0.2 for farm in farms},
        'food_group_constraints': {
            g: {'min_foods': 2, 'max_foods': len(lst)}
            for g, lst in food_groups.items()
        },
        'weights': {
            'nutritional_value': 0.25,
            'nutrient_density': 0.2,
            'environmental_impact': 0.25,
            'affordability': 0.15,
            'sustainability': 0.15
        }
    }
    
    config = {'parameters': parameters}
    
    return farms, foods, food_groups, config


def power_law(x, a, b):
    """Power law: y = a * x^b"""
    return a * np.power(x, b)


def polynomial(x, a, b, c):
    """Polynomial: y = a * x^2 + b * x + c"""
    return a * x**2 + b * x + c


def exponential(x, a, b, c):
    """Exponential: y = a * exp(b * x) + c"""
    return a * np.exp(b * x) + c


def fit_scaling_models(n_values, time_values):
    """
    Fit multiple scaling models to the data.
    
    Args:
        n_values: Problem size (n = farms × foods)
        time_values: Solve times in seconds
        
    Returns:
        dict: Fitted parameters and quality metrics for each model
    """
    models = {}
    
    # Convert to numpy arrays
    x = np.array(n_values)
    y = np.array(time_values)
    
    # Power law fit
    try:
        popt_power, _ = curve_fit(power_law, x, y, p0=[1e-6, 2], maxfev=10000)
        y_pred_power = power_law(x, *popt_power)
        r2_power = 1 - np.sum((y - y_pred_power)**2) / np.sum((y - np.mean(y))**2)
        models['power_law'] = {
            'params': popt_power,
            'r2': r2_power,
            'formula': f'y = {popt_power[0]:.2e} * x^{popt_power[1]:.3f}',
            'func': lambda x_val: power_law(x_val, *popt_power)
        }
    except Exception as e:
        print(f"Power law fit failed: {e}")
        models['power_law'] = None
    
    # Polynomial fit
    try:
        popt_poly, _ = curve_fit(polynomial, x, y, p0=[1e-6, 1e-4, 0], maxfev=10000)
        y_pred_poly = polynomial(x, *popt_poly)
        r2_poly = 1 - np.sum((y - y_pred_poly)**2) / np.sum((y - np.mean(y))**2)
        models['polynomial'] = {
            'params': popt_poly,
            'r2': r2_poly,
            'formula': f'y = {popt_poly[0]:.2e} * x^2 + {popt_poly[1]:.2e} * x + {popt_poly[2]:.2e}',
            'func': lambda x_val: polynomial(x_val, *popt_poly)
        }
    except Exception as e:
        print(f"Polynomial fit failed: {e}")
        models['polynomial'] = None
    
    # Exponential fit (only if data looks exponential)
    try:
        if max(y) / min(y) > 10:  # Only try if there's significant range
            popt_exp, _ = curve_fit(exponential, x, y, p0=[0.001, 0.001, 0], maxfev=10000)
            y_pred_exp = exponential(x, *popt_exp)
            r2_exp = 1 - np.sum((y - y_pred_exp)**2) / np.sum((y - np.mean(y))**2)
            models['exponential'] = {
                'params': popt_exp,
                'r2': r2_exp,
                'formula': f'y = {popt_exp[0]:.2e} * exp({popt_exp[1]:.2e} * x) + {popt_exp[2]:.2e}',
                'func': lambda x_val: exponential(x_val, *popt_exp)
            }
        else:
            models['exponential'] = None
    except Exception as e:
        print(f"Exponential fit failed: {e}")
        models['exponential'] = None
    
    return models


def find_n_for_target_time(models, target_time, max_n=100000):
    """
    Find n value that gives target solve time using best model.
    
    Args:
        models: Dictionary of fitted models
        target_time: Target solve time in seconds
        max_n: Maximum n to search
        
    Returns:
        dict: Results for each model
    """
    results = {}
    
    for model_name, model_data in models.items():
        if model_data is None:
            continue
        
        func = model_data['func']
        
        # Binary search for n
        n_low = 1
        n_high = max_n
        tolerance = 0.01  # 1% tolerance
        
        for _ in range(100):  # Max iterations
            n_mid = (n_low + n_high) // 2
            predicted_time = func(n_mid)
            
            if abs(predicted_time - target_time) / target_time < tolerance:
                results[model_name] = {
                    'n': n_mid,
                    'predicted_time': predicted_time
                }
                break
            
            if predicted_time < target_time:
                n_low = n_mid
            else:
                n_high = n_mid
            
            if n_high - n_low <= 1:
                results[model_name] = {
                    'n': n_mid,
                    'predicted_time': predicted_time
                }
                break
    
    return results


def plot_scaling_results(n_values, time_values, models, output_path='scaling_plot.png'):
    """
    Create visualization of scaling results.
    
    Args:
        n_values: Problem sizes
        time_values: Solve times
        models: Fitted models
        output_path: Path to save plot
    """
    plt.figure(figsize=(14, 10))
    
    # Subplot 1: Linear scale
    plt.subplot(2, 2, 1)
    plt.scatter(n_values, time_values, color='blue', s=50, alpha=0.6, label='Measured')
    
    x_fit = np.linspace(min(n_values), max(n_values), 200)
    for model_name, model_data in models.items():
        if model_data is not None:
            y_fit = model_data['func'](x_fit)
            plt.plot(x_fit, y_fit, label=f"{model_name} (R²={model_data['r2']:.4f})", linewidth=2)
    
    plt.xlabel('n (farms × foods)', fontsize=12)
    plt.ylabel('Solve Time (seconds)', fontsize=12)
    plt.title('PuLP Solver Scaling - Linear Scale', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Log-log scale
    plt.subplot(2, 2, 2)
    plt.scatter(n_values, time_values, color='blue', s=50, alpha=0.6, label='Measured')
    
    for model_name, model_data in models.items():
        if model_data is not None:
            y_fit = model_data['func'](x_fit)
            plt.plot(x_fit, y_fit, label=f"{model_name} (R²={model_data['r2']:.4f})", linewidth=2)
    
    plt.xlabel('n (farms × foods)', fontsize=12)
    plt.ylabel('Solve Time (seconds)', fontsize=12)
    plt.title('PuLP Solver Scaling - Log-Log Scale', fontsize=14, fontweight='bold')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3, which='both')
    
    # Subplot 3: Residuals
    plt.subplot(2, 2, 3)
    for model_name, model_data in models.items():
        if model_data is not None:
            y_pred = model_data['func'](np.array(n_values))
            residuals = np.array(time_values) - y_pred
            plt.scatter(n_values, residuals, label=model_name, alpha=0.6)
    
    plt.axhline(y=0, color='k', linestyle='--', linewidth=1)
    plt.xlabel('n (farms × foods)', fontsize=12)
    plt.ylabel('Residual (seconds)', fontsize=12)
    plt.title('Fit Residuals', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 4: Extended prediction
    plt.subplot(2, 2, 4)
    plt.scatter(n_values, time_values, color='blue', s=50, alpha=0.6, label='Measured', zorder=5)
    
    # Extend prediction range
    x_extended = np.linspace(min(n_values), max(n_values) * 2, 300)
    
    for model_name, model_data in models.items():
        if model_data is not None:
            y_extended = model_data['func'](x_extended)
            plt.plot(x_extended, y_extended, label=f"{model_name}", linewidth=2, alpha=0.7)
    
    # Mark 5s and 6.5s lines
    plt.axhline(y=5.0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='5s target')
    plt.axhline(y=6.5, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label='6.5s target')
    
    plt.xlabel('n (farms × foods)', fontsize=12)
    plt.ylabel('Solve Time (seconds)', fontsize=12)
    plt.title('Extended Prediction', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    plt.close()


def main():
    """Main execution function."""
    print("=" * 80)
    print("PULP SOLVER SCALING ANALYSIS - FULL_FAMILY SCENARIO")
    print("=" * 80)
    
    # Create output directory
    os.makedirs('Scaling_Analysis', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Generate farm counts on log scale
    print("\nGenerating farm counts on log scale...")
    # Extended range to reach 5-6.5 second solve times
    farm_counts = generate_farm_counts_log_scale(min_farms=1, max_farms=5000, num_points=25)
    print(f"Farm counts to test: {farm_counts}")
    
    # Run experiments
    print("\n" + "=" * 80)
    print("RUNNING SCALING EXPERIMENTS")
    print("=" * 80)
    
    results = []
    
    for i, n_farms in enumerate(farm_counts, 1):
        print(f"\n[{i}/{len(farm_counts)}] Testing with {n_farms} farms...")
        
        try:
            # Load scenario
            farms, foods, food_groups, config = load_full_family_with_n_farms(n_farms, seed=42)
            num_foods = len(foods)
            n = n_farms * num_foods
            
            print(f"  - Foods: {num_foods}")
            print(f"  - n = {n_farms} × {num_foods} = {n}")
            
            # Solve
            result = solve_with_pulp_timed(farms, foods, food_groups, config)
            
            print(f"  - Status: {result['status']}")
            print(f"  - Solve time: {result['solve_time']:.3f} seconds")
            print(f"  - Variables: {result['num_variables']}")
            print(f"  - Constraints: {result['num_constraints']}")
            
            results.append(result)
            
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
    
    # Save raw results
    results_path = f'Scaling_Analysis/scaling_results_{timestamp}.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nRaw results saved to: {results_path}")
    
    # Extract data for fitting
    n_values = [r['n'] for r in results]
    time_values = [r['solve_time'] for r in results]
    
    # Fit scaling models
    print("\n" + "=" * 80)
    print("FITTING SCALING MODELS")
    print("=" * 80)
    
    models = fit_scaling_models(n_values, time_values)
    
    for model_name, model_data in models.items():
        if model_data is not None:
            print(f"\n{model_name.upper()}:")
            print(f"  Formula: {model_data['formula']}")
            print(f"  R² score: {model_data['r2']:.6f}")
    
    # Find best model
    best_model_name = max(
        [m for m in models.keys() if models[m] is not None],
        key=lambda m: models[m]['r2']
    )
    print(f"\nBest model: {best_model_name} (R² = {models[best_model_name]['r2']:.6f})")
    
    # Find n for target times
    print("\n" + "=" * 80)
    print("EXTRAPOLATING TO TARGET SOLVE TIMES")
    print("=" * 80)
    
    target_times = [5.0, 6.5]
    
    for target_time in target_times:
        print(f"\nTarget time: {target_time} seconds")
        n_results = find_n_for_target_time(models, target_time, max_n=100000)
        
        for model_name, data in n_results.items():
            n_val = data['n']
            pred_time = data['predicted_time']
            
            # Estimate number of farms (assuming same number of foods)
            num_foods_estimate = len(foods) if 'foods' in locals() else 10
            n_farms_estimate = n_val / num_foods_estimate
            
            print(f"  {model_name}:")
            print(f"    n = {n_val}")
            print(f"    Estimated farms (assuming {num_foods_estimate} foods): {n_farms_estimate:.1f}")
            print(f"    Predicted time: {pred_time:.3f} seconds")
    
    # Create visualization
    print("\n" + "=" * 80)
    print("CREATING VISUALIZATION")
    print("=" * 80)
    
    plot_path = f'Scaling_Analysis/scaling_plot_{timestamp}.png'
    plot_scaling_results(n_values, time_values, models, output_path=plot_path)
    
    # Create summary report
    report_path = f'Scaling_Analysis/scaling_report_{timestamp}.md'
    
    with open(report_path, 'w') as f:
        f.write("# PuLP Solver Scaling Analysis - Full Family Scenario\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Experimental Setup\n\n")
        f.write(f"- **Scenario**: full_family\n")
        f.write(f"- **Farm counts tested**: {len(farm_counts)} ({min(farm_counts)} to {max(farm_counts)})\n")
        f.write(f"- **Number of foods**: ~{len(foods) if 'foods' in locals() else 'varies'}\n")
        f.write(f"- **Problem size (n)**: {min(n_values)} to {max(n_values)}\n\n")
        
        f.write("## Measured Data\n\n")
        f.write("| Farms | Foods | n | Variables | Constraints | Solve Time (s) | Status |\n")
        f.write("|-------|-------|---|-----------|-------------|----------------|--------|\n")
        for r in results:
            f.write(f"| {r['num_farms']} | {r['num_foods']} | {r['n']} | {r['num_variables']} | {r['num_constraints']} | {r['solve_time']:.3f} | {r['status']} |\n")
        
        f.write("\n## Fitted Models\n\n")
        for model_name, model_data in models.items():
            if model_data is not None:
                f.write(f"### {model_name.replace('_', ' ').title()}\n\n")
                f.write(f"- **Formula**: `{model_data['formula']}`\n")
                f.write(f"- **R² score**: {model_data['r2']:.6f}\n\n")
        
        f.write(f"\n### Best Model: {best_model_name.replace('_', ' ').title()}\n\n")
        f.write(f"The {best_model_name} model provides the best fit with R² = {models[best_model_name]['r2']:.6f}\n\n")
        
        f.write("## Extrapolation Results\n\n")
        for target_time in target_times:
            f.write(f"### Target Solve Time: {target_time} seconds\n\n")
            n_results = find_n_for_target_time(models, target_time, max_n=100000)
            
            f.write("| Model | n | Estimated Farms | Predicted Time (s) |\n")
            f.write("|-------|---|-----------------|--------------------|\n")
            
            for model_name, data in n_results.items():
                n_val = data['n']
                pred_time = data['predicted_time']
                num_foods_estimate = len(foods) if 'foods' in locals() else 10
                n_farms_estimate = n_val / num_foods_estimate
                
                f.write(f"| {model_name} | {n_val} | {n_farms_estimate:.1f} | {pred_time:.3f} |\n")
            
            f.write("\n")
        
        f.write("## Visualization\n\n")
        f.write(f"![Scaling Plot]({os.path.basename(plot_path)})\n\n")
        
        f.write("## Conclusions\n\n")
        f.write(f"Based on the {best_model_name} model:\n\n")
        
        for target_time in target_times:
            n_results = find_n_for_target_time(models, target_time, max_n=100000)
            if best_model_name in n_results:
                n_val = n_results[best_model_name]['n']
                num_foods_estimate = len(foods) if 'foods' in locals() else 10
                n_farms_estimate = n_val / num_foods_estimate
                f.write(f"- To achieve a solve time of **{target_time} seconds**, you need approximately:\n")
                f.write(f"  - **n = {n_val}** (farms × foods)\n")
                f.write(f"  - **~{int(n_farms_estimate)} farms** (assuming {num_foods_estimate} foods)\n\n")
    
    print(f"\nSummary report saved to: {report_path}")
    
    print("\n" + "=" * 80)
    print("SCALING ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nResults directory: Scaling_Analysis/")
    print(f"Raw data: {results_path}")
    print(f"Plot: {plot_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
