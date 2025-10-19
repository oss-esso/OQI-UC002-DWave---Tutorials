#!/usr/bin/env python3
"""
Comparison script for the three optimization approaches:
1. PuLP solver
2. CQM D-Wave solver  
3. QUBO D-Wave solver

Compares solutions and validates against all constraints.
"""

import sys
import os
import time
import json
from typing import Dict, List, Tuple, Any

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from scenarios import load_food_data

# Import the solver functions
try:
    from solve_pulp import solve_food_optimization_pulp, validate_solution
    PULP_AVAILABLE = True
except ImportError as e:
    print(f"Warning: PuLP solver not available: {e}")
    PULP_AVAILABLE = False

try:
    from solve_cqm import solve_food_optimization_cqm
    CQM_AVAILABLE = True
except ImportError as e:
    print(f"Warning: CQM solver not available: {e}")
    CQM_AVAILABLE = False

try:
    from solve_qubo import solve_food_optimization_qubo
    QUBO_AVAILABLE = True
except ImportError as e:
    print(f"Warning: QUBO solver not available: {e}")
    QUBO_AVAILABLE = False

def format_solution_summary(result: Dict[str, Any], solver_name: str) -> str:
    """Format a solution summary for display."""
    if result is None:
        return f"{solver_name}: Not available"
    
    status = result.get('status', 'UNKNOWN')
    obj_val = result.get('objective_value', None)
    solve_time = result.get('solve_time', 0)
    
    summary = f"{solver_name}:\n"
    summary += f"  Status: {status}\n"
    summary += f"  Objective Value: {obj_val:.6f if obj_val is not None else 'N/A'}\n"
    summary += f"  Solve Time: {solve_time:.2f}s\n"
    
    if 'note' in result:
        summary += f"  Note: {result['note']}\n"
    
    return summary

def compare_solutions(results: Dict[str, Dict[str, Any]], 
                     farms: List[str], foods: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    """
    Compare the solutions from different solvers.
    
    Args:
        results: Dictionary with solver results
        farms: List of farm names  
        foods: Dictionary of food data
        
    Returns:
        Dictionary with comparison metrics
    """
    comparison = {
        'solver_comparison': {},
        'solution_differences': {},
        'objective_comparison': {},
        'timing_comparison': {}
    }
    
    # Extract valid results
    valid_results = {name: result for name, result in results.items() 
                    if result is not None and result.get('solution') is not None}
    
    if len(valid_results) == 0:
        comparison['error'] = "No valid solutions to compare"
        return comparison
    
    # Compare objective values
    objective_values = {}
    for solver, result in valid_results.items():
        obj_val = result.get('objective_value')
        if obj_val is not None:
            objective_values[solver] = obj_val
    
    if objective_values:
        best_solver = max(objective_values, key=objective_values.get)
        worst_solver = min(objective_values, key=objective_values.get)
        
        comparison['objective_comparison'] = {
            'best_solver': best_solver,
            'best_value': objective_values[best_solver],
            'worst_solver': worst_solver,
            'worst_value': objective_values[worst_solver],
            'relative_gap': ((objective_values[best_solver] - objective_values[worst_solver]) / 
                           abs(objective_values[best_solver]) * 100) if objective_values[best_solver] != 0 else 0,
            'all_values': objective_values
        }
    
    # Compare solve times
    solve_times = {}
    for solver, result in valid_results.items():
        solve_time = result.get('solve_time', 0)
        solve_times[solver] = solve_time
    
    if solve_times:
        fastest_solver = min(solve_times, key=solve_times.get)
        slowest_solver = max(solve_times, key=solve_times.get)
        
        comparison['timing_comparison'] = {
            'fastest_solver': fastest_solver,
            'fastest_time': solve_times[fastest_solver],
            'slowest_solver': slowest_solver,
            'slowest_time': solve_times[slowest_solver],
            'all_times': solve_times
        }
    
    # Compare solution allocations
    if len(valid_results) >= 2:
        solver_names = list(valid_results.keys())
        for i, solver1 in enumerate(solver_names):
            for solver2 in solver_names[i+1:]:
                solution1 = valid_results[solver1]['solution']
                solution2 = valid_results[solver2]['solution']
                
                # Calculate differences
                differences = {}
                max_diff = 0
                total_diff = 0
                num_comparisons = 0
                
                for farm in farms:
                    differences[farm] = {}
                    for food in foods.keys():
                        area1 = solution1[farm][food]
                        area2 = solution2[farm][food]
                        diff = abs(area1 - area2)
                        differences[farm][food] = {
                            'solver1_area': area1,
                            'solver2_area': area2,
                            'absolute_diff': diff,
                            'relative_diff': (diff / max(area1, area2, 1e-6)) * 100
                        }
                        
                        max_diff = max(max_diff, diff)
                        total_diff += diff
                        num_comparisons += 1
                
                comparison['solution_differences'][f"{solver1}_vs_{solver2}"] = {
                    'max_absolute_difference': max_diff,
                    'average_absolute_difference': total_diff / num_comparisons,
                    'detailed_differences': differences
                }
    
    return comparison

def comprehensive_constraint_validation(solution: Dict, binary_solution: Dict, 
                                      farms: List[str], foods: Dict[str, Dict[str, float]], 
                                      food_groups: Dict[str, List[str]], config: Dict) -> Dict[str, Any]:
    """
    Comprehensive validation of solution against all constraints.
    
    Args:
        solution: Area allocation solution
        binary_solution: Binary planting decisions
        farms: List of farm names
        foods: Dictionary of food data
        food_groups: Dictionary mapping food groups to foods
        config: Configuration parameters
        
    Returns:
        Detailed validation results
    """
    if solution is None or binary_solution is None:
        return {'valid': False, 'reason': 'No solution to validate', 'constraint_violations': []}
    
    params = config['parameters']
    land_availability = params['land_availability']
    min_planting_area = params['minimum_planting_area']
    max_percentage_per_crop = params['max_percentage_per_crop']
    social_benefit = params['social_benefit']
    food_group_constraints = params['food_group_constraints']
    
    validation = {
        'valid': True,
        'constraint_violations': [],
        'constraint_details': {
            'land_availability': {},
            'social_benefit': {},
            'planting_area': {},
            'food_groups': {}
        },
        'summary': {
            'total_violations': 0,
            'violation_types': {}
        }
    }
    
    tolerance = 1e-6
    
    # 1. Land availability constraints
    for farm in farms:
        farm_total = sum(solution[farm].values())
        max_land = land_availability[farm]
        
        validation['constraint_details']['land_availability'][farm] = {
            'used_land': farm_total,
            'available_land': max_land,
            'utilization_rate': (farm_total / max_land) * 100,
            'violated': farm_total > max_land + tolerance
        }
        
        if farm_total > max_land + tolerance:
            violation = f"Land availability violated for {farm}: {farm_total:.2f} > {max_land}"
            validation['constraint_violations'].append(violation)
            validation['valid'] = False
            validation['summary']['violation_types']['land_availability'] = validation['summary']['violation_types'].get('land_availability', 0) + 1
    
    # 2. Social benefit constraints (minimum land utilization)
    for farm in farms:
        farm_total = sum(solution[farm].values())
        min_land = social_benefit[farm] * land_availability[farm]
        
        validation['constraint_details']['social_benefit'][farm] = {
            'used_land': farm_total,
            'required_minimum': min_land,
            'social_benefit_rate': social_benefit[farm],
            'violated': farm_total < min_land - tolerance
        }
        
        if farm_total < min_land - tolerance:
            violation = f"Social benefit violated for {farm}: {farm_total:.2f} < {min_land:.2f}"
            validation['constraint_violations'].append(violation)
            validation['valid'] = False
            validation['summary']['violation_types']['social_benefit'] = validation['summary']['violation_types'].get('social_benefit', 0) + 1
    
    # 3. Planting area constraints
    for farm in farms:
        for food in foods.keys():
            area = solution[farm][food]
            planted = binary_solution[farm][food]
            min_area = min_planting_area[food]
            max_area = max_percentage_per_crop[food] * land_availability[farm]
            
            constraint_detail = {
                'area': area,
                'planted': bool(planted),
                'min_area': min_area,
                'max_area': max_area,
                'violations': []
            }
            
            # Check planting logic: area > 0 iff planted = 1
            if planted == 1:
                if area < min_area - tolerance:
                    violation = f"Minimum planting area violated for {farm}-{food}: {area:.2f} < {min_area}"
                    validation['constraint_violations'].append(violation)
                    constraint_detail['violations'].append('minimum_area')
                    validation['valid'] = False
                
                if area > max_area + tolerance:
                    violation = f"Maximum percentage violated for {farm}-{food}: {area:.2f} > {max_area:.2f}"
                    validation['constraint_violations'].append(violation)
                    constraint_detail['violations'].append('maximum_percentage')
                    validation['valid'] = False
            else:
                if area > tolerance:
                    violation = f"Area should be 0 when not planted for {farm}-{food}: {area:.2f}"
                    validation['constraint_violations'].append(violation)
                    constraint_detail['violations'].append('zero_when_not_planted')
                    validation['valid'] = False
            
            if constraint_detail['violations']:
                validation['summary']['violation_types']['planting_area'] = validation['summary']['violation_types'].get('planting_area', 0) + len(constraint_detail['violations'])
            
            validation['constraint_details']['planting_area'][f"{farm}_{food}"] = constraint_detail
    
    # 4. Food group constraints
    for farm in farms:
        for group, foods_in_group in food_groups.items():
            group_count = sum(binary_solution[farm][food] for food in foods_in_group)
            constraints = food_group_constraints[group]
            min_foods = constraints['min_foods']
            max_foods = constraints['max_foods']
            
            constraint_detail = {
                'selected_foods': group_count,
                'min_required': min_foods,
                'max_allowed': max_foods,
                'foods_in_group': foods_in_group,
                'selected_food_list': [food for food in foods_in_group if binary_solution[farm][food] == 1],
                'violations': []
            }
            
            if group_count < min_foods:
                violation = f"Minimum foods violated for {farm}-{group}: {group_count} < {min_foods}"
                validation['constraint_violations'].append(violation)
                constraint_detail['violations'].append('minimum_foods')
                validation['valid'] = False
            
            if group_count > max_foods:
                violation = f"Maximum foods violated for {farm}-{group}: {group_count} > {max_foods}"
                validation['constraint_violations'].append(violation)
                constraint_detail['violations'].append('maximum_foods')
                validation['valid'] = False
            
            if constraint_detail['violations']:
                validation['summary']['violation_types']['food_groups'] = validation['summary']['violation_types'].get('food_groups', 0) + len(constraint_detail['violations'])
            
            validation['constraint_details']['food_groups'][f"{farm}_{group}"] = constraint_detail
    
    validation['summary']['total_violations'] = len(validation['constraint_violations'])
    
    return validation

def generate_comparison_report(results: Dict[str, Dict[str, Any]], 
                             comparison: Dict[str, Any],
                             validations: Dict[str, Dict[str, Any]],
                             farms: List[str], foods: Dict[str, Dict[str, float]],
                             food_groups: Dict[str, List[str]], config: Dict) -> str:
    """Generate a comprehensive comparison report."""
    
    report = []
    report.append("=" * 80)
    report.append("FOOD OPTIMIZATION - SOLVER COMPARISON REPORT")
    report.append("=" * 80)
    report.append("")
    
    # Scenario summary
    report.append("SCENARIO SUMMARY:")
    report.append(f"  Farms: {len(farms)} ({', '.join(farms)})")
    report.append(f"  Foods: {len(foods)} ({', '.join(foods.keys())})")
    report.append(f"  Food Groups: {len(food_groups)}")
    for group, foods_list in food_groups.items():
        report.append(f"    {group}: {', '.join(foods_list)}")
    report.append("")
    
    # Solver results summary
    report.append("SOLVER RESULTS SUMMARY:")
    for solver_name, result in results.items():
        if result is not None:
            status = result.get('status', 'UNKNOWN')
            obj_val = result.get('objective_value', None)
            solve_time = result.get('solve_time', 0)
            
            report.append(f"  {solver_name}:")
            report.append(f"    Status: {status}")
            obj_val_str = f"{obj_val:.6f}" if obj_val is not None else "N/A"
            report.append(f"    Objective Value: {obj_val_str}")
            report.append(f"    Solve Time: {solve_time:.2f}s")
            
            if 'note' in result:
                report.append(f"    Note: {result['note']}")
        else:
            report.append(f"  {solver_name}: Not available")
    report.append("")
    
    # Objective comparison
    if 'objective_comparison' in comparison:
        obj_comp = comparison['objective_comparison']
        report.append("OBJECTIVE VALUE COMPARISON:")
        report.append(f"  Best Solver: {obj_comp['best_solver']} ({obj_comp['best_value']:.6f})")
        report.append(f"  Worst Solver: {obj_comp['worst_solver']} ({obj_comp['worst_value']:.6f})")
        report.append(f"  Relative Gap: {obj_comp['relative_gap']:.2f}%")
        report.append("")
    
    # Timing comparison
    if 'timing_comparison' in comparison:
        time_comp = comparison['timing_comparison']
        report.append("SOLVE TIME COMPARISON:")
        report.append(f"  Fastest Solver: {time_comp['fastest_solver']} ({time_comp['fastest_time']:.2f}s)")
        report.append(f"  Slowest Solver: {time_comp['slowest_solver']} ({time_comp['slowest_time']:.2f}s)")
        report.append("")
    
    # Constraint validation summary
    report.append("CONSTRAINT VALIDATION SUMMARY:")
    for solver_name, validation in validations.items():
        if validation is not None:
            is_valid = validation.get('valid', False)
            num_violations = validation.get('summary', {}).get('total_violations', 0)
            
            report.append(f"  {solver_name}:")
            report.append(f"    Valid: {'✓' if is_valid else '✗'}")
            report.append(f"    Violations: {num_violations}")
            
            if num_violations > 0:
                violation_types = validation.get('summary', {}).get('violation_types', {})
                for vtype, count in violation_types.items():
                    report.append(f"      {vtype}: {count}")
    report.append("")
    
    # Detailed solution comparison (if multiple solutions available)
    valid_results = {name: result for name, result in results.items() 
                    if result is not None and result.get('solution') is not None}
    
    if len(valid_results) >= 2:
        report.append("DETAILED SOLUTION ALLOCATION:")
        
        for farm in farms:
            report.append(f"  {farm}:")
            report.append(f"    {'Food':<12} " + " ".join(f"{solver:<12}" for solver in valid_results.keys()))
            report.append(f"    {'-'*12} " + " ".join(f"{'-'*12}" for _ in valid_results.keys()))
            
            for food in foods.keys():
                food_line = f"    {food:<12} "
                for solver in valid_results.keys():
                    area = valid_results[solver]['solution'][farm][food]
                    food_line += f"{area:<12.2f} "
                report.append(food_line)
            
            # Farm totals
            report.append(f"    {'TOTAL':<12} " + " ".join(f"{sum(valid_results[solver]['solution'][farm].values()):<12.2f}" for solver in valid_results.keys()))
            report.append("")
    
    # Detailed constraint violations (for invalid solutions)
    invalid_solvers = [name for name, validation in validations.items() 
                      if validation is not None and not validation.get('valid', True)]
    
    if invalid_solvers:
        report.append("DETAILED CONSTRAINT VIOLATIONS:")
        for solver_name in invalid_solvers:
            validation = validations[solver_name]
            violations = validation.get('constraint_violations', [])
            
            if violations:
                report.append(f"  {solver_name}:")
                for violation in violations:
                    report.append(f"    - {violation}")
                report.append("")
    
    return "\n".join(report)

def main():
    """Main function to run all solvers and compare results."""
    print("Loading custom food optimization scenario...")
    
    # Load the custom scenario
    farms, foods, food_groups, config = load_food_data('custom')
    
    print(f"Loaded scenario with {len(farms)} farms, {len(foods)} foods, {len(food_groups)} food groups")
    print("\nRunning all available solvers...")
    
    # Results storage
    results = {}
    validations = {}
    
    # Run PuLP solver
    if PULP_AVAILABLE:
        print("\n" + "="*50)
        print("Running PuLP Solver...")
        print("="*50)
        try:
            results['PuLP'] = solve_food_optimization_pulp(farms, foods, food_groups, config)
            if results['PuLP'].get('solution') is not None:
                validations['PuLP'] = comprehensive_constraint_validation(
                    results['PuLP']['solution'], results['PuLP']['binary_solution'],
                    farms, foods, food_groups, config
                )
        except Exception as e:
            print(f"Error running PuLP solver: {e}")
            results['PuLP'] = None
            validations['PuLP'] = None
    else:
        results['PuLP'] = None
        validations['PuLP'] = None
    
    # Run CQM solver
    if CQM_AVAILABLE:
        print("\n" + "="*50)
        print("Running CQM Solver...")
        print("="*50)
        try:
            results['CQM'] = solve_food_optimization_cqm(farms, foods, food_groups, config, dwave_token="dummy")
            if results['CQM'].get('solution') is not None:
                validations['CQM'] = comprehensive_constraint_validation(
                    results['CQM']['solution'], results['CQM']['binary_solution'],
                    farms, foods, food_groups, config
                )
        except Exception as e:
            print(f"Error running CQM solver: {e}")
            results['CQM'] = None
            validations['CQM'] = None
    else:
        results['CQM'] = None
        validations['CQM'] = None
    
    # Run QUBO solver
    if QUBO_AVAILABLE:
        print("\n" + "="*50)
        print("Running QUBO Solver...")
        print("="*50)
        try:
            results['QUBO'] = solve_food_optimization_qubo(farms, foods, food_groups, config, dwave_token="dummy", num_levels=3)
            if results['QUBO'].get('solution') is not None:
                validations['QUBO'] = comprehensive_constraint_validation(
                    results['QUBO']['solution'], results['QUBO']['binary_solution'],
                    farms, foods, food_groups, config
                )
        except Exception as e:
            print(f"Error running QUBO solver: {e}")
            results['QUBO'] = None
            validations['QUBO'] = None
    else:
        results['QUBO'] = None  
        validations['QUBO'] = None
    
    # Compare results
    print("\n" + "="*50)
    print("Comparing Results...")
    print("="*50)
    
    comparison = compare_solutions(results, farms, foods)
    
    # Generate comprehensive report
    report = generate_comparison_report(results, comparison, validations, farms, foods, food_groups, config)
    
    # Display report
    print("\n" + report)
    
    # Save report to file
    report_filename = "comparison_report.txt"
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\nDetailed report saved to: {report_filename}")
    
    # Save results as JSON for further analysis
    results_filename = "solver_results.json"
    
    # Prepare results for JSON serialization (remove non-serializable objects)
    json_results = {}
    for solver, result in results.items():
        if result is not None:
            json_result = {k: v for k, v in result.items() if k not in ['discretization']}  # Remove complex objects
            json_results[solver] = json_result
        else:
            json_results[solver] = None
    
    with open(results_filename, 'w', encoding='utf-8') as f:
        json.dump({
            'results': json_results,
            'validations': validations,
            'comparison': comparison,
            'scenario': {
                'farms': farms,
                'foods': {name: data for name, data in foods.items()},
                'food_groups': food_groups,
                'config': config
            }
        }, f, indent=2, default=str)
    
    print(f"Results data saved to: {results_filename}")
    
    return results, comparison, validations

if __name__ == "__main__":
    results, comparison, validations = main()