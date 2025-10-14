"""
Tutorial 5: Complete Workflow Example

This tutorial brings everything together in a complete end-to-end workflow.
You'll see:

1. Loading real scenario data
2. Formulating the problem as QUBO
3. Converting to DIMOD BQM
4. Solving with multiple samplers
5. Interpreting and validating solutions
6. Comparing results
7. Production-ready code structure

This demonstrates a realistic workflow for solving optimization problems
with D-Wave quantum annealers.
"""

import sys
import os
import numpy as np
import dimod
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import time

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from src.scenarios import load_food_data


@dataclass
class OptimizationSolution:
    """Container for optimization solution results."""
    energy: float
    sample: Dict[int, int]
    assignment: Dict[str, List[str]]  # farm -> list of foods
    objective_value: float
    constraint_violations: Dict[str, Any]
    is_feasible: bool
    solver_info: Dict[str, Any]


class FoodProductionQUBOBuilder:
    """
    Builder class for constructing QUBO formulations of food production problems.
    
    This encapsulates the QUBO formulation logic in a reusable class.
    """
    
    def __init__(self, farms: List[str], foods: Dict[str, Dict[str, float]], 
                 food_groups: Dict[str, List[str]], config: Dict):
        """Initialize with scenario data."""
        self.farms = farms
        self.foods = foods
        self.food_groups = food_groups
        self.config = config
        self.weights = config.get('weights', config.get('parameters', {}).get('weights', {}))
        self.land_availability = config.get('land_availability', config.get('parameters', {}).get('land_availability', {}))
        
        # Create variable mapping
        self.variables = []
        self.variable_map = {}
        self.reverse_map = {}
        idx = 0
        
        for farm in farms:
            for food in foods.keys():
                var_name = f"{farm}_{food}"
                self.variables.append(var_name)
                self.variable_map[var_name] = idx
                self.reverse_map[idx] = var_name
                idx += 1
    
    def build_objective(self) -> Dict[Tuple[int, int], float]:
        """Build objective terms of the QUBO."""
        Q = {}
        
        for farm in self.farms:
            for food, attributes in self.foods.items():
                var_name = f"{farm}_{food}"
                var_idx = self.variable_map[var_name]
                
                # Calculate weighted score
                score = 0
                for criterion, weight in self.weights.items():
                    if weight > 0 and criterion in attributes:
                        score += weight * attributes[criterion]
                
                # Negate for minimization
                Q[(var_idx, var_idx)] = -score
        
        return Q
    
    def add_land_constraints(self, Q: Dict[Tuple[int, int], float], 
                            penalty_weight: float = 5.0,
                            land_per_food: int = 20) -> Dict[Tuple[int, int], float]:
        """Add land availability constraints as penalties."""
        for farm in self.farms:
            if farm not in self.land_availability:
                continue
            
            capacity = self.land_availability[farm]
            max_foods = capacity // land_per_food
            
            # Get variables for this farm
            farm_vars = [self.variable_map[f"{farm}_{food}"] 
                        for food in self.foods.keys()]
            
            # Add penalty for selecting too many foods
            # Simplified: penalize pairs
            for i, var_i in enumerate(farm_vars):
                for j in range(i+1, len(farm_vars)):
                    var_j = farm_vars[j]
                    key = (var_i, var_j) if var_i < var_j else (var_j, var_i)
                    Q[key] = Q.get(key, 0) + penalty_weight * 0.5
        
        return Q
    
    def add_diversity_bonus(self, Q: Dict[Tuple[int, int], float], 
                           bonus_weight: float = -0.5) -> Dict[Tuple[int, int], float]:
        """Add bonus for selecting diverse food groups."""
        for farm in self.farms:
            for group1, foods1 in self.food_groups.items():
                for group2, foods2 in self.food_groups.items():
                    if group1 >= group2:
                        continue
                    
                    for food1 in foods1:
                        for food2 in foods2:
                            var1_name = f"{farm}_{food1}"
                            var2_name = f"{farm}_{food2}"
                            
                            if var1_name not in self.variable_map or var2_name not in self.variable_map:
                                continue
                            
                            var1 = self.variable_map[var1_name]
                            var2 = self.variable_map[var2_name]
                            
                            key = (var1, var2) if var1 < var2 else (var2, var1)
                            Q[key] = Q.get(key, 0) + bonus_weight
        
        return Q
    
    def build_complete_qubo(self, land_penalty: float = 5.0, 
                           diversity_bonus: float = -0.5) -> Dict[Tuple[int, int], float]:
        """Build complete QUBO with objective and constraints."""
        Q = self.build_objective()
        Q = self.add_land_constraints(Q, penalty_weight=land_penalty)
        Q = self.add_diversity_bonus(Q, bonus_weight=diversity_bonus)
        return Q
    
    def interpret_solution(self, sample: Dict[int, int], energy: float) -> OptimizationSolution:
        """Interpret a solution sample in the context of the original problem."""
        # Convert to assignment
        assignment = {farm: [] for farm in self.farms}
        
        for var_idx, value in sample.items():
            if value == 1:
                var_name = self.reverse_map[var_idx]
                farm, food = var_name.split('_', 1)
                assignment[farm].append(food)
        
        # Calculate objective value
        objective_value = 0
        for farm, foods_selected in assignment.items():
            for food in foods_selected:
                for criterion, weight in self.weights.items():
                    if weight > 0 and criterion in self.foods[food]:
                        objective_value += weight * self.foods[food][criterion]
        
        # Check constraints
        constraint_violations = {}
        is_feasible = True
        
        # Check land constraints
        land_per_food = 20
        for farm, foods_selected in assignment.items():
            if farm in self.land_availability:
                land_used = len(foods_selected) * land_per_food
                capacity = self.land_availability[farm]
                if land_used > capacity:
                    constraint_violations[f"land_{farm}"] = {
                        'used': land_used,
                        'capacity': capacity,
                        'violation': land_used - capacity
                    }
                    is_feasible = False
        
        return OptimizationSolution(
            energy=energy,
            sample=sample,
            assignment=assignment,
            objective_value=objective_value,
            constraint_violations=constraint_violations,
            is_feasible=is_feasible,
            solver_info={}
        )


def workflow_step_1_load_data():
    """Step 1: Load scenario data."""
    print("\n" + "="*70)
    print("STEP 1: Load Scenario Data")
    print("="*70)
    
    farms, foods, food_groups, config = load_food_data('simple')
    
    print(f"\nLoaded scenario:")
    print(f"  Farms: {len(farms)}")
    print(f"  Foods: {len(foods)}")
    print(f"  Food groups: {len(food_groups)}")
    print(f"  Total variables: {len(farms) * len(foods)}")
    
    return farms, foods, food_groups, config


def workflow_step_2_build_qubo(farms, foods, food_groups, config):
    """Step 2: Build QUBO formulation."""
    print("\n" + "="*70)
    print("STEP 2: Build QUBO Formulation")
    print("="*70)
    
    builder = FoodProductionQUBOBuilder(farms, foods, food_groups, config)
    
    print(f"\nVariable mapping created:")
    print(f"  Total variables: {len(builder.variables)}")
    print(f"  Example: {builder.variables[0]} -> index {builder.variable_map[builder.variables[0]]}")
    
    Q = builder.build_complete_qubo(land_penalty=5.0, diversity_bonus=-0.5)
    
    print(f"\nQUBO statistics:")
    print(f"  Total entries: {len(Q)}")
    linear_terms = sum(1 for (i, j) in Q.keys() if i == j)
    quadratic_terms = sum(1 for (i, j) in Q.keys() if i != j)
    print(f"  Linear terms: {linear_terms}")
    print(f"  Quadratic terms: {quadratic_terms}")
    
    return builder, Q


def workflow_step_3_convert_to_bqm(Q):
    """Step 3: Convert QUBO to DIMOD BQM."""
    print("\n" + "="*70)
    print("STEP 3: Convert to DIMOD BQM")
    print("="*70)
    
    bqm = dimod.BinaryQuadraticModel.from_qubo(Q)
    
    print(f"\nBQM created:")
    print(f"  Variables: {len(bqm.variables)}")
    print(f"  Linear coefficients: {len(bqm.linear)}")
    print(f"  Quadratic coefficients: {len(bqm.quadratic)}")
    print(f"  Offset: {bqm.offset}")
    print(f"  Variable type: {bqm.vartype}")
    
    return bqm


def workflow_step_4_solve(bqm):
    """Step 4: Solve using multiple samplers."""
    print("\n" + "="*70)
    print("STEP 4: Solve with Multiple Samplers")
    print("="*70)
    
    results = {}
    
    # 1. Simulated Annealing
    print("\n1. Simulated Annealing Sampler:")
    start_time = time.time()
    sa_sampler = dimod.SimulatedAnnealingSampler()
    sa_sampleset = sa_sampler.sample(bqm, num_reads=200, seed=42)
    sa_time = time.time() - start_time
    
    print(f"   Time: {sa_time:.3f} seconds")
    print(f"   Best energy: {sa_sampleset.first.energy:.3f}")
    print(f"   Unique solutions: {len([tuple(s.values()) for s in sa_sampleset.samples()])}")
    
    results['simulated_annealing'] = {
        'sampleset': sa_sampleset,
        'time': sa_time
    }
    
    # 2. Exact Solver (if problem is small enough)
    if len(bqm.variables) <= 20:
        print("\n2. Exact Solver:")
        start_time = time.time()
        exact_sampler = dimod.ExactSolver()
        exact_sampleset = exact_sampler.sample(bqm)
        exact_time = time.time() - start_time
        
        print(f"   Time: {exact_time:.3f} seconds")
        print(f"   Optimal energy: {exact_sampleset.first.energy:.3f}")
        
        results['exact'] = {
            'sampleset': exact_sampleset,
            'time': exact_time
        }
        
        # Compare
        print(f"\n   SA found optimal: {abs(sa_sampleset.first.energy - exact_sampleset.first.energy) < 1e-6}")
    else:
        print("\n2. Exact Solver: Skipped (problem too large)")
    
    return results


def workflow_step_5_interpret(builder, results):
    """Step 5: Interpret solutions."""
    print("\n" + "="*70)
    print("STEP 5: Interpret Solutions")
    print("="*70)
    
    solutions = {}
    
    for solver_name, result in results.items():
        print(f"\n{solver_name.upper()} SOLUTION:")
        sampleset = result['sampleset']
        best_sample = sampleset.first.sample
        best_energy = sampleset.first.energy
        
        solution = builder.interpret_solution(dict(best_sample), best_energy)
        solution.solver_info = {
            'name': solver_name,
            'time': result['time']
        }
        
        print(f"  Energy: {solution.energy:.3f}")
        print(f"  Objective value: {solution.objective_value:.3f}")
        print(f"  Feasible: {solution.is_feasible}")
        print(f"  Solve time: {solution.solver_info['time']:.3f}s")
        
        print(f"\n  Assignment:")
        for farm, foods_selected in solution.assignment.items():
            if foods_selected:
                print(f"    {farm}: {foods_selected}")
        
        if solution.constraint_violations:
            print(f"\n  Constraint violations:")
            for constraint, info in solution.constraint_violations.items():
                print(f"    {constraint}: {info}")
        
        solutions[solver_name] = solution
    
    return solutions


def workflow_step_6_compare(solutions):
    """Step 6: Compare solutions."""
    print("\n" + "="*70)
    print("STEP 6: Compare Solutions")
    print("="*70)
    
    print("\nComparison Table:")
    print(f"{'Solver':<25} {'Energy':<12} {'Obj Value':<12} {'Feasible':<10} {'Time (s)':<10}")
    print("-" * 70)
    
    for solver_name, solution in solutions.items():
        print(f"{solver_name:<25} {solution.energy:<12.3f} {solution.objective_value:<12.3f} "
              f"{str(solution.is_feasible):<10} {solution.solver_info['time']:<10.3f}")
    
    # Find best feasible solution
    feasible_solutions = {k: v for k, v in solutions.items() if v.is_feasible}
    
    if feasible_solutions:
        best_solver = max(feasible_solutions.items(), 
                         key=lambda x: x[1].objective_value)
        print(f"\nBest feasible solution: {best_solver[0]}")
        print(f"  Objective value: {best_solver[1].objective_value:.3f}")
    else:
        print("\nNo feasible solutions found (consider adjusting penalty weights)")
    
    return solutions


def complete_workflow():
    """Execute the complete workflow."""
    print("\n" + "="*70)
    print("COMPLETE WORKFLOW: SCENARIO TO SOLUTION")
    print("="*70)
    print("\nThis demonstrates the full end-to-end process:")
    print("  1. Load scenario data")
    print("  2. Build QUBO formulation")
    print("  3. Convert to DIMOD BQM")
    print("  4. Solve with multiple samplers")
    print("  5. Interpret solutions")
    print("  6. Compare and select best solution")
    
    # Execute workflow
    farms, foods, food_groups, config = workflow_step_1_load_data()
    builder, Q = workflow_step_2_build_qubo(farms, foods, food_groups, config)
    bqm = workflow_step_3_convert_to_bqm(Q)
    results = workflow_step_4_solve(bqm)
    solutions = workflow_step_5_interpret(builder, results)
    workflow_step_6_compare(solutions)
    
    return solutions


def example_production_code():
    """
    Example: Production-ready code structure.
    
    Shows how to structure code for production use.
    """
    print("\n" + "="*70)
    print("PRODUCTION CODE STRUCTURE")
    print("="*70)
    
    print("\nRecommended structure:")
    print("""
    project/
    ├── data/
    │   └── scenarios.py          # Scenario data loading
    ├── models/
    │   ├── qubo_builder.py       # QUBO formulation
    │   └── solution.py           # Solution classes
    ├── solvers/
    │   ├── base.py              # Abstract solver interface
    │   ├── simulator.py         # Simulated annealing
    │   ├── qpu.py              # QPU solver
    │   └── hybrid.py           # Hybrid solver
    ├── config/
    │   └── solver_config.yaml   # Solver configurations
    ├── main.py                  # Entry point
    └── tests/
        └── test_workflow.py     # Unit tests
    """)
    
    print("\nKey principles:")
    print("  1. Separation of concerns (data, models, solvers)")
    print("  2. Abstract interfaces for flexibility")
    print("  3. Configuration management")
    print("  4. Comprehensive testing")
    print("  5. Logging and monitoring")
    print("  6. Error handling and fallbacks")


def main():
    """Run the complete tutorial."""
    print("\n" + "="*70)
    print("TUTORIAL 5: COMPLETE WORKFLOW")
    print("="*70)
    print("\nThis tutorial demonstrates the complete end-to-end workflow")
    print("for solving optimization problems with D-Wave quantum annealers.")
    
    # Run complete workflow
    solutions = complete_workflow()
    
    # Show production code structure
    example_production_code()
    
    print("\n" + "="*70)
    print("TUTORIAL 5 COMPLETE")
    print("="*70)
    print("\nKey Takeaways:")
    print("1. Start with clear data loading and validation")
    print("2. Use builder classes for complex QUBO formulations")
    print("3. Convert to DIMOD BQM for solver compatibility")
    print("4. Test with multiple samplers to verify results")
    print("5. Always interpret solutions in the original problem context")
    print("6. Compare results to select the best approach")
    print("7. Structure code for maintainability and extensibility")
    print("\nYou now have a complete understanding of the workflow!")
    print("Use these tutorials as templates for your own optimization problems.")


if __name__ == "__main__":
    main()
