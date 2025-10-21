"""
Benders Decomposition for Crop Allocation MILP

This script implements a hybrid optimization approach:
- Master Problem: Binary crop selection (Y variables) solved via Simulated Annealing
- Subproblem: Continuous area allocation (A variables) solved via PuLP
- Iterative cut generation until convergence

Author: Autonomous Agent
Date: October 21, 2025
"""

import sys
import os
import time
import numpy as np
import pulp as pl
from typing import Dict, List, Tuple, Optional, Any
import logging
import json
from dataclasses import dataclass, asdict

# Add src directory to path for scenarios import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'my_functions'))

from scenarios import load_food_data
from simulated_annealing import anneal as classical_anneal
from simulated_Qannealing import anneal as quantum_anneal


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class BendersIteration:
    """Store information about each Benders iteration"""
    iteration: int
    master_time: float
    subproblem_time: float
    lower_bound: float
    upper_bound: float
    gap: float
    num_cuts: int
    binary_solution: List[int]
    is_feasible: bool
    cut_type: str  # 'optimality' or 'feasibility'


@dataclass
class BendersSolution:
    """Store final Benders solution"""
    status: str
    objective_value: float
    lower_bound: float
    upper_bound: float
    gap: float
    iterations: int
    total_time: float
    binary_variables: Dict[Tuple[str, str], int]  # (farm, crop) -> 0/1
    area_variables: Dict[Tuple[str, str], float]  # (farm, crop) -> area
    iteration_history: List[BendersIteration]


class BendersDecomposition:
    """
    Benders Decomposition for MILP Crop Allocation Problem
    
    Decomposes the MILP into:
    - Master Problem: Binary Y variables (crop selection) + Benders cuts
    - Subproblem: Continuous A variables (area allocation) given fixed Y
    """
    
    def __init__(
        self,
        farms: List[str],
        crops: Dict[str, Dict[str, float]],
        food_groups: Dict[str, List[str]],
        config: Dict[str, Any],
        use_quantum: bool = False,
        annealing_params: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Benders Decomposition
        
        Args:
            farms: List of farm names
            crops: Dictionary of crop properties
            food_groups: Dictionary mapping food group to crops
            config: Configuration with parameters
            use_quantum: If True, use quantum annealing; else classical
            annealing_params: Parameters for annealing algorithm
        """
        self.farms = farms
        self.crops = crops
        self.crop_names = list(crops.keys())
        self.food_groups = food_groups
        self.config = config
        self.params = config['parameters']
        self.use_quantum = use_quantum
        
        # Set default annealing parameters
        default_params = {
            'T0': 100.0,
            'alpha': 0.95,
            'max_iter': 5000,
            'seed': 42
        }
        if use_quantum:
            default_params.update({
                'num_replicas': 10,
                'gamma0': 50.0,
                'beta': 0.1
            })
        self.annealing_params = annealing_params or default_params
        
        # Benders parameters
        self.tolerance = config.get('benders_tolerance', 1e-3)
        self.max_iterations = config.get('benders_max_iterations', 100)
        self.time_limit = config.get('pulp_time_limit', 120)
        
        # Storage for cuts and solutions
        self.cuts = []  # List of (coefficients, rhs) tuples
        self.iteration_history = []
        self.best_upper_bound = float('inf')
        self.best_solution = None
        self.best_binary_solution = None
        
        # Track theta (objective value approximation from cuts)
        self.theta_values = []  # Track theta over iterations
        
        # Create index mappings
        self.create_index_mappings()
        
        # Extract weights for objective
        self.weights = self.params['weights']
        self.land_availability = self.params['land_availability']
        
        # Get additional constraints
        self.A_min = self.params.get('minimum_planting_area', {})
        if not self.A_min:
            # Default minimum area if not specified
            self.A_min = {crop: 5.0 for crop in self.crop_names}
        
        self.FG_min = {}
        self.FG_max = {}
        for group, crops_in_group in self.food_groups.items():
            fg_constraints = self.params.get('food_group_constraints', {}).get(group, {})
            self.FG_min[group] = fg_constraints.get('min_foods', 1)
            self.FG_max[group] = fg_constraints.get('max_foods', len(crops_in_group))
        
        logger.info(f"Initialized Benders Decomposition:")
        logger.info(f"  Farms: {len(self.farms)}")
        logger.info(f"  Crops: {len(self.crop_names)}")
        logger.info(f"  Binary variables: {len(self.binary_indices)}")
        logger.info(f"  Annealing mode: {'Quantum' if use_quantum else 'Classical'}")
    
    def create_index_mappings(self):
        """Create mappings between binary arrays and (farm, crop) pairs"""
        self.binary_indices = []
        self.index_to_pair = {}
        self.pair_to_index = {}
        
        idx = 0
        for farm in self.farms:
            for crop in self.crop_names:
                self.binary_indices.append((farm, crop))
                self.index_to_pair[idx] = (farm, crop)
                self.pair_to_index[(farm, crop)] = idx
                idx += 1
        
        self.num_binary_vars = len(self.binary_indices)
    
    def decode_binary_solution(self, binary_array: np.ndarray) -> Dict[Tuple[str, str], int]:
        """Convert binary array to dictionary of Y variables"""
        Y = {}
        for idx, val in enumerate(binary_array):
            farm, crop = self.index_to_pair[idx]
            Y[(farm, crop)] = int(val)
        return Y
    
    def encode_binary_solution(self, Y: Dict[Tuple[str, str], int]) -> np.ndarray:
        """Convert dictionary of Y variables to binary array"""
        binary_array = np.zeros(self.num_binary_vars, dtype=int)
        for (farm, crop), val in Y.items():
            idx = self.pair_to_index[(farm, crop)]
            binary_array[idx] = val
        return binary_array
    
    def check_binary_feasibility(self, Y: Dict[Tuple[str, str], int]) -> Tuple[bool, List[str]]:
        """
        Check if binary solution satisfies food group constraints
        
        Returns:
            (is_feasible, list of violation messages)
        """
        violations = []
        
        # Check food group constraints per farm
        for farm in self.farms:
            for group, crops_in_group in self.food_groups.items():
                count = sum(Y.get((farm, crop), 0) for crop in crops_in_group)
                
                if count < self.FG_min[group]:
                    violations.append(
                        f"{farm}: {group} has {count} crops, needs >= {self.FG_min[group]}"
                    )
                if count > self.FG_max[group]:
                    violations.append(
                        f"{farm}: {group} has {count} crops, needs <= {self.FG_max[group]}"
                    )
        
        return len(violations) == 0, violations
    
    def calculate_penalty(self, Y: Dict[Tuple[str, str], int]) -> float:
        """
        Calculate penalty for constraint violations in binary solution
        Used in master problem objective function
        """
        penalty = 0.0
        penalty_weight = 1000.0  # Large penalty for violations
        
        # Food group constraint penalties
        for farm in self.farms:
            for group, crops_in_group in self.food_groups.items():
                count = sum(Y.get((farm, crop), 0) for crop in crops_in_group)
                
                # Penalty for violating minimum
                if count < self.FG_min[group]:
                    penalty += penalty_weight * (self.FG_min[group] - count)
                
                # Penalty for violating maximum
                if count > self.FG_max[group]:
                    penalty += penalty_weight * (count - self.FG_max[group])
        
        return penalty
    
    def master_objective_function(self, binary_array: np.ndarray) -> float:
        """
        Objective function for master problem (for annealing)
        
        Returns energy (lower is better) = -objective + penalty + cuts
        """
        Y = self.decode_binary_solution(binary_array)
        
        # Large penalty weight for violations
        penalty_weight = 1000.0
        
        # Calculate penalty for constraint violations
        penalty = self.calculate_penalty(Y)
        
        # Estimate objective value (simplified, actual comes from subproblem)
        # For now, use a heuristic based on crop properties
        estimated_obj = 0.0
        total_area = sum(self.land_availability.values())
        
        for (farm, crop), y_val in Y.items():
            if y_val > 0:
                # Use minimum area as estimate when crop is selected
                area_est = self.A_min.get(crop, 5.0)
                
                # Calculate contribution to objective
                N = self.crops[crop].get('nutritional_value', 0.5)
                D = self.crops[crop].get('nutrient_density', 0.5)
                E = self.crops[crop].get('environmental_impact', 0.5)
                P = self.crops[crop].get('sustainability', 0.5)
                
                obj_contribution = (
                    self.weights.get('nutritional_value', 0.25) * N * area_est +
                    self.weights.get('nutrient_density', 0.25) * D * area_est -
                    self.weights.get('environmental_impact', 0.25) * E * area_est +
                    self.weights.get('sustainability', 0.25) * P * area_est
                ) / total_area
                
                estimated_obj += obj_contribution
        
        # Add Benders cuts
        cut_value = 0.0
        for cut_coeffs, cut_rhs in self.cuts:
            # Evaluate cut: sum(coeff * y) >= rhs
            cut_lhs = sum(cut_coeffs.get((farm, crop), 0.0) * Y[(farm, crop)] 
                         for farm, crop in self.binary_indices)
            # Penalty if cut is violated
            if cut_lhs < cut_rhs:
                cut_value += penalty_weight * (cut_rhs - cut_lhs)
        
        # Return energy (negative objective + penalties)
        energy = -estimated_obj + penalty + cut_value
        
        return energy
    
    def solve_master_problem(self) -> Dict[Tuple[str, str], int]:
        """
        Solve master problem using simulated annealing
        
        Returns:
            Dictionary of binary Y variables
        """
        # Create initial solution (random feasible if possible)
        initial_binary = self.generate_initial_solution()
        
        # Choose annealing algorithm
        if self.use_quantum:
            logger.info("Solving master problem with Quantum Annealing...")
            best_binary, best_energy = quantum_anneal(
                objective_function=self.master_objective_function,
                initial_solution=initial_binary,
                T0=self.annealing_params['T0'],
                alpha=self.annealing_params['alpha'],
                max_iterations=self.annealing_params['max_iter'],
                num_replicas=self.annealing_params['num_replicas'],
                gamma0=self.annealing_params['gamma0'],
                beta=self.annealing_params['beta'],
                seed=self.annealing_params.get('seed')
            )
        else:
            logger.info("Solving master problem with Classical Annealing...")
            best_binary, best_energy = classical_anneal(
                objective_function=self.master_objective_function,
                initial_state=initial_binary,
                T0=self.annealing_params['T0'],
                alpha=self.annealing_params['alpha'],
                max_iter=self.annealing_params['max_iter'],
                seed=self.annealing_params.get('seed')
            )
        
        Y = self.decode_binary_solution(best_binary)
        logger.info(f"Master problem energy: {best_energy:.6f}")
        
        return Y
    
    def generate_initial_solution(self) -> np.ndarray:
        """Generate initial binary solution (attempt feasible)"""
        binary = np.zeros(self.num_binary_vars, dtype=int)
        
        # Try to satisfy food group minimums
        for farm in self.farms:
            for group, crops_in_group in self.food_groups.items():
                # Select minimum required crops from this group
                available_crops = [c for c in crops_in_group if c in self.crop_names]
                num_to_select = min(self.FG_min[group], len(available_crops))
                selected = np.random.choice(available_crops, num_to_select, replace=False)
                
                for crop in selected:
                    idx = self.pair_to_index[(farm, crop)]
                    binary[idx] = 1
        
        return binary
    
    def solve_subproblem(
        self, 
        Y: Dict[Tuple[str, str], int]
    ) -> Tuple[str, float, Optional[Dict[Tuple[str, str], float]], Optional[Dict]]:
        """
        Solve subproblem given fixed binary Y variables
        
        Args:
            Y: Fixed binary crop selection variables
            
        Returns:
            (status, objective_value, A_variables, duals)
            status: 'Optimal', 'Infeasible', or 'Error'
            objective_value: Value of subproblem objective
            A_variables: Dictionary of area allocation variables (if feasible)
            duals: Dictionary of dual variables for cut generation (if feasible)
        """
        logger.info("Solving subproblem with PuLP...")
        
        # Create subproblem
        subproblem = pl.LpProblem("Benders_Subproblem", pl.LpMaximize)
        
        # Create continuous area variables
        A = pl.LpVariable.dicts(
            "Area",
            [(f, c) for f in self.farms for c in self.crop_names],
            lowBound=0
        )
        
        # Objective function
        total_area = sum(self.land_availability.values())
        
        objective_terms = []
        for farm in self.farms:
            for crop in self.crop_names:
                N = self.crops[crop].get('nutritional_value', 0.5)
                D = self.crops[crop].get('nutrient_density', 0.5)
                E = self.crops[crop].get('environmental_impact', 0.5)
                P = self.crops[crop].get('sustainability', 0.5)
                
                obj_coeff = (
                    self.weights.get('nutritional_value', 0.25) * N +
                    self.weights.get('nutrient_density', 0.25) * D -
                    self.weights.get('environmental_impact', 0.25) * E +
                    self.weights.get('sustainability', 0.25) * P
                ) / total_area
                
                objective_terms.append(obj_coeff * A[(farm, crop)])
        
        subproblem += pl.lpSum(objective_terms), "Objective"
        
        # Constraints
        # 1. Land availability per farm
        for farm in self.farms:
            subproblem += (
                pl.lpSum([A[(farm, crop)] for crop in self.crop_names]) 
                <= self.land_availability[farm],
                f"Land_Limit_{farm}"
            )
        
        # 2. Linking constraints: A >= A_min * Y
        for farm in self.farms:
            for crop in self.crop_names:
                y_val = Y.get((farm, crop), 0)
                a_min = self.A_min.get(crop, 5.0)
                subproblem += (
                    A[(farm, crop)] >= a_min * y_val,
                    f"MinArea_{farm}_{crop}"
                )
        
        # 3. Linking constraints: A <= L * Y
        for farm in self.farms:
            for crop in self.crop_names:
                y_val = Y.get((farm, crop), 0)
                subproblem += (
                    A[(farm, crop)] <= self.land_availability[farm] * y_val,
                    f"MaxArea_{farm}_{crop}"
                )
        
        # Solve subproblem
        solver = pl.PULP_CBC_CMD(msg=0, timeLimit=self.time_limit)
        subproblem.solve(solver)
        
        status = pl.LpStatus[subproblem.status]
        
        if status == 'Optimal':
            obj_value = pl.value(subproblem.objective)
            
            # Extract solution
            A_solution = {}
            for farm in self.farms:
                for crop in self.crop_names:
                    val = A[(farm, crop)].varValue
                    A_solution[(farm, crop)] = val if val is not None else 0.0
            
            # Extract dual variables for cut generation
            duals = {}
            for name, constraint in subproblem.constraints.items():
                duals[name] = constraint.pi if hasattr(constraint, 'pi') else 0.0
            
            logger.info(f"Subproblem optimal, objective: {obj_value:.6f}")
            return 'Optimal', obj_value, A_solution, duals
        
        elif status == 'Infeasible':
            logger.warning("Subproblem infeasible!")
            return 'Infeasible', float('inf'), None, None
        
        else:
            logger.error(f"Subproblem status: {status}")
            return 'Error', float('inf'), None, None
    
    def generate_optimality_cut(
        self,
        Y: Dict[Tuple[str, str], int],
        subproblem_obj: float,
        duals: Dict[str, float]
    ) -> Tuple[Dict[Tuple[str, str], float], float]:
        """
        Generate Benders optimality cut from subproblem duals
        
        Cut form: theta <= subproblem_obj + sum(dual * (y - y_current))
        Rearranged: sum(dual * y) >= rhs
        
        Returns:
            (cut_coefficients, cut_rhs)
        """
        cut_coeffs = {}
        
        # Extract duals from linking constraints
        for farm in self.farms:
            for crop in self.crop_names:
                # Dual from MinArea constraint: A >= A_min * Y
                dual_min_key = f"MinArea_{farm}_{crop}"
                dual_min = duals.get(dual_min_key, 0.0)
                
                # Dual from MaxArea constraint: A <= L * Y
                dual_max_key = f"MaxArea_{farm}_{crop}"
                dual_max = duals.get(dual_max_key, 0.0)
                
                # Coefficient for Y variable in cut
                a_min = self.A_min.get(crop, 5.0)
                coeff = dual_min * a_min - dual_max * self.land_availability[farm]
                
                cut_coeffs[(farm, crop)] = coeff
        
        # Calculate RHS
        rhs = subproblem_obj - sum(
            cut_coeffs.get((farm, crop), 0.0) * Y[(farm, crop)]
            for farm, crop in self.binary_indices
        )
        
        return cut_coeffs, rhs
    
    def generate_feasibility_cut(
        self,
        Y: Dict[Tuple[str, str], int]
    ) -> Tuple[Dict[Tuple[str, str], float], float]:
        """
        Generate Benders feasibility cut when subproblem is infeasible
        
        Simple cut: require at least one Y variable to change
        
        Returns:
            (cut_coefficients, cut_rhs)
        """
        cut_coeffs = {}
        
        # Find which Y variables are 1
        ones = [(farm, crop) for (farm, crop), val in Y.items() if val == 1]
        zeros = [(farm, crop) for (farm, crop), val in Y.items() if val == 0]
        
        # Cut: sum(y for y=1) - sum(y for y=0) <= |ones| - 1
        # Rearranged: -sum(y for y=1) + sum(y for y=0) >= -|ones| + 1
        
        for farm, crop in ones:
            cut_coeffs[(farm, crop)] = -1.0
        for farm, crop in zeros:
            cut_coeffs[(farm, crop)] = 1.0
        
        rhs = -len(ones) + 1
        
        return cut_coeffs, rhs
    
    def calculate_theta_lower_bound(self, Y: Dict[Tuple[str, str], int]) -> float:
        """
        Calculate theta (lower bound) from all cuts for given Y
        
        Theta represents the best lower bound on the objective for the current Y
        based on all generated cuts
        """
        if not self.cuts:
            return -float('inf')
        
        # For each cut, calculate: sum(coeff * y) - rhs
        # Theta is bounded by all cuts, so we take the minimum violation
        theta = -float('inf')
        
        for cut_coeffs, cut_rhs in self.cuts:
            cut_value = sum(
                cut_coeffs.get((farm, crop), 0.0) * Y[(farm, crop)]
                for farm, crop in self.binary_indices
            )
            # The cut states: theta <= cut_value
            # So theta is bounded by the minimum of all such values
            theta = max(theta, cut_value)
        
        return theta
    
    def solve(self) -> BendersSolution:
        """
        Main Benders Decomposition algorithm
        
        Returns:
            BendersSolution with complete results
        """
        logger.info("="*80)
        logger.info("Starting Benders Decomposition")
        logger.info("="*80)
        
        start_time = time.time()
        lower_bound = -float('inf')
        
        # Initialize upper bound for maximization problem
        if self.best_upper_bound == float('inf'):
            self.best_upper_bound = -float('inf')
        
        for iteration in range(1, self.max_iterations + 1):
            logger.info(f"\n{'='*80}")
            logger.info(f"Iteration {iteration}")
            logger.info(f"{'='*80}")
            
            # Solve master problem
            master_start = time.time()
            Y = self.solve_master_problem()
            master_time = time.time() - master_start
            
            # Check binary feasibility
            is_feasible, violations = self.check_binary_feasibility(Y)
            if not is_feasible:
                logger.warning(f"Binary solution violates constraints:")
                for v in violations[:5]:  # Show first 5 violations
                    logger.warning(f"  {v}")
            
            # Solve subproblem
            subproblem_start = time.time()
            status, subproblem_obj, A_solution, duals = self.solve_subproblem(Y)
            subproblem_time = time.time() - subproblem_start
            
            # Generate cut and update bounds
            if status == 'Optimal':
                # For MAXIMIZATION: upper_bound = best objective found (from subproblem)
                # lower_bound = master problem bound (gets better with cuts)
                
                # Update upper bound (best feasible solution objective)
                if subproblem_obj > self.best_upper_bound or self.best_upper_bound == float('inf'):
                    if self.best_upper_bound == float('inf'):
                        self.best_upper_bound = -float('inf')  # Initialize properly for max problem
                    if subproblem_obj > self.best_upper_bound:
                        self.best_upper_bound = subproblem_obj
                        self.best_solution = (Y, A_solution)
                        self.best_binary_solution = Y.copy()
                        logger.info(f"New best solution found! Objective: {subproblem_obj:.6f}")
                
                # Generate optimality cut
                cut_coeffs, cut_rhs = self.generate_optimality_cut(Y, subproblem_obj, duals)
                self.cuts.append((cut_coeffs, cut_rhs))
                
                # Calculate theta (lower bound from cuts)
                theta = self.calculate_theta_lower_bound(Y)
                self.theta_values.append(theta)
                lower_bound = max(lower_bound, theta)
                
                cut_type = 'optimality'
                
            else:  # Infeasible or Error
                # Generate feasibility cut
                cut_coeffs, cut_rhs = self.generate_feasibility_cut(Y)
                self.cuts.append((cut_coeffs, cut_rhs))
                
                subproblem_obj = -float('inf')  # Infeasible solution has no valid objective
                cut_type = 'feasibility'
            
            # Calculate gap (for maximization)
            if self.best_upper_bound > -float('inf') and self.best_upper_bound < float('inf'):
                if lower_bound > -float('inf'):
                    gap = abs(self.best_upper_bound - lower_bound) / (abs(self.best_upper_bound) + 1e-10)
                else:
                    gap = float('inf')
            else:
                gap = float('inf')
            
            # Store iteration info
            iter_info = BendersIteration(
                iteration=iteration,
                master_time=master_time,
                subproblem_time=subproblem_time,
                lower_bound=lower_bound,
                upper_bound=self.best_upper_bound,
                gap=gap,
                num_cuts=len(self.cuts),
                binary_solution=[Y[(farm, crop)] for farm, crop in self.binary_indices],
                is_feasible=(status == 'Optimal'),
                cut_type=cut_type
            )
            self.iteration_history.append(iter_info)
            
            # Log progress
            logger.info(f"Lower Bound: {lower_bound:.6f}")
            logger.info(f"Upper Bound: {self.best_upper_bound:.6f}")
            logger.info(f"Gap: {gap:.6%}")
            logger.info(f"Cuts: {len(self.cuts)}")
            logger.info(f"Master time: {master_time:.2f}s, Subproblem time: {subproblem_time:.2f}s")
            
            # Check convergence
            if gap < self.tolerance and self.best_upper_bound < float('inf'):
                logger.info(f"\n{'='*80}")
                logger.info(f"CONVERGED! Gap {gap:.6%} < tolerance {self.tolerance:.6%}")
                logger.info(f"{'='*80}")
                break
            
            # Check time limit
            if time.time() - start_time > self.time_limit * 5:  # 5x subproblem time limit
                logger.warning("Time limit reached!")
                break
        
        total_time = time.time() - start_time
        
        # Create solution object
        if self.best_solution is not None and self.best_upper_bound > -float('inf'):
            Y_best, A_best = self.best_solution
            solution = BendersSolution(
                status='Optimal' if gap < self.tolerance else 'SubOptimal',
                objective_value=self.best_upper_bound,
                lower_bound=lower_bound,
                upper_bound=self.best_upper_bound,
                gap=gap,
                iterations=len(self.iteration_history),
                total_time=total_time,
                binary_variables=Y_best,
                area_variables=A_best,
                iteration_history=self.iteration_history
            )
        else:
            solution = BendersSolution(
                status='NoSolution',
                objective_value=-float('inf'),
                lower_bound=lower_bound,
                upper_bound=self.best_upper_bound,
                gap=gap,
                iterations=len(self.iteration_history),
                total_time=total_time,
                binary_variables={},
                area_variables={},
                iteration_history=self.iteration_history
            )
        
        logger.info(f"\n{'='*80}")
        logger.info("Benders Decomposition Complete")
        logger.info(f"{'='*80}")
        logger.info(f"Status: {solution.status}")
        logger.info(f"Objective: {solution.objective_value:.6f}")
        logger.info(f"Iterations: {solution.iterations}")
        logger.info(f"Total Time: {solution.total_time:.2f}s")
        
        return solution


def print_solution(solution: BendersSolution, farms: List[str], crops: List[str]):
    """Print detailed solution information"""
    print("\n" + "="*80)
    print("BENDERS DECOMPOSITION SOLUTION")
    print("="*80)
    print(f"Status: {solution.status}")
    print(f"Objective Value: {solution.objective_value:.6f}")
    print(f"Gap: {solution.gap:.6%}")
    print(f"Iterations: {solution.iterations}")
    print(f"Total Time: {solution.total_time:.2f}s")
    
    if solution.status in ['Optimal', 'SubOptimal']:
        print("\n" + "="*80)
        print("CROP SELECTION (Y variables)")
        print("="*80)
        
        for farm in farms:
            selected = []
            for crop in crops:
                if solution.binary_variables.get((farm, crop), 0) == 1:
                    selected.append(crop)
            print(f"{farm}: {', '.join(selected) if selected else 'None'}")
        
        print("\n" + "="*80)
        print("AREA ALLOCATION (A variables)")
        print("="*80)
        
        for farm in farms:
            print(f"\n{farm}:")
            print(f"{'Crop':<15} {'Selected':<10} {'Area (ha)':<15}")
            print("-" * 40)
            for crop in crops:
                y_val = solution.binary_variables.get((farm, crop), 0)
                a_val = solution.area_variables.get((farm, crop), 0.0)
                if y_val > 0 or a_val > 0.01:
                    print(f"{crop:<15} {y_val:<10} {a_val:<15.4f}")
    
    # Print convergence history
    print("\n" + "="*80)
    print("CONVERGENCE HISTORY")
    print("="*80)
    print(f"{'Iter':<6} {'LB':<12} {'UB':<12} {'Gap':<10} {'Cuts':<6} {'Type':<12}")
    print("-" * 70)
    for iter_info in solution.iteration_history[-10:]:  # Last 10 iterations
        print(f"{iter_info.iteration:<6} "
              f"{iter_info.lower_bound:<12.4f} "
              f"{iter_info.upper_bound:<12.4f} "
              f"{iter_info.gap:<10.4%} "
              f"{iter_info.num_cuts:<6} "
              f"{iter_info.cut_type:<12}")


def save_solution(solution: BendersSolution, filename: str):
    """Save solution to JSON file"""
    # Convert solution to serializable format
    solution_dict = {
        'status': solution.status,
        'objective_value': solution.objective_value,
        'lower_bound': solution.lower_bound,
        'upper_bound': solution.upper_bound,
        'gap': solution.gap,
        'iterations': solution.iterations,
        'total_time': solution.total_time,
        'binary_variables': {f"{k[0]}_{k[1]}": v for k, v in solution.binary_variables.items()},
        'area_variables': {f"{k[0]}_{k[1]}": v for k, v in solution.area_variables.items()},
        'iteration_history': [
            {
                'iteration': it.iteration,
                'lower_bound': it.lower_bound,
                'upper_bound': it.upper_bound,
                'gap': it.gap,
                'num_cuts': it.num_cuts,
                'cut_type': it.cut_type,
                'is_feasible': it.is_feasible
            }
            for it in solution.iteration_history
        ]
    }
    
    with open(filename, 'w') as f:
        json.dump(solution_dict, f, indent=2)
    
    logger.info(f"Solution saved to {filename}")


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Benders Decomposition for Crop Allocation')
    parser.add_argument(
        '--scenario',
        type=str,
        default='simple',
        choices=['simple', 'intermediate', 'custom', 'full', 'full_family'],
        help='Scenario complexity level'
    )
    parser.add_argument(
        '--quantum',
        action='store_true',
        help='Use quantum annealing for master problem'
    )
    parser.add_argument(
        '--max-iter',
        type=int,
        default=50,
        help='Maximum Benders iterations'
    )
    parser.add_argument(
        '--tolerance',
        type=float,
        default=1e-3,
        help='Convergence tolerance'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='benders_solution.json',
        help='Output file for solution'
    )
    
    args = parser.parse_args()
    
    # Load scenario
    logger.info(f"Loading scenario: {args.scenario}")
    farms, crops, food_groups, config = load_food_data(args.scenario)
    
    # Override config with command line args
    config['benders_max_iterations'] = args.max_iter
    config['benders_tolerance'] = args.tolerance
    
    # Create and solve
    benders = BendersDecomposition(
        farms=farms,
        crops=crops,
        food_groups=food_groups,
        config=config,
        use_quantum=args.quantum
    )
    
    solution = benders.solve()
    
    # Print and save results
    print_solution(solution, farms, list(crops.keys()))
    save_solution(solution, args.output)


if __name__ == "__main__":
    main()
