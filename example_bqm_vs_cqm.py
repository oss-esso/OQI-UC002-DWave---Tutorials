"""
BQM vs CQM: Side-by-Side Comparison

This script demonstrates the key differences between:
- BQM with penalty-based constraints (soft constraints)
- CQM with built-in hard constraints

Problem: Resource Allocation
- Assign 3 tasks to 2 processors
- Each task must go to exactly one processor
- Minimize total cost
"""

import dimod
from dimod import ConstrainedQuadraticModel, Binary, BinaryQuadraticModel


def solve_with_bqm():
    """
    BQM Approach: Encode constraints as penalties
    """
    print("\n" + "="*70)
    print("APPROACH 1: BQM WITH PENALTY-BASED CONSTRAINTS")
    print("="*70)
    
    # Task costs on each processor
    costs = {
        ('t1', 'p1'): 2.0,
        ('t1', 'p2'): 3.0,
        ('t2', 'p1'): 1.5,
        ('t2', 'p2'): 2.5,
        ('t3', 'p1'): 3.0,
        ('t3', 'p2'): 1.0,
    }
    
    print("\nStep 1: Create BQM and add objective")
    bqm = BinaryQuadraticModel('BINARY')
    
    # Add objective costs
    for (task, proc), cost in costs.items():
        var = f"{task}_{proc}"
        bqm.add_variable(var, cost)
    
    print(f"  Variables created: {list(bqm.variables)}")
    print(f"  Linear terms (objective only):")
    for var in ['t1_p1', 't1_p2', 't2_p1', 't2_p2']:
        print(f"    {var}: {bqm.get_linear(var)}")
    
    print("\nStep 2: Add constraints as penalties")
    print("  Constraint: Each task assigned to exactly one processor")
    print("  Formula: penalty * (t_p1 + t_p2 - 1)^2")
    print("  Expanded: penalty * (-t_p1 - t_p2 + 2*t_p1*t_p2 + 1)")
    
    penalty = 10.0
    print(f"  Penalty weight: {penalty}")
    
    for task in ['t1', 't2', 't3']:
        var1 = f"{task}_p1"
        var2 = f"{task}_p2"
        
        # Add penalty terms
        bqm.add_variable(var1, -penalty)
        bqm.add_variable(var2, -penalty)
        bqm.add_interaction(var1, var2, 2*penalty)
        bqm.offset += penalty
    
    print(f"\n  After adding penalties:")
    print(f"  Linear terms (objective + penalty):")
    for var in ['t1_p1', 't1_p2', 't2_p1', 't2_p2']:
        print(f"    {var}: {bqm.get_linear(var)}")
    print(f"  Quadratic interactions: {len(bqm.quadratic)}")
    print(f"  Offset: {bqm.offset}")
    
    print("\nStep 3: Solve with SimulatedAnnealing")
    sampler = dimod.SimulatedAnnealingSampler()
    sampleset = sampler.sample(bqm, num_reads=100)
    
    best = sampleset.first
    print(f"\n  Best solution:")
    print(f"    Energy: {best.energy:.2f}")
    
    # Interpret solution
    assignments = {}
    violations = []
    for task in ['t1', 't2', 't3']:
        selected = []
        if best.sample[f"{task}_p1"] == 1:
            selected.append('p1')
        if best.sample[f"{task}_p2"] == 1:
            selected.append('p2')
        
        assignments[task] = selected
        if len(selected) != 1:
            violations.append(f"{task}: {len(selected)} processors")
    
    print(f"    Assignments:")
    for task, procs in assignments.items():
        print(f"      {task} -> {procs}")
    
    if violations:
        print(f"    CONSTRAINT VIOLATIONS: {violations}")
    else:
        print(f"    All constraints satisfied!")
    
    # Calculate actual cost
    actual_cost = 0
    for task, procs in assignments.items():
        for proc in procs:
            actual_cost += costs[(task, proc)]
    print(f"    Actual cost (excluding penalties): {actual_cost:.2f}")
    
    print("\n  BQM Characteristics:")
    print("    - Constraints are 'soft' (can be violated)")
    print("    - Penalty terms added to objective")
    print("    - Need to tune penalty weight")
    print("    - Works with all solvers (SA, QPU, Hybrid)")
    
    return bqm, sampleset


def solve_with_cqm():
    """
    CQM Approach: Use built-in hard constraints
    """
    print("\n" + "="*70)
    print("APPROACH 2: CQM WITH HARD CONSTRAINTS")
    print("="*70)
    
    # Task costs on each processor
    costs = {
        ('t1', 'p1'): 2.0,
        ('t1', 'p2'): 3.0,
        ('t2', 'p1'): 1.5,
        ('t2', 'p2'): 2.5,
        ('t3', 'p1'): 3.0,
        ('t3', 'p2'): 1.0,
    }
    
    print("\nStep 1: Create CQM and define binary variables")
    cqm = ConstrainedQuadraticModel()
    
    # Create binary variables
    variables = {}
    for task in ['t1', 't2', 't3']:
        for proc in ['p1', 'p2']:
            var_name = f"{task}_{proc}"
            variables[var_name] = Binary(var_name)
    
    print(f"  Variables created: {list(variables.keys())}")
    
    print("\nStep 2: Set objective (just costs, NO penalties needed!)")
    objective = sum(costs[(task, proc)] * variables[f"{task}_{proc}"]
                   for task in ['t1', 't2', 't3'] 
                   for proc in ['p1', 'p2'])
    
    cqm.set_objective(objective)
    
    print("  Objective expression:")
    print(f"    2.0*t1_p1 + 3.0*t1_p2 + 1.5*t2_p1 + 2.5*t2_p2 + 3.0*t3_p1 + 1.0*t3_p2")
    print("  Note: Pure objective, no penalty terms!")
    
    print("\nStep 3: Add hard constraints")
    print("  Each task must be assigned to exactly one processor")
    
    for task in ['t1', 't2', 't3']:
        constraint = variables[f"{task}_p1"] + variables[f"{task}_p2"] == 1
        cqm.add_constraint(constraint, label=f'{task}_assignment')
        print(f"    {task}_p1 + {task}_p2 == 1 (HARD CONSTRAINT)")
    
    print(f"\n  CQM Summary:")
    print(f"    Number of variables: {len(cqm.variables)}")
    print(f"    Number of constraints: {len(cqm.constraints)}")
    print(f"    Constraint labels: {list(cqm.constraints.keys())}")
    
    print("\nStep 4: Solve with ExactCQMSolver (for demonstration)")
    print("  Note: In production, use LeapHybridCQMSampler")
    print("  Note: ExactCQMSolver may not find optimal CQM solutions")
    
    try:
        # Try exact solver (small problems only)
        from dimod import ExactCQMSolver
        sampler = ExactCQMSolver()
        sampleset = sampler.sample_cqm(cqm)
        
        # Find first feasible solution
        best = None
        for sample in sampleset.data(['sample', 'energy', 'is_feasible']):
            if sample.is_feasible:
                best = sample
                break
        
        if best is None and len(sampleset) > 0:
            best = sampleset.first
        
        if best is not None:
            print(f"\n  Best solution:")
            print(f"    Energy: {best.energy:.2f}")
            print(f"    Is feasible: {best.is_feasible}")
            
            # Interpret solution
            assignments = {}
            for task in ['t1', 't2', 't3']:
                selected = []
                if best.sample[f"{task}_p1"] == 1:
                    selected.append('p1')
                if best.sample[f"{task}_p2"] == 1:
                    selected.append('p2')
                assignments[task] = selected
            
            print(f"    Assignments:")
            for task, procs in assignments.items():
                print(f"      {task} -> {procs}")
            
            # Check constraint satisfaction
            print(f"\n    Constraint check:")
            for task in ['t1', 't2', 't3']:
                count = len(assignments[task])
                status = "[OK] SATISFIED" if count == 1 else "[X] VIOLATED"
                print(f"      {task}: {count} processor(s) - {status}")
            
            # Calculate actual cost
            actual_cost = 0
            for task, procs in assignments.items():
                for proc in procs:
                    actual_cost += costs[(task, proc)]
            print(f"    Total cost: {actual_cost:.2f}")
            
        else:
            print("\n  No feasible solution found (constraints cannot be satisfied)")
            
    except ImportError:
        print("\n  ExactCQMSolver not available in this dimod version")
        print("  CQM requires: pip install dimod>=0.12.0")
        
    # Show production usage
    print("\n" + "-"*70)
    print("Production CQM Usage (with D-Wave Leap):")
    print("-"*70)
    print("\n  from dwave.system import LeapHybridCQMSampler")
    print("  ")
    print("  sampler = LeapHybridCQMSampler()")
    print("  sampleset = sampler.sample_cqm(cqm, label='Task Assignment')")
    print("  ")
    print("  best = sampleset.first")
    print("  if best.is_feasible:")
    print("      print('Feasible solution found!')")
    print("      print(f'Cost: {best.energy}')")
    print("  else:")
    print("      print('No feasible solution found')")
    
    print("\n  CQM Characteristics:")
    print("    - Constraints are 'hard' (ALWAYS satisfied in feasible solutions)")
    print("    - Constraints separate from objective")
    print("    - No penalty tuning needed")
    print("    - Requires CQM-capable solver (Hybrid CQM)")
    print("    - May return infeasible if constraints impossible")
    
    return cqm


def demonstrate_cqm_simplicity():
    """
    Show how simple CQM code is compared to BQM
    """
    print("\n" + "="*70)
    print("CODE COMPARISON: BQM vs CQM")
    print("="*70)
    
    print("\n" + "-"*70)
    print("BQM Code (Manual Penalty Calculation):")
    print("-"*70)
    print("""
# Create BQM
bqm = BinaryQuadraticModel('BINARY')

# Add objective
bqm.add_variable('x', 2.0)
bqm.add_variable('y', 3.0)

# Add constraint x + y = 1 as penalty
# Must manually expand: P * (x + y - 1)^2
penalty = 10.0
bqm.add_variable('x', -penalty)      # Add -P term
bqm.add_variable('y', -penalty)      # Add -P term  
bqm.add_interaction('x', 'y', 2*penalty)  # Add 2P term
bqm.offset += penalty                # Add P term

# Solve
sampler = SimulatedAnnealingSampler()
result = sampler.sample(bqm, num_reads=100)
# No guarantee constraints are satisfied!
""")
    
    print("\n" + "-"*70)
    print("CQM Code (Declarative Constraints):")
    print("-"*70)
    print("""
# Create CQM
cqm = ConstrainedQuadraticModel()

# Define variables
x = Binary('x')
y = Binary('y')

# Set objective (clean!)
cqm.set_objective(2.0*x + 3.0*y)

# Add constraint (declarative!)
cqm.add_constraint(x + y == 1, label='assignment')

# Solve
sampler = LeapHybridCQMSampler()
result = sampler.sample_cqm(cqm)
# Constraints guaranteed in feasible solutions!
""")
    
    print("\n" + "-"*70)
    print("Key Differences:")
    print("-"*70)
    print("  BQM: Manual penalty math, 7 lines of penalty code")
    print("  CQM: One line constraint, no math needed")
    print("  BQM: No feasibility guarantee")
    print("  CQM: Feasibility explicitly checked")
    print("  BQM: Penalty tuning critical")
    print("  CQM: No tuning needed")


def compare_approaches():
    """
    Summary comparison of both approaches
    """
    print("\n" + "="*70)
    print("SIDE-BY-SIDE COMPARISON")
    print("="*70)
    
    print("\nModel Formulation:")
    print("  BQM:")
    print("    Energy = Objective + Penalty * (Constraint_Violation)^2")
    print("    Energy = (2*t1_p1 + 3*t1_p2 + ...) + 10*(t1_p1 + t1_p2 - 1)^2 + ...")
    print("    All mixed together!")
    
    print("\n  CQM:")
    print("    Objective = 2*t1_p1 + 3*t1_p2 + ...")
    print("    Constraints = t1_p1 + t1_p2 == 1, t2_p1 + t2_p2 == 1, ...")
    print("    Separate and clean!")
    
    print("\n" + "-"*70)
    print("Feature Comparison:")
    print("-"*70)
    
    comparison = [
        ("Constraint Type", "Soft (may violate)", "Hard (always met)"),
        ("Penalty Tuning", "Required (critical!)", "Not needed"),
        ("Code Complexity", "Manual penalty math", "Declarative syntax"),
        ("Solver Support", "All (SA, QPU, Hybrid)", "Hybrid CQM only"),
        ("Feasibility", "Always returns solution", "May be infeasible"),
        ("Model Size", "Larger (extra terms)", "Smaller (cleaner)"),
        ("Development", "Easy (local testing)", "Needs Hybrid access"),
        ("Production", "Needs tuning", "Ready to use"),
    ]
    
    print(f"\n{'Feature':<20} {'BQM (Penalties)':<25} {'CQM (Hard Constraints)':<25}")
    print("-" * 70)
    for feature, bqm_val, cqm_val in comparison:
        print(f"{feature:<20} {bqm_val:<25} {cqm_val:<25}")
    
    print("\n" + "-"*70)
    print("When to Choose:")
    print("-"*70)
    
    print("\nChoose BQM if:")
    print("  • Developing and testing with local simulator")
    print("  • Using QPU directly (Advantage, Advantage2)")
    print("  • Soft constraints acceptable (near-optimal okay)")
    print("  • Need flexibility in constraint satisfaction")
    print("  • Problem size fits on QPU")
    
    print("\nChoose CQM if:")
    print("  • Constraints MUST be satisfied (hard requirements)")
    print("  • Complex constraints (inequalities, multiple types)")
    print("  • Don't want to tune penalty weights")
    print("  • Production system with Hybrid solver access")
    print("  • Large problems (1000+ variables)")
    
    print("\n" + "="*70)


def main():
    """Run both approaches and compare"""
    print("\n" + "="*70)
    print("BQM vs CQM: PRACTICAL DEMONSTRATION")
    print("="*70)
    print("\nProblem: Assign 3 tasks to 2 processors")
    print("  Objective: Minimize total cost")
    print("  Constraint: Each task to exactly one processor")
    
    # Solve with BQM
    bqm, bqm_result = solve_with_bqm()
    
    # Solve with CQM
    cqm = solve_with_cqm()
    
    # Show code comparison
    demonstrate_cqm_simplicity()
    
    # Compare
    compare_approaches()
    
    print("\n" + "="*70)
    print("KEY INSIGHT")
    print("="*70)
    print("\nBQM constraints are added, but as PENALTY TERMS:")
    print("  Before: t1_p1 = 2.0 (objective)")
    print("  After:  t1_p1 = -8.0 (objective + penalty: 2.0 + (-10.0))")
    print("\nCQM constraints are SEPARATE:")
    print("  Objective: t1_p1 = 2.0 (stays pure)")
    print("  Constraint: t1_p1 + t1_p2 == 1 (separate rule)")
    print("\nBoth solve the same problem, but with different approaches!")
    print("="*70)


if __name__ == "__main__":
    main()
