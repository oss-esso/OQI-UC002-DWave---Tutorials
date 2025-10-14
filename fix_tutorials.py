"""
Quick patch to fix tutorial bugs found in testing.
This script will be run once to fix the issues.
"""

import re

def fix_file(filepath, replacements):
    """Apply replacements to a file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    for old, new in replacements:
        content = content.replace(old, new)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Fixed: {filepath}")

# Fix Tutorial 1 - generator slicing
fix_file('tutorial_01_basic_dimod.py', [
    ("for i, sample in enumerate(sampleset.data(['sample', 'energy', 'num_occurrences'])[:5]):",
     "for i, sample in enumerate(list(sampleset.data(['sample', 'energy', 'num_occurrences']))[:5]):"),
])

# Fix Tutorial 2 - generator slicing and SampleView hashing
fix_file('tutorial_02_qubo_basics.py', [
    ("for i, sample_data in enumerate(sampleset.data(['sample', 'energy'])[:5]):",
     "for i, sample_data in enumerate(list(sampleset.data(['sample', 'energy']))[:5]):"),
    ("for i, sample_data in enumerate(sampleset.data(['sample', 'energy', 'num_occurrences'])[:5]):",
     "for i, sample_data in enumerate(list(sampleset.data(['sample', 'energy', 'num_occurrences']))[:5]):"),
    ("print(f\"   Unique solutions found: {len(set(sa_sampleset.samples()))}\")",
     "print(f\"   Unique solutions found: {len([tuple(s.values()) for s in sa_sampleset.samples()])}\")"),
    ("bqm_qubo = dimod.BinaryQuadraticModel.from_qubo(Q)",
     "bqm_qubo = dimod.BinaryQuadraticModel.from_qubo(Q)\n    bqm_qubo.change_vartype(dimod.BINARY, inplace=True)"),
])

# Fix Tutorial 3 - config['weights'] to config['parameters']['weights']
fix_file('tutorial_03_scenario_to_qubo.py', [
    ("weights = config['weights']",
     "weights = config.get('weights', config.get('parameters', {}).get('weights', {}))"),
    ("land_availability = config['land_availability']",
     "land_availability = config.get('land_availability', config.get('parameters', {}).get('land_availability', {}))"),
])

# Fix Tutorial 4 - SampleView hashing
fix_file('tutorial_04_dwave_integration.py', [
    ("print(f\"  Unique solutions found: {len(set(sampleset.samples()))}\")",
     "print(f\"  Unique solutions found: {len([tuple(s.values()) for s in sampleset.samples()])}\")"),
])

# Fix Tutorial 5 - config['weights'] to config['parameters']['weights']
fix_file('tutorial_05_complete_workflow.py', [
    ("self.weights = config['weights']",
     "self.weights = config.get('weights', config.get('parameters', {}).get('weights', {}))"),
    ("self.land_availability = config.get('land_availability', {})",
     "self.land_availability = config.get('land_availability', config.get('parameters', {}).get('land_availability', {}))"),
    ("weights = config['weights']",
     "weights = config.get('weights', config.get('parameters', {}).get('weights', {}))"),
])

print("\nAll fixes applied successfully!")
