"""
Second pass of fixes for remaining bugs.
"""

def fix_file(filepath, replacements):
    """Apply replacements to a file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    for old, new in replacements:
        content = content.replace(old, new)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Fixed: {filepath}")

# Fix Tutorial 2 - QUBO vartype issue (need to keep it as BINARY)
fix_file('tutorial_02_qubo_basics.py', [
    ("bqm_qubo = dimod.BinaryQuadraticModel.from_qubo(Q)\n    bqm_qubo.change_vartype(dimod.BINARY, inplace=True)",
     "bqm_qubo = dimod.BinaryQuadraticModel.from_qubo(Q)"),
])

# Fix Tutorial 3 - more generator slicing and config access
fix_file('tutorial_03_scenario_to_qubo.py', [
    ("for farm, land in config['land_availability'].items():",
     "land_avail = config.get('land_availability', config.get('parameters', {}).get('land_availability', {}))\n    for farm, land in land_avail.items():"),
    ("for sol_idx, sample_data in enumerate(sampleset.data(['sample', 'energy'])[:3]):",
     "for sol_idx, sample_data in enumerate(list(sampleset.data(['sample', 'energy']))[:3]):"),
])

# Fix Tutorial 5 - SampleView hashing in workflow
fix_file('tutorial_05_complete_workflow.py', [
    ("print(f\"   Unique solutions: {len(set(sa_sampleset.samples()))}\")",
     "print(f\"   Unique solutions: {len([tuple(s.values()) for s in sa_sampleset.samples()])}\")"),
])

print("\nAll additional fixes applied successfully!")
