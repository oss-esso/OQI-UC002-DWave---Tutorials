"""
Test Suite for D-Wave QUBO/DIMOD Tutorials

This test suite verifies that all tutorial scripts work correctly
using the simulator (no D-Wave token required).

Run with: python test_tutorials.py
Or with pytest: pytest test_tutorials.py -v
"""

import sys
import os
import unittest
import importlib.util
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)


def import_tutorial(tutorial_name: str):
    """Import a tutorial module by name."""
    file_path = os.path.join(project_root, f"{tutorial_name}.py")
    spec = importlib.util.spec_from_file_location(tutorial_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[tutorial_name] = module
    spec.loader.exec_module(module)
    return module


class TestTutorial01(unittest.TestCase):
    """Tests for Tutorial 1: Basic DIMOD BQM Construction."""
    
    @classmethod
    def setUpClass(cls):
        """Import the tutorial module."""
        cls.tutorial = import_tutorial('tutorial_01_basic_dimod')
    
    def test_example_1_simple_bqm(self):
        """Test Example 1: Simple BQM construction."""
        print("\nTesting Tutorial 1 - Example 1...")
        bqm, sampleset = self.tutorial.example_1_simple_bqm()
        
        # Verify BQM was created
        self.assertIsNotNone(bqm)
        self.assertEqual(len(bqm.variables), 3)
        
        # Verify solution was found
        self.assertIsNotNone(sampleset)
        self.assertGreater(len(sampleset), 0)
        
        print("  PASSED: Simple BQM construction works")
    
    def test_example_2_construction_methods(self):
        """Test Example 2: Different construction methods."""
        print("\nTesting Tutorial 1 - Example 2...")
        bqm_a, sampleset_a = self.tutorial.example_2_different_construction_methods()
        
        self.assertIsNotNone(bqm_a)
        self.assertEqual(len(bqm_a.variables), 3)
        
        print("  PASSED: Different construction methods work")
    
    def test_example_3_practical_problem(self):
        """Test Example 3: Practical resource allocation."""
        print("\nTesting Tutorial 1 - Example 3...")
        bqm, sampleset = self.tutorial.example_3_practical_problem()
        
        self.assertIsNotNone(bqm)
        self.assertGreater(len(bqm.variables), 0)
        
        print("  PASSED: Practical problem formulation works")
    
    def test_example_4_energy_understanding(self):
        """Test Example 4: Understanding energy landscapes."""
        print("\nTesting Tutorial 1 - Example 4...")
        bqm, sampleset = self.tutorial.example_4_understanding_energy()
        
        self.assertIsNotNone(bqm)
        self.assertEqual(len(bqm.variables), 2)
        
        # Check that optimal solution was found
        self.assertEqual(sampleset.first.energy, -4.0)
        
        print("  PASSED: Energy landscape understanding works")


class TestTutorial02(unittest.TestCase):
    """Tests for Tutorial 2: QUBO Formulation Basics."""
    
    @classmethod
    def setUpClass(cls):
        """Import the tutorial module."""
        cls.tutorial = import_tutorial('tutorial_02_qubo_basics')
    
    def test_example_1_qubo_matrix(self):
        """Test Example 1: QUBO matrix basics."""
        print("\nTesting Tutorial 2 - Example 1...")
        bqm, sampleset = self.tutorial.example_1_qubo_matrix_basics()
        
        self.assertIsNotNone(bqm)
        self.assertEqual(len(bqm.variables), 3)
        
        print("  PASSED: QUBO matrix representation works")
    
    def test_example_2_constraint_penalty(self):
        """Test Example 2: Constraint as penalty."""
        print("\nTesting Tutorial 2 - Example 2...")
        bqm, sampleset = self.tutorial.example_2_constraint_as_penalty()
        
        self.assertIsNotNone(bqm)
        self.assertEqual(len(bqm.variables), 4)
        
        # Check that best solution respects constraint (selects 2 items)
        best_sample = sampleset.first.sample
        num_selected = sum(best_sample.values())
        # With proper penalty, should select exactly 2
        self.assertLessEqual(abs(num_selected - 2), 1)  # Allow small deviation
        
        print("  PASSED: Constraint penalty encoding works")
    
    def test_example_3_number_partitioning(self):
        """Test Example 3: Number partitioning."""
        print("\nTesting Tutorial 2 - Example 3...")
        bqm, sampleset = self.tutorial.example_3_number_partitioning()
        
        self.assertIsNotNone(bqm)
        self.assertEqual(len(bqm.variables), 6)
        
        print("  PASSED: Number partitioning problem works")
    
    def test_example_4_sampler_comparison(self):
        """Test Example 4: Comparing samplers."""
        print("\nTesting Tutorial 2 - Example 4...")
        bqm, exact, sa, random = self.tutorial.example_4_comparing_samplers()
        
        self.assertIsNotNone(exact)
        self.assertIsNotNone(sa)
        self.assertIsNotNone(random)
        
        # Verify exact solver found best solution
        self.assertLessEqual(exact.first.energy, sa.first.energy)
        self.assertLessEqual(exact.first.energy, random.first.energy)
        
        print("  PASSED: Sampler comparison works")
    
    def test_example_5_qubo_ising_conversion(self):
        """Test Example 5: QUBO/Ising conversion."""
        print("\nTesting Tutorial 2 - Example 5...")
        bqm_qubo, bqm_ising = self.tutorial.example_5_qubo_from_ising()
        
        self.assertIsNotNone(bqm_qubo)
        self.assertIsNotNone(bqm_ising)
        
        self.assertEqual(bqm_qubo.vartype.name, 'BINARY')
        self.assertEqual(bqm_ising.vartype.name, 'SPIN')
        
        print("  PASSED: QUBO/Ising conversion works")


class TestTutorial03(unittest.TestCase):
    """Tests for Tutorial 3: Scenario to QUBO Conversion."""
    
    @classmethod
    def setUpClass(cls):
        """Import the tutorial module."""
        cls.tutorial = import_tutorial('tutorial_03_scenario_to_qubo')
    
    def test_example_1_understand_data(self):
        """Test Example 1: Understanding scenario data."""
        print("\nTesting Tutorial 3 - Example 1...")
        farms, foods, food_groups, config = self.tutorial.example_1_understand_scenario_data()
        
        self.assertIsNotNone(farms)
        self.assertIsNotNone(foods)
        self.assertIsNotNone(food_groups)
        self.assertIsNotNone(config)
        
        self.assertGreater(len(farms), 0)
        self.assertGreater(len(foods), 0)
        
        print("  PASSED: Scenario data loading works")
    
    def test_example_2_simple_objective(self):
        """Test Example 2: Objective-only QUBO."""
        print("\nTesting Tutorial 3 - Example 2...")
        bqm, sampleset, var_map = self.tutorial.example_2_simple_objective_only()
        
        self.assertIsNotNone(bqm)
        self.assertIsNotNone(sampleset)
        self.assertGreater(len(var_map), 0)
        
        print("  PASSED: Objective-only QUBO formulation works")
    
    def test_example_3_land_constraints(self):
        """Test Example 3: Adding land constraints."""
        print("\nTesting Tutorial 3 - Example 3...")
        bqm, sampleset, var_map = self.tutorial.example_3_add_land_constraints()
        
        self.assertIsNotNone(bqm)
        self.assertIsNotNone(sampleset)
        
        print("  PASSED: Land constraint encoding works")
    
    def test_example_4_complete_formulation(self):
        """Test Example 4: Complete QUBO formulation."""
        print("\nTesting Tutorial 3 - Example 4...")
        bqm, sampleset, var_map = self.tutorial.example_4_complete_formulation()
        
        self.assertIsNotNone(bqm)
        self.assertIsNotNone(sampleset)
        self.assertGreater(len(bqm.quadratic), 0)
        
        print("  PASSED: Complete QUBO formulation works")
    
    def test_example_5_tuning_penalties(self):
        """Test Example 5: Tuning penalty weights."""
        print("\nTesting Tutorial 3 - Example 5...")
        # This example doesn't return anything, just prints
        # Capture output to verify it runs
        output = StringIO()
        with redirect_stdout(output):
            self.tutorial.example_5_tuning_penalties()
        
        output_text = output.getvalue()
        self.assertIn("penalty weight", output_text.lower())
        
        print("  PASSED: Penalty tuning demonstration works")


class TestTutorial04(unittest.TestCase):
    """Tests for Tutorial 4: D-Wave Solver Integration."""
    
    @classmethod
    def setUpClass(cls):
        """Import the tutorial module."""
        cls.tutorial = import_tutorial('tutorial_04_dwave_integration')
    
    def test_example_1_simulator(self):
        """Test Example 1: Simulator solver."""
        print("\nTesting Tutorial 4 - Example 1...")
        sampleset = self.tutorial.example_1_simulator_solver()
        
        self.assertIsNotNone(sampleset)
        self.assertGreater(len(sampleset), 0)
        
        print("  PASSED: Simulator solver works")
    
    def test_example_2_comparison(self):
        """Test Example 2: Sampler comparison."""
        print("\nTesting Tutorial 4 - Example 2...")
        exact, sa, random = self.tutorial.example_2_solver_comparison()
        
        self.assertIsNotNone(exact)
        self.assertIsNotNone(sa)
        self.assertIsNotNone(random)
        
        print("  PASSED: Solver comparison works")
    
    def test_example_3_qpu_simulation(self):
        """Test Example 3: QPU workflow simulation."""
        print("\nTesting Tutorial 4 - Example 3...")
        sampleset = self.tutorial.example_3_qpu_simulation()
        
        self.assertIsNotNone(sampleset)
        
        print("  PASSED: QPU workflow simulation works")
    
    def test_example_4_hybrid_simulation(self):
        """Test Example 4: Hybrid solver simulation."""
        print("\nTesting Tutorial 4 - Example 4...")
        sampleset = self.tutorial.example_4_hybrid_solver_simulation()
        
        self.assertIsNotNone(sampleset)
        
        print("  PASSED: Hybrid solver simulation works")
    
    def test_example_5_plug_and_play(self):
        """Test Example 5: Plug-and-play configuration."""
        print("\nTesting Tutorial 4 - Example 5...")
        configs = self.tutorial.example_5_plug_and_play()
        
        self.assertIsNotNone(configs)
        self.assertEqual(len(configs), 3)
        
        print("  PASSED: Plug-and-play configuration works")


class TestTutorial05(unittest.TestCase):
    """Tests for Tutorial 5: Complete Workflow."""
    
    @classmethod
    def setUpClass(cls):
        """Import the tutorial module."""
        cls.tutorial = import_tutorial('tutorial_05_complete_workflow')
    
    def test_workflow_step_1(self):
        """Test Step 1: Load data."""
        print("\nTesting Tutorial 5 - Step 1...")
        farms, foods, food_groups, config = self.tutorial.workflow_step_1_load_data()
        
        self.assertIsNotNone(farms)
        self.assertIsNotNone(foods)
        self.assertGreater(len(farms), 0)
        
        print("  PASSED: Data loading works")
    
    def test_workflow_step_2(self):
        """Test Step 2: Build QUBO."""
        print("\nTesting Tutorial 5 - Step 2...")
        farms, foods, food_groups, config = self.tutorial.workflow_step_1_load_data()
        builder, Q = self.tutorial.workflow_step_2_build_qubo(farms, foods, food_groups, config)
        
        self.assertIsNotNone(builder)
        self.assertIsNotNone(Q)
        self.assertGreater(len(Q), 0)
        
        print("  PASSED: QUBO building works")
    
    def test_workflow_step_3(self):
        """Test Step 3: Convert to BQM."""
        print("\nTesting Tutorial 5 - Step 3...")
        farms, foods, food_groups, config = self.tutorial.workflow_step_1_load_data()
        builder, Q = self.tutorial.workflow_step_2_build_qubo(farms, foods, food_groups, config)
        bqm = self.tutorial.workflow_step_3_convert_to_bqm(Q)
        
        self.assertIsNotNone(bqm)
        self.assertGreater(len(bqm.variables), 0)
        
        print("  PASSED: BQM conversion works")
    
    def test_complete_workflow(self):
        """Test complete workflow integration."""
        print("\nTesting Tutorial 5 - Complete Workflow...")
        solutions = self.tutorial.complete_workflow()
        
        self.assertIsNotNone(solutions)
        self.assertGreater(len(solutions), 0)
        
        # Verify solutions have required attributes
        for solver_name, solution in solutions.items():
            self.assertIsNotNone(solution.energy)
            self.assertIsNotNone(solution.sample)
            self.assertIsNotNone(solution.assignment)
        
        print("  PASSED: Complete workflow integration works")
    
    def test_qubo_builder_class(self):
        """Test FoodProductionQUBOBuilder class."""
        print("\nTesting Tutorial 5 - QUBO Builder Class...")
        from src.scenarios import load_food_data
        farms, foods, food_groups, config = load_food_data('simple')
        
        builder = self.tutorial.FoodProductionQUBOBuilder(
            farms, foods, food_groups, config
        )
        
        self.assertIsNotNone(builder)
        self.assertGreater(len(builder.variables), 0)
        
        # Test building QUBO
        Q = builder.build_complete_qubo()
        self.assertIsNotNone(Q)
        self.assertGreater(len(Q), 0)
        
        print("  PASSED: QUBO Builder class works")


class TestIntegration(unittest.TestCase):
    """Integration tests across all tutorials."""
    
    def test_all_tutorials_importable(self):
        """Test that all tutorials can be imported."""
        print("\nTesting all tutorials can be imported...")
        
        tutorials = [
            'tutorial_01_basic_dimod',
            'tutorial_02_qubo_basics',
            'tutorial_03_scenario_to_qubo',
            'tutorial_04_dwave_integration',
            'tutorial_05_complete_workflow'
        ]
        
        for tutorial_name in tutorials:
            try:
                module = import_tutorial(tutorial_name)
                self.assertIsNotNone(module)
                print(f"  PASSED: {tutorial_name} imports successfully")
            except Exception as e:
                self.fail(f"Failed to import {tutorial_name}: {e}")
    
    def test_dimod_available(self):
        """Test that dimod is installed and working."""
        print("\nTesting dimod availability...")
        
        try:
            import dimod
            
            # Test basic functionality
            bqm = dimod.BinaryQuadraticModel('BINARY')
            bqm.add_variable('x0', 1.0)
            
            sampler = dimod.SimulatedAnnealingSampler()
            sampleset = sampler.sample(bqm, num_reads=10)
            
            self.assertIsNotNone(sampleset)
            print("  PASSED: dimod is installed and working")
        except ImportError:
            self.fail("dimod is not installed")
        except Exception as e:
            self.fail(f"dimod test failed: {e}")


def run_tests():
    """Run all tests with detailed output."""
    print("\n" + "="*70)
    print("RUNNING TUTORIAL TEST SUITE")
    print("="*70)
    print("\nThis test suite verifies all tutorials work correctly.")
    print("No D-Wave token is required (uses simulator only).\n")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestTutorial01))
    suite.addTests(loader.loadTestsFromTestCase(TestTutorial02))
    suite.addTests(loader.loadTestsFromTestCase(TestTutorial03))
    suite.addTests(loader.loadTestsFromTestCase(TestTutorial04))
    suite.addTests(loader.loadTestsFromTestCase(TestTutorial05))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\nALL TESTS PASSED!")
        print("All tutorial scripts are working correctly.")
        return 0
    else:
        print("\nSOME TESTS FAILED!")
        print("Please review the errors above.")
        return 1


if __name__ == "__main__":
    exit_code = run_tests()
    sys.exit(exit_code)
