# Task: Implement Linear-Quadratic Objective in Solver

This document provides instructions for a coding agent to modify the `solver_runner_NLQ.py` script. The goal is to replace the current non-linear objective function with a new objective that is a combination of a linear term (based on area) and a quadratic term that models a synergy bonus for planting similar crops together.

## Agent Workflow

To ensure a structured and safe implementation, please follow this workflow:

1.  **Understand the Goal:** Review this document carefully to understand the required changes to the objective function.
2.  **Create a Task List:** Before you begin, create a checklist of the specific steps you will take. This helps in tracking progress and ensuring all requirements are met.
3.  **Iterative Implementation:** Implement the changes one function at a time. Start by creating a copy of the file to work on.
4.  **Test (if possible):** After modifying the script, if you have the capability, run it on a simple scenario to ensure it executes without errors.
5.  **Finalize:** Once all changes are implemented and verified, clean up the code and ensure the documentation (docstrings, comments) is updated to reflect the new functionality.

---

## Implementation Prompts

### Prompt 1: Setup and File Preparation

**Goal:** Create a new working file and add the necessary configuration parameter.

**Prompt:**

"First, create a copy of `solver_runner_NLQ.py` and name it `solver_runner_LQ.py`. All subsequent modifications will be made to this new file.

Next, you will need to introduce a new parameter for the synergy bonus. In the `main` function, locate the `config` dictionary and add a new key `synergy_boost` to the `parameters` dictionary. A default value of `0.1` is a good starting point. This will be used to control the strength of the quadratic bonus."

### Prompt 2: Modify the `create_cqm` Function

**Goal:** Replace the non-linear objective in the CQM with the new linear-quadratic objective.

**Prompt:**

"In `solver_runner_LQ.py`, locate the `create_cqm` function. You need to perform the following changes:

1.  **Remove Non-Linear Logic:** Delete all code related to the piecewise approximation. This includes the `power` and `num_breakpoints` arguments, the `PiecewiseApproximation` class, the `Lambda` variables, `f_approx`, and any constraints associated with them (`Piecewise_A_Definition`, `Piecewise_Convexity`). Simplify the function signature accordingly.

2.  **Implement Linear Objective:** Re-implement the linear part of the objective function, which is directly proportional to the allocated area `A`. This should be identical to the objective in `solver_runner.py`. The code is:
    ```python
    objective = 0
    for farm in farms:
        for food in foods:
            objective += (
                weights.get('nutritional_value', 0) * foods[food].get('nutritional_value', 0) * A[(farm, food)] +
                weights.get('nutrient_density', 0) * foods[food].get('nutrient_density', 0) * A[(farm, food)] -
                weights.get('environmental_impact', 0) * foods[food].get('environmental_impact', 0) * A[(farm, food)] +
                weights.get('affordability', 0) * foods[food].get('affordability', 0) * A[(farm, food)] +
                weights.get('sustainability', 0) * foods[food].get('sustainability', 0) * A[(farm, food)]
            )
    ```

3.  **Implement Quadratic Synergy Bonus:** Add a quadratic term to the `objective`. This term should reward planting pairs of different but similar crops on the *same farm*. Use the `food_groups` dictionary to determine similarity. The bonus for each pair should be `synergy_boost * Y[(farm, crop1)] * Y[(farm, crop2)]`.

    The logic should be:
    ```python
    synergy_boost = params.get('synergy_boost', 0.1)
    for farm in farms:
        for group in food_groups.values():
            # Create pairs of distinct crops in the same group
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    crop1 = group[i]
                    crop2 = group[j]
                    if crop1 in foods and crop2 in foods:
                        objective += synergy_boost * Y[(farm, crop1)] * Y[(farm, crop2)]
    ```

4.  **Set Final Objective:** Combine the linear and quadratic terms and set it as the CQM objective. Remember to negate it for maximization: `cqm.set_objective(-objective)`."

### Prompt 3: Modify `solve_with_pulp` and `solve_with_pyomo`

**Goal:** Update the classical solver functions to use the new linear-quadratic objective.

**Prompt:**

"Now, apply the same objective function logic to the `solve_with_pulp` and `solve_with_pyomo` functions.

1.  **For `solve_with_pulp`:**
    *   Remove all piecewise approximation logic (`Lambda_pulp`, etc.).
    *   Construct the objective `goal` by first calculating the linear sum based on `A_pulp`, and then adding the quadratic synergy bonus using `Y_pulp`. The structure will be very similar to the changes made in `create_cqm`.

2.  **For `solve_with_pyomo`:**
    *   In the `objective_rule`, replace the `m.A[f, c] ** power` term with the linear term `m.A[f, c]`.
    *   Add the quadratic synergy bonus to the `obj` expression by summing `synergy_boost * m.Y[f, crop1] * m.Y[f, crop2]` over the relevant pairs."

### Prompt 4: Modify `solve_with_pulp` and `solve_with_pyomo`

**Goal:** Update the classical solver functions to use the new linear-quadratic objective.

**Prompt:**

"Now, apply the same objective function logic to the `solve_with_pulp` and `solve_with_pyomo` functions.

1.  **For `solve_with_pulp`:**
    *   Remove all piecewise approximation logic (`Lambda_pulp`, etc.).
    *   Construct the objective `goal` by first calculating the linear sum based on `A_pulp`.
    *   Then, read the `synergy_bonus_weight` and `synergy_matrix` from the config.
    *   Add the quadratic synergy bonus to the `goal` by iterating through the matrix, similar to the `create_cqm` modification.

2.  **For `solve_with_pyomo`:**
    *   In the `objective_rule`, replace the `m.A[f, c] ** power` term with the linear term `m.A[f, c]`.
    *   Add the quadratic synergy bonus to the `obj` expression by iterating through the `synergy_matrix` and adding `synergy_bonus_weight * boost_value * m.Y[f, crop1] * m.Y[f, crop2]` for each farm `f` and crop pair."

### Prompt 5: Update Scenarios for Synergy Bonus

**Goal:** Modify the `src/scenarios.py` file to generate the synergy matrix and add the corresponding weight to the configuration.

**Prompt:**

"The new quadratic term requires data from the scenario configuration. You need to update the functions in `src/scenarios.py` (e.g., `_load_simple_food_data`, `_load_intermediate_food_data`, etc.).

1.  **Add Synergy Weight:** In the `weights` dictionary within the `parameters`, add a new weight for the synergy bonus.
    ```python
    'weights': {
        'nutritional_value': 0.25,
        'nutrient_density': 0.2,
        'environmental_impact': 0.25,
        'affordability': 0.15,
        'sustainability': 0.15,
        'synergy_bonus': 0.1 # New weight
    },
    ```

2.  **Generate and Add Synergy Matrix:** After defining `foods` and `food_groups`, generate a `synergy_matrix`. This matrix will define the boost factor for each pair of crops. The matrix should be sparse, with non-zero values only for distinct crops within the same food group. Add the following logic to the scenario function:

    ```python
    # --- Begin new section for synergy matrix ---
    synergy_matrix = {}
    default_boost = 0.1 # A default boost value for pairs in the same group

    for group_name, crops_in_group in food_groups.items():
        for i in range(len(crops_in_group)):
            for j in range(i + 1, len(crops_in_group)):
                crop1 = crops_in_group[i]
                crop2 = crops_in_group[j]

                if crop1 not in synergy_matrix:
                    synergy_matrix[crop1] = {}
                if crop2 not in synergy_matrix:
                    synergy_matrix[crop2] = {}

                # Add symmetric entries for the pair
                synergy_matrix[crop1][crop2] = default_boost
                synergy_matrix[crop2][crop1] = default_boost
    
    # Add the matrix to the parameters
    parameters['synergy_matrix'] = synergy_matrix
    # --- End new section ---
    ```

3.  **Apply to all Scenarios:** Ensure this logic is added to all relevant scenario-loading functions (`_load_simple_food_data`, `_load_intermediate_food_data`, `_load_custom_food_data`, etc.) so that the `synergy_matrix` and `synergy_bonus` weight are always available in the `config`."

### Prompt 6: Final Cleanup

**Goal:** Finalize the script by removing obsolete code and updating documentation.

**Prompt:**

"To complete the task, please clean up the `solver_runner_LQ.py` file:

1.  **Update Docstrings:** Change the main docstring of the file to reflect that it now implements a 'Linear-Quadratic' objective, driven by a synergy matrix.
2.  **Remove Unused Imports:** Delete any imports that are no longer necessary (e.g., `PiecewiseApproximation`, `numpy` if unused).
3.  **Adjust `main` function:** Remove the `--power` and `--breakpoints` command-line arguments from the `argparse` setup, as they are no longer used.
4.  **Rename Functions (Optional but Recommended):** Consider renaming functions like `solve_with_pulp_nln` to `solve_with_pulp_lq` to reflect the new model type."
