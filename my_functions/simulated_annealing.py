import numpy as np
import random

def anneal(objective_function, initial_state, T0, alpha, max_iter, seed=None):
    """
    Classical Simulated Annealing (SA) for binary optimization.
    
    Parameters:
        objective_fn: function(state) -> energy (lower is better)
        initial_state: np.array of 0/1
        max_iter: number of iterations
        T0: initial temperature
        alpha: cooling rate (0 < alpha < 1)
        seed: RNG seed for reproducibility
    """

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    state=initial_state.copy()
    best_state = state.copy()
    best_energy = objective_function(state)
    current_energy = best_energy
    T = T0

    for iteration in range(max_iter):

        idx=random.randint(0, len(state) - 1)
        new_state = state.copy()
        new_state[idx] = 1 - new_state[idx]  # Flip bit

        new_energy = objective_function(new_state)
        deltaE = new_energy - current_energy

        if deltaE < 0 or random.uniform(0, 1) < np.exp(-deltaE / T):
            state = new_state
            current_energy = new_energy

            if new_energy < best_energy:
                best_state = new_state
                best_energy = new_energy

        T *= alpha  # Cool down

    return best_state, best_energy