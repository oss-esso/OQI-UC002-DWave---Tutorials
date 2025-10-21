import numpy as np
import random

def anneal(objective_function, initial_solution, T0, alpha, max_iterations, num_replicas, gamma0, beta, seed=None):
    """
    Simulated Quantum Annealing (SQA) for binary optimization.
    
    Parameters:
        objective_function: function(solution) -> energy (lower is better)
        initial_solution: np.array of 0/1
        max_iterations: number of iterations
        T0: initial temperature
        alpha: cooling rate (0 < alpha < 1)
        num_replicas: number of replicas for quantum fluctuations
        gamma0: initial quantum fluctuation strength
        beta: inverse temperature scaling factor
        seed: RNG seed for reproducibility
    """

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    n = len(initial_solution)
    replicas = [initial_solution.copy() for _ in range(num_replicas)]
    best_solution = initial_solution.copy()
    best_energy = objective_function(initial_solution)
    T = T0
    gamma = gamma0

    for iteration in range(max_iterations):
        for replica in range(num_replicas):
            idx = random.randint(0, n - 1)
            new_solution = replicas[replica].copy()
            new_solution[idx] = 1 - new_solution[idx]  # Flip bit

            E_classical = objective_function(new_solution)
            E_quantum = 0
            for other_replica in range(num_replicas):
                if other_replica != replica:
                    E_quantum += gamma * (np.sum(new_solution != replicas[other_replica]))
            
            total_energy = E_classical + E_quantum
            deltaE = total_energy - (objective_function(replicas[replica]) + 
                                    sum(gamma * np.sum(replicas[replica] != replicas[other_replica]) 
                                        for other_replica in range(num_replicas) if other_replica != replica))
            
            if deltaE < 0 or random.random() < np.exp(-deltaE * beta):
                replicas[replica] = new_solution

            if E_classical < best_energy:
                best_solution = new_solution
                best_energy = E_classical

        T *= alpha  # Cool down
        gamma *= alpha  # Reduce quantum fluctuations

    return best_solution, best_energy