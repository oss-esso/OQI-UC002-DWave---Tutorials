import numpy as np
import pennylane as qml
from math import pi
from collections import Counter


n_qubits = 4
N=2**n_qubits
L=5.0
dx = 2*L/N

dt = 0.1
n_steps = 100

m_i=1.0
m_f=10.0
sample_every = 20

shots=512


#Example Goal

def styblinski_tang(x):
    return 0.5 - (16*x**2 - 5)**2 + 0.2*(x + 5)**2

xs=np.linspace(-L,L-dx,N)

V_vals = np.array([styblinski_tang(x) for x in xs])

#normalize and scale
V_vals = V_vals - np.min(V_vals)
V_vals = V_vals / np.max(V_vals) * 5.0

j, k = np.meshgrid(np.arange(N), np.arange(N))
F = (1/np.sqrt(N)) * np.exp(-2j * pi * j * k / N)

k_vec = np.fft.fftfreq(N, d=dx)*2*pi
p_vals = k_vec

def kinetic_diag(m):
    return 0.5 * (p_vals**2) / m


def potential_unitary(dt, Vdiag):
    return np.diag(np.exp(-1j * Vdiag * dt))


def kinetic_unitary(dt, m):
    Kdiag = kinetic_diag(m)
    return np.diag(np.exp(-1j * Kdiag * dt))


dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def apply_unitary_and_measure(U):
    qml.StatePrep(np.ones(N)/np.sqrt(N), wires=range(n_qubits))
    qml.QubitUnitary(U, wires=range(n_qubits))
    return qml.probs(wires=range(n_qubits))


@qml.qnode(dev)
def apply_unitary_to_state(U, input_state):
    qml.StatePrep(input_state, wires=range(n_qubits))
    qml.QubitUnitary(U, wires=range(n_qubits))
    return qml.state()


def gaussian_grid(x0, sigma):
    psi = np.exp(-0.5 * ((xs - x0) / sigma) ** 2)
    psi /= np.linalg.norm(psi)
    return psi.astype(np.complex128)

psi0=gaussian_grid(x0=1.5, sigma=0.5)
state = psi0.copy()

best_val = np.inf
best_x = None


for step in range(n_steps):
    m_t = m_i + (m_f - m_i) * step / n_steps
    U_pot = potential_unitary(dt/2, V_vals)
    U_kin = kinetic_unitary(dt, m_t)

    # Apply potential unitary
    state = apply_unitary_to_state(U_pot, state)
    
    # Apply Fourier transform
    state = apply_unitary_to_state(F, state)
    
    # Apply kinetic unitary
    state = apply_unitary_to_state(U_kin, state)
    
    # Apply inverse Fourier transform
    state = apply_unitary_to_state(np.conj(F.T), state)
    
    # Apply potential unitary again
    state = apply_unitary_to_state(U_pot, state)
    
    if step % sample_every == 0:
        probs = np.abs(state)**2
        x_expectation = np.sum(xs * probs)
        val = styblinski_tang(x_expectation)
        
        if val < best_val:
            best_val = val
            best_x = x_expectation


print(f"Best found value: {best_val} at x = {best_x}")






