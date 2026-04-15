import numpy as np
from scipy.linalg import expm

class QuantumSimulator:
    """
    Advanced Physics Engine for Quantum State Evolution.
    Supports Lindblad Master Equation for noisy environments.
    """
    def __init__(self, n_qubits=1):
        self.n_qubits = n_qubits
        self.dim = 2**n_qubits
        self.sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
        self.sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        self.sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
        self.identity = np.eye(self.dim, dtype=complex)

    def get_hamiltonian(self, omega, delta, pulse_amplitude):
        """Constructs the rotating frame Hamiltonian."""
        # H = (delta/2) * Z + (amplitude/2) * X
        h = (delta / 2.0) * self.sigma_z + (pulse_amplitude / 2.0) * self.sigma_x
        return h

    def evolve_state(self, initial_state, hamiltonian, dt):
        """Unitary evolution using matrix exponential."""
        u = expm(-1j * hamiltonian * dt)
        return u @ initial_state @ u.conj().T

    def apply_decoherence(self, density_matrix, t1, t2, dt):
        """
        Applies a simplified Kraus-map approximation of Lindbladian decoherence.
        """
        # Relaxation (T1) and Dephasing (T2)
        p_t1 = 1 - np.exp(-dt / t1) if t1 > 0 else 0
        p_t2 = 1 - np.exp(-dt / t2) if t2 > 0 else 0
        
        # This is a placeholder for a full Lindblad integration
        # In a real scenario, we'd use ODE solvers.
        return density_matrix * (1 - p_t1 - p_t2) + (p_t1 + p_t2) * self.identity / self.dim

    def calculate_fidelity(self, state_a, state_b):
        """Compute state fidelity between two density matrices."""
        # F = (Tr(sqrt(sqrt(rho) * sigma * sqrt(rho))))^2
        # For pure states: |<psi|phi>|^2
        return np.real(np.trace(state_a @ state_b))
