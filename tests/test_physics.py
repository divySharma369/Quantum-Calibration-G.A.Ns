import unittest
import numpy as np
from core.quantum_engine import QuantumSimulator

class TestQuantumPhysics(unittest.TestCase):
    def setUp(self):
        self.sim = QuantumSimulator(n_qubits=1)

    def test_identity_evolution(self):
        """Test that evolving with zero Hamiltonian preserves the state."""
        initial_state = np.array([[1, 0], [0, 0]], dtype=complex)
        h = np.zeros((2, 2), dtype=complex)
        final_state = self.sim.evolve_state(initial_state, h, dt=1.0)
        np.testing.assert_array_almost_equal(initial_state, final_state)

    def test_fidelity_range(self):
        """Fidelity should always be between 0 and 1."""
        state_a = np.array([[1, 0], [0, 0]], dtype=complex)
        state_b = np.array([[0, 0], [0, 1]], dtype=complex)
        f = self.sim.calculate_fidelity(state_a, state_b)
        self.assertTrue(0 <= f <= 1)

if __name__ == '__main__':
    unittest.main()
