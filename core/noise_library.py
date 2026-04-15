import numpy as np

class QuantumNoiseLibrary:
    """
    Advanced noise generators for realistic QPU simulation.
    """
    @staticmethod
    def get_amplitude_damping_kraus(gamma):
        """Returns Kraus operators for amplitude damping (T1)."""
        e0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]])
        e1 = np.array([[0, np.sqrt(gamma)], [0, 0]])
        return [e0, e1]

    @staticmethod
    def get_phase_damping_kraus(lambda_):
        """Returns Kraus operators for phase damping (T2)."""
        e0 = np.array([[1, 0], [0, np.sqrt(1 - lambda_)]])
        e1 = np.array([[0, 0], [0, np.sqrt(lambda_)]])
        return [e0, e1]

    @staticmethod
    def apply_channel(rho, operators):
        """Apply a quantum channel defined by Kraus operators."""
        new_rho = np.zeros_like(rho)
        for op in operators:
            new_rho += op @ rho @ op.conj().T
        return new_rho
