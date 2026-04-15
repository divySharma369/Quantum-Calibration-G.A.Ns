import torch
import torch.nn as nn

class PulseGenerator(nn.Module):
    """
    The Generator: Maps target gate parameters to time-domain control pulses.
    Input: [Target Angle, Phase, Noise_Vector]
    Output: [Amplitude_Sequence, Phase_Sequence]
    """
    def __init__(self, input_dim=64, sequence_length=128):
        super(PulseGenerator, self).__init__()
        self.seq_len = sequence_length
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(128),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            nn.Linear(256, sequence_length * 2), # Amp + Phase
            nn.Tanh() # Normalized pulses between -1 and 1
        )

    def forward(self, z):
        pulses = self.network(z)
        pulses = pulses.view(-1, 2, self.seq_len)
        return pulses

class FidelityDiscriminator(nn.Module):
    """
    The Discriminator/Critic: Evaluates if a pulse sequence results in 
    a high-fidelity "real" gate vs a "fake" noisy/uncalibrated gate.
    """
    def __init__(self, sequence_length=128):
        super(FidelityDiscriminator, self).__init__()
        
        self.conv_blocks = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=7, stride=2, padding=3),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(128 * (sequence_length // 4), 1)
        )

    def forward(self, pulse_seq):
        validity = self.conv_blocks(pulse_seq)
        return validity
