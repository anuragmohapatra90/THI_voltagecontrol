import torch
import torch.nn as nn

class VoltageEstimator(nn.Module):
    """
    A Simple Multi-Layer Perceptron (MLP) to act as a Surrogate Model.
    
    Mapping:
    [P_0, ..., P_14, Q_0, ..., Q_14] -> [V_0, ..., V_14]
    """
    def __init__(self, input_dim=30, output_dim=15):
        super(VoltageEstimator, self).__init__()
        
        self.model = nn.Sequential(
            # Input Layer -> Hidden Layer 1
            nn.Linear(input_dim, 64),
            nn.ReLU(),  # Activation function (adds non-linearity)
            
            # Hidden Layer 1 -> Hidden Layer 2
            nn.Linear(64, 32),
            nn.ReLU(),
            
            # Hidden Layer 2 -> Output Layer
            nn.Linear(32, output_dim)
            # No activation at the end (we want raw voltage values, e.g., 0.98)
        )
    
    def forward(self, x):
        return self.model(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)