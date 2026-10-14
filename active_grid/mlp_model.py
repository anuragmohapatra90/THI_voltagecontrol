"""Architecture from ADG_datadriven/active_grid/mlp_model.py."""
import torch.nn as nn


class VoltageEstimator(nn.Module):
    """[P at every bus, Q at every bus] -> voltage at every bus."""
    def __init__(self,input_dim=30,output_dim=15):
        super().__init__()
        self.model=nn.Sequential(
            nn.Linear(input_dim,64),nn.ReLU(),
            nn.Linear(64,32),nn.ReLU(),
            nn.Linear(32,output_dim))

    def forward(self,x):return self.model(x)

    def count_parameters(self):return sum(p.numel() for p in self.parameters() if p.requires_grad)
