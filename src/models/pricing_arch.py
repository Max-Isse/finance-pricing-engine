import torch
import torch.nn as nn

class MacroCalibratedPricingNet(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=64, output_dim=1):
        super(MacroCalibratedPricingNet, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        # Gamma adjustment layer scaling outputs based on macro yield input index
        self.gamma_layer = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x, macro_yields):
        out = self.layer1(x)
        out = self.relu(out)
        out = self.layer2(out)
        base_price = self.relu(out)
        
        # Inject intraday macro adjustments dynamically
        adjustment = self.gamma_layer(base_price) * macro_yields
        final_price = base_price + adjustment
        return final_price