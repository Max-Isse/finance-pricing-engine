import torch.nn as nn


class MacroCalibratedPricingNet(nn.Module):
    """Pricing network with a macro adjustment.

    price = base_head(hidden) + gamma_layer(hidden) * macro_yields

    `base_head` gives one base price per row. `gamma_layer` gives the row's
    sensitivity to the macro input (the scaled 10-year yield), so the adjustment
    can be recalibrated on its own when the macro regime moves.
    """

    def __init__(self, input_dim=10, hidden_dim=64, output_dim=1):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.base_head = nn.Linear(hidden_dim, output_dim)
        self.gamma_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, macro_yields):
        hidden = self.relu(self.layer1(x))
        hidden = self.relu(self.layer2(hidden))
        base_price = self.base_head(hidden)
        adjustment = self.gamma_layer(hidden) * macro_yields.reshape(-1, 1)
        return base_price + adjustment
