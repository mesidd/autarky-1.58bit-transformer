import torch
import torch.nn as nn
import torch.nn.functional as F

class BitLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(BitLinear, self).__init__(in_features, out_features, bias)

    def forward(self, x):
        # 1. Get the weights
        w = self.weight
        
        # 2. Quantize weights to {-1, 0, 1}
        # We scale weights to be centered first
        scale = 1.0 / w.abs().mean().clamp(min=1e-5)
        w_scaled = w * scale
        w_quant = w_scaled.round().clamp(-1, 1)
        
        # 3. THE TRICK (Straight Through Estimator)
        # In forward, use w_quant. In backward, use w (gradients flow to w).
        # This line says: "Value is w_quant, but Gradient is derived from w"
        w_effective = (w_quant - w).detach() + w
        
        # 4. Run the Linear function using Quantized weights
        # Note: We technically multiply by 'scale' later to keep magnitude correct
        output = F.linear(x, w_effective)
        return output

# --- TEST IT ---
# Create a standard layer vs your BitLinear layer
layer = BitLinear(10, 5)
input_data = torch.randn(1, 10)
output = layer(input_data)

print("Input:", input_data)
print("Output from 1.58-bit Layer:", output)
print("Actual Used Weights (First 5):", layer.weight.data.flatten()[:5])
print("Quantized Weights:", layer.weight.data.round().clamp(-1, 1)[:10])
