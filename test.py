import torch
from torch.utils.cpp_extension import load

# Compile the CUDA extension
flash_attn = load(
    name="flash_attn_lib",
    sources=["flash_attn_v2.cu"],
    verbose=True
)

# Configuration
N, d = 512, 64
Br, Bc = 32, 32 # Br * d * 4 + 2 * Bc * d * 4 + Br * Bc * 4 approx 28KB < 48KB

Q = torch.randn(N, d, device="cuda")
K = torch.randn(N, d, device="cuda")
V = torch.randn(N, d, device="cuda")

# Custom Implementation
output_custom = flash_attn.forward(Q, K, V, Br, Bc)

# Reference Implementation (PyTorch FlashAttention)
output_ref = torch.nn.functional.scaled_dot_product_attention(
    Q.unsqueeze(0), K.unsqueeze(0), V.unsqueeze(0)
).squeeze(0)

# Accuracy Check
max_diff = torch.abs(output_custom - output_ref).max().item()
print(f"Verification Results:")
print(f"Max Absolute Error: {max_diff:.6e}")
if max_diff < 1e-4:
    print("STATUS: SUCCESS")
else:
    print("STATUS: FAILED - check scaling or statistics update")
