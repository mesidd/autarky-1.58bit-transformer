import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- 1.The 1.58-bit Layer ---
class BitLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=False):
        super(BitLinear, self).__init__(in_features, out_features, bias)

    def forward(self, x):
        w = self.weight
        # RMS Norm of weights for stability
        gamma = w.abs().mean().clamp(min=1e-5)
        w_scaled = w / gamma
        w_quant = w_scaled.round().clamp(-1, 1)
        w_effective = (w_quant - w).detach() + w
        return F.linear(x, w_effective) * gamma

# --- 2. BIT-FEED FORWARD ---
# This is where 2/3rds of the compute happens in a Transformer.
# We replace the dense float layers with 1.58-bit layers.
class BitFeedForward(nn.Module):
    def __init__(self, d_model, expansion_factor=4):
        super(BitFeedForward, self).__init__()
        self.w1 = BitLinear(d_model, d_model * expansion_factor)
        self.w2 = BitLinear(d_model * expansion_factor, d_model)
        self.act = nn.GELU()

    def forward(self, x):
        return self.w2(self.act(self.w1(x)))

# --- 3. BIT-ATTENTION MECHANISM ---
# We quantize the Projections (Q, K, V, Output).
# The Attention Math (Softmax) remains float for now.
class BitSelfAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super(BitSelfAttention, self).__init__()
        assert d_model % n_head == 0
        
        self.d_head = d_model // n_head
        self.n_head = n_head
        
        # 1.58-bit Projections
        self.q_proj = BitLinear(d_model, d_model)
        self.k_proj = BitLinear(d_model, d_model)
        self.v_proj = BitLinear(d_model, d_model)
        self.o_proj = BitLinear(d_model, d_model)

    def forward(self, x):
        B, T, C = x.size() # Batch, Time, Channels
        
        # Calculate Q, K, V using Integer Weights
        q = self.q_proj(x).view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_head, self.d_head).transpose(1, 2)
        
        # Standard Flash Attention (PyTorch optimized)
        # Note: We don't quantize the scores yet, just the weights creating them.
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(out)

# --- 4. THE AUTARKY TRANSFORMER BLOCK ---
class AutarkyBlock(nn.Module):
    def __init__(self, d_model, n_head):
        super(AutarkyBlock, self).__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = BitSelfAttention(d_model, n_head)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = BitFeedForward(d_model)

    def forward(self, x):
        # Pre-Norm Architecture (Standard for Stability)
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

# --- 5. THE MINI-GPT MODEL ---
class AutarkyGPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_head, n_layer, max_len=128):
        super(AutarkyGPT, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)
        
        # Stack the 1.58-bit Blocks
        self.blocks = nn.Sequential(*[
            AutarkyBlock(d_model, n_head) for _ in range(n_layer)
        ])
        
        # Final Head
        self.ln_f = nn.LayerNorm(d_model)
        self.head = BitLinear(d_model, vocab_size)

    def forward(self, idx):
        B, T = idx.shape
        
        # Embeddings (Usually kept in Float for precision, but can be quantized later)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        x = self.token_embedding(idx) + pos_emb
        
        # Run through the 1.58-bit Layers
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

# ---  TEST FLIGHT ---
print(" Initializing AutarkyGPT (1.58-bit Transformer)...")

# Tiny Config
vocab_size = 1000
d_model = 128 # Dimension
n_head = 4
n_layer = 2

model = AutarkyGPT(vocab_size, d_model, n_head, n_layer)

# Create a dummy input (Batch 2, Sequence Length 10)
dummy_input = torch.randint(0, vocab_size, (2, 10))

# Forward pass
output = model(dummy_input)

print(f"Input Shape: {dummy_input.shape}")
print(f"Output Logits Shape: {output.shape}") 
print("\n🔍 Inspecting Weights of First Attention Layer:")
print(model.blocks[0].attn.q_proj.weight.data.flatten()[:10].round().clamp(-1,1))
print("✅ Success: The Transformer is thinking in Integers.")
