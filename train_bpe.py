import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import requests

# IMPORT YOUR MODULES
from bit_transformer import AutarkyGPT
from tokenizer import AutarkyTokenizer 

# --- 1. SETUP & DATA ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cpu' and torch.backends.mps.is_available():
    device = 'mps' # Use Apple M1 Metal Performance Shaders if available

print(f"⚙️  Running on: {device.upper()}")

# Check for data
if not os.path.exists('input.txt'):
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with open('input.txt', 'w') as f:
        f.write(requests.get(data_url).text)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# --- 2. TRAIN TOKENIZER ---
# We train it fresh every time for simplicity in this demo
print("📚 Training BPE Tokenizer (Target: 1024 tokens)...")
tokenizer = AutarkyTokenizer()
tokenizer.train(text, vocab_size=1024, verbose=False)
vocab_size = 1024
print("✅ Tokenizer Ready.")

# Encode Data
data_ids = tokenizer.encode(text)
data = torch.tensor(data_ids, dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

print(f"📊 Dataset: {len(text)} chars compressed to {len(data)} tokens.")

# --- 3. HYPERPARAMETERS ---
batch_size = 32
block_size = 128
max_iters = 1000  # Increased slightly as BPE steps cover more ground
learning_rate = 3e-4

# Model Config
d_model = 128
n_head = 4
n_layer = 4

# --- 4. INITIALIZE MODEL ---
model = AutarkyGPT(vocab_size, d_model, n_head, n_layer, max_len=block_size)
model = model.to(device)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

print(f"🧠 Model Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f} M")

# Helper function
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

# --- 5. TRAINING ---
print("\n🔥 Training Started (BPE Version)...")

for iter in range(max_iters):
    xb, yb = get_batch('train')
    
    logits = model(xb)
    B, T, C = logits.shape
    logits = logits.view(B*T, C)
    targets = yb.view(B*T)
    loss = F.cross_entropy(logits, targets)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if iter % 100 == 0:
        print(f"Step {iter}: Loss = {loss.item():.4f}")

print(f"🏁 Final Loss: {loss.item():.4f}")

# --- 6. GENERATION ---
print("\n🗣️  AUTARKY SPEAKS (BPE Mode):")
# Start with a simple prompt, e.g., newline
start_ids = tokenizer.encode("\n")
context = torch.tensor([start_ids], dtype=torch.long, device=device)

model.eval()
with torch.no_grad():
    for _ in range(100): # Generate 100 TOKENS (approx 300-400 chars)
        logits = model(context[:, -block_size:])
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        context = torch.cat((context, idx_next), dim=1)

output_ids = context[0].tolist()
print(tokenizer.decode(output_ids))

print("💾 Saving Autarky-BPE Model...")
torch.save(model.state_dict(), 'autarky_bpe.pth')
print("✅ Saved to 'autarky_bpe.pth'")

print("\n--- END ---")
