import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import requests

# --- 1. IMPORT YOUR ENGINE ---
from bit_transformer import AutarkyGPT

# --- 2. PREPARE THE DATA (The Knowledge) ---
print(" Downloading 'Tiny Shakespeare' dataset...")
file_path = 'input.txt'
if not os.path.exists(file_path):
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with open(file_path, 'w') as f:
        f.write(requests.get(data_url).text)

with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()

print(f"✅ Data Loaded. Length: {len(text)} characters.")

# --- 3. THE TOKENIZER (The Translator) ---
# We use Character-Level tokenization (Simple and Fast)
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"🔤 Vocabulary Size: {vocab_size} unique characters.")

stoi = { ch:i for i,ch in enumerate(chars) } # String to Integer
itos = { i:ch for i,ch in enumerate(chars) } # Integer to String
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Move data to Tensor
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# --- 4. HYPERPARAMETERS (The Settings) ---
batch_size = 32      # How many sequences to learn at once
block_size = 128     # Maximum context length (Time)
max_iters = 500     # How long to train (Short run for testing)
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_interval = 100

# Model Config (Small for Laptop CPU)
d_model = 128
n_head = 4
n_layer = 4

print(f" Running on: {device.upper()}")

# --- 5. INITIALIZE THE 1.58-BIT MODEL ---
model = AutarkyGPT(vocab_size, d_model, n_head, n_layer, max_len=block_size)
model = model.to(device)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

print(f"🧠 Model Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f} M")

# Helper function to get a batch of data
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

# --- 6. THE TRAINING LOOP (The Meditation) ---
print("\n Training Started...")

for iter in range(max_iters):
    # Sample a batch of data
    xb, yb = get_batch('train')

    # Evaluate the loss
    logits = model(xb)
    B, T, C = logits.shape
    logits = logits.view(B*T, C)
    targets = yb.view(B*T)
    loss = F.cross_entropy(logits, targets)

    # Optimization (The Learning)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    # Reporting
    if iter % 50 == 0:
        print(f"Step {iter}: Loss = {loss.item():.4f}")

print(f" Final Loss: {loss.item():.4f}")

# --- 7. THE GENERATION (The Speaking) ---
print("\n AUTARKY SPEAKS:")
context = torch.zeros((1, 1), dtype=torch.long, device=device) # Start with '0' token
generated = []

model.eval()
with torch.no_grad():
    for _ in range(300): # Generate 300 characters
        # Get predictions
        logits = model(context[:, -block_size:]) 
        logits = logits[:, -1, :] # Focus on last time step
        probs = F.softmax(logits, dim=-1) # Get probabilities
        
        # Sample from the distribution
        idx_next = torch.multinomial(probs, num_samples=1)
        
        # Append to sequence
        context = torch.cat((context, idx_next), dim=1)
        generated.append(idx_next.item())

print(decode(generated))
print("\n--- END OF TRANSMISSION ---")

# --- SAVING THE ENGINE ---
print(" Saving Autarky-1.58bit Model...")
torch.save(model.state_dict(), 'autarky_158.pth')
print("✅ Saved to 'autarky_158.pth'")
