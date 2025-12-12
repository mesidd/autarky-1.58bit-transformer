# Autarky 1.58-bit: Proving Intelligence is Compressible

> "We aren't losing detail; we are losing noise."

### The Theory: Harmonic Resonance
We often assume that `More Precision = Higher Intelligence`. However, biological and physical systems suggest that nature optimizes for **Resonance**, not Resolution.

In my simulation **"Harmonic Explorer"**, I demonstrated that low-precision systems (1:8 ratios) preserve semantic topology (structure), while high-precision systems (16:8) often drown that structure in noise.

### ⚙️ The Proof: Autarky Engine
To test this theory on Language, I built **Autarky**, a custom Transformer trained from scratch using **1.58-bit Quantization** (Ternary Weights: `{-1, 0, 1}`).

Instead of using high-precision floating point numbers (FP16/FP32), this model forces the neural network to make "hard" decisions. It cannot memorize noise; it must learn the invariant grammar of the dataset.

---

### 📊 The Results (Character-Level)
I trained the v1 engine on the "Tiny Shakespeare" dataset using a standard Apple M1 chip.

* **Architecture:** Custom 1.58-bit Transformer (4 Layers, 4 Heads)
* **Precision:** Ternary (Weights are strictly -1, 0, or 1 during forward pass)
* **Training Time:** ~3 hours (50,000 steps)
* **Final Loss:** **1.3776** (Converged)

The model successfully learned Shakespearean grammar, vocabulary, and syntax using **only three integer values** to represent its knowledge.

**Sample Output (Loss 1.37):**
> "By Lord what returns us...
> Weep of the won! Unto my fother.
> HENRY BOLINGBRVLLAND:"

---

### 📂 Repository Structure

| File | Description |
| :--- | :--- |
| `bitlinear.py` | The Core Invention. Custom PyTorch layer implementing **1.58-bit quantization** with STE. |
| `bit_transformer.py` | The Model Architecture. GPT-style Transformer utilizing BitLinear layers. |
| **`train_char.py`** | **v1 (Proof):** The character-level training script that achieved 1.37 loss. |
| **`train_bpe.py`** | **v2 (Upgrade):** Advanced training script using **Byte-Pair Encoding (BPE)**. |
| `tokenizer.py` | Custom BPE Tokenizer class (similar to GPT-4) for the v2 upgrade. |

---

### 🚀 Usage

#### 1. Install Dependencies
```bash
pip install torch requests regex

# For Version 1: Character-Level Transformer
python train_char.py

# For Version 2: BPE
python train_bpe.py
```

### 🧠 Why This Matters
If a model can reach **1.37 Loss** using only **1.58 bits** per weight, we are potentially wasting massive amounts of energy on high-precision Floating Point arithmetic. 
