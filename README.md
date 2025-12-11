# Autarky 1.58-bit: Proving Intelligence is Compressible

> "We aren't losing detail; we are losing noise."

### The Theory: Harmonic Resonance
We often assume that `More Precision = Higher Intelligence`.
However, biological and physical systems suggest that nature optimizes for **Resonance**, not Resolution.

In my simulation **"Harmonic Explorer"**, I demonstrated that low-precision systems (1:8 ratios) preserve semantic topology (structure), while high-precision systems (16:8) often drown that structure in noise.

### ⚙️ The Proof: Autarky Engine
To test this theory on Language, I built **Autarky**, a custom Transformer trained from scratch using **1.58-bit Quantization** (Ternary Weights: `{-1, 0, 1}`).

Instead of using high-precision floating point numbers (FP16/FP32), this model forces the neural network to make "hard" decisions. It cannot memorize noise; it must learn the invariant grammar of the dataset.

### 📊 The Results
I trained this model on the "Tiny Shakespeare" dataset.
* **Architecture:** Custom 1.58-bit Transformer (4 Layers, 4 Heads)
* **Precision:** Ternary (Weights are strictly -1, 0, or 1 during forward pass)
* **Final Loss:** **1.3776** (Converged)

The model successfully learned Shakespearean grammar, vocabulary, and syntax using **only three integer values** to represent its knowledge.

**Sample Output (Loss 1.37):**
> "By Lord what returns us...
> Weep of the won! Unto my fother.
> HENRY BOLINGBRVLLAND:"

### 📂 File Structure
* `bitlinear.py`: The custom PyTorch layer implementing **1.58-bit quantization** with the Straight-Through Estimator (STE).
* `bit_transformer.py`: The GPT-style Transformer architecture utilizing BitLinear layers.
* `train_autarky.py`: The training script that reproduces the 1.37 loss result.

### Usage
 **Install Dependencies and Run**
   ```bash
   pip install torch requests

   python train_autarky.py
```
