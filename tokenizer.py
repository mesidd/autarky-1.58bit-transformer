"""
Minimal Byte Pair Encoding (BPE) Tokenizer
Reference: Based on minbpe by Andrej Karpathy
"""
import regex as re
import torch
import json
import os

# 1. THE PATTERN (How to split text before merging)
# This regex splits text into words/punctuation so we don't merge across boundaries.
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

class AutarkyTokenizer:
    def __init__(self):
        self.merges = {} # {(p0, p1): idx}
        self.vocab = {}  # {idx: bytes}
        self.special_tokens = {} # e.g. <|endoftext|>

    def train(self, text, vocab_size=512, verbose=True):
        """
        Train the tokenizer on text to reach target vocab_size.
        We start with 256 raw bytes, then merge common pairs.
        """
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        # Helper: Get stats of pair counts
        def get_stats(ids):
            counts = {}
            for pair in zip(ids, ids[1:]):
                counts[pair] = counts.get(pair, 0) + 1
            return counts

        # Helper: Merge a pair into a new ID
        def merge(ids, pair, idx):
            newids = []
            i = 0
            while i < len(ids):
                if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                    newids.append(idx)
                    i += 2
                else:
                    newids.append(ids[i])
                    i += 1
            return newids

        # 1. Pre-tokenize (convert to UTF-8 bytes)
        # In a real GPT-4 tokenizer, we split by regex first. 
        # For simplicity/speed here, we just use raw bytes of the whole text.
        train_ids = list(text.encode("utf-8"))

        print(f"Initial bytes length: {len(train_ids)}")

        # 2. Iteratively find best pair and merge
        for i in range(num_merges):
            stats = get_stats(train_ids)
            if not stats:
                break
            
            # Find most common pair
            pair = max(stats, key=stats.get)
            idx = 256 + i
            
            # Record the merge
            self.merges[pair] = idx
            
            # Apply merge to training data
            train_ids = merge(train_ids, pair, idx)
            
            if verbose and i % 10 == 0:
                print(f"Merge {i+1}/{num_merges}: {pair} -> {idx} (Count: {stats[pair]})")

        print(f"Training Complete. Compression ratio: {len(text.encode('utf-8')) / len(train_ids):.2f}X")
        self.build_vocab()

    def build_vocab(self):
        # Base vocab (0-255)
        vocab = {idx: bytes([idx]) for idx in range(256)}
        # Merged vocab
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        self.vocab = vocab

    def encode(self, text):
        # Simple encode: bytes -> merge
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            stats = {}
            for pair in zip(ids, ids[1:]):
                stats[pair] = stats.get(pair, 0) + 1
            
            # Find the pair that was merged earliest (lowest index)
            pair_to_merge = None
            min_idx = float('inf')
            
            for pair in stats:
                if pair in self.merges:
                    if self.merges[pair] < min_idx:
                        min_idx = self.merges[pair]
                        pair_to_merge = pair
            
            if pair_to_merge is None:
                break # No more merges possible
            
            # Apply just this merge
            # (Optimization: We could apply all valid merges, but order matters)
            newids = []
            i = 0
            while i < len(ids):
                if i < len(ids) - 1 and ids[i] == pair_to_merge[0] and ids[i+1] == pair_to_merge[1]:
                    newids.append(self.merges[pair_to_merge])
                    i += 2
                else:
                    newids.append(ids[i])
                    i += 1
            ids = newids
            
        return ids

    def decode(self, ids):
        tokens = b"".join(self.vocab[idx] for idx in ids)
        text = tokens.decode("utf-8", errors="replace")
        return text

    def save(self, prefix):
        # Save merges
        with open(prefix + ".model", 'w') as f:
            f.write("minbpe v1\n")
            f.write(f"{json.dumps(list(self.merges.keys()))}\n")
            f.write(f"{json.dumps(list(self.merges.values()))}\n")
    
    def load(self, prefix):
        # (Simplified load logic would go here, 
        # but we usually just train fresh for this size)
        pass

# --- RUN THE TRAINING ---
if __name__ == "__main__":
    with open('input.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    
    # We will aim for a vocabulary of 1024 tokens (Small but potent for Shakespeare)
    # Standard GPT-4 is 100k. 
    tokenizer = AutarkyTokenizer()
    tokenizer.train(text, vocab_size=1024)
    
    # Test it
    sample = "The King said: Hello World!"
    encoded = tokenizer.encode(sample)
    decoded = tokenizer.decode(encoded)
    
    print(f"\nTest String: '{sample}'")
    print(f"Encoded: {encoded}")
    print(f"Decoded: '{decoded}'")
    
    # Save the object to be imported by training script
    # For now, we'll just pickle it or let the training script run this.
    torch.save(tokenizer, "autarky_tokenizer.pkl")
    print("💾 Tokenizer saved to 'autarky_tokenizer.pkl'")
