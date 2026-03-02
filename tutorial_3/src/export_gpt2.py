#!/usr/bin/env python3
"""Export GPT-2 weights + BPE vocab to binary files for gpt2.cpp.
   pip install torch transformers
   python export_gpt2.py                       # 124M
   python export_gpt2.py --model gpt2-medium   # 345M
"""
import argparse, struct
import numpy as np
from transformers import GPT2Model, GPT2Tokenizer

def write_tensor(f, arr, name=""):
    arr = arr.astype("float32")
    f.write(struct.pack('<I', arr.ndim))
    for d in arr.shape: f.write(struct.pack('<I', int(d)))
    f.write(arr.tobytes())
    print(f"  {name}  {arr.shape}")

def export_weights(model_name, out_path):
    model = GPT2Model.from_pretrained(model_name).eval()
    c = model.config
    with open(out_path, 'wb') as f:
        f.write(struct.pack('<II', 0x67707432, 1))           # magic, version
        f.write(struct.pack('<IIIII',
            c.vocab_size, c.n_ctx, c.n_embd, c.n_layer, c.n_head))

        sd = {k: v.float().numpy() for k, v in model.state_dict().items()}
        L  = c.n_layer

        # HF Conv1D stores weights as (in, out); transpose to (out, in) for our matmul
        stack   = lambda keys: np.stack([sd[k]   for k in keys])
        stack_T = lambda keys: np.stack([sd[k].T for k in keys])

        write_tensor(f, sd['wte.weight'],  'wte')
        write_tensor(f, sd['wpe.weight'],  'wpe')
        write_tensor(f, stack  ([f'h.{l}.ln_1.weight'        for l in range(L)]), 'ln1_w')
        write_tensor(f, stack  ([f'h.{l}.ln_1.bias'          for l in range(L)]), 'ln1_b')
        write_tensor(f, stack_T([f'h.{l}.attn.c_attn.weight' for l in range(L)]), 'c_attn_w')
        write_tensor(f, stack  ([f'h.{l}.attn.c_attn.bias'   for l in range(L)]), 'c_attn_b')
        write_tensor(f, stack_T([f'h.{l}.attn.c_proj.weight' for l in range(L)]), 'c_proj_w')
        write_tensor(f, stack  ([f'h.{l}.attn.c_proj.bias'   for l in range(L)]), 'c_proj_b')
        write_tensor(f, stack  ([f'h.{l}.ln_2.weight'        for l in range(L)]), 'ln2_w')
        write_tensor(f, stack  ([f'h.{l}.ln_2.bias'          for l in range(L)]), 'ln2_b')
        write_tensor(f, stack_T([f'h.{l}.mlp.c_fc.weight'    for l in range(L)]), 'mlp_fc_w')
        write_tensor(f, stack  ([f'h.{l}.mlp.c_fc.bias'      for l in range(L)]), 'mlp_fc_b')
        write_tensor(f, stack_T([f'h.{l}.mlp.c_proj.weight'  for l in range(L)]), 'mlp_pj_w')
        write_tensor(f, stack  ([f'h.{l}.mlp.c_proj.bias'    for l in range(L)]), 'mlp_pj_b')
        write_tensor(f, sd['ln_f.weight'], 'ln_f_w')
        write_tensor(f, sd['ln_f.bias'],   'ln_f_b')
    print("Weights ->", out_path)

def export_vocab(model_name, out_path):
    tok   = GPT2Tokenizer.from_pretrained(model_name)
    vocab = tok.get_vocab()
    id2str = [''] * len(vocab)
    for s, i in vocab.items():
        id2str[i] = bytes([tok.byte_decoder[c] for c in s]).decode('utf-8', errors='replace')

    with open(out_path, 'wb') as f:
        f.write(struct.pack('<II', 0x62706532, len(id2str)))
        for s in id2str:
            b = s.encode('utf-8')
            f.write(struct.pack('<I', len(b)))
            f.write(b)
        merges = sorted(tok.bpe_ranks.items(), key=lambda x: x[1])
        f.write(struct.pack('<I', len(merges)))
        for (a, b), _ in merges:
            f.write(struct.pack('<II', vocab.get(a,0), vocab.get(b,0)))
    print("Vocab ->", out_path)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model',   default='gpt2')
    ap.add_argument('--weights', default='gpt2_weights.bin')
    ap.add_argument('--vocab',   default='gpt2_vocab.bin')
    a = ap.parse_args()
    export_weights(a.model, a.weights)
    export_vocab(a.model,   a.vocab)