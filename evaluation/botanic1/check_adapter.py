"""Check the adapter against the released Botanic1 class on GPU.

    python check_adapter.py models/Botanic1-S

1. hidden_states layout: index 0 = embeddings, i = block i, last = post-norm.
2. a right-padded batch gives each row the states of that row encoded alone.
3. long inputs (8,192 / 32,000 / 128,000 bp) run, with time and peak memory.
Runs under torch.use_deterministic_algorithms(True), as the harness does.
"""

import random
import sys
import time

import torch
from transformers import AutoModel, AutoTokenizer

torch.use_deterministic_algorithms(True)
model_dir = sys.argv[1]
tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModel.from_pretrained(model_dir, torch_dtype=torch.bfloat16,
                                  trust_remote_code=True).cuda().eval()
base = type(model).__mro__[1]
print(type(model).__name__, "base", base.__name__,
      "layers", model.config.num_hidden_layers, "hidden", model.config.hidden_size)
rng = random.Random(0)


def seq(n):
    return "".join(rng.choice("ACGT") for _ in range(n))


def max_rel(a, b):
    return ((a.float() - b.float()).abs().max() / b.float().abs().max()).item()


with torch.no_grad():
    # 1. layout
    ids = tok([seq(1000)], return_tensors="pt")["input_ids"].cuda()
    out = model(input_ids=ids, output_hidden_states=True)
    post, blocks = base.encode(model, input_ids=ids, output_hidden_states=True)
    hs = out.hidden_states
    n = model.config.num_hidden_layers
    assert len(hs) == n + 1, len(hs)
    assert torch.equal(hs[0], model.tok_embeddings(ids))
    worst = max(max_rel(hs[i], blocks[i - 1]) for i in range(1, n))
    print(f"[1] blocks 1..{n - 1} max rel diff {worst:.2e}; "
          f"last vs post-norm {max_rel(hs[n], post):.2e}")
    assert worst == 0 and torch.equal(hs[n], post)

    # 2. padding
    seqs = [seq(n_bp) for n_bp in (500, 300, 300, 37)]
    batch = tok(seqs, padding=True, return_tensors="pt")
    assert tok.padding_side == "right"
    out = model(input_ids=batch["input_ids"].cuda(), output_hidden_states=True)
    worst = 0.0
    for i, s in enumerate(seqs):
        one = tok([s], return_tensors="pt")["input_ids"].cuda()
        ref = model(input_ids=one, output_hidden_states=True).hidden_states
        length = one.shape[1]
        for layer in (1, n // 2, n):
            worst = max(worst, max_rel(out.hidden_states[layer][i, :length], ref[layer][0]))
            assert out.hidden_states[layer][i, length:].abs().sum() == 0
    print(f"[2] padded batch vs single rows: max rel diff {worst:.2e}")
    assert worst < 1e-2

    # 3. long inputs (single sequence, all hidden states kept as the harness does)
    for n_bp, bs in ((8192, 8), (32000, 1), (128000, 1)):
        ids = tok([seq(n_bp)] * bs, return_tensors="pt")["input_ids"].cuda()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t = time.time()
        out = model(input_ids=ids, output_hidden_states=True)
        torch.cuda.synchronize()
        dt = time.time() - t
        print(f"[3] {bs}x{n_bp} bp: {dt:.2f}s, {bs * n_bp / dt / 1e3:.0f} kbp/s, "
              f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB, "
              f"finite={bool(torch.isfinite(out.hidden_states[-1]).all())}")
print("OK")
