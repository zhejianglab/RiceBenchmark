"""Sharded embedding extraction that reproduces the harness's own.

The harness extracts each dataset split in one process on one GPU. At Botanic1's
throughput the 32-128 kb tasks would take weeks that way, so we split the rows
into shards, extract them on separate GPUs, and merge the shards into the file
the harness reads (``{model}/{dataset}-{layer}layer_{split}.pt``). With those
files present the harness skips extraction and only trains and scores the MLPs.

Per batch this follows ``benchmarks/embedding_extract.py`` line for line: the same
rows in the same order, the same batch size rule (``batch_size``, halved for
``long_sequence_dataset``, 1 for ``super_long_sequence_dataset``), the
tokenizer called with ``padding=True``, a masked mean over each requested
``hidden_states[layer]``, float32 embeddings and ``torch.tensor(labels)`` per
batch. Shard boundaries are multiples of the batch size, so every batch is the
same as in a single-process run.

    python extract_embeddings.py shard --model-dir M --data D --info I \
        --dataset X --split train --layers 6,12,18,24 --shard 3 --num-shards 16 --out W
    python extract_embeddings.py merge --model-dir M ... --num-shards 16 --out W
"""

import argparse
import json
import math
from pathlib import Path

import torch
import yaml

torch.use_deterministic_algorithms(True)


def count_rows(path):
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def load_rows(path, feature, start, stop):
    """Rows [start, stop) of the non-empty lines, parsing only those."""
    rows = []
    with open(path, encoding="utf-8") as f:
        i = 0
        for line in f:
            line = line.strip()
            if not line:
                continue
            if start <= i < stop:
                item = json.loads(line)
                rows.append((item[feature["seq_key"]], item[feature["label_key"]]))
            i += 1
            if i >= stop:
                break
    return rows


def batch_size_for(dataset, info, batch_size):
    if dataset in info["long_sequence_dataset"]:
        batch_size = batch_size // 2
    if dataset in info["super_long_sequence_dataset"]:
        batch_size = 1
    return batch_size


def shard_path(out, model, dataset, layer, split, shard, num_shards):
    return (Path(out) / "shards" / model /
            f"{dataset}-{layer}layer_{split}.{shard:04d}-of-{num_shards:04d}.pt")


def run_shard(args, info):
    from transformers import AutoModel, AutoTokenizer

    feature = info["dataset_feature"][args.dataset]
    assert feature["seq_for_item"] == 1, "multi-sequence items are not handled"
    model_name = Path(args.model_dir).name
    layers = [int(x) for x in args.layers.split(",")]
    todo = [layer for layer in layers if not shard_path(
        args.out, model_name, args.dataset, layer, args.split, args.shard,
        args.num_shards).exists()]
    if not todo:
        print("shard already done")
        return

    path = Path(args.data) / args.dataset / f"{args.split}.jsonl"
    bs = batch_size_for(args.dataset, info, args.batch_size)
    per_shard = math.ceil(math.ceil(count_rows(path) / bs) / args.num_shards) * bs
    rows = load_rows(path, feature, args.shard * per_shard, (args.shard + 1) * per_shard)
    print(f"{args.dataset} {args.split} shard {args.shard}/{args.num_shards}: "
          f"{len(rows)} rows, batch {bs}, layers {todo}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model_dir, dtype=torch.bfloat16,
                                      trust_remote_code=True).cuda().eval()
    hidden = model.config.hidden_size
    embeddings = {layer: [] for layer in todo}
    labels = []
    with torch.no_grad():
        for start in range(0, len(rows), bs):
            batch = rows[start:start + bs]
            enc = tokenizer([seq for seq, _ in batch], padding=True, return_tensors="pt")
            mask = enc["attention_mask"].unsqueeze(-1).cuda()
            out = model(input_ids=enc["input_ids"].cuda(non_blocking=True),
                        output_hidden_states=True)
            for layer in todo:
                h = out.hidden_states[layer]
                embeddings[layer].append(((h * mask).sum(1) / mask.sum(1)).float().cpu())
            labels.append(torch.tensor([label for _, label in batch]))
            if (start // bs) % 200 == 0:
                print(f"  {start + len(batch)}/{len(rows)}", flush=True)
    labels = torch.cat(labels) if labels else torch.empty(0)
    for layer in todo:
        emb = torch.cat(embeddings[layer]) if embeddings[layer] else torch.empty(0, hidden)
        path = shard_path(args.out, model_name, args.dataset, layer, args.split,
                          args.shard, args.num_shards)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        torch.save({"embeddings": emb, "labels": labels}, tmp)
        tmp.rename(path)
    print("done", flush=True)


def merge(args, info):
    model_name = Path(args.model_dir).name
    feature = info["dataset_feature"][args.dataset]
    n_rows = count_rows(Path(args.data) / args.dataset / f"{args.split}.jsonl")
    for layer in (int(x) for x in args.layers.split(",")):
        parts = [torch.load(shard_path(args.out, model_name, args.dataset, layer,
                                       args.split, i, args.num_shards))
                 for i in range(args.num_shards)]
        emb = torch.cat([p["embeddings"] for p in parts if len(p["labels"])])
        lab = torch.cat([p["labels"] for p in parts if len(p["labels"])])
        assert len(emb) == len(lab) == n_rows, (len(emb), len(lab), n_rows)
        # The harness stores reshape(-1, seq_number, H) squeezed: [n, H] here.
        emb = torch.squeeze(torch.reshape(emb, (-1, feature["seq_for_item"], emb.shape[-1])))
        dst = Path(args.out) / "embeddings" / model_name / \
            f"{args.dataset}-{layer}layer_{args.split}.pt"
        dst.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"embeddings": emb, "labels": lab}, dst)
        print(dst, tuple(emb.shape), tuple(lab.shape))


parser = argparse.ArgumentParser()
parser.add_argument("mode", choices=["shard", "merge"])
parser.add_argument("--model-dir", required=True)
parser.add_argument("--data", required=True)
parser.add_argument("--info", required=True, help="datasets_info.yaml")
parser.add_argument("--dataset", required=True)
parser.add_argument("--split", required=True)
parser.add_argument("--layers", required=True)
parser.add_argument("--shard", type=int, default=0)
parser.add_argument("--num-shards", type=int, default=1)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--out", required=True, help="WORK root (embeddings/ and shards/)")
args = parser.parse_args()
info = yaml.safe_load(open(args.info))
run_shard(args, info) if args.mode == "shard" else merge(args, info)
