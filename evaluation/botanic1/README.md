# Botanic1 on RiceBenchmark

Scripts to evaluate the four Botanic1 models on all 26 RiceBenchmark datasets with
the released OneGenome-Rice evaluation code. Results are in
[`results/`](../../results/).

| Model | Hugging Face | Layers × width | Parameters | Training context |
|---|---|---|---|---|
| Botanic1-S | [`living-models/Botanic1-S`](https://huggingface.co/living-models/Botanic1-S) | 24 × 1024 | 318M | 8,192 bp |
| Botanic1-M | [`living-models/Botanic1-M`](https://huggingface.co/living-models/Botanic1-M) | 52 × 1024 | 688M | 8,192 bp |
| Botanic1-L | [`living-models/Botanic1-L`](https://huggingface.co/living-models/Botanic1-L) | 72 × 1536 | 2.1B | 8,192 bp |
| Botanic1-XL | [`living-models/Botanic1-XL`](https://huggingface.co/living-models/Botanic1-XL) | 80 × 1792 | 3.2B | 8,192 bp |

Botanic1 is a bidirectional Mamba-2 masked language model over single nucleotides,
trained on plant genomes ([technical report](https://doi.org/10.64898/2026.09.04.749355)).

## Protocol

- **Harness:** `evaluation/benchmark_code` from
  [zhejianglab/OneGenome-Rice](https://github.com/zhejianglab/OneGenome-Rice) at
  commit `79984da2`, unmodified. Embeddings come from a sharded copy of its
  extraction step (below); the harness itself trains and scores every probe.
- **Data:** [`ZhejiangLab/RiceBenchmark`](https://huggingface.co/datasets/ZhejiangLab/RiceBenchmark)
  at revision `1aeac87a`, all 26 datasets.
- **Settings:** the harness defaults in `benchmark_code/config.yaml` (MLP probe,
  100 epochs, lr 1e-4, dropout 0.2, seed 42, mean pooling, bf16 weights). Only the
  paths, the dataset list, `layer_to_eval`, `gpu_list` and `batch_size` differ;
  see [`run_botanic1.sh`](./run_botanic1.sh).
- **Layers:** the harness reads `hidden_states[layer]`. Each model is probed at the
  last layer and at ¼, ½ and ¾ depth (S: 6, 12, 18, 24). The headline numbers use
  the last layer, as the released config does for OneGenome-Rice (`layer_to_eval: [12]`).
  The harness's `best_*` tables pick the best layer on the test split; we report
  those separately, labelled as test-selected.

## What differs from a stock run

1. **Adapter** ([`ricebench_botanic1.py`](./ricebench_botanic1.py)). `make_model_dir.py`
   builds a model directory whose `config.json` loads `Botanic1ForRiceBenchmark`, a
   subclass of the released class that uses the same weights. It changes three
   things:
   - `hidden_states` has `num_hidden_layers + 1` entries in the Hugging Face layout
     (embeddings first, final normalized state last).
   - Padded rows are re-encoded at their own length. Botanic1 does not mask pad
     tokens, and the harness pads each batch without passing an attention mask, so
     without this a sequence's embedding would depend on what it was batched with.
   - Inputs longer than the 8,194-token limit in the released class are accepted.
     The harness passes whole sequences, so the 32 kb, 100 kb and 128 kb tasks run as
     single scans well beyond the 8,192-bp training length.

   [`check_adapter.py`](./check_adapter.py) verifies the first two points against the
   released class and times long inputs.
2. **`datasets_info.yaml`.** The released file has no `long_sequence_dataset` or
   `super_long_sequence_dataset` lists, and `save_embedding()` raises `KeyError`
   without them. [Our copy](./datasets_info.yaml) is the dataset's file with those
   two lists added. They only lower the batch size for long inputs; with the adapter,
   batch size does not change the embeddings.
3. **Inputs are passed unchanged.** All sequences are uppercase `ACGTN` except
   15,924 `U` characters in `smallRNA`, which the Botanic1 tokenizer maps to `<unk>`.
4. **Sharded extraction** ([`extract_embeddings.py`](./extract_embeddings.py)). The
   harness embeds each split in one process on one GPU. Botanic1-S runs at about
   80 kbp/s on one RTX PRO 6000, and the benchmark holds about 29 Gbp, 19 Gbp of
   it in `varieties_classification_128k`. A single-process run would take about
   100 GPU-hours for S and about ten times that for XL. The script splits the rows into
   shards on separate GPUs and merges them into the files the harness reads
   (`{model}/{dataset}-{layer}layer_{split}.pt`). The harness then skips extraction
   (its documented pre-extracted-embedding path). Each batch is built and pooled
   exactly as in `benchmarks/embedding_extract.py`. Check against the harness's own
   files on a 100-row-per-split subsample of every dataset: *pending*.
   Batches differ from a harness run only at its 50,000-row part boundaries.
5. **`wandb`.** `evaluation.py` imports `wandb` even with `wandb_report: False`. We
   ran with a no-op `wandb` module on `PYTHONPATH`; installing `wandb` works too.

## Reproduce

```bash
pip install -r OneGenome-Rice/evaluation/benchmark_code/requirements.txt mamba-ssm causal-conv1d
python make_model_dir.py --size S --out $WORK/models
python check_adapter.py $WORK/models/Botanic1-S
# embeddings, one call per (dataset, split, shard), then one merge per (dataset, split)
python extract_embeddings.py shard --model-dir $WORK/models/Botanic1-S --data RiceBenchmark \
  --info datasets_info.yaml --dataset smallRNA --split train --layers 6,12,18,24 \
  --num-shards 4 --shard 0 --out $WORK
python extract_embeddings.py merge ... --num-shards 4 --out $WORK
# probes and reports (extraction is skipped when the merged files exist)
HARNESS=OneGenome-Rice/evaluation/benchmark_code DATA=RiceBenchmark WORK=$WORK PYTHON=python \
  ./run_botanic1.sh S 6,12,18,24 chromatin_access_512 smallRNA ...   # dataset names
```

`run_botanic1.sh` alone also runs the whole pipeline, extraction included, in one
process.

Without `mamba_ssm`, the model falls back to a pure-PyTorch scan that is much slower.
