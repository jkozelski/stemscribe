# Chart Formatter Training Results

**Date:** 2026-04-13
**Model:** microsoft/Phi-3-mini-4k-instruct
**Method:** QLoRA (4-bit NF4 + LoRA r=64, alpha=16)

---

## Training Metrics

| Metric | Value |
|--------|-------|
| Train Loss | 0.3476 |
| Best Eval Loss | 0.3540 (step 300) |
| Final Eval Loss | 0.3637 (step 900) |
| Runtime | 14,173s (3h 56min) |
| Samples/sec | 1.094 |
| Steps/sec | 0.068 |
| Trainable Params | 35,651,584 / 3,856,731,136 (0.92%) |

## Configuration

| Parameter | Value |
|-----------|-------|
| Base model | microsoft/Phi-3-mini-4k-instruct |
| GPU | NVIDIA A100-SXM4-40GB |
| Quantization | 4-bit NF4 + double quant |
| LoRA rank | 64 |
| LoRA alpha | 16 |
| LoRA dropout | 0.1 |
| Target modules | q,k,v,o,gate,up,down proj |
| Learning rate | 2e-4 (cosine decay) |
| Warmup ratio | 0.05 |
| Epochs | 1 |
| Batch size | 4 (effective 16 with grad_accum=4) |
| Max seq length | 2048 |
| Attention | eager (flash-attn not available on Modal) |

## Training Data

| Dataset | Examples | Purpose |
|---------|----------|---------|
| Chord library (15,414 charts x 12 keys) | 184,968 total | Training pool |
| Subsampled for training | 15,500 | Actually trained on |
| Kozelski charts (19 songs x 3 formats) | 57 | Validation only |

Format distribution: ~60% JSON, ~25% slash notation, ~15% ChordPro

## Loss Curve

```
Step  Train  Eval   LR        Epoch
50    0.540  -      6.8e-5    0.05
100   0.406  -      1.4e-4    0.10
150   0.354  -      1.9e-4    0.15  (LR peak)
200   0.348  -      1.9e-4    0.21
250   0.346  -      1.8e-4    0.26
300   0.338  0.354  1.7e-4    0.31  ← best eval
350   0.339  -      1.5e-4    0.36
400   0.338  -      1.4e-4    0.41
450   0.335  -      1.2e-4    0.46
500   0.337  -      1.0e-4    0.52
550   0.337  -      8.6e-5    0.57
600   0.336  0.360  6.9e-5    0.62
650   0.338  -      5.3e-5    0.67
700   0.334  -      3.9e-5    0.72
750   0.332  -      2.7e-5    0.77
800   0.331  -      1.6e-5    0.83
850   0.331  -      8.0e-6    0.88
900   0.331  0.364  2.7e-6    0.93
950   0.336  -      1.9e-7    0.98
968   0.348  -      0         1.00  (final avg)
```

## Adapter Location

`backend/models/pretrained/chart_formatter_lora/`

Files:
- `adapter_model.safetensors` (136 MB) — LoRA weights
- `adapter_config.json` — LoRA configuration
- `tokenizer.json` + `tokenizer.model` — tokenizer
- `training_config.json` — hyperparameters used

## Issues Encountered

1. **transformers 5.x incompatible** with Phi-3 custom code (rope_scaling["type"] → ["rope_type"]). Fix: pin transformers<5.0
2. **flash-attn won't build on Modal** — CUDA toolkit not available during image build. Fix: use eager attention
3. **A10G too slow** — eager attention makes training ~4x slower. Fix: use A100
4. **OOM on A10G** — batch=2 + seq=4096 exceeded 24GB VRAM. Fix: batch=1, seq=2048
5. **185K examples timeout** — 4h not enough for full dataset. Fix: subsample to 15.5K
6. **Worker preemption** — A100-80GB preempted mid-training by Modal. Fix: auto-restart
7. **Eval timeout** — 57 examples at ~30s each = 28 min, exceeded 1h function timeout. Fix: download adapter separately

## Next Steps

1. **More epochs** — resume from saved adapter, push loss lower (target: 0.15-0.20)
2. **Full dataset** — train on all 185K examples with checkpoint resumption
3. **Wire into pipeline** — replace rule-based chart_formatter.py with Phi-3 inference
4. **Deploy to Modal** — serve alongside stem separation
5. **A/B test** — compare AI formatter output vs rule-based output
