"""
Modal GPU training script for StemScriber chart formatter.

QLoRA fine-tune Phi-3-mini-4k-instruct to format chord charts.
All data prep happens on CPU first (prep_chart_training_data.py).
This script uploads data to a Modal Volume, then trains on GPU.

Usage:
    cd ~/stemscribe/backend
    ../venv311/bin/python prep_chart_training_data.py              # CPU data prep
    ../venv311/bin/python -m modal run train_chart_formatter.py    # Upload + GPU train

The trained LoRA adapter is saved to a Modal Volume and downloaded locally to:
    backend/models/pretrained/chart_formatter_lora/
"""

import modal
import os
import json
from pathlib import Path

# ─── Modal Setup ─────────────────────────────────────────────────────────────

training_image = (
    modal.Image.from_registry("nvidia/cuda:12.1.0-devel-ubuntu22.04", add_python="3.11")
    .pip_install(
        "torch>=2.3,<2.5",
        "packaging",
        "wheel",
        "setuptools",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "flash-attn>=2.5",
        extra_options="--no-build-isolation",
    )
    .pip_install(
        "transformers>=4.44,<5.0",
        "peft>=0.12,<0.14",
        "bitsandbytes>=0.43,<0.45",
        "datasets>=2.20,<3.0",
        "accelerate>=0.33,<1.0",
        "trl>=0.10,<0.13",
        "scipy",
        "sentencepiece",
        "protobuf",
    )
)

# Volumes
model_cache = modal.Volume.from_name("chart-formatter-model-cache", create_if_missing=True)
output_volume = modal.Volume.from_name("chart-formatter-output", create_if_missing=True)
data_volume = modal.Volume.from_name("chart-formatter-training-data", create_if_missing=True)

app = modal.App("stemscribe-chart-formatter-train", image=training_image)

# ─── Training Configuration ─────────────────────────────────────────────────

TRAIN_CONFIG = {
    "base_model": "microsoft/Phi-3-mini-4k-instruct",
    "quant_bits": 4,
    "lora_r": 64,       # r=64 matches adapter_config.json from successful direct test
    "lora_alpha": 16,    # Match original training plan spec
    "lora_dropout": 0.1,
    "lora_target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    "learning_rate": 2e-4,  # Full LR for fresh 3-epoch training (per training plan)
    "warmup_ratio": 0.05,
    "lr_scheduler": "cosine",
    "max_train_examples": 15500,  # 15K library examples (all 3 formats)
    "num_epochs": 3,     # 3 epochs (was 1 — insufficient for learning the task)
    "per_device_batch_size": 8,    # Doubled — A100 80GB has headroom with QLoRA
    "gradient_accumulation_steps": 2,  # Halved — same effective batch (16), 2x faster
    "max_seq_length": 2048,
    "logging_steps": 50,
    "save_steps": 500,
    "eval_steps": 500,
    "bf16": True,
    "gradient_checkpointing": True,
}


# ─── Data Upload Function (CPU, no GPU) ─────────────────────────────────────

@app.function(
    volumes={"/data": data_volume},
    timeout=1800,
    memory=8192,
)
def upload_chunk(chunk: bytes, filename: str, append: bool = False) -> str:
    """Upload a single chunk of data to the data volume."""
    os.makedirs("/data/chart_formatter", exist_ok=True)
    fpath = f"/data/chart_formatter/{filename}"
    mode = "ab" if append else "wb"
    with open(fpath, mode) as f:
        f.write(chunk)
    data_volume.commit()
    size = os.path.getsize(fpath)
    return f"Wrote {len(chunk)/1e6:.1f} MB to {fpath} (total: {size/1e6:.1f} MB)"
    return {"train_count": train_count, "val_count": val_count}


# ─── GPU Training Function ──────────────────────────────────────────────────

@app.function(
    gpu="A100-80GB",
    timeout=21600,  # 6 hours
    volumes={
        "/model-cache": model_cache,
        "/output": output_volume,
        "/data": data_volume,
    },
    memory=32768,
    # secrets=[modal.Secret.from_name("huggingface-secret")],  # uncomment if HF token needed
)
def train_chart_formatter() -> dict:
    """QLoRA fine-tune Phi-3-mini on chart formatting data."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset

    config = TRAIN_CONFIG

    print("=" * 60)
    print("StemScriber Chart Formatter - QLoRA Training")
    print("=" * 60)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Load data ────────────────────────────────────────────────────────
    print("\n[1/5] Loading training data...")

    def load_jsonl(path):
        examples = []
        with open(path) as f:
            for line in f:
                if line.strip():
                    examples.append(json.loads(line))
        return examples

    train_data = load_jsonl("/data/chart_formatter/train.jsonl")
    val_data = load_jsonl("/data/chart_formatter/val.jsonl")
    print(f"  Loaded: {len(train_data)} train, {len(val_data)} val")

    # Subsample if needed to fit in timeout
    max_examples = config.get("max_train_examples")
    if max_examples and len(train_data) > max_examples:
        import random
        random.seed(42)
        random.shuffle(train_data)
        train_data = train_data[:max_examples]
        print(f"  Subsampled to {len(train_data)} training examples")

    def format_chat(ex):
        text = ""
        for msg in ex["messages"]:
            text += f"<|{msg['role']}|>\n{msg['content']}\n<|end|>\n"
        return {"text": text}

    train_formatted = [format_chat(ex) for ex in train_data]
    val_formatted = [format_chat(ex) for ex in val_data]
    del train_data, val_data

    train_dataset = Dataset.from_list(train_formatted)
    val_dataset = Dataset.from_list(val_formatted)
    del train_formatted, val_formatted

    # ── Load model ───────────────────────────────────────────────────────
    print("\n[2/5] Loading Phi-3-mini with 4-bit quantization...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        config["base_model"],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        cache_dir="/model-cache",
        attn_implementation="flash_attention_2",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        config["base_model"], trust_remote_code=True, cache_dir="/model-cache",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    print(f"  Model: {model.num_parameters() / 1e9:.2f}B params")

    # ── Configure LoRA ───────────────────────────────────────────────────
    print("\n[3/5] Configuring QLoRA...")

    # Fresh training — clear any old adapter to avoid resume
    # (Previous 1-epoch adapter produced poor output; starting over with 3 epochs)
    resume_adapter = "/output/chart_formatter_lora/final_adapter"
    if os.path.exists(resume_adapter):
        import shutil
        print(f"  Clearing old adapter at {resume_adapter} for fresh 3-epoch training")
        shutil.rmtree(resume_adapter, ignore_errors=True)

    if False:  # Resume disabled — always start fresh
        pass
    else:
        print("  Starting fresh LoRA training")
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        lora_config = LoraConfig(
            r=config["lora_r"], lora_alpha=config["lora_alpha"],
            lora_dropout=config["lora_dropout"],
            target_modules=config["lora_target_modules"],
            bias="none", task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    trainable, total = model.get_nb_trainable_parameters()
    print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # ── Train ────────────────────────────────────────────────────────────
    print("\n[4/5] Training...")

    output_dir = "/output/chart_formatter_lora"
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=config["num_epochs"],
        per_device_train_batch_size=config["per_device_batch_size"],
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        warmup_ratio=config["warmup_ratio"],
        lr_scheduler_type=config["lr_scheduler"],
        bf16=config["bf16"],
        gradient_checkpointing=config["gradient_checkpointing"],
        logging_steps=config["logging_steps"],
        save_steps=config["save_steps"],
        eval_steps=config["eval_steps"],
        eval_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        max_seq_length=config["max_seq_length"],
        dataset_text_field="text",
        report_to="none",
        packing=False,  # Packing creates fewer but slower steps — worse for timeouts
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    trainer = SFTTrainer(
        model=model, args=training_args,
        train_dataset=train_dataset, eval_dataset=val_dataset,
        processing_class=tokenizer,
    )

    train_result = trainer.train()
    print(f"  Train loss: {train_result.training_loss:.4f}")
    print(f"  Runtime: {train_result.metrics['train_runtime']:.0f}s")

    # ── Save ─────────────────────────────────────────────────────────────
    print("\n[5/5] Saving adapter...")

    adapter_path = "/output/chart_formatter_lora/final_adapter"
    trainer.save_model(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    with open(f"{adapter_path}/training_config.json", "w") as f:
        json.dump(config, f, indent=2)

    eval_result = trainer.evaluate()
    print(f"  Eval loss: {eval_result['eval_loss']:.4f}")

    output_volume.commit()

    return {
        "train_loss": train_result.training_loss,
        "eval_loss": eval_result["eval_loss"],
        "train_runtime_seconds": train_result.metrics["train_runtime"],
        "train_samples_per_second": train_result.metrics["train_samples_per_second"],
        "trainable_params": trainable,
        "total_params": total,
        "trainable_pct": round(100 * trainable / total, 2),
        "adapter_path": adapter_path,
        "config": config,
    }


# ─── Evaluation Function ────────────────────────────────────────────────────

@app.function(
    gpu="A100-80GB", timeout=7200,
    volumes={"/model-cache": model_cache, "/output": output_volume, "/data": data_volume},
    memory=32768,
)
def evaluate_chart_formatter() -> dict:
    """Run inference on validation set, compare to expected output."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    config = TRAIN_CONFIG
    print("Loading model + adapter for evaluation...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        config["base_model"], quantization_config=bnb_config,
        device_map="auto", trust_remote_code=True, cache_dir="/model-cache",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config["base_model"], trust_remote_code=True, cache_dir="/model-cache",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = PeftModel.from_pretrained(base_model, "/output/chart_formatter_lora/final_adapter")
    model.eval()

    val_data = []
    with open("/data/chart_formatter/val.jsonl") as f:
        for line in f:
            if line.strip():
                val_data.append(json.loads(line))

    print(f"Evaluating {len(val_data)} examples...")
    results = []

    for i, example in enumerate(val_data):
        messages = example["messages"]
        expected = messages[2]["content"]

        prompt = f"<|system|>\n{messages[0]['content']}\n<|end|>\n<|user|>\n{messages[1]['content']}\n<|end|>\n<|assistant|>\n"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=3072).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=2048, temperature=0.1,
                top_p=0.95, do_sample=True, pad_token_id=tokenizer.pad_token_id,
            )

        generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        if "<|end|>" in generated:
            generated = generated[:generated.index("<|end|>")]

        song = example.get("_song", f"example_{i}")
        fmt = example.get("_format", "unknown")

        results.append({
            "song": song, "format": fmt,
            "expected": expected[:500], "generated": generated[:500],
            "expected_length": len(expected), "generated_length": len(generated),
        })

        if i < 6:
            print(f"\n--- {song} ({fmt}) ---")
            print(f"Generated:\n{generated[:200]}")

    return {"evaluations": results, "count": len(results)}


# ─── Download Function ──────────────────────────────────────────────────────

@app.function(volumes={"/output": output_volume}, timeout=300)
def download_adapter() -> dict:
    """Download trained adapter files as {filename: bytes}."""
    adapter_path = "/output/chart_formatter_lora/final_adapter"
    files = {}
    if not os.path.exists(adapter_path):
        print(f"ERROR: {adapter_path} not found")
        return files
    for fname in os.listdir(adapter_path):
        fpath = os.path.join(adapter_path, fname)
        if os.path.isfile(fpath):
            with open(fpath, "rb") as f:
                files[fname] = f.read()
            print(f"  {fname} ({os.path.getsize(fpath)/1024:.1f} KB)")
    return files


# ─── Local Entry Point ──────────────────────────────────────────────────────

CHUNK_SIZE = 200 * 1024 * 1024  # 200 MB

@app.local_entrypoint()
def main():
    """Upload data to Modal Volume, train on GPU, download adapter."""
    data_dir = Path(__file__).parent / "training_data" / "chart_formatter"
    train_path = data_dir / "train.jsonl"
    val_path = data_dir / "val.jsonl"

    if not train_path.exists():
        print("ERROR: Run data prep first:  ../venv311/bin/python prep_chart_training_data.py")
        return

    train_size = train_path.stat().st_size
    val_size = val_path.stat().st_size
    print(f"  Train: {train_size/1e6:.1f} MB, Val: {val_size/1e6:.3f} MB")

    # Upload training data in chunks (skip if SKIP_UPLOAD=1 and data already on volume)
    if os.environ.get("SKIP_UPLOAD"):
        print("\nSKIP_UPLOAD set — assuming data is already on Modal Volume.")
    else:
        print(f"\nUploading train.jsonl ({train_size/1e6:.1f} MB) in {CHUNK_SIZE//1024//1024} MB chunks...")
        with open(train_path, "rb") as f:
            chunk_num = 0
            while True:
                chunk = f.read(CHUNK_SIZE)
                if not chunk:
                    break
                result = upload_chunk.remote(chunk, "train.jsonl", append=(chunk_num > 0))
                print(f"  Chunk {chunk_num}: {result}")
                chunk_num += 1

        # Upload validation data (small, single chunk)
        print(f"\nUploading val.jsonl ({val_size/1e6:.3f} MB)...")
        result = upload_chunk.remote(val_path.read_bytes(), "val.jsonl", append=False)
        print(f"  {result}")

    print("\nStarting GPU training...")
    results = train_chart_formatter.remote()
    print("\nTRAINING COMPLETE")
    print(json.dumps(results, indent=2, default=str))

    print("\nEvaluating on Kozelski validation set...")
    eval_results = evaluate_chart_formatter.remote()
    print(f"Evaluated {eval_results['count']} examples")

    print("\nDownloading adapter weights...")
    adapter_files = download_adapter.remote()
    local_dir = Path(__file__).parent / "models" / "pretrained" / "chart_formatter_lora"
    local_dir.mkdir(parents=True, exist_ok=True)
    for fname, fbytes in adapter_files.items():
        (local_dir / fname).write_bytes(fbytes)
        print(f"  Saved: {local_dir / fname} ({len(fbytes)/1024:.1f} KB)")

    results["evaluation"] = eval_results
    results_json = Path(__file__).parent.parent / "docs" / "chart-formatter-training-results.json"
    results_json.write_text(json.dumps(results, indent=2, default=str))

    _write_results_markdown(results, eval_results, local_dir)
    print(f"\nAdapter saved to: {local_dir}")


def _write_results_markdown(results, eval_results, adapter_dir):
    md_path = Path(__file__).parent.parent / "docs" / "chart-formatter-training-results.md"
    config = results.get("config", {})

    def fv(key, fs="{:.4f}"):
        v = results.get(key)
        return fs.format(v) if isinstance(v, (int, float)) else str(v or "N/A")

    lines = [
        "# Chart Formatter Training Results", "",
        f"**Date:** {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Model:** {config.get('base_model', 'Phi-3-mini-4k-instruct')}",
        f"**Method:** QLoRA (4-bit NF4 + LoRA r={config.get('lora_r', 64)})", "",
        "---", "",
        "## Training Metrics", "",
        "| Metric | Value |", "|--------|-------|",
        f"| Train Loss | {fv('train_loss')} |",
        f"| Eval Loss | {fv('eval_loss')} |",
        f"| Runtime | {fv('train_runtime_seconds', '{:.0f}')}s |",
        f"| Samples/sec | {fv('train_samples_per_second', '{:.1f}')} |",
        f"| Trainable Params | {fv('trainable_params')} / {fv('total_params')} ({fv('trainable_pct', '{:.2f}')}%) |",
        "", "## Configuration", "", "```json", json.dumps(config, indent=2), "```", "",
        "## Kozelski Validation Examples", "",
    ]

    for ev in eval_results.get("evaluations", [])[:10]:
        lines += [
            f"### {ev.get('song','?')} ({ev.get('format','?')})", "",
            "**Generated (first 300 chars):**", "```",
            ev.get("generated", "")[:300], "```", "",
        ]

    lines += [
        "## Adapter Location", "", f"`{adapter_dir}`", "",
        "## Next Steps", "",
        "1. Deploy adapter on Modal for inference",
        "2. Wire into pipeline after chord detection + Whisper",
        "3. A/B test against rule-based chart_formatter.py",
        "4. Iterate on training data if eval loss is high",
    ]

    md_path.write_text("\n".join(lines))
    print(f"Results markdown: {md_path}")
