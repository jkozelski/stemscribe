"""
Modal GPU training script for StemScriber chord recall model.

QLoRA fine-tune Phi-3-mini-4k-instruct to recall chord charts from lyrics.
This is a SEPARATE training run from the chart formatter — different Modal app.

Usage:
    cd ~/stemscribe/backend
    ../venv311/bin/python -m modal run train_chart_recall.py

The trained LoRA adapter is saved to a Modal Volume and downloaded locally to:
    backend/models/pretrained/chart_recall_lora/
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

# Volumes — all separate from formatter to avoid conflicts
model_cache = modal.Volume.from_name("chart-recall-model-cache", create_if_missing=True)
output_volume = modal.Volume.from_name("chart-recall-checkpoints", create_if_missing=True)
data_volume = modal.Volume.from_name("chart-recall-training-data", create_if_missing=True)

app = modal.App("stemscribe-chart-recall-train", image=training_image)

# ─── Training Configuration ─────────────────────────────────────────────────

TRAIN_CONFIG = {
    "base_model": "microsoft/Phi-3-mini-4k-instruct",
    "quant_bits": 4,
    "lora_r": 128,
    "lora_alpha": 32,
    "lora_dropout": 0.1,
    "lora_target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    "learning_rate": 5e-5,  # Lower LR for continued training (was 2e-4 for epoch 1)
    "warmup_ratio": 0.05,
    "lr_scheduler": "cosine",
    "max_train_examples": 30000,  # Subsample from 97K to fit in 6h timeout
    "num_epochs": 1,
    "per_device_batch_size": 4,
    "gradient_accumulation_steps": 4,
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
    os.makedirs("/data/chart_recall", exist_ok=True)
    fpath = f"/data/chart_recall/{filename}"
    mode = "ab" if append else "wb"
    with open(fpath, mode) as f:
        f.write(chunk)
    data_volume.commit()
    size = os.path.getsize(fpath)
    return f"Wrote {len(chunk)/1e6:.1f} MB to {fpath} (total: {size/1e6:.1f} MB)"


# ─── GPU Training Function ──────────────────────────────────────────────────

@app.function(
    gpu="A100",
    timeout=21600,  # 6 hours
    volumes={
        "/model-cache": model_cache,
        "/output": output_volume,
        "/data": data_volume,
    },
    memory=32768,
)
def train_chart_recall() -> dict:
    """QLoRA fine-tune Phi-3-mini on chord recall data (97K examples, 3 epochs)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset

    config = TRAIN_CONFIG

    print("=" * 60)
    print("StemScriber Chord Recall - QLoRA Training")
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

    train_data = load_jsonl("/data/chart_recall/recall_train.jsonl")
    val_data = load_jsonl("/data/chart_recall/recall_val.jsonl")
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

    # Check if we're resuming from a previous adapter or checkpoint
    output_dir = "/output/chart_recall_lora"
    resume_adapter = f"{output_dir}/final_adapter"
    resume_checkpoint = None

    # First check for final_adapter, then latest checkpoint
    if os.path.exists(resume_adapter) and os.path.exists(f"{resume_adapter}/adapter_model.safetensors"):
        print(f"  RESUMING from final adapter at {resume_adapter}")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, resume_adapter, is_trainable=True)
    else:
        # Find latest checkpoint
        import glob
        checkpoints = sorted(glob.glob(f"{output_dir}/checkpoint-*"), key=lambda x: int(x.split("-")[-1]))
        if checkpoints:
            latest_ckpt = checkpoints[-1]
            print(f"  RESUMING from checkpoint {latest_ckpt}")
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, latest_ckpt, is_trainable=True)
            resume_checkpoint = latest_ckpt
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
    print("\n[4/5] Training (3 epochs, 97K examples)...")
    # Effective batch = 2 * 8 = 16
    # Steps per epoch = 97476 / 16 ≈ 6092
    # Total steps ≈ 18276 across 3 epochs

    output_dir = "/output/chart_recall_lora"
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
        packing=False,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    trainer = SFTTrainer(
        model=model, args=training_args,
        train_dataset=train_dataset, eval_dataset=val_dataset,
        processing_class=tokenizer,
    )

    if resume_checkpoint:
        print(f"  Resuming training from {resume_checkpoint}")
        train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
    else:
        train_result = trainer.train()
    print(f"  Train loss: {train_result.training_loss:.4f}")
    print(f"  Runtime: {train_result.metrics['train_runtime']:.0f}s")

    # ── Save ─────────────────────────────────────────────────────────────
    print("\n[5/5] Saving adapter...")

    adapter_path = "/output/chart_recall_lora/final_adapter"
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
    gpu="A10G", timeout=3600,
    volumes={"/model-cache": model_cache, "/output": output_volume, "/data": data_volume},
    memory=32768,
)
def evaluate_chart_recall() -> dict:
    """Run inference on Kozelski validation set + Thunderhead test."""
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

    model = PeftModel.from_pretrained(base_model, "/output/chart_recall_lora/final_adapter")
    model.eval()

    # ── Evaluate Kozelski validation set ─────────────────────────────────
    val_data = []
    with open("/data/chart_recall/recall_val.jsonl") as f:
        for line in f:
            if line.strip():
                val_data.append(json.loads(line))

    print(f"Evaluating {len(val_data)} Kozelski validation examples...")
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
        results.append({
            "song": song,
            "expected": expected[:500], "generated": generated[:500],
            "expected_length": len(expected), "generated_length": len(generated),
        })

        if i < 8:
            print(f"\n--- {song} ---")
            print(f"Generated:\n{generated[:300]}")

    # ── Thunderhead test ─────────────────────────────────────────────────
    print("\n\n=== THUNDERHEAD TEST ===")
    thunderhead_prompt = (
        "<|system|>\nYou are a chord chart expert with encyclopedic knowledge of 15,000+ songs. "
        "Given lyrics and optional detected notes, identify the song and recall the correct chord "
        "chart from your training. If the song is not recognized, say so.\n<|end|>\n"
        "<|user|>\nIdentify this song and recall the correct chord chart:\n\n<recall>\n"
        "Lyrics: Another night stirring with the boys in a cloud\n"
        "We were loud and alive, burning proud\n"
        "Thunder rolling through the hills\n"
        "Expected chords include: Am7, Bm7, Cm7, E9, Bm6\n</recall>\n<|end|>\n<|assistant|>\n"
    )

    inputs = tokenizer(thunderhead_prompt, return_tensors="pt", truncation=True, max_length=3072).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=2048, temperature=0.1,
            top_p=0.95, do_sample=True, pad_token_id=tokenizer.pad_token_id,
        )
    thunderhead_output = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    if "<|end|>" in thunderhead_output:
        thunderhead_output = thunderhead_output[:thunderhead_output.index("<|end|>")]

    print(f"Thunderhead output:\n{thunderhead_output[:500]}")

    # Check if expected chords appear
    expected_chords = ["Am7", "Bm7", "Cm7", "E9", "Bm6"]
    found_chords = [c for c in expected_chords if c in thunderhead_output]
    print(f"Expected chords found: {found_chords} / {expected_chords}")

    results.append({
        "song": "Thunderhead (manual test)",
        "expected": f"Chords: {', '.join(expected_chords)}",
        "generated": thunderhead_output[:500],
        "expected_length": 0,
        "generated_length": len(thunderhead_output),
        "chords_found": found_chords,
        "chords_expected": expected_chords,
    })

    return {"evaluations": results, "count": len(results)}


# ─── Download Function ──────────────────────────────────────────────────────

@app.function(volumes={"/output": output_volume}, timeout=300)
def download_adapter() -> dict:
    """Download trained adapter files as {filename: bytes}."""
    adapter_path = "/output/chart_recall_lora/final_adapter"
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
    data_dir = Path(__file__).parent / "training_data" / "chart_recall"
    train_path = data_dir / "recall_train.jsonl"
    val_path = data_dir / "recall_val.jsonl"

    if not train_path.exists():
        print(f"ERROR: Training data not found at {train_path}")
        return

    train_size = train_path.stat().st_size
    val_size = val_path.stat().st_size
    print(f"  Train: {train_size/1e6:.1f} MB ({train_size:,} bytes)")
    print(f"  Val: {val_size/1e6:.3f} MB ({val_size:,} bytes)")

    # Upload training data in chunks (CPU, no GPU cost)
    if os.environ.get("SKIP_UPLOAD"):
        print("\nSKIP_UPLOAD set — assuming data is already on Modal Volume.")
    else:
        print(f"\nUploading recall_train.jsonl ({train_size/1e6:.1f} MB) in {CHUNK_SIZE//1024//1024} MB chunks...")
        with open(train_path, "rb") as f:
            chunk_num = 0
            while True:
                chunk = f.read(CHUNK_SIZE)
                if not chunk:
                    break
                result = upload_chunk.remote(chunk, "recall_train.jsonl", append=(chunk_num > 0))
                print(f"  Chunk {chunk_num}: {result}")
                chunk_num += 1

        # Upload validation data (small, single chunk)
        print(f"\nUploading recall_val.jsonl ({val_size/1e6:.3f} MB)...")
        result = upload_chunk.remote(val_path.read_bytes(), "recall_val.jsonl", append=False)
        print(f"  {result}")

    # Train on GPU
    print("\nStarting GPU training (3 epochs, ~97K examples)...")
    print("Estimated time: 2-4 hours on A100")
    results = train_chart_recall.remote()
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(json.dumps(results, indent=2, default=str))

    # Evaluate on Kozelski validation set + Thunderhead test
    print("\nEvaluating on Kozelski validation set + Thunderhead test...")
    eval_results = evaluate_chart_recall.remote()
    print(f"Evaluated {eval_results['count']} examples")

    # Download adapter weights
    print("\nDownloading adapter weights...")
    adapter_files = download_adapter.remote()
    local_dir = Path(__file__).parent / "models" / "pretrained" / "chart_recall_lora"
    local_dir.mkdir(parents=True, exist_ok=True)
    for fname, fbytes in adapter_files.items():
        (local_dir / fname).write_bytes(fbytes)
        print(f"  Saved: {local_dir / fname} ({len(fbytes)/1024:.1f} KB)")

    # Save full results JSON
    results["evaluation"] = eval_results
    results_json = Path(__file__).parent.parent / "docs" / "chart-recall-training-results.json"
    results_json.parent.mkdir(parents=True, exist_ok=True)
    results_json.write_text(json.dumps(results, indent=2, default=str))

    # Write markdown report
    _write_results_markdown(results, eval_results, local_dir)
    print(f"\nAdapter saved to: {local_dir}")


def _write_results_markdown(results, eval_results, adapter_dir):
    md_path = Path(__file__).parent.parent / "docs" / "chart-recall-training-results.md"
    config = results.get("config", {})

    def fv(key, fs="{:.4f}"):
        v = results.get(key)
        return fs.format(v) if isinstance(v, (int, float)) else str(v or "N/A")

    # Find Thunderhead result
    thunderhead = None
    for ev in eval_results.get("evaluations", []):
        if "Thunderhead" in ev.get("song", ""):
            thunderhead = ev
            break

    lines = [
        "# Chord Recall Training Results", "",
        f"**Date:** {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Model:** {config.get('base_model', 'Phi-3-mini-4k-instruct')}",
        f"**Method:** QLoRA (4-bit NF4 + LoRA r={config.get('lora_r', 64)})",
        f"**Training examples:** 97,476",
        f"**Validation examples:** 108 (Kozelski catalog)",
        f"**Epochs:** {config.get('num_epochs', 3)}", "",
        "---", "",
        "## Training Metrics", "",
        "| Metric | Value |", "|--------|-------|",
        f"| Train Loss | {fv('train_loss')} |",
        f"| Eval Loss | {fv('eval_loss')} |",
        f"| Runtime | {fv('train_runtime_seconds', '{:.0f}')}s |",
        f"| Samples/sec | {fv('train_samples_per_second', '{:.1f}')} |",
        f"| Trainable Params | {fv('trainable_params')} / {fv('total_params')} ({fv('trainable_pct', '{:.2f}')}%) |",
        "", "## Configuration", "", "```json", json.dumps(config, indent=2), "```", "",
    ]

    # Thunderhead test
    if thunderhead:
        found = thunderhead.get("chords_found", [])
        expected = thunderhead.get("chords_expected", [])
        lines += [
            "## Thunderhead Chord Recall Test", "",
            f"**Input:** \"Another night stirring with the boys in a cloud...\"",
            f"**Expected chords:** {', '.join(expected)}",
            f"**Found chords:** {', '.join(found)} ({len(found)}/{len(expected)})",
            "", "**Model output:**", "```",
            thunderhead.get("generated", "")[:500], "```", "",
        ]

    # Kozelski validation examples
    lines += ["## Kozelski Validation Examples", ""]
    for ev in eval_results.get("evaluations", [])[:12]:
        if "Thunderhead" in ev.get("song", ""):
            continue
        lines += [
            f"### {ev.get('song','?')}", "",
            "**Generated (first 300 chars):**", "```",
            ev.get("generated", "")[:300], "```", "",
        ]

    lines += [
        "## Adapter Location", "", f"`{adapter_dir}`", "",
        "## Next Steps", "",
        "1. Wire recall adapter into chord pipeline (after Whisper lyrics + basic-pitch notes)",
        "2. A/B test: recall model vs raw chord detection",
        "3. Combine with formatter: recall -> format -> final chart",
        "4. Test on non-Kozelski songs to check generalization",
    ]

    md_path.write_text("\n".join(lines))
    print(f"Results markdown: {md_path}")
