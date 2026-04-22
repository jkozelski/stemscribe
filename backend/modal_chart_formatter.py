"""
Modal GPU inference for StemScriber Phi-3 chart formatter.

Deploys a serverless GPU function that runs the fine-tuned Phi-3-mini-4k-instruct
model with QLoRA adapter for chord chart formatting.

Deploy:  cd ~/stemscribe/backend && ../venv311/bin/python -m modal deploy modal_chart_formatter.py
Test:    cd ~/stemscribe/backend && ../venv311/bin/python -m modal run modal_chart_formatter.py
"""

import modal
import os

# ---------------------------------------------------------------------------
# Modal Image
# ---------------------------------------------------------------------------

formatter_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.0-runtime-ubuntu22.04", add_python="3.11"
    )
    .pip_install(
        "torch>=2.3,<2.5",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "transformers==4.44.2",
        "peft==0.12.0",
        "bitsandbytes>=0.43,<0.45",
        "accelerate>=0.33,<1.0",
        "scipy",
        "sentencepiece",
        "protobuf",
    )
)

# Volumes
model_cache = modal.Volume.from_name("chart-formatter-model-cache", create_if_missing=True)
adapter_volume = modal.Volume.from_name("chart-formatter-output", create_if_missing=True)

app = modal.App("stemscribe-chart-formatter", image=formatter_image)

# ---------------------------------------------------------------------------
# GPU Function
# ---------------------------------------------------------------------------

@app.function(
    gpu="A10G",
    timeout=300,
    volumes={
        "/model-cache": model_cache,
        "/adapter": adapter_volume,
    },
    memory=16384,
)
def format_chart_gpu(
    system_prompt: str,
    user_prompt: str,
    max_new_tokens: int = 2048,
) -> str:
    """Run Phi-3 chart formatter inference on Modal GPU."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel

    base_model_id = "microsoft/Phi-3-mini-4k-instruct"

    # Find adapter on the volume — list what's there for debugging
    import subprocess
    listing = subprocess.run(["find", "/adapter", "-maxdepth", "4", "-type", "f"], capture_output=True, text=True)
    print(f"[DEBUG] Files on adapter volume:\n{listing.stdout[:2000]}")

    # Try several possible paths
    adapter_candidates = [
        "/adapter/chart_formatter_lora/final_adapter",
        "/adapter/chart_formatter_lora",
        "/adapter/final_adapter",
        "/adapter",
    ]
    adapter_path = None
    for candidate in adapter_candidates:
        check = os.path.join(candidate, "adapter_model.safetensors")
        if os.path.exists(check):
            adapter_path = candidate
            print(f"[DEBUG] Found adapter at: {adapter_path}")
            break

    if not adapter_path:
        raise FileNotFoundError(
            f"Adapter not found. Checked: {adapter_candidates}. "
            f"Volume listing: {listing.stdout[:500]}"
        )

    # Log adapter config for verification
    adapter_config_path = os.path.join(adapter_path, "adapter_config.json")
    if os.path.exists(adapter_config_path):
        import json as _json
        with open(adapter_config_path) as f:
            cfg = _json.load(f)
        print(f"[DEBUG] Adapter config: base_model={cfg.get('base_model_name_or_path')}, "
              f"r={cfg.get('r')}, lora_alpha={cfg.get('lora_alpha')}, "
              f"inference_mode={cfg.get('inference_mode')}")

    # Load base model in 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
        cache_dir="/model-cache",
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        cache_dir="/model-cache",
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # Load LoRA adapter
    print(f"[DEBUG] Loading LoRA adapter from {adapter_path}...")
    try:
        model = PeftModel.from_pretrained(model, adapter_path)
        model.eval()
        # Verify adapter is actually active
        active_adapters = model.active_adapters if hasattr(model, 'active_adapters') else 'unknown'
        print(f"[DEBUG] LoRA adapter loaded successfully. Active adapters: {active_adapters}")
    except Exception as e:
        print(f"[ERROR] Failed to load LoRA adapter: {e}")
        raise

    # Build chat messages in Phi-3 format
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    inputs = tokenizer.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
            repetition_penalty=1.25,
            no_repeat_ngram_size=5,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the generated tokens (skip input)
    generated = outputs[0][inputs.shape[1]:]
    result = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return result


# ---------------------------------------------------------------------------
# Local entrypoint for testing
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main():
    """Test the formatter with a simple example."""
    import json

    system = "You are a professional chord chart formatter. Given raw chord detection data with timestamps, format a clean, readable chord chart."

    user = """Format this as a StemScriber JSON chord chart:

Title: Test Song
Artist: Test Artist
Key: Am
Tempo: 120 BPM
Time Signature: 4/4

Chord events:
[{"time":0.0,"duration":4.0,"chord":"Am","confidence":0.9},{"time":4.0,"duration":4.0,"chord":"G","confidence":0.85},{"time":8.0,"duration":4.0,"chord":"F","confidence":0.88},{"time":12.0,"duration":4.0,"chord":"E","confidence":0.9}]

Lyrics with timing:
[{"word":"Hello","start":0.5,"end":1.0},{"word":"darkness","start":1.2,"end":1.8},{"word":"my","start":2.0,"end":2.2},{"word":"old","start":2.3,"end":2.5},{"word":"friend","start":2.6,"end":3.0}]"""

    print("Calling Modal GPU function...")
    result = format_chart_gpu.remote(system, user)
    print("Result:")
    print(result)

    # Try to parse as JSON
    try:
        chart = json.loads(result)
        print(f"\nParsed successfully: {len(chart.get('sections', []))} sections")
    except json.JSONDecodeError:
        print("\nNote: output is not JSON (may be slash/chordpro format)")
