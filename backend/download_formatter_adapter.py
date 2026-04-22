"""Download the trained chart formatter LoRA adapter from Modal volume."""
import modal
import os
import json
from pathlib import Path

output_volume = modal.Volume.from_name("chart-formatter-output")
app = modal.App("formatter-adapter-download")


@app.function(volumes={"/output": output_volume}, timeout=300)
def download_adapter() -> dict:
    output_volume.reload()
    adapter_path = "/output/chart_formatter_lora/final_adapter"
    files = {}
    if not os.path.exists(adapter_path):
        print(f"ERROR: {adapter_path} not found!")
        # Try listing what's there
        for root, dirs, fnames in os.walk("/output"):
            for f in fnames:
                print(f"  Found: {os.path.join(root, f)}")
        return files

    for fname in os.listdir(adapter_path):
        fpath = os.path.join(adapter_path, fname)
        if os.path.isfile(fpath):
            with open(fpath, "rb") as f:
                files[fname] = f.read()
            print(f"  Packed: {fname} ({os.path.getsize(fpath) / 1024:.1f} KB)")
    return files


@app.local_entrypoint()
def main():
    print("Downloading formatter adapter from Modal volume...")
    adapter_files = download_adapter.remote()

    if not adapter_files:
        print("No adapter files found!")
        return

    local_dir = Path(__file__).parent / "models" / "pretrained" / "chart_formatter_lora"
    local_dir.mkdir(parents=True, exist_ok=True)

    for fname, fbytes in adapter_files.items():
        out_path = local_dir / fname
        out_path.write_bytes(fbytes)
        print(f"  Saved: {out_path} ({len(fbytes) / 1024:.1f} KB)")

    print(f"\nDone! Adapter at: {local_dir}")
