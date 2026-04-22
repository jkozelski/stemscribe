"""
Modal GPU training script for trimplexx CRNN guitar tablature model.

Uploads preprocessed GuitarSet data + raw audio (for augmentation) to a
Modal volume, trains the model on an A10G GPU using the run_72
hyperparameters (best known config), and downloads the trained checkpoint.

Usage:
  cd ~/stemscribe/train_tab_model
  ../../venv311/bin/python -m modal run modal_train_trimplexx.py

Deploy as persistent app (for monitoring):
  ../../venv311/bin/python -m modal deploy modal_train_trimplexx.py

Cost estimate: ~$4-5 for 3-4 hours on A10G ($1.10/hr)
"""

import modal
import os
import json
import time
import subprocess

# ---------------------------------------------------------------------------
# Modal setup
# ---------------------------------------------------------------------------

app = modal.App("trimplexx-guitar-training")

# Volume for preprocessed data + training artifacts
training_vol = modal.Volume.from_name("trimplexx-training-data", create_if_missing=True)

# Image with all training dependencies
training_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libsndfile1")
    .pip_install(
        "torch==2.5.1",
        "torchaudio==2.5.1",
        "librosa==0.11.0",
        "mirdata==0.3.9",
        "numpy==1.26.3",
        "scikit-learn==1.6.1",
        "tqdm==4.67.1",
        "pretty_midi>=0.2.10",
        "jams>=0.3.4",
        "noisereduce>=3.0.3",
        "matplotlib>=3.9.4",
        "mir_eval",
        "scipy",
    )
)

# ---------------------------------------------------------------------------
# Local paths
# ---------------------------------------------------------------------------

PROCESSED_DATA_DIR = os.path.join(
    os.path.dirname(__file__),
    "trimplexx", "python", "_processed_guitarset_data"
)

TRIMPLEXX_CODE_DIR = os.path.join(
    os.path.dirname(__file__),
    "trimplexx", "python"
)

GUITARSET_DATA_HOME = os.path.join(
    os.path.dirname(__file__),
    "trimplexx", "python", "_mir_datasets_storage"
)


@app.local_entrypoint()
def main():
    """Upload data, run training, download model."""
    import time as _time

    print("=" * 60)
    print("trimplexx CRNN Training Pipeline on Modal A10G")
    print("=" * 60)

    # Step 1: Upload preprocessed data + raw audio to volume
    print("\n[1/4] Uploading data to Modal volume...")
    upload_start = _time.time()
    _upload_data_to_volume()
    print(f"      Upload complete in {_time.time() - upload_start:.1f}s")

    # Step 2: Run training on GPU
    print("\n[2/4] Starting training on A10G GPU...")
    print("      Config: GRU, hidden=768, 2 layers, bidirectional")
    print("      Augmentations: time stretch, noise, reverb, EQ, clipping, SpecAugment")
    print("      Epochs: 300 (with early stopping, patience=25)")
    print("      Checkpoint metric: val_tdr_f1")
    train_start = _time.time()
    results = train_model.remote()
    train_duration = _time.time() - train_start
    print(f"\n      Training complete in {train_duration / 60:.1f} minutes")
    print(f"      Results: {json.dumps(results, indent=2)}")

    # Step 3: Download model
    print("\n[3/4] Downloading trained model...")
    _download_model()

    # Step 4: Summary
    print("\n[4/4] Done!")
    print(f"      Model saved to: ~/stemscribe/backend/models/pretrained/trimplexx_guitar_model.pt")
    print(f"      Config saved to: ~/stemscribe/backend/models/pretrained/trimplexx_run_config.json")
    print(f"      Total cost estimate: ${train_duration / 3600 * 1.10:.2f}")


def _upload_data_to_volume():
    """Upload preprocessed .pt files, raw audio, and code to Modal volume."""
    vol = modal.Volume.from_name("trimplexx-training-data", create_if_missing=True)

    with vol.batch_upload(force=True) as batch:
        # Upload preprocessed data directories (train/val/test .pt files)
        print("      Uploading preprocessed GuitarSet data...")
        for split in ["train", "validation", "test"]:
            split_dir = os.path.join(PROCESSED_DATA_DIR, split)
            if os.path.exists(split_dir):
                n_files = len([f for f in os.listdir(split_dir) if f.endswith(".pt")])
                print(f"        {split}: {n_files} files")
                batch.put_directory(split_dir, f"/data/processed/{split}")

        # Upload split ID files
        for split in ["train", "validation", "test"]:
            ids_file = os.path.join(PROCESSED_DATA_DIR, f"{split}_ids.txt")
            if os.path.exists(ids_file):
                batch.put_file(ids_file, f"/data/processed/{split}_ids.txt")

        # Upload raw GuitarSet audio (needed for audio augmentation during training)
        print("      Uploading GuitarSet raw audio for augmentation...")
        audio_dir = os.path.join(GUITARSET_DATA_HOME, "audio_mono-pickup_mix")
        if os.path.exists(audio_dir):
            wav_files = [f for f in os.listdir(audio_dir) if f.endswith(".wav")]
            print(f"        audio_mono-pickup_mix: {len(wav_files)} WAV files")
            for f in wav_files:
                batch.put_file(
                    os.path.join(audio_dir, f),
                    f"/data/guitarset/audio_mono-pickup_mix/{f}",
                )

        # Upload JAMS annotations
        print("      Uploading JAMS annotations...")
        annotation_dir = os.path.join(GUITARSET_DATA_HOME, "annotation")
        if os.path.exists(annotation_dir):
            jams_files = [f for f in os.listdir(annotation_dir) if f.endswith(".jams")]
            print(f"        annotation: {len(jams_files)} JAMS files")
            batch.put_directory(annotation_dir, "/data/guitarset/annotation")

        # Upload trimplexx Python code
        print("      Uploading trimplexx code...")
        code_dirs = ["model", "training", "evaluation", "data_processing", "vizualization"]
        for d in code_dirs:
            dir_path = os.path.join(TRIMPLEXX_CODE_DIR, d)
            if os.path.exists(dir_path):
                batch.put_directory(dir_path, f"/code/{d}")

        # Upload config.py and hyperparam file
        batch.put_file(os.path.join(TRIMPLEXX_CODE_DIR, "config.py"), "/code/config.py")
        hp_path = os.path.join(TRIMPLEXX_CODE_DIR, "hyperparam_set_v1.json")
        if os.path.exists(hp_path):
            batch.put_file(hp_path, "/code/hyperparam_set_v1.json")

    print("      Upload committed.")


def _download_model():
    """Download trained model from Modal volume."""
    vol = modal.Volume.from_name("trimplexx-training-data")

    output_dir = os.path.expanduser(
        "~/stemscribe/backend/models/pretrained"
    )
    os.makedirs(output_dir, exist_ok=True)

    # Download best model
    model_remote = "/artifacts/best_model.pth"
    model_local = os.path.join(output_dir, "trimplexx_guitar_model.pt")
    try:
        data = b""
        for chunk in vol.read_file(model_remote):
            data += chunk
        with open(model_local, "wb") as f:
            f.write(data)
        print(f"      Model: {len(data) / 1024 / 1024:.1f} MB -> {model_local}")
    except Exception as e:
        print(f"      ERROR downloading model: {e}")

    # Download run config
    config_remote = "/artifacts/run_configuration.json"
    config_local = os.path.join(output_dir, "trimplexx_run_config.json")
    try:
        data = b""
        for chunk in vol.read_file(config_remote):
            data += chunk
        with open(config_local, "wb") as f:
            f.write(data)
        print(f"      Config: {len(data)} bytes -> {config_local}")
    except Exception as e:
        print(f"      ERROR downloading config: {e}")

    # Download training log
    log_remote = "/artifacts/training_log.txt"
    log_local = os.path.join(output_dir, "trimplexx_training_log.txt")
    try:
        data = b""
        for chunk in vol.read_file(log_remote):
            data += chunk
        with open(log_local, "wb") as f:
            f.write(data)
        print(f"      Log: {len(data)} bytes -> {log_local}")
    except Exception as e:
        print(f"      (training log not found, skipping)")

    # Download test metrics
    metrics_remote = "/artifacts/test_metrics.json"
    metrics_local = os.path.join(output_dir, "trimplexx_test_metrics.json")
    try:
        data = b""
        for chunk in vol.read_file(metrics_remote):
            data += chunk
        with open(metrics_local, "wb") as f:
            f.write(data)
        print(f"      Test metrics: {len(data)} bytes -> {metrics_local}")
    except Exception as e:
        print(f"      (test metrics not found, skipping)")


# ---------------------------------------------------------------------------
# GPU training function
# ---------------------------------------------------------------------------

@app.function(
    image=training_image,
    gpu="A10G",
    volumes={"/vol": training_vol},
    timeout=6 * 3600,  # 6 hour max
    memory=32768,
)
def train_model():
    """Run full training pipeline on Modal A10G GPU."""
    import sys
    import torch
    import numpy as np
    import json
    import os
    import time

    # Add code to path
    sys.path.insert(0, "/vol/code")
    os.chdir("/vol/code")

    # Ensure matplotlib doesn't try to open display
    import matplotlib
    matplotlib.use("Agg")

    import config
    from model import architecture
    from training import pipeline, loss_functions, epoch_processing
    from evaluation import metrics as eval_metrics, performance_metrics
    from data_processing.dataset import GuitarSetTabDataset, create_frame_level_labels
    from data_processing.batching import collate_fn_pad

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Paths
    processed_data_dir = "/vol/data/processed"
    guitarset_data_home = "/vol/data/guitarset"
    artifacts_dir = "/vol/artifacts"
    os.makedirs(artifacts_dir, exist_ok=True)

    # Verify data exists
    for split in ["train", "validation", "test"]:
        split_dir = os.path.join(processed_data_dir, split)
        ids_file = os.path.join(processed_data_dir, f"{split}_ids.txt")
        n_files = len([f for f in os.listdir(split_dir) if f.endswith(".pt")]) if os.path.exists(split_dir) else 0
        print(f"  {split}: {n_files} .pt files, ids_file exists: {os.path.exists(ids_file)}")

    # Verify raw audio for augmentation
    audio_dir = os.path.join(guitarset_data_home, "audio_mono-pickup_mix")
    has_raw_audio = os.path.exists(audio_dir) and len([f for f in os.listdir(audio_dir) if f.endswith(".wav")]) > 0 if os.path.exists(audio_dir) else False
    if has_raw_audio:
        n_wavs = len([f for f in os.listdir(audio_dir) if f.endswith(".wav")])
        print(f"  Raw audio: {n_wavs} WAV files (augmentation enabled)")
        # Download mirdata index so it can resolve track IDs to audio paths
        import mirdata
        gs = mirdata.initialize("guitarset", data_home=guitarset_data_home)
        try:
            gs.download(partial_download=["index"])
            print("  mirdata index downloaded")
        except Exception as e:
            print(f"  mirdata index download issue: {e}")
    else:
        print("  WARNING: No raw audio found — audio augmentation disabled")

    # Build datasets with full augmentation (matching run_72)
    common_params = config.DATASET_COMMON_PARAMS.copy()

    print("\nBuilding datasets...")
    train_dataset = GuitarSetTabDataset(
        processed_data_base_dir=processed_data_dir,
        data_split_name="train",
        label_transform_function=create_frame_level_labels,
        guitarset_data_home=guitarset_data_home if has_raw_audio else None,
        enable_audio_augmentations=has_raw_audio,
        enable_specaugment=True,
        specaug_time_mask_param=40,
        specaug_freq_mask_param=26,
        **({
            "aug_p_time_stretch": 0.6,
            "aug_time_stretch_limits": (0.8, 1.2),
            "aug_p_add_noise": 0.7,
            "aug_noise_level_limits": (0.001, 0.01),
            "aug_p_random_gain": 0.7,
            "aug_gain_limits": (0.6, 1.4),
        } if has_raw_audio else {}),
        **common_params,
    )

    val_dataset = GuitarSetTabDataset(
        processed_data_base_dir=processed_data_dir,
        data_split_name="validation",
        label_transform_function=create_frame_level_labels,
        guitarset_data_home=guitarset_data_home if has_raw_audio else None,
        enable_audio_augmentations=False,
        enable_specaugment=False,
        **common_params,
    )

    test_dataset = GuitarSetTabDataset(
        processed_data_base_dir=processed_data_dir,
        data_split_name="test",
        label_transform_function=create_frame_level_labels,
        guitarset_data_home=guitarset_data_home if has_raw_audio else None,
        enable_audio_augmentations=False,
        enable_specaugment=False,
        **common_params,
    )

    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples")

    batch_size = config.BATCH_SIZE_DEFAULT  # 2

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn_pad,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_pad,
        num_workers=2,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_pad,
        num_workers=2,
        pin_memory=True,
    )

    # Best hyperparameters from run_72
    hyperparams = {
        "run_description": "stemscriber_run72_modal",
        "LEARNING_RATE_INIT": config.LEARNING_RATE_INIT_DEFAULT,  # 0.0003
        "ONSET_LOSS_WEIGHT": config.ONSET_LOSS_WEIGHT_DEFAULT,  # 9.0
        "ONSET_POS_WEIGHT_MANUAL_VALUE": config.ONSET_POS_WEIGHT_MANUAL_VALUE_DEFAULT,  # 6.0
        "RNN_TYPE": config.RNN_TYPE_DEFAULT,  # "GRU"
        "RNN_HIDDEN_SIZE": config.RNN_HIDDEN_SIZE_DEFAULT,  # 768
        "RNN_LAYERS": config.RNN_LAYERS_DEFAULT,  # 2
        "RNN_DROPOUT": config.RNN_DROPOUT_DEFAULT,  # 0.5
        "RNN_BIDIRECTIONAL": config.RNN_BIDIRECTIONAL_DEFAULT,  # True
        "WEIGHT_DECAY": config.WEIGHT_DECAY_DEFAULT,  # 0.0001
        "SCHEDULER_PATIENCE": config.SCHEDULER_PATIENCE_DEFAULT,  # 10
        "SCHEDULER_FACTOR": config.SCHEDULER_FACTOR_DEFAULT,  # 0.2
    }

    # Calculate CNN output dimension
    temp_cnn = architecture.TabCNN()
    with torch.no_grad():
        dummy = torch.randn(1, 1, config.N_BINS_CQT, 32)
        cnn_out = temp_cnn(dummy)
        cnn_out_dim = cnn_out.shape[1] * cnn_out.shape[2]
    del temp_cnn
    print(f"\nCNN output dim: {cnn_out_dim}")

    # Build model
    model = architecture.GuitarTabCRNN(
        num_frames_rnn_input_dim=cnn_out_dim,
        rnn_type=hyperparams["RNN_TYPE"],
        rnn_hidden_size=hyperparams["RNN_HIDDEN_SIZE"],
        rnn_layers=hyperparams["RNN_LAYERS"],
        rnn_dropout=hyperparams["RNN_DROPOUT"],
        rnn_bidirectional=hyperparams["RNN_BIDIRECTIONAL"],
    )
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {total_params:,} total, {trainable_params:,} trainable")
    print(f"Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB (float32)")

    # Loss, optimizer, scheduler
    onset_pos_weight = torch.tensor([hyperparams["ONSET_POS_WEIGHT_MANUAL_VALUE"]], device=device)
    criterion = loss_functions.CombinedLoss(
        onset_pos_weight=onset_pos_weight,
        onset_loss_weight=hyperparams["ONSET_LOSS_WEIGHT"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=hyperparams["LEARNING_RATE_INIT"],
        weight_decay=hyperparams["WEIGHT_DECAY"],
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=hyperparams["SCHEDULER_FACTOR"],
        patience=hyperparams["SCHEDULER_PATIENCE"],
    )

    # Save run config
    run_config = {
        "static_parameters": {
            "SAMPLE_RATE": config.SAMPLE_RATE,
            "HOP_LENGTH": config.HOP_LENGTH,
            "MAX_FRETS": config.MAX_FRETS,
            "N_BINS_CQT": config.N_BINS_CQT,
            "BINS_PER_OCTAVE_CQT": config.BINS_PER_OCTAVE_CQT,
            "VALIDATION_SPLIT_SIZE": config.VALIDATION_SPLIT_SIZE,
            "TEST_SPLIT_SIZE": config.TEST_SPLIT_SIZE,
            "RANDOM_SEED": config.RANDOM_SEED,
        },
        "default_training_parameters": {
            "NUM_EPOCHS_DEFAULT": config.NUM_EPOCHS_DEFAULT,
            "BATCH_SIZE_DEFAULT": config.BATCH_SIZE_DEFAULT,
            "FRET_LOSS_WEIGHT_DEFAULT": config.FRET_LOSS_WEIGHT_DEFAULT,
            "EARLY_STOPPING_PATIENCE_DEFAULT": config.EARLY_STOPPING_PATIENCE_DEFAULT,
            "CHECKPOINT_METRIC_DEFAULT": config.CHECKPOINT_METRIC_DEFAULT,
        },
        "hyperparameters_tuned": hyperparams,
        "augmentations": {
            "enable_audio_augmentations": has_raw_audio,
            "enable_specaugment": True,
            **({"aug_p_time_stretch": 0.6, "aug_p_add_noise": 0.7, "aug_p_random_gain": 0.7,
                "reverb_params": {"enabled": True, "probability": 0.4},
                "eq_params": {"enabled": True, "probability": 0.5},
                "clipping_params": {"enabled": True, "probability": 0.3},
            } if has_raw_audio else {}),
        },
    }

    config_path = os.path.join(artifacts_dir, "run_configuration.json")
    with open(config_path, "w") as f:
        json.dump(run_config, f, indent=4)

    # Training loop
    log_path = os.path.join(artifacts_dir, "training_log.txt")
    training_config = {
        **hyperparams,
        "ARTIFACTS_DIR": artifacts_dir,
        "LOG_FILE_PATH": log_path,
        "CHECKPOINT_METRIC": config.CHECKPOINT_METRIC_DEFAULT,
        "EARLY_STOPPING_PATIENCE": config.EARLY_STOPPING_PATIENCE_DEFAULT,
        "NUM_EPOCHS": config.NUM_EPOCHS_DEFAULT,
    }

    with open(log_path, "w") as f:
        f.write(f"--- StemScriber trimplexx Training Run ---\n")
        f.write(f"Device: {device}\n")
        if torch.cuda.is_available():
            f.write(f"GPU: {torch.cuda.get_device_name(0)}\n")
        f.write(json.dumps(run_config, indent=4))
        f.write("\n\n" + "=" * 50 + "\n")

    print(f"\nStarting training: {config.NUM_EPOCHS_DEFAULT} epochs, batch_size={batch_size}")
    print(f"Early stopping patience: {config.EARLY_STOPPING_PATIENCE_DEFAULT}")
    print(f"Checkpoint metric: {config.CHECKPOINT_METRIC_DEFAULT}")
    print("=" * 60)

    train_start = time.time()

    training_history = pipeline.run_training_loop(
        model_instance=model,
        device_to_use=device,
        train_loader=train_loader,
        validation_loader=val_loader,
        optimizer_instance=optimizer,
        scheduler_instance=scheduler,
        criterion_instance_combined=criterion,
        run_training_config=training_config,
        config_obj=config,
    )

    train_duration = time.time() - train_start
    stopped_epoch = len(training_history.get("train_total_loss", []))
    print(f"\nTraining finished after {stopped_epoch} epochs in {train_duration / 60:.1f} minutes")

    # Get best metrics from training history
    tdr_history = training_history.get("val_tdr_f1_at_0.5", [])
    mpe_history = training_history.get("val_mpe_f1", [])
    best_val_tdr = max(tdr_history) if tdr_history else 0.0
    best_val_mpe = max(mpe_history) if mpe_history else 0.0
    print(f"Best val TDR F1: {best_val_tdr:.4f}")
    print(f"Best val MPE F1: {best_val_mpe:.4f}")

    # Evaluate on test set
    print("\nEvaluating on test set...")
    best_model_path = os.path.join(artifacts_dir, "best_model.pth")
    test_metrics = {}

    if os.path.exists(best_model_path):
        from model import utils as model_utils
        loaded_model = model_utils.load_best_model(
            model_class=architecture.GuitarTabCRNN,
            model_path=best_model_path,
            run_config_path=config_path,
            device=device,
        )
        if loaded_model:
            test_metrics = performance_metrics.evaluate_model_on_test_set(
                model_to_eval=loaded_model,
                test_dataloader=test_loader,
                device_to_use=device,
                config_obj=config,
                optimal_onset_threshold=0.5,
            )
            print(f"Test metrics: {json.dumps(test_metrics, indent=2)}")

            # Save test metrics
            test_metrics_path = os.path.join(artifacts_dir, "test_metrics.json")
            with open(test_metrics_path, "w") as f:
                json.dump(test_metrics, f, indent=4)
    else:
        print("WARNING: No best_model.pth found!")

    # Commit volume so artifacts persist
    training_vol.commit()

    return {
        "stopped_epoch": stopped_epoch,
        "train_duration_minutes": train_duration / 60,
        "best_val_tdr_f1": best_val_tdr,
        "best_val_mpe_f1": best_val_mpe,
        "test_metrics": test_metrics,
        "model_path": best_model_path,
    }
