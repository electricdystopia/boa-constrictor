import argparse
import os
import time
from pathlib import Path
from networkx import config
import yaml
import numpy as np
import torch
from tqdm import tqdm

from model import BoaConstrictor, ByteDataloader, make_splits
from boa import BOA
from train import train


def load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, 'r') as f:
        return yaml.safe_load(f) or {}


def resolve_config_path(config_arg: str, experiments_root: Path = Path('experiments')) -> Path:
    """Resolve a --config argument which may be a path or an experiment name.

    Order:
      1. If the argument is an existing file path, return it.
      2. If it's a simple experiment name (no existing path), look for
         experiments/<name>/<name>.yaml and return if exists.
      3. Fallback to configs/<name>.yaml if present.
      4. Raise FileNotFoundError.
    """
    if config_arg is None:
        return None
    p = Path(config_arg)
    # Direct file path provided
    if p.exists():
        return p

    # Try experiments/<name>/<name>.yaml
    name = p.stem
    exp_cfg = experiments_root / name / f"{name}.yaml"
    if exp_cfg.exists():
        return exp_cfg

    # Try configs/<name>.yaml
    cfg_cfg = Path('configs') / f"{name}.yaml"
    if cfg_cfg.exists():
        return cfg_cfg

    raise FileNotFoundError(f"Could not resolve config argument '{config_arg}' to a config file")


def parse_args():
    p = argparse.ArgumentParser(description="Run BoaConstrictor experiments from a config file")
    p.add_argument('--config', '-c', type=Path, required=False, help='Path to YAML experiment config')
    p.add_argument('--no-progress', action='store_true', help='Disable progress bars')
    p.add_argument('--device', type=str, default="cuda", help='Torch device override (cpu|cuda)')
    p.add_argument('--precision', type=str, default="fp32", choices=['fp32','fp16', 'fp8'], help='Precision override')
    p.add_argument('--new-experiment', action='store_true', help='Create a new experiment config interactively and run it')
    p.add_argument('--train-only', action='store_true', help='Only run training')
    p.add_argument('--compress-only', action='store_true', help='Only run compression')
    p.add_argument('--decompress-only', action='store_true', help='Only run decompression')
    p.add_argument('--show-timings', action='store_true', help='Print timings for each major operation')
    p.add_argument('--verify', action='store_true', help='After decompression, verify bytes match the input file used for compression')
    p.add_argument('--evaluate', action='store_true', help='After decompression, run evaluation metrics on the compressor model')
    p.add_argument('--evaluate-only', action='store_true', help='After decompression, run evaluation metrics on the compressor model')
    p.add_argument('--comparison-baseline-only', action='store_true', help='Run LZMA and ZLIB (ultra) baseline compressions on the compression input file, print results, and exit')
    p.add_argument('--model-path', type=str, default=None, help='Path to a pre-trained model .pt file (state_dict or full model). If provided, training is skipped and the model is loaded')
    return p.parse_args()


def main():
    args = parse_args()

    # If user requests a new experiment, run interactive creator and obtain a config path
    if args.new_experiment:
        def _prompt(prompt, default=None, cast=str):
            if default is None:
                resp = input(f"{prompt}: ").strip()
            else:
                resp = input(f"{prompt} [{default}]: ").strip()
                if resp == "":
                    resp = str(default)
            try:
                return cast(resp)
            except Exception:
                return resp

        print("Creating a new experiment config interactively. Press enter to accept the default shown in brackets.")
        name = _prompt("Experiment name", "example_experiment")
        file_path = _prompt("Path to dataset file (binary)", "/path/to/dataset.bin")
        progress = _prompt("Show progress bars (true/false)", "true", lambda s: s.lower() in ("1","true","yes"))
        device = _prompt("Device (cpu|cuda)", "cuda")
        precision = _prompt("Precision (fp32|fp16|fp8)", "fp32")
        seq_len = _prompt("Sequence length (seq_len)", 32768, int)
        batch_size = _prompt("Batch size", 3, int)
        d_model = _prompt("Model d_model", 256, int)
        num_layers = _prompt("Model num_layers", 2, int)
        lr = _prompt("Learning rate", 5e-4, float)
        epochs = _prompt("Epochs", 10, int)
        chunks_count = _prompt("Compression chunks_count", 1000, int)
        use_vocab_subset = _prompt("Use vocab subset (true/false)", "false", lambda s: s.lower() in ("1","true","yes"))
        compress_file = _prompt("File to compress (leave blank to use dataset file)", "", lambda s: s if s != "" else "")
        splits_in = _prompt("Data splits as comma-separated (train,val,test)", "0.8,0.1,0.1")
        try:
            splits = [float(x.strip()) for x in splits_in.split(',')]
            if len(splits) != 3 or abs(sum(splits) - 1.0) > 1e-6:
                print("Warning: splits do not sum to 1. Using default [0.8,0.1,0.1].")
                splits = [0.8, 0.1, 0.1]
        except Exception:
            splits = [0.8, 0.1, 0.1]

        cfg = {
            'name': name,
            'file_path': file_path,
            'progress': bool(progress),
            'device': device,
            'precision': precision,
            'dataloader': {'seq_len': int(seq_len), 'batch_size': int(batch_size)},
            'model': {'d_model': int(d_model), 'num_layers': int(num_layers)},
            'training': {'lr': float(lr), 'epochs': int(epochs)},
            'compression': {'chunks_count': int(chunks_count), 'file_to_compress': compress_file},
            'use_vocab_subset': bool(use_vocab_subset),
            'splits': splits
        }

        # Decide where to save the config: store it under experiments/<name>/<name>.yaml
        cfg_path = Path('experiments') / name / f"{name}.yaml"
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cfg_path, 'w') as f:
            yaml.safe_dump(cfg, f)
        print(f"Wrote new experiment config to: {cfg_path}")

        # Use the newly created config for the rest of the run
        args.config = str(cfg_path)

    if args.config is None:
        raise ValueError('Either --config must be provided or use --new-experiment to create one interactively')

    # Resolve the config argument: allow passing an experiment name which maps
    # to experiments/<name>/<name>.yaml, or a direct path.
    args.config = resolve_config_path(str(args.config))
    config = load_config(args.config)

    # Apply CLI overrides
    progress = not args.no_progress and config.get('progress', True)
    # device =  config.get('device', 'cuda' if torch.cuda.is_available() else 'cuda') or args.device
    device = args.device or config.get('device', 'cpu')
    
    print(device)
    precision = args.precision or config.get('precision', 'fp32')
    verify = args.verify or bool(config.get('verify', False))
    # Model path can be provided via CLI or config (either top-level 'model_path' or under 'model.path')
    model_path_cfg = config.get('model_path') or config.get('model', {}).get('path')
    model_path = Path(args.model_path).expanduser() if args.model_path else (Path(model_path_cfg).expanduser() if model_path_cfg else None)
    if model_path is not None:
        try:
            cfg_dir = Path(args.config).parent if args.config is not None else Path.cwd()
        except Exception:
            cfg_dir = Path.cwd()
        if not model_path.is_absolute():
            model_path = (cfg_dir / model_path).resolve()

    # Experiment parameters (with sensible defaults)
    # Use the config filename stem as the canonical experiment/model name
    # so checkpoints are consistently named and retraining can be skipped.
    name = Path(args.config).stem
    file_path = config.get('file_path', '')
    # Resolve file_path: if it's absolute, use as-is; if relative, interpret
    # it relative to the directory of the resolved config file (so passing
    # --config <experiment_name> works and paths inside the YAML are relative
    # to that YAML file).
    if file_path:
        file_path = Path(file_path)
        try:
            cfg_dir = Path(args.config).parent if args.config is not None else Path.cwd()
        except Exception:
            cfg_dir = Path.cwd()
        if not file_path.is_absolute():
            file_path = (cfg_dir / file_path).resolve()
    seq_len = config.get('dataloader', {}).get('seq_len', 32768)
    batch_size = config.get('dataloader', {}).get('batch_size', 3)
    d_model = config.get('model', {}).get('d_model', 256)
    num_layers = config.get('model', {}).get('num_layers', 8)
    lr = float(config.get('training', {}).get('lr', 5e-4))
    num_epochs = config.get('training', {}).get('epochs', 50)
    use_vocab_subset = config.get('use_vocab_subset', False)

    timings = {}

    # Read file
    t0 = time.perf_counter()
    if not file_path:
        raise ValueError('file_path must be set in the config or passed via CLI')

    # file_path is already a Path (resolved above when possible)
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    with open(file_path, 'rb') as f:
        data_bytes = f.read()

    timings['read_bytes'] = time.perf_counter() - t0
    print(f"Read {len(data_bytes)} bytes from {file_path} in {timings['read_bytes']:.2f}s")
    compress_file_cfg = config.get('compression', {}).get('file_to_compress', '')
    # If blank, use the original dataset file we already loaded
    if not compress_file_cfg:
        compress_file_path = file_path
    else:
        # Resolve compress_file relative to config dir when relative
        cfp = Path(compress_file_cfg)
        cfg_dir = Path(args.config).parent if args.config is not None else Path.cwd()
        if not cfp.is_absolute():
            cfp = (cfg_dir / cfp).resolve()
        if not cfp.exists():
            raise FileNotFoundError(f"Compression input file not found: {cfp}")
        compress_file_path = cfp

    # Compute vocabulary and remap (training data only)
    if use_vocab_subset:
        unique_bytes = sorted(list(set(data_bytes)))
        vocab_size = len(unique_bytes)
        print(f"Using vocab subset of size {vocab_size} out of 256 possible bytes.")
        byte_to_idx = {b: i for i, b in enumerate(unique_bytes)}
        idx_to_byte = {i: b for i, b in enumerate(unique_bytes)}

        # Remap training data
        arr = np.frombuffer(data_bytes, dtype=np.uint8)
        lookup = np.zeros(256, dtype=np.uint8)
        for b, idx in byte_to_idx.items():
            lookup[b] = idx
        data_bytes = lookup[arr].tobytes()
    else:
        vocab_size = 256
        unique_bytes = None
        byte_to_idx = None
        idx_to_byte = None
        lookup = None

    # Prepare experiment output directory and filenames (needed before optional training)
    experiments_root = Path(config.get('experiments_root', 'experiments'))
    exp_dir = experiments_root / name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Setup model, dataloaders, optimizer, loss
    model = BoaConstrictor(d_model=d_model, num_layers=num_layers, vocab_size=vocab_size, device=device)

    dataloader = ByteDataloader(data_bytes, seq_len=seq_len, batch_size=batch_size, device=device)

    train_b, val_b, test_b = make_splits(data_bytes, dataloader.seq_len, dataloader.batch_size,
                                         splits=tuple(config.get('splits', (0.8, 0.1, 0.1))))

    train_loader = ByteDataloader(train_b, seq_len=dataloader.seq_len, batch_size=dataloader.batch_size, device=device)
    val_loader = ByteDataloader(val_b, seq_len=dataloader.seq_len, batch_size=dataloader.batch_size, device=device)
    test_loader = ByteDataloader(test_b, seq_len=dataloader.seq_len, batch_size=dataloader.batch_size, device=device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # If a model path is provided and exists, load it and skip training
    def _load_model_from_path(model, path: Path):
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        obj = torch.load(path, map_location='cpu')
        try:
            # Try state_dict first
            if isinstance(obj, dict) and all(isinstance(k, str) for k in obj.keys()):
                model.load_state_dict(obj, strict=False)
                return model
            # If whole model was saved
            if hasattr(obj, 'state_dict') and hasattr(obj, 'parameters'):
                return obj
        except Exception:
            pass
        # Fallback: if torch.save was used with state_dict under a key
        if isinstance(obj, dict) and 'state_dict' in obj:
            model.load_state_dict(obj['state_dict'], strict=False)
            return model
        raise ValueError(f"Unrecognized checkpoint format at {path}")

    # If no explicit model_path was provided, check for an existing final checkpoint
    start_epoch = 1
    resume_training = False
    
    default_ckpt = exp_dir / f"{name}_final_model_{precision}.pt"
    if model_path is None:
        if default_ckpt.exists():
            model_path = default_ckpt
            print(f"Found existing checkpoint at {model_path}. Will load and skip training.")
        else:
            # Check for intermediate checkpoints
            candidates = list(exp_dir.glob(f"{name}_*_Checkpoint_epoch_*_{precision}.pt"))
            if candidates:
                def _get_epoch_from_path(p: Path) -> int:
                    parts = p.stem.split('_')
                    if 'epoch' in parts:
                        try:
                            return int(parts[parts.index('epoch') + 1])
                        except (ValueError, IndexError):
                            pass
                    return 0

                latest_ckpt = max(candidates, key=_get_epoch_from_path)
                latest_epoch = _get_epoch_from_path(latest_ckpt)
                
                if latest_epoch > 0:
                    model_path = latest_ckpt
                    if latest_epoch < num_epochs:
                        start_epoch = latest_epoch + 1
                        resume_training = True
                        print(f"Found intermediate checkpoint at {model_path}. Resuming training from epoch {start_epoch}.")
                    else:
                        print(f"Found checkpoint at {model_path} (epoch {latest_epoch} >= {num_epochs}). Will load and skip training.")

    # Training or loading
    if model_path is not None and Path(model_path).exists() and not args.comparison_baseline_only:
        print(f"Loading pre-trained model from {model_path}")
        if not resume_training:
            print("Skipping training (final model or explicit path provided).")
        
        t_start = time.perf_counter()
        model = _load_model_from_path(model, Path(model_path))
        model = model.to(device)
        timings['load_model'] = time.perf_counter() - t_start
        print(f"Model loaded in {timings['load_model']:.2f}s")

    if not args.compress_only and not args.decompress_only and not args.comparison_baseline_only:
        if model_path is None or resume_training:
            print(f"Starting training on device=={device}, precision={precision}, epochs={num_epochs}, start_epoch={start_epoch}")
            t_start = time.perf_counter()
            train(model, train_loader, val_loader, test_loader, optimizer, criterion,
                  device=device, name=str(exp_dir / name), NUM_EPOCHS=num_epochs, PRECISION=precision, progress=progress, start_epoch=start_epoch, vocab_size=vocab_size)
            timings['training'] = time.perf_counter() - t_start
            print(f"Training complete in {timings['training']:.2f}s")

            # After successful training, persist model_path into the YAML config
            trained_ckpt = exp_dir / f"{name}_final_model_{precision}.pt"
            final_model_name = f"{name}_final_model_{precision}.pt"
            try:
                cfg_path: Path = Path(args.config)
                # Write model_path relative to the config directory when possible
                rel_ckpt = trained_ckpt
                if trained_ckpt.is_absolute():
                    try:
                        rel_ckpt = trained_ckpt.relative_to(cfg_path.parent)
                    except Exception:
                        rel_ckpt = trained_ckpt
                with open(cfg_path, 'r') as f:
                    cfg_data = yaml.safe_load(f) or {}
                cfg_data['model_path'] = str(final_model_name)
                with open(cfg_path, 'w') as f:
                    yaml.safe_dump(cfg_data, f)
                print(f"Updated config with model_path: {rel_ckpt}")
            except Exception as e:
                print(f"[WARN] Failed to update config with model_path: {e}")

    compress_file_cfg = config.get('compression', {}).get('file_to_compress', '')
    # If blank, use the original dataset file we already loaded
    if not compress_file_cfg:
        compress_file_path = file_path
    else:
        # Resolve compress_file relative to config dir when relative
        cfp = Path(compress_file_cfg)
        cfg_dir = Path(args.config).parent if args.config is not None else Path.cwd()
        if not cfp.is_absolute():
            cfp = (cfg_dir / cfp).resolve()
        if not cfp.exists():
            raise FileNotFoundError(f"Compression input file not found: {cfp}")
        compress_file_path = cfp

    # If the user only wants baseline comparisons, run quick LZMA and ZLIB (ultra) compressions
    # on the compression input file and print results, then exit.
    def _run_baseline_comparisons(in_path: Path, out_dir: Path, exp_name: str):
        import lzma
        import zlib
        import uproot
        import time

        with open(in_path, 'rb') as rf:
            data = rf.read()

        orig_size = len(data)
        results = {}

        # LZMA (try EXTREME if available, fall back to preset=9)
        try:
            t0 = time.perf_counter()
            try:
                comp_lz = lzma.compress(data, preset=9 | getattr(lzma, 'PRESET_EXTREME', 0))
            except Exception:
                comp_lz = lzma.compress(data, preset=9)
            t_lz = time.perf_counter() - t0
            lz_size = len(comp_lz)
            lz_path = out_dir / f"{exp_name}.lzma"
            with open(lz_path, 'wb') as wf:
                wf.write(comp_lz)
            results['lzma'] = {'path': str(lz_path), 'size': lz_size, 'time_s': t_lz}
        except Exception as e:
            results['lzma'] = {'error': str(e)}

        # ZLIB (max compression level = 9)
        try:
            t0 = time.perf_counter()
            comp_z = zlib.compress(data, level=9)
            t_z = time.perf_counter() - t0
            z_size = len(comp_z)
            z_path = out_dir / f"{exp_name}.zlib"
            with open(z_path, 'wb') as wf:
                wf.write(comp_z)
            results['zlib'] = {'path': str(z_path), 'size': z_size, 'time_s': t_z}
        except Exception as e:
            results['zlib'] = {'error': str(e)}
        if config.get('baseline', {}).get('rntuple', False):
            try:
                rntuple_path = out_dir / f"{exp_name}.root"
                t0 = time.perf_counter()
                file = uproot.recreate(rntuple_path)
                rn_data = np.frombuffer(data_bytes)
                file.mkrntuple("tuple6", {"data": rn_data})
                file.close()
                t_rntuple = time.perf_counter() - t0
                rntuple_size = os.path.getsize(rntuple_path)
                results['rntuple'] = {'path': str(rntuple_path), 'size': rntuple_size, 'time_s': t_rntuple}
            except Exception as e:
                results['rntuple'] = {'error': str(e)}

        # Print a concise summary
        print("\nBaseline compression results:")
        print(f"  Original size: {orig_size} bytes")
        for k in ('lzma', 'zlib', 'rntuple'):
            r = results.get(k, {})
            if 'error' in r:
                print(f"  {k.upper()}: ERROR: {r['error']}")
                continue
            size = r['size']
            t = r['time_s']
            ratio = orig_size / size if size > 0 else float('inf')
            print(f"  {k.upper():5} -> size={size} bytes, ratio={ratio:.2f}, time={t:.3f}s, written={r.get('path')}")

        return results

    if args.comparison_baseline_only:
        try:
            os.makedirs(exp_dir, exist_ok=True)
            _run_baseline_comparisons(compress_file_path, exp_dir, name)
            print("\n--comparison-baseline-only complete. Exiting.")
            return
        except Exception as e:
            print(f"[ERROR] Baseline comparison failed: {e}")
            return

    boa = BOA(device, str(exp_dir / f"{name}.boa"), model)
    file_format = compress_file_path.suffix.lstrip('.') or 'bin'
    # Compression
    if not args.train_only and not args.decompress_only and not args.evaluate_only:
        print("Starting compression...")
        
        target_compress_path = compress_file_path
        temp_compress_path = None
        
        if vocab_size < 256:
            print(f"Remapping compression input to {vocab_size} vocab size...")
            with open(compress_file_path, 'rb') as f:
                c_data = f.read()
            
            # Check if all bytes in c_data are in vocab
            c_unique = set(c_data)
            if not c_unique.issubset(set(unique_bytes)):
                 print("[ERROR] Compression input contains bytes not seen in training data! Cannot compress.")
                 target_compress_path = None
            else:
                c_arr = np.frombuffer(c_data, dtype=np.uint8)
                c_remapped = lookup[c_arr].tobytes()
                temp_compress_path = exp_dir / f"temp_remapped_{compress_file_path.name}"
                with open(temp_compress_path, 'wb') as f:
                    f.write(c_remapped)
                target_compress_path = temp_compress_path

        if target_compress_path:
            t_start = time.perf_counter()
            # Create BOA that writes into the experiment directory
            boa.compress(
                data_path=str(target_compress_path),
                chunks_count=config.get('compression', {}).get('chunks_count', 1000),
                progress=progress,
            )
            with open(exp_dir / f"{name}.boa", 'rb') as bf:
                boa_size = len(bf.read())
            with open(compress_file_path, 'rb') as rf:
                original_size = len(rf.read())
            compression_ratio = original_size / boa_size if boa_size > 0 else float('inf')
            print(f"Compression ratio: {compression_ratio:.2f}")

            timings['compression'] = time.perf_counter() - t_start
            print(f"Compression complete in {timings['compression']:.2f}s")
            
            if temp_compress_path and temp_compress_path.exists():
                temp_compress_path.unlink()

    # Decompression (write decompressed bytes into the experiment directory)
    if not args.train_only and not args.compress_only and not args.evaluate_only:
        print("Starting decompression...")
        t_start = time.perf_counter()
        # BoaFile.decompress() returns the original bytes (which are remapped indices here)
        decompressed_bytes = boa.decompress(progress=progress)
        
        if vocab_size < 256:
             print(f"Remapping decompressed output back to original bytes...")
             # Inverse mapping
             inv_lookup = np.zeros(256, dtype=np.uint8)
             for idx, b in idx_to_byte.items():
                 inv_lookup[idx] = b
             
             d_arr = np.frombuffer(decompressed_bytes, dtype=np.uint8)
             decompressed_bytes = inv_lookup[d_arr].tobytes()

        out_path = exp_dir / f"{name}_decompressed.{file_format}"
        with open(out_path, 'wb') as outf:
            outf.write(decompressed_bytes)
        timings['decompression'] = time.perf_counter() - t_start
        print(f"Decompression complete in {timings['decompression']:.2f}s")

        # Optional verification: compare decompressed bytes with original compression input
        if verify:
            # Compare against the bytes we actually compressed (compress_file_path)
            with open(compress_file_path, 'rb') as rf:
                ref_bytes = rf.read()
            same = decompressed_bytes == ref_bytes
            if same:
                print(f"VERIFY: OK — decompressed output matches input ({len(decompressed_bytes)} bytes)")
            else:
                # Provide small diagnostic: print sizes and first mismatch position (bounded)
                print("VERIFY: MISMATCH — decompressed output differs from input")
                if len(decompressed_bytes) != len(ref_bytes):
                    print(f"  Sizes differ: decompressed={len(decompressed_bytes)} vs input={len(ref_bytes)}")
                else:
                    # Find first mismatch up to a cap
                    cap = min(len(decompressed_bytes), 1_000_000)
                    for i in range(cap):
                        if decompressed_bytes[i] != ref_bytes[i]:
                            print(f"  First differing byte at offset {i}: dec={decompressed_bytes[i]} input={ref_bytes[i]}")
                            break

    # Note: configs are stored under experiments/<name>/<name>.yaml when created
    # and can be referenced by experiment name via --config <name>. No copy is necessary.
    if (args.evaluate or args.evaluate_only) and torch.cuda.is_available():
        from evaluator import CompressionEvaluator
        print("Starting evaluation...")
        print("Loading model and data...")

        # Data
        with open(compress_file_path, 'rb') as rf:
            data_bytes = rf.read()
            print(f"Data loaded: {len(data_bytes)/1024/1024:.2f} MB")

        if use_vocab_subset and vocab_size < 256 and lookup is not None:
            print(f"Remapping evaluation data to {vocab_size} vocab size...")
            arr = np.frombuffer(data_bytes, dtype=np.uint8)
            data_bytes = lookup[arr].tobytes()


        # Splits
        n = len(data_bytes)
        train_end = int(0.8 * n)
        val_end   = int(0.9 * n)
        train_bytes = data_bytes[:train_end]
        val_bytes   = data_bytes[train_end:val_end]
        test_bytes  = data_bytes[val_end:]

        eval_seq_len = 1024
        eval_batch_size = 1
        train_loader = ByteDataloader(train_bytes, seq_len=eval_seq_len, batch_size=eval_batch_size)
        val_loader   = ByteDataloader(val_bytes,   seq_len=eval_seq_len, batch_size=eval_batch_size)
        test_loader  = ByteDataloader(test_bytes,  seq_len=eval_seq_len, batch_size=eval_batch_size)

        # Evaluate & plot all on one figure
        evaluator = CompressionEvaluator(model, device=device)
        os.makedirs(f"experiments/{name}/plots", exist_ok=True)
        curves = evaluator.plot_calibration_curves_multi(
            {"train": train_loader, "val": val_loader, "test": test_loader},
            n_bins=20,
            max_batches=20,            # subset for speed
            savepath=f"experiments/{name}/plots/calibration_all.png",
            quantile_bins=False        # set True for equal-mass bins
        )
        res = evaluator.plot_topk_accuracy(
            test_loader, k_max=20, step=1,
            savepath=f"experiments/{name}/plots/top_k_accuracy.png",
            annotate_ks=(1, 5, 10)
        )
        res = evaluator.plot_confusion_top_bytes(test_loader, top_n=20, normalize="true",
                                savepath=f"experiments/{name}/plots/byte_confusion_matrix.png")
        # Also plot original vs decompressed comparison for first few columns to show bit-exactness
        try:
            decompressed_path = exp_dir / f"{name}_decompressed.{file_format}"
            if decompressed_path.exists():
                evaluator.plot_bit_exact_columns(
                    original_file=str(compress_file_path),
                    decompressed_file=str(decompressed_path),
                    num_cols=4,
                    dtype='float32',
                    max_rows=2000,
                    savepath=f"experiments/{name}/plots/bit_exact_columns.png",
                )
            else:
                print(f"[INFO] Decompressed file not found at {decompressed_path}; skipping bit-exact columns plot.")
        except Exception as e:
            print(f"[WARN] Failed to generate bit-exact columns plot: {e}")
        print("Evaluation complete.")
    elif not torch.cuda.is_available() and (args.evaluate or args.evaluate_only):
        print("[WARN] Evaluation requires CUDA; skipping evaluation as no CUDA device is available.")
        
    if args.show_timings:
        print('\nTimings:')
        for k, v in timings.items():
            print(f"  {k}: {v:.2f}s")


if __name__ == '__main__':
    main()


