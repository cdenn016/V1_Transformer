"""
Checkpoint Utilities
====================

Save, load, and inspect GaugeTransformerLM checkpoints.

Handles legacy config migration (kappa_beta_base -> kappa_beta,
diagonal_covariance -> use_diagonal_covariance) and supports both
experiment_config.json and in-checkpoint config extraction.
"""

import torch
import json
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

from transformer.core.model import GaugeTransformerLM


def save_checkpoint(
    model: GaugeTransformerLM,
    optimizer,
    config: Dict[str, Any],
    epoch: int,
    step: int,
    save_path: str,
    **kwargs
):
    """
    Save a model checkpoint.

    Args:
        model: GaugeTransformerLM instance to save.
        optimizer: Optimizer whose state_dict is saved (or None).
        config: Model configuration dict (BlockConfig fields, irrep_spec, etc.).
        epoch: Current training epoch.
        step: Current global training step.
        save_path: Destination file path.
        **kwargs: Additional items to include (e.g., best_val_loss).
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'config': config,
        'epoch': epoch,
        'step': step,
        **kwargs
    }
    torch.save(checkpoint, save_path)
    print(f"Saved checkpoint to {save_path}")


def load_checkpoint(checkpoint_path: str, device: str = 'cpu', trusted: bool = True) -> Dict[str, Any]:
    """
    Load a raw checkpoint dictionary.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load tensors to
        trusted: If True (default), uses weights_only=False which permits
            arbitrary code execution via pickle. Set to False for untrusted
            checkpoints to use weights_only=True (safe but may fail on
            checkpoints containing non-tensor objects like configs).

    Returns:
        checkpoint: Dictionary with model_state_dict, config, etc.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=not trusted)
    return checkpoint


def load_model(checkpoint_path: str, trusted: bool = True) -> Tuple[GaugeTransformerLM, Dict[str, Any]]:
    """
    Load a trained GaugeTransformerLM from checkpoint.

    Config resolution order:
    1. experiment_config.json in the checkpoint directory (preferred).
    2. Config dict embedded in the checkpoint file.
    3. Hardcoded defaults (last resort).

    Applies legacy config migrations (kappa_beta_base, diagonal_covariance).

    Args:
        checkpoint_path: Path to best_model.pt or similar checkpoint.

    Returns:
        model: GaugeTransformerLM in eval mode on CPU.
        config: Configuration dict (embed_dim, irrep_spec, etc.).

    Raises:
        FileNotFoundError: If checkpoint file does not exist.
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint_dir = checkpoint_path.parent
    config_json_path = checkpoint_dir / "experiment_config.json"

    # Default config (fallback values)
    config = {
        'vocab_size': 50257,
        'embed_dim': 25,
        'n_layers': 1,
        'irrep_spec': [('ℓ0', 5, 1), ('ℓ1', 3, 3), ('ℓ2', 1, 5)],
        'hidden_dim': 112,
        'max_seq_len': 128,
        'kappa_beta': 1.0,
        'dropout': 0.1,
        'pos_encoding_mode': 'learned',
        'evolve_sigma': True,
        'evolve_phi': False,
        'tie_embeddings': True,
        'diagonal_covariance': True,
        'ffn_mode': 'VFE_dynamic',
    }

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=not trusted)

    # Try loading from experiment_config.json first (more reliable)
    if config_json_path.exists():
        print(f"Loading config from {config_json_path}")
        with open(config_json_path, 'r') as f:
            json_data = json.load(f)

        # Check if config is nested under a 'config' key
        if 'config' in json_data and isinstance(json_data['config'], dict):
            config.update(json_data['config'])
            print("Loaded nested config from experiment_config.json")
        else:
            config.update(json_data)
            print("Loaded config from experiment_config.json")
    else:
        # Try to extract config from checkpoint pickle
        print(f"Warning: {config_json_path} not found, trying to extract from checkpoint...")

        if 'config' in checkpoint:
            ckpt_config = checkpoint['config']
            if isinstance(ckpt_config, dict):
                config.update(ckpt_config)
            elif hasattr(ckpt_config, '__dict__'):
                config.update(vars(ckpt_config))
            print("Extracted config from checkpoint")
        else:
            print("Warning: No config found, using defaults")

    # Handle config key translations for backward compatibility
    # Legacy checkpoint compat: old configs stored kappa_beta_base instead of kappa_beta
    if 'kappa_beta' not in config:
        config['kappa_beta'] = config.pop('kappa_beta_base', 1.0)
    # Remove defunct auto-scale keys if present
    config.pop('kappa_beta_auto_scale', None)
    config.pop('kappa_beta_base', None)
    config.pop('kappa_beta_k_ref', None)
    # Legacy compat: old configs may use 'use_diagonal_covariance' but model reads 'diagonal_covariance'
    if 'diagonal_covariance' not in config and 'use_diagonal_covariance' in config:
        config['diagonal_covariance'] = config['use_diagonal_covariance']

    print(f"Config: K={config['embed_dim']}, vocab={config['vocab_size']}, "
          f"layers={config['n_layers']}")

    # Create model
    model = GaugeTransformerLM(config)

    # Load checkpoint weights with strict=False for graceful handling of
    # architecture mismatches (e.g., added/removed layers between versions).
    if 'model_state_dict' in checkpoint:
        result = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        result = model.load_state_dict(checkpoint, strict=False)
    if result.missing_keys:
        print(f"  Warning: missing keys in checkpoint: {result.missing_keys[:10]}"
              + (f" ... ({len(result.missing_keys)} total)" if len(result.missing_keys) > 10 else ""))
    if result.unexpected_keys:
        print(f"  Warning: unexpected keys in checkpoint: {result.unexpected_keys[:10]}"
              + (f" ... ({len(result.unexpected_keys)} total)" if len(result.unexpected_keys) > 10 else ""))

    print(f"Loaded checkpoint from {checkpoint_path}")
    model.eval()

    return model, config


def get_tokenizer(config: Dict[str, Any], dataset_name: Optional[str] = None):
    """
    Get tokenizer matching the vocabulary a model was trained with.

    Encoding selection:
    - wiki-ja (or vocab_size > 50257): cl100k_base (GPT-4, better CJK).
    - All others: gpt2 encoding.

    Falls back to WikiTextDataset tokenizer if tiktoken is unavailable.

    Args:
        config: Model configuration dict (reads 'dataset' and 'vocab_size').
        dataset_name: Override dataset name (default: from config or 'wikitext-103').

    Returns:
        Tokenizer with encode/decode methods, or None if unavailable.
    """
    if dataset_name is None:
        dataset_name = config.get('dataset', 'wikitext-103')

    # Auto-detect wiki-ja from vocab_size if dataset not in config
    # cl100k_base has 100277 tokens; GPT-2 has 50257
    vocab_size = config.get('vocab_size', 50257)
    if dataset_name is None and vocab_size > 50257:
        dataset_name = 'wiki-ja'

    is_japanese = (dataset_name == 'wiki-ja')

    # Try tiktoken first (faster, lighter)
    try:
        import tiktoken
        if is_japanese:
            enc = tiktoken.get_encoding("cl100k_base")
            print(f"Using tiktoken cl100k_base tokenizer for wiki-ja (vocab_size={enc.n_vocab})")
        else:
            enc = tiktoken.get_encoding("gpt2")
            print(f"Using tiktoken GPT-2 tokenizer (vocab_size={enc.n_vocab})")
        return enc
    except ImportError:
        pass

    # Fall back to WikiTextDataset
    try:
        from transformer.data.datasets import WikiTextDataset
        dataset = WikiTextDataset(
            split='train',
            max_seq_len=128,
            dataset=dataset_name,
        )
        print(f"Using WikiTextDataset tokenizer")
        return dataset
    except Exception as e:
        print(f"Warning: Could not load dataset tokenizer: {e}")

    print("Warning: No tokenizer available. Install tiktoken: pip install tiktoken")
    return None


def load_checkpoint_info(checkpoint_path: str, trusted: bool = True) -> Dict[str, Any]:
    """
    Load metadata from a checkpoint without instantiating the model.

    Args:
        checkpoint_path: Path to checkpoint file.
        trusted: If True (default), uses weights_only=False. Set to False
            for untrusted checkpoints (see load_checkpoint docstring).

    Returns:
        Dict with keys: config, epoch, step, has_optimizer, n_parameters.

    Raises:
        FileNotFoundError: If checkpoint file does not exist.
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=not trusted)

    info = {}

    # Extract config
    if 'config' in checkpoint:
        info['config'] = checkpoint['config']

    # Extract training state
    info['epoch'] = checkpoint.get('epoch', 'unknown')
    info['step'] = checkpoint.get('step', 'unknown')

    # Check for optimizer state (indicates training checkpoint vs inference)
    info['has_optimizer'] = 'optimizer_state_dict' in checkpoint

    # Model parameter count
    if 'model_state_dict' in checkpoint:
        n_params = sum(p.numel() for p in checkpoint['model_state_dict'].values())
        info['n_parameters'] = n_params

    return info
