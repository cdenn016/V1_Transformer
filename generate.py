#!/usr/bin/env python3
"""
Simple text generation script.

Usage:
    python generate.py path/to/best_model.pt
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from transformer.utils.checkpoint import load_model, get_tokenizer


def main():
    if len(sys.argv) < 2:
        print("Usage: python generate.py <checkpoint_path>")
        print("Example: python generate.py experiments/best_model.pt")
        sys.exit(1)

    checkpoint_path = sys.argv[1]

    if not Path(checkpoint_path).exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    # Load model
    print(f"Loading model from {checkpoint_path}...")
    model, config = load_model(checkpoint_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()
    print(f"Model loaded ({device})")

    # Load tokenizer
    tokenizer = get_tokenizer(config)
    if tokenizer is None:
        print("Error: Could not load tokenizer. Install tiktoken: pip install tiktoken")
        sys.exit(1)

    # Settings
    max_tokens = 50
    temperature = 0.8

    print("\n" + "=" * 50)
    print("GAUGE TRANSFORMER TEXT GENERATION")
    print("=" * 50)
    print(f"Temperature: {temperature}, Max tokens: {max_tokens}")
    print("Type 'quit' to exit, 'temp X' to change temperature")
    print("=" * 50)

    while True:
        try:
            text = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not text:
            continue

        if text.lower() == 'quit':
            print("Goodbye!")
            break

        if text.lower().startswith('temp '):
            try:
                temperature = float(text.split()[1])
                print(f"Temperature set to {temperature}")
            except (IndexError, ValueError):
                print("Usage: temp 0.8")
            continue

        if text.lower().startswith('tokens '):
            try:
                max_tokens = int(text.split()[1])
                print(f"Max tokens set to {max_tokens}")
            except (IndexError, ValueError):
                print("Usage: tokens 50")
            continue

        # Encode input
        token_ids = tokenizer.encode(text)
        input_ids = torch.tensor([token_ids], device=device)

        # Generate
        with torch.no_grad():
            output_ids = model.generate(
                prompt_ids=input_ids,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=40,
                top_p=0.9,
            )

        # Decode and print
        output_text = tokenizer.decode(output_ids[0].tolist())
        generated = output_text[len(text):]

        print(f"\nModel: {text}\033[92m{generated}\033[0m")


if __name__ == '__main__':
    main()
