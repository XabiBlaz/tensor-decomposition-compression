"""Convert a trusted legacy full-model .pt into tensor-only state_dict weights.

Run this in the environment that defines the model's original Python classes.
Never run it on files from an untrusted source: legacy pickle deserialization
can execute code. The web service intentionally does not use this path.
"""

import argparse
from pathlib import Path

import torch
from torch import nn


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--trusted", action="store_true",
                        help="Confirm you trust this pickled file and have its Python model code installed.")
    args = parser.parse_args(argv)
    if not args.trusted:
        parser.error("Pass --trusted only for a file you created or fully trust.")
    if args.output.exists():
        parser.error("Output already exists; choose a new path.")
    model = torch.load(args.input, map_location="cpu", weights_only=False)
    if not isinstance(model, nn.Module):
        parser.error("The input did not deserialize to an nn.Module.")
    state = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, args.output)
    print(f"Saved {len(state)} tensor entries to {args.output}. Select the same architecture and class count in the UI.")


if __name__ == "__main__":
    main()
