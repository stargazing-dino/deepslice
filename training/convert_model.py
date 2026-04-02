# /// script
# requires-python = ">=3.11"
# dependencies = ["torch", "safetensors", "packaging", "numpy"]
# ///
"""Convert the TorchScript ninepatch model to safetensors for candle."""

import sys
from pathlib import Path

import torch
from safetensors.torch import save_file


def convert_name(name: str) -> str | None:
    if "num_batches_tracked" in name:
        return None

    # Stem conv + batchnorm
    if name.startswith("backbone.conv_stem."):
        return name.replace("backbone.conv_stem.", "features.0.0.")
    if name.startswith("backbone.bn1."):
        return name.replace("backbone.bn1.", "features.0.1.")

    # Final conv head + batchnorm
    if name.startswith("backbone.conv_head."):
        return name.replace("backbone.conv_head.", "features.8.0.")
    if name.startswith("backbone.bn2."):
        return name.replace("backbone.bn2.", "features.8.1.")

    # Regressor layers (keep as-is)
    if name.startswith("regressor."):
        return name

    # MBConv blocks
    if name.startswith("backbone.blocks."):
        rest = name[len("backbone.blocks."):]
        parts = rest.split(".")
        stage = int(parts[0])
        repeat = int(parts[1])
        param_path = ".".join(parts[2:])

        candle_stage = stage + 1
        is_no_expand = stage == 0  # expand_ratio = 1

        if is_no_expand:
            # No expansion conv: depthwise=block.0, SE=block.1, project=block.2
            mapping = {
                "conv_dw.weight": "block.0.0.weight",
                "bn1.weight": "block.0.1.weight",
                "bn1.bias": "block.0.1.bias",
                "bn1.running_mean": "block.0.1.running_mean",
                "bn1.running_var": "block.0.1.running_var",
                "se.conv_reduce.weight": "block.1.fc1.weight",
                "se.conv_reduce.bias": "block.1.fc1.bias",
                "se.conv_expand.weight": "block.1.fc2.weight",
                "se.conv_expand.bias": "block.1.fc2.bias",
                "conv_pw.weight": "block.2.0.weight",
                "bn2.weight": "block.2.1.weight",
                "bn2.bias": "block.2.1.bias",
                "bn2.running_mean": "block.2.1.running_mean",
                "bn2.running_var": "block.2.1.running_var",
            }
        else:
            # With expansion: expand=block.0, dw=block.1, SE=block.2, project=block.3
            mapping = {
                "conv_pw.weight": "block.0.0.weight",
                "bn1.weight": "block.0.1.weight",
                "bn1.bias": "block.0.1.bias",
                "bn1.running_mean": "block.0.1.running_mean",
                "bn1.running_var": "block.0.1.running_var",
                "conv_dw.weight": "block.1.0.weight",
                "bn2.weight": "block.1.1.weight",
                "bn2.bias": "block.1.1.bias",
                "bn2.running_mean": "block.1.1.running_mean",
                "bn2.running_var": "block.1.1.running_var",
                "se.conv_reduce.weight": "block.2.fc1.weight",
                "se.conv_reduce.bias": "block.2.fc1.bias",
                "se.conv_expand.weight": "block.2.fc2.weight",
                "se.conv_expand.bias": "block.2.fc2.bias",
                "conv_pwl.weight": "block.3.0.weight",
                "bn3.weight": "block.3.1.weight",
                "bn3.bias": "block.3.1.bias",
                "bn3.running_mean": "block.3.1.running_mean",
                "bn3.running_var": "block.3.1.running_var",
            }

        candle_param = mapping.get(param_path)
        if candle_param is None:
            return None
        return f"features.{candle_stage}.{repeat}.{candle_param}"

    return None


def main():
    input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("ninepatch_model.pt")
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else input_path.with_suffix(".safetensors")

    print(f"Loading {input_path}...")
    model = torch.jit.load(input_path, map_location="cpu")
    state_dict = model.state_dict()

    converted = {}
    skipped = []
    for name, tensor in state_dict.items():
        new_name = convert_name(name)
        if new_name is None:
            skipped.append(name)
            continue
        converted[new_name] = tensor.contiguous()
        print(f"  {name:55s} -> {new_name}")

    if skipped:
        print(f"\nSkipped {len(skipped)} params (num_batches_tracked)")

    print(f"\nSaving {len(converted)} tensors to {output_path}...")
    save_file(converted, str(output_path))
    print("Done!")


if __name__ == "__main__":
    main()
