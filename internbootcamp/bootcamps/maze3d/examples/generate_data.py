#!/usr/bin/env python3
"""
Example script for generating 3D maze training data
"""

import os
import sys

# Add repo root to path (go up 4 levels from examples/generate_data.py)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../..'))

from internbootcamp.utils.data_generation import generate_data_with_config


if __name__ == "__main__":
    print("=" * 60)
    print("Generating 3D Maze Training Data")
    print("=" * 60)

    # Configuration
    instruction_config = "internbootcamp/bootcamps/maze3d/configs/instruction_config.yaml"
    output_dir = "maze3d_dataset"
    tool_config = "internbootcamp/bootcamps/maze3d/configs/tools_config.yaml"
    num_samples = 100

    print(f"Instruction config: {instruction_config}")
    print(f"Tools config: {tool_config}")
    print(f"Number of samples: {num_samples}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)

    # Generate data using the actual function
    generate_data_with_config(
        instruction_config_path=instruction_config,
        output_dir=output_dir,
        tool_config_path=tool_config,
        interaction_config_path=None,
        split_samples={'train': num_samples},  # Generate 'num_samples' training samples
        shuffle=True,
        global_config_overrides=None
    )

    print("=" * 60)
    print(f"Data generation complete!")
    print(f"Output saved to: {output_dir}/train.jsonl")
    print("=" * 60)
