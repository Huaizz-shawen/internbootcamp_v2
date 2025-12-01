#!/usr/bin/env python3
"""
Example script for generating 3D maze training data
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from internbootcamp.utils.data_generation import main as generate_data


if __name__ == "__main__":
    # Configuration
    config = {
        "instruction_config": "internbootcamp/bootcamps/maze3d/configs/instruction_config.yaml",
        "tools_config": "internbootcamp/bootcamps/maze3d/configs/tools_config.yaml",
        "output_file": "maze3d_training_data.jsonl",
        "num_samples": 100,
        "data_source": "maze3d",
    }

    # Generate data
    print("=" * 60)
    print("Generating 3D Maze Training Data")
    print("=" * 60)
    print(f"Instruction config: {config['instruction_config']}")
    print(f"Tools config: {config['tools_config']}")
    print(f"Number of samples: {config['num_samples']}")
    print(f"Output file: {config['output_file']}")
    print("=" * 60)

    # Create argument list for data generation
    sys.argv = [
        "generate_data.py",
        "--instruction_config", config["instruction_config"],
        "--tools_config", config["tools_config"],
        "--output_file", config["output_file"],
        "--num_samples", str(config["num_samples"]),
        "--data_source", config["data_source"],
    ]

    # Run data generation
    generate_data()

    print("=" * 60)
    print(f"Data generation complete! Output saved to: {config['output_file']}")
    print("=" * 60)
