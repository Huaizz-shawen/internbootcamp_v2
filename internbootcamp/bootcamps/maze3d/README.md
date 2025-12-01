# 3D Maze RLVR Bootcamp

A comprehensive implementation of **Reinforcement Learning with Verifiable Rewards (RLVR)** for 3D maze navigation and spatial reasoning tasks.

## Overview

This bootcamp implements a multi-round tool-calling environment where LLMs learn to navigate 3D mazes while being rewarded for physically grounded reasoning. The implementation features a sophisticated **three-level reward system** that evaluates both outcomes and reasoning processes.

## Key Features

### 🎮 Multiple Puzzle Types

1. **Path Finding**: Navigate through branch points to reach the goal
2. **Sequence Finding**: Visit numbered checkpoints in the correct order
3. **Height Comparison**: Analyze and compare heights of different positions
4. **Main Path**: Identify which blocks are on the main path vs dead ends

### 🛠️ Multi-Round Tool Calling

Six navigation tools for step-by-step maze exploration:
- `move`: Navigate horizontally (X/Y directions)
- `climb`: Climb ladders to gain height
- `check_position`: Query current position
- `get_height`: Check height at any location
- `describe_surroundings`: Get information about nearby cubes
- `analyze_path`: Validate proposed move sequences

### 🎯 Three-Level Reward System (RLVR Core)

#### Level 1: Outcome Reward (Weight: 0.4)
- Direct correctness check of final answer
- Partial credit for partially correct solutions
- Binary or soft reward modes

#### Level 2: Physics & Logic Reward (Weight: 0.4) ⭐ **RLVR Essence**
- Validates each reasoning step against physical constraints
- Detects **hallucinations** (invalid moves, non-existent ladders)
- Verifies height comparisons against ground truth
- Penalizes physics violations while rewarding valid reasoning

#### Level 3: State Tracking Reward (Weight: 0.2)
- Evaluates model's awareness of positions
- Rewards accurate state tracking throughout reasoning
- Detects when model loses track of its location

## Architecture

```
maze3d/
├── maze3d_instruction_generator.py    # Generates puzzle instances
├── maze3d_tools.py                    # Navigation tools with state tracking
├── maze3d_reward_calculator.py        # Three-level RLVR reward system
├── configs/
│   ├── instruction_config.yaml        # Puzzle generation config
│   ├── tools_config.yaml             # Tool schemas
│   └── reward_config.yaml            # Reward system config
└── examples/
    ├── generate_data.py              # Data generation script
    └── test_maze3d.py               # Test suite
```

## Quick Start

### 1. Test the Implementation

```bash
cd internbootcamp/bootcamps/maze3d
python examples/test_maze3d.py
```

This runs a comprehensive test suite covering:
- Puzzle generation for all types
- Three-level reward calculation
- Multi-round tool execution
- Full RLVR pipeline

### 2. Generate Training Data

```bash
python examples/generate_data.py
```

This generates a JSONL file with 100 maze puzzles configured with:
- Mixed difficulty levels
- All puzzle types
- Tool schemas included
- Ground truth answers

### 3. Customize Configuration

Edit `configs/instruction_config.yaml` to adjust:
- Grid sizes for different difficulties
- Puzzle type distribution
- Path complexity parameters
- Number of checkpoints/branches

Edit `configs/reward_config.yaml` to tune RLVR:
- Enable/disable reward levels
- Adjust weights for each level
- Configure soft reward mode

## RLVR in Action

### Example: Physics Violation Detection

**Scenario**: Model tries to move through a wall

```python
# Model attempts invalid move
move_action = {"action": "move", "direction": "forward", "distance": 5}

# RLVR Level 2 detects this violates physics
# - Checks if path exists using is_path_valid()
# - Finds no cubes in the path
# - Records as hallucination
# - Applies negative reward: -0.5

reward_breakdown = {
    "level1_outcome": 0.0,      # Wrong path, no outcome reward
    "level2_physics": 0.0,      # Hallucination penalty applied
    "level3_state": 0.5,        # Neutral (no position claims)
    "total": 0.1                # 0.4*0.0 + 0.4*0.0 + 0.2*0.5
}
```

### Example: Correct Reasoning with State Tracking

**Scenario**: Model navigates correctly and tracks position

```python
# Model output with position awareness
output = """
I start at position (7, 7, 0).
I move left by 2, now at (5, 7, 0).
I climb the ladder by 3, now at (5, 7, 3).
...
Answer: 1-left-forward, 2-left-forward, 3-up
"""

reward_breakdown = {
    "level1_outcome": 1.0,      # Correct answer
    "level2_physics": 1.0,      # All moves valid
    "level3_state": 1.0,        # Accurate position tracking
    "total": 1.0                # 0.4*1.0 + 0.4*1.0 + 0.2*1.0
}
```

## Integration with VERL

The bootcamp is designed to integrate with the VERL (Versatile Reinforcement Learning) framework:

```python
from internbootcamp.reward_manager.bootcamp import BootcampRewardManager
from verl.trainer import Trainer

# The reward manager will automatically load the Maze3DRewardCalculator
# based on the data_source field in your training data

trainer = Trainer(
    reward_manager=BootcampRewardManager(),
    # ... other trainer config
)
```

## Puzzle Generation Parameters

### Grid Size
- **Easy**: (6, 6, 5) - Smaller space, simpler navigation
- **Medium**: (8, 8, 7) - Standard maze size
- **Hard**: (10, 10, 9) - Larger, more complex mazes

### Path Complexity
- `main_path_length`: (min, max) segments for main path
- `side_path_num`: (min, max) number of branch points
- `side_path_length`: (min, max) segments for dead-end branches

### Checkpoint Configuration
- `label_num_range`: (min, max) numbered checkpoints
- Automatically ensures checkpoints aren't too close
- Balanced distribution along the path

## Tool Execution Flow

1. **LLM receives prompt** with maze description and rules
2. **LLM calls tools** to explore and navigate
   - Each tool call receives immediate feedback
   - State is tracked across all tool calls
3. **LLM provides final answer** in JSON format
4. **Three-level reward** is calculated:
   - Level 1: Answer correctness
   - Level 2: Reasoning validity (hallucination detection)
   - Level 3: State awareness
5. **Cumulative tool reward** added for efficiency

## Advanced Features

### Hallucination Detection

The system tracks multiple types of hallucinations:
- Moving through walls or off the grid
- Climbing non-existent ladders
- Claiming incorrect heights
- Stating positions that don't exist

### Adaptive Difficulty

Puzzle difficulty automatically scales based on:
- Number of cubes (grid complexity)
- Number of branch points
- Path length and verticality
- Checkpoint distribution

### Visualization Support

Enable puzzle visualization in config:
```yaml
save_images: true
output_dir: "maze3d_visualizations"
```

Generates matplotlib 3D plots showing:
- Blue cube: Start position
- Red cube: Goal position
- Green cubes: Checkpoints/branches
- Black lines: Ladders

## Troubleshooting

### Import Errors
Make sure you're running from the project root and the `3d_maze` module is accessible:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Path Generation Failures
If you see "Cannot generate valid path" errors:
- Reduce path complexity (shorter paths)
- Use larger grid sizes
- Decrease number of branches

### Reward Calculation Issues
Check `configs/reward_config.yaml`:
- Weights should sum to 1.0
- All levels should be enabled for full RLVR
- Adjust weights based on your training objectives

## Citation

If you use this bootcamp in your research, please cite:

```bibtex
@software{maze3d_rlvr_bootcamp,
  title={3D Maze RLVR Bootcamp},
  author={InternBootcamp Contributors},
  year={2025},
  description={Reinforcement Learning with Verifiable Rewards for 3D Spatial Reasoning}
}
```

## License

This bootcamp is part of the InternBootcamp v2 project.
