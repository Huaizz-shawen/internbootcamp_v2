# 3D Maze RLVR - Complete Usage Guide

This guide shows the complete workflow from data generation to LLM interaction and evaluation.

## Workflow Overview

```
1. Generate Data → 2. Evaluate with LLM → 3. Analyze Results → 4. Train with VERL
     (JSONL)            (API Calls)           (Scores)          (RL Training)
```

## Step 1: Generate Training Data

### 1.1 Basic Generation

```bash
# Activate environment
conda activate maze3d

# Generate 100 maze puzzles
python internbootcamp/bootcamps/maze3d/examples/generate_data.py
```

**Output:** `maze3d_training_data.jsonl` with this format:

```json
{
  "prompt": "You are an expert at solving 3D maze puzzles...",
  "identity": {
    "puzzle_type": "path_finding",
    "puzzle_state": {...},
    "answer": "1-left-forward, 2-up, 3-left-forward",
    "grid_size": [8, 8, 7],
    "difficulty": "Medium"
  },
  "tools": [
    {"type": "function", "function": {"name": "move", ...}},
    {"type": "function", "function": {"name": "climb", ...}},
    ...
  ],
  "data_source": "maze3d"
}
```

### 1.2 Custom Generation

Modify `generate_data.py` to customize:

```python
config = {
    "instruction_config": "internbootcamp/bootcamps/maze3d/configs/instruction_config.yaml",
    "tools_config": "internbootcamp/bootcamps/maze3d/configs/tools_config.yaml",
    "output_file": "maze3d_hard.jsonl",  # Custom output
    "num_samples": 500,  # More samples
    "data_source": "maze3d",
}
```

Or edit `configs/instruction_config.yaml`:

```yaml
config_groups:
  - name: "hard_puzzles"
    weight: 1.0
    config:
      grid_size: [10, 10, 9]  # Larger maze
      puzzle_types: ["path_finding"]
      main_path_length: [7, 10]  # Longer paths
      side_path_num: [5, 7]  # More branches
```

## Step 2: Evaluate LLM with Generated Data

### 2.1 Basic Evaluation (OpenAI API)

```bash
python -m internbootcamp.utils.run_evaluation \
    --dataset-path maze3d_training_data.jsonl \
    --output-dir results/maze3d_eval/ \
    --api-key "sk-your-api-key" \
    --api-model "gpt-4o" \
    --api-url "https://api.openai.com/v1" \
    --reward-calculator-class "internbootcamp.bootcamps.maze3d.maze3d_reward_calculator.Maze3DRewardCalculator" \
    --tool-config "internbootcamp/bootcamps/maze3d/configs/tools_config.yaml" \
    --max-assistant-turns 20 \
    --max-user-turns 10 \
    --max-concurrent 5 \
    --verbose
```

**What happens:**
1. Loads each puzzle from JSONL
2. Sends prompt + tools to LLM
3. LLM makes multi-round tool calls (move, climb, etc.)
4. Each tool call validated with physics rules
5. Final answer evaluated with three-level RLVR
6. Results saved to `results/maze3d_eval/`

### 2.2 Evaluation with Custom LLM Endpoint

```bash
# For local models or custom APIs
python -m internbootcamp.utils.run_evaluation \
    --dataset-path maze3d_training_data.jsonl \
    --output-dir results/local_llm/ \
    --api-key "dummy-key" \
    --api-model "internlm2.5-20b-chat" \
    --api-url "http://localhost:8000/v1" \
    --api-extra-headers "Authorization:Bearer custom-token" \
    --api-extra-params '{"temperature": 0.7, "max_tokens": 4096}' \
    --reward-calculator-class "internbootcamp.bootcamps.maze3d.maze3d_reward_calculator.Maze3DRewardCalculator" \
    --tool-config "internbootcamp/bootcamps/maze3d/configs/tools_config.yaml" \
    --max-assistant-turns 20
```

### 2.3 Advanced: Custom Reward Configuration

```bash
python -m internbootcamp.utils.run_evaluation \
    --dataset-path maze3d_training_data.jsonl \
    --output-dir results/custom_rewards/ \
    --api-key "sk-xxx" \
    --api-model "gpt-4o" \
    --reward-calculator-class "internbootcamp.bootcamps.maze3d.maze3d_reward_calculator.Maze3DRewardCalculator" \
    --verify-correction-kwargs '{
        "enable_level1_outcome": true,
        "enable_level2_physics": true,
        "enable_level3_state": true,
        "weight_level1": 0.3,
        "weight_level2": 0.5,
        "weight_level3": 0.2
    }' \
    --tool-config "internbootcamp/bootcamps/maze3d/configs/tools_config.yaml"
```

**Reward Configuration Options:**
- `enable_level1_outcome`: Enable outcome correctness check
- `enable_level2_physics`: Enable physics validation (RLVR core)
- `enable_level3_state`: Enable state tracking check
- `weight_level1/2/3`: Adjust weights (must sum to 1.0)

### 2.4 Resume from Interruption

```bash
# If evaluation was interrupted, resume from checkpoint
python -m internbootcamp.utils.run_evaluation \
    --dataset-path maze3d_training_data.jsonl \
    --output-dir results/maze3d_eval/ \
    --api-key "sk-xxx" \
    --resume-from-result-path "results/maze3d_eval/gpt-4o/eval_results_20250101_120000.jsonl" \
    --reward-calculator-class "internbootcamp.bootcamps.maze3d.maze3d_reward_calculator.Maze3DRewardCalculator" \
    --tool-config "internbootcamp/bootcamps/maze3d/configs/tools_config.yaml"
```

## Step 3: Understand the Evaluation Results

### 3.1 Output Files

After evaluation, you'll find:

```
results/maze3d_eval/
├── gpt-4o/
│   ├── eval_results_20250101_120000.jsonl  # Main results
│   ├── eval_summary_20250101_120000.json   # Summary statistics
│   └── failed_samples_20250101_120000.jsonl # Failed cases
```

### 3.2 Result Format

Each line in `eval_results_*.jsonl`:

```json
{
  "index": 0,
  "prompt": "...",
  "identity": {...},
  "model_output": "I will navigate the maze...\n\n```json\n{\"answer\": \"1-left, 2-up\"}```",
  "extracted_output": {
    "answer": "1-left, 2-up",
    "reasoning_steps": [...],
    "move_actions": [...],
    "position_claims": [...]
  },
  "score": 0.85,
  "reward_breakdown": {
    "level1_outcome": 1.0,
    "level2_physics": 0.7,
    "level3_state": 0.8
  },
  "tool_calls": [
    {
      "tool": "check_position",
      "parameters": {},
      "result": "Current position: (7, 7, 0)",
      "reward": 0.0
    },
    {
      "tool": "move",
      "parameters": {"direction": "left", "distance": 2},
      "result": "Moved left by 2. Current position: (5, 7, 0)",
      "reward": 0.1
    },
    ...
  ],
  "metadata": {
    "total_tokens": 2543,
    "completion_time": 3.2,
    "num_tool_calls": 8
  }
}
```

### 3.3 Summary Statistics

`eval_summary_*.json`:

```json
{
  "total_samples": 100,
  "completed": 98,
  "failed": 2,
  "average_score": 0.73,
  "score_distribution": {
    "0.0-0.2": 5,
    "0.2-0.4": 10,
    "0.4-0.6": 15,
    "0.6-0.8": 35,
    "0.8-1.0": 33
  },
  "average_by_puzzle_type": {
    "path_finding": 0.68,
    "sequence_finding": 0.75,
    "height_comparison": 0.82,
    "main_path": 0.71
  },
  "hallucination_rate": 0.12,
  "average_tool_calls": 6.5
}
```

## Step 4: Analyze Results

### 4.1 Post-Processing

```bash
# Convert results to analysis format
python -m internbootcamp.utils.data_postprocess \
    --input-file results/maze3d_eval/gpt-4o/eval_results_20250101_120000.jsonl \
    --output-file analysis/maze3d_analysis.jsonl \
    --add-metrics \
    --filter-failed
```

### 4.2 Analyze by Difficulty

```python
import json
import pandas as pd

# Load results
results = []
with open('results/maze3d_eval/gpt-4o/eval_results_20250101_120000.jsonl') as f:
    for line in f:
        results.append(json.loads(line))

df = pd.DataFrame(results)

# Group by difficulty
difficulty_stats = df.groupby('identity.difficulty').agg({
    'score': ['mean', 'std', 'count'],
    'metadata.num_tool_calls': 'mean'
})

print(difficulty_stats)
```

### 4.3 Analyze Hallucinations

```python
# Find cases with hallucinations
hallucinations = []
for result in results:
    tool_calls = result.get('tool_calls', [])
    for call in tool_calls:
        if call.get('reward', 0) < 0:  # Negative reward = hallucination
            hallucinations.append({
                'index': result['index'],
                'tool': call['tool'],
                'parameters': call['parameters'],
                'reason': call.get('result')
            })

print(f"Total hallucinations: {len(hallucinations)}")
print(f"Hallucination rate: {len(hallucinations) / sum(len(r.get('tool_calls', [])) for r in results):.2%}")
```

## Step 5: Train with VERL (RL Training)

### 5.1 Prepare Training Data

The generated JSONL is already in VERL format! Key fields:
- `data_source: "maze3d"` → automatically loads `Maze3DRewardCalculator`
- `tools` → passed to LLM for tool calling
- `identity` → used for reward calculation

### 5.2 Training Configuration

Create `configs/maze3d_training.yaml`:

```yaml
# Model configuration
model:
  name: "internlm2-chat-7b"
  path: "internlm/internlm2-chat-7b"

# Data configuration
data:
  train_files: ["maze3d_training_data.jsonl"]
  eval_files: ["maze3d_eval_data.jsonl"]
  data_source: "maze3d"  # Important: links to Maze3DRewardCalculator

# RL configuration
rl:
  algorithm: "grpo"  # Group Relative Policy Optimization
  learning_rate: 1e-5
  batch_size: 64
  epochs: 3

# Reward configuration
reward:
  calculator_class: "internbootcamp.bootcamps.maze3d.maze3d_reward_calculator.Maze3DRewardCalculator"
  tool_config: "internbootcamp/bootcamps/maze3d/configs/tools_config.yaml"
  max_tool_calls: 20
```

### 5.3 Start Training

```bash
# Using VERL framework
python -m verl.trainer.main \
    --config configs/maze3d_training.yaml \
    --output-dir checkpoints/maze3d_rl/
```

**What happens during training:**
1. LLM generates responses with tool calls
2. Tools execute with physics validation
3. Three-level RLVR rewards computed
4. Policy updated based on rewards
5. Repeat for multiple epochs

### 5.4 Monitor Training

```bash
# TensorBoard monitoring
tensorboard --logdir checkpoints/maze3d_rl/logs/

# Key metrics to watch:
# - average_reward: Overall performance
# - level2_physics_reward: Hallucination reduction
# - hallucination_rate: Physics violations
# - tool_efficiency: Average tool calls per puzzle
```

## Step 6: Evaluate Trained Model

### 6.1 Compare Before/After Training

```bash
# Evaluate base model
python -m internbootcamp.utils.run_evaluation \
    --dataset-path maze3d_test_data.jsonl \
    --output-dir results/base_model/ \
    --api-model "internlm2-chat-7b" \
    --reward-calculator-class "internbootcamp.bootcamps.maze3d.maze3d_reward_calculator.Maze3DRewardCalculator" \
    --tool-config "internbootcamp/bootcamps/maze3d/configs/tools_config.yaml"

# Evaluate RL-trained model
python -m internbootcamp.utils.run_evaluation \
    --dataset-path maze3d_test_data.jsonl \
    --output-dir results/rl_trained/ \
    --api-model "checkpoints/maze3d_rl/final_model" \
    --reward-calculator-class "internbootcamp.bootcamps.maze3d.maze3d_reward_calculator.Maze3DRewardCalculator" \
    --tool-config "internbootcamp/bootcamps/maze3d/configs/tools_config.yaml"
```

### 6.2 Compare Results

```python
import json

def load_results(path):
    with open(path) as f:
        return [json.loads(line) for line in f]

base = load_results('results/base_model/eval_results.jsonl')
trained = load_results('results/rl_trained/eval_results.jsonl')

print(f"Base Model:")
print(f"  Average Score: {sum(r['score'] for r in base) / len(base):.3f}")
print(f"  Hallucination Rate: {sum(1 for r in base if any(t['reward'] < 0 for t in r.get('tool_calls', []))) / len(base):.2%}")

print(f"\nRL-Trained Model:")
print(f"  Average Score: {sum(r['score'] for r in trained) / len(trained):.3f}")
print(f"  Hallucination Rate: {sum(1 for r in trained if any(t['reward'] < 0 for t in r.get('tool_calls', []))) / len(trained):.2%}")
```

## Common Use Cases

### Use Case 1: Quick Testing (Small Dataset)

```bash
# Generate 10 samples
python internbootcamp/bootcamps/maze3d/examples/generate_data.py
# Edit to set num_samples: 10

# Evaluate with GPT-4
python -m internbootcamp.utils.run_evaluation \
    --dataset-path maze3d_training_data.jsonl \
    --output-dir results/quick_test/ \
    --api-key "sk-xxx" \
    --api-model "gpt-4o" \
    --reward-calculator-class "internbootcamp.bootcamps.maze3d.maze3d_reward_calculator.Maze3DRewardCalculator" \
    --tool-config "internbootcamp/bootcamps/maze3d/configs/tools_config.yaml" \
    --max-concurrent 5
```

### Use Case 2: Ablation Study (RLVR Levels)

```bash
# Test Level 1 only (outcome)
python -m internbootcamp.utils.run_evaluation \
    --dataset-path maze3d_test.jsonl \
    --output-dir results/level1_only/ \
    --verify-correction-kwargs '{"enable_level1_outcome": true, "enable_level2_physics": false, "enable_level3_state": false, "weight_level1": 1.0}' \
    ...

# Test Level 2 only (physics)
python -m internbootcamp.utils.run_evaluation \
    --dataset-path maze3d_test.jsonl \
    --output-dir results/level2_only/ \
    --verify-correction-kwargs '{"enable_level1_outcome": false, "enable_level2_physics": true, "enable_level3_state": false, "weight_level2": 1.0}' \
    ...

# Compare results to measure RLVR impact
```

### Use Case 3: Difficulty Scaling

```bash
# Generate easy, medium, hard datasets
# Edit instruction_config.yaml for each

# Evaluate model on each difficulty
for difficulty in easy medium hard; do
    python -m internbootcamp.utils.run_evaluation \
        --dataset-path maze3d_${difficulty}.jsonl \
        --output-dir results/${difficulty}/ \
        ...
done

# Analyze scaling behavior
```

## Troubleshooting

### Issue: "No module named 'verl'"

**Solution:** VERL is only needed for RL training, not for data generation or evaluation.

```bash
# For data generation and evaluation only
pip install -e .

# For full RL training
# Install VERL separately (see main README)
```

### Issue: "Matplotlib not found"

**Solution:** Install visualization dependencies

```bash
conda activate maze3d
# OR
pip install matplotlib numpy
```

### Issue: API Rate Limiting

**Solution:** Reduce concurrency

```bash
python -m internbootcamp.utils.run_evaluation \
    --max-concurrent 1 \  # Slower but avoids rate limits
    --timeout-per-query 120 \  # Increase timeout
    ...
```

### Issue: High Hallucination Rate

**Possible causes:**
1. **Model not trained** - Base models have high hallucination rates
2. **Puzzle too hard** - Reduce difficulty in config
3. **Max turns too low** - Increase `--max-assistant-turns`

**Solutions:**
- Use RL-trained model
- Start with easier puzzles
- Allow more tool calls

## Next Steps

1. **Generate diverse data** - Try all puzzle types and difficulties
2. **Baseline evaluation** - Test multiple LLMs (GPT-4, Claude, local models)
3. **RL training** - Train with VERL to reduce hallucinations
4. **Ablation studies** - Measure impact of each RLVR level
5. **Transfer learning** - Test on unseen maze configurations

## Summary: Complete Workflow

```bash
# 1. Setup environment
conda activate maze3d

# 2. Generate training data
python internbootcamp/bootcamps/maze3d/examples/generate_data.py

# 3. Evaluate LLM
python -m internbootcamp.utils.run_evaluation \
    --dataset-path maze3d_training_data.jsonl \
    --output-dir results/ \
    --api-key "sk-xxx" \
    --api-model "gpt-4o" \
    --reward-calculator-class "internbootcamp.bootcamps.maze3d.maze3d_reward_calculator.Maze3DRewardCalculator" \
    --tool-config "internbootcamp/bootcamps/maze3d/configs/tools_config.yaml" \
    --max-assistant-turns 20 \
    --verbose

# 4. Analyze results
python analyze_results.py results/gpt-4o/eval_results_*.jsonl

# 5. Train with VERL (optional)
python -m verl.trainer.main --config configs/maze3d_training.yaml

# 6. Evaluate trained model
python -m internbootcamp.utils.run_evaluation \
    --api-model "checkpoints/maze3d_rl/final_model" \
    ...
```

That's it! You now have a complete RLVR pipeline for 3D maze navigation.
