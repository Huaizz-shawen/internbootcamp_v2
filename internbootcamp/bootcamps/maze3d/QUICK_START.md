# 3D Maze RLVR - Quick Start Guide

## 🚀 5-Minute Quick Start

### Step 1: Generate Data (30 seconds)

```bash
conda activate maze3d
cd /media/user/B29202FA9202C2B91/internbootcamp_v2
python internbootcamp/bootcamps/maze3d/examples/generate_data.py
```

**Output:** `maze3d_training_data.jsonl` (100 puzzles)

### Step 2: Test with LLM (2 minutes)

```bash
export OPENAI_API_KEY="sk-your-key-here"

python -m internbootcamp.utils.run_evaluation \
    --dataset-path maze3d_training_data.jsonl \
    --output-dir results/maze3d/ \
    --api-key $OPENAI_API_KEY \
    --api-model "gpt-4o-mini" \
    --reward-calculator-class "internbootcamp.bootcamps.maze3d.maze3d_reward_calculator.Maze3DRewardCalculator" \
    --tool-config "internbootcamp/bootcamps/maze3d/configs/tools_config.yaml" \
    --max-assistant-turns 20 \
    --max-concurrent 3
```

### Step 3: Check Results

```bash
ls results/maze3d/gpt-4o-mini/
# eval_results_*.jsonl - Full results
# eval_summary_*.json  - Statistics

cat results/maze3d/gpt-4o-mini/eval_summary_*.json
```

## 📊 What You'll See

```json
{
  "total_samples": 100,
  "average_score": 0.73,
  "hallucination_rate": 0.12,
  "average_tool_calls": 6.5,
  "average_by_puzzle_type": {
    "path_finding": 0.68,
    "sequence_finding": 0.75,
    "height_comparison": 0.82,
    "main_path": 0.71
  }
}
```

## 🎯 Key Commands

### Generate More Data

```bash
# Edit num_samples in examples/generate_data.py
python internbootcamp/bootcamps/maze3d/examples/generate_data.py
```

### Test Different Models

```bash
# GPT-4
--api-model "gpt-4o"

# Claude (via OpenAI-compatible API)
--api-url "https://api.anthropic.com/v1" \
--api-model "claude-3-5-sonnet-20241022"

# Local model
--api-url "http://localhost:8000/v1" \
--api-model "internlm2-chat-20b"
```

### Adjust Difficulty

Edit `configs/instruction_config.yaml`:

```yaml
config_groups:
  - name: "easy"
    config:
      grid_size: [6, 6, 5]      # Smaller
      main_path_length: [3, 5]  # Shorter
```

### Custom Rewards

```bash
--verify-correction-kwargs '{
    "weight_level1": 0.4,  # Outcome
    "weight_level2": 0.4,  # Physics (RLVR core)
    "weight_level3": 0.2   # State tracking
}'
```

## 🔧 Common Options

| Option | Description | Example |
|--------|-------------|---------|
| `--max-assistant-turns` | Max LLM responses | `20` |
| `--max-concurrent` | Parallel evaluations | `5` |
| `--verbose` | Show details | Add flag |
| `--dry-run` | Test without API calls | Add flag |
| `--resume-from-result-path` | Continue interrupted eval | `results/.../eval_results_*.jsonl` |

## 📈 Typical Results

**Before RL Training:**
- Average Score: **0.55-0.70**
- Hallucination Rate: **15-25%**
- Tool Efficiency: **8-12 calls/puzzle**

**After RL Training:**
- Average Score: **0.75-0.90**
- Hallucination Rate: **5-10%**
- Tool Efficiency: **5-8 calls/puzzle**

## 🐛 Quick Troubleshooting

```bash
# Test installation
python internbootcamp/bootcamps/maze3d/examples/simple_test.py
# Should show: ALL TESTS PASSED! ✓

# Check API connection
curl $API_URL/v1/models \
  -H "Authorization: Bearer $API_KEY"

# Reduce rate limits
--max-concurrent 1 --timeout-per-query 120
```

## 📚 Documentation

- Full guide: `USAGE_GUIDE.md`
- Implementation: `IMPLEMENTATION_SUMMARY.md`
- Features: `README.md`

## 💡 Pro Tips

1. **Start small**: Test with 10 samples first
2. **Use --dry-run**: Verify setup before spending API credits
3. **Monitor hallucinations**: They indicate model quality
4. **Save checkpoints**: Use `--resume-from-result-path` for long runs
5. **Compare models**: Evaluate multiple LLMs to find best performer

## Next: Full RL Training

See main README for VERL training setup:
```bash
# After generating data
python -m verl.trainer.main --config configs/maze3d_training.yaml
```

---

**Need help?** Check `USAGE_GUIDE.md` for detailed explanations!
