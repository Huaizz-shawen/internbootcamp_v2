
# 评测数据集路径
DATASET_PATH="internbootcamp/bootcamps/example_bootcamp/data/example_arithmetic/example_20251113160818_test.jsonl"
# 评测输出目录
OUTPUT_DIR="internbootcamp/bootcamps/example_bootcamp/data/eval_output/"
# API 密钥
API_KEY="sk-jncgtocbccfvawgfmiuewwmxpmrwsthijvxkqdfekploaany"
# API 地址
API_URL="https://api.siliconflow.cn/v1/"
# API 模型名称
API_MODEL="Qwen/Qwen3-8B"
# 评测器类
EVALUATOR_CLASS="internbootcamp.src.base_evaluator.BaseEvaluator"
# 奖励计算器类
REWARD_CALCULATOR_CLASS="internbootcamp.bootcamps.example_bootcamp.example_reward_calculator.ExampleRewardCalculator"
# 工具配置文件路径
## mcp tool
TOOL_CONFIG="internbootcamp/bootcamps/example_bootcamp/configs/example_tool_config.yaml"
## tool
# TOOL_CONFIG="internbootcamp/bootcamps/example_bootcamp/configs/example_tool_config.yaml" 
# 交互配置文件路径
INTERACTION_CONFIG="internbootcamp/bootcamps/example_bootcamp/configs/example_interaction_config.yaml"

# RESUME="internbootcamp/bootcamps/example_bootcamp/data/eval_output/deepseekv3-1-terminus/eval_results_20251022134402.jsonl"
    # --resume-from-result-path $RESUME \
# 最大LLM轮数
MAX_ASSISTANT_TURNS=4
MAX_USER_TURNS=512
# 最大并发数
MAX_CONCURRENT=16

TOKENIZER_PATH="/mnt/shared-storage-user/large-model-center-share-weights/hf_hub/models--Qwen--Qwen3-8B/snapshots/9c925d64d72725edaf899c6cb9c377fd0709d9c5"

# 执行评测命令
python -m internbootcamp.utils.run_evaluation \
    --dataset-path "$DATASET_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --api-key "$API_KEY" \
    --api-url "$API_URL" \
    --api-model "$API_MODEL" \
    --evaluator-class "$EVALUATOR_CLASS" \
    --reward-calculator-class "$REWARD_CALCULATOR_CLASS" \
    --tool-config "$TOOL_CONFIG" \
    --interaction-config "$INTERACTION_CONFIG" \
    --max-assistant-turns $MAX_ASSISTANT_TURNS \
    --max-user-turns $MAX_USER_TURNS \
    --max-concurrent $MAX_CONCURRENT \
    --api-extra-params '{"temperature":0.7,"max_completion_tokens":32768}' \
    --verify-correction-kwargs '{"soft_reward": true}' \
    --tokenizer-path $TOKENIZER_PATH \
    --verbose 


# --dry-run \
# --api-extra-params '{"temperature":0.7, "max_completion_tokens":65536, "extra_body": {"enable_thinking": true}}' \