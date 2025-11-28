# 使用bootcamp注册表进行批量评测
python -m verl.Bootcampv2.utils.run_evaluation \
  --dataset-path internbootcamp/bootcamps/NP_MM/data/TSP_20251118152919_train.jsonl \
  --bootcamp-registry internbootcamp/bootcamps/NP_MM/configs/bootcamp_registry_NP_MM.jsonl \
  --output-dir results/batch_evaluation/ \
  --api-key "null" \
  --api-url "http://10.102.249.85:30001/v1" \
  --api-model "Qwen/Qwen3VL-8B-Instruct" \
  --max-concurrent 32 \
  --verbose 