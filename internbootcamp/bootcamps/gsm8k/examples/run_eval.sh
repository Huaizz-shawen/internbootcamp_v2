python3 internbootcamp/examples/unittest/run_eval.py \
    data/gsm8k_w_tool/test.jsonl \
    --output_path data/gsm8k_w_tool/eval_results_qwen3-32B.jsonl \
    --api-url https://180.163.156.43:21020/qwen3-32B/v1/ \
    --api-key "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoiZ3VvcWlwZW5nIiwiZXhwIjoxNzk4NzYxNjAwfQ.9An25TNKrhC6K8OIluuSjFGCBEtrr6414vmTqC5q6FQ" \
    --api-model "qwen" \
    --max-turns 10 \
    --max-concurrent 32