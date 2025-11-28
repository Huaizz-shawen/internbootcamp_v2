python -m internbootcamp.utils.batch_data_generation \
    --bootcamp-registry internbootcamp/bootcamps/NP_MM/bootcamp_registry/bootcamp_registry_NP_MM.jsonl \
    --output-dir data/NP_MM_test/ \
    --split-samples train:500,test:50 \
    --max-workers 64 \
    --log-level DEBUG \
    --continue-on-error \
    --concat-files \
    --no-tool \
    --no-interaction