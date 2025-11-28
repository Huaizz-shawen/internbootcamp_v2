#! /bin/bash

python -m internbootcamp.utils.data_generation \
    --instruction-config internbootcamp/bootcamps/NP_MM/configs/TSP_instruction_config.yaml \
    --output-dir internbootcamp/bootcamps/NP_MM/data \
    --split-samples train:1,test:0 \
    --shuffle


# python -m internbootcamp.utils.data_generation \
#     --instruction-config internbootcamp/bootcamps/NP_MM/configs/gcp_instruction_config.yaml \
#     --output-dir internbootcamp/bootcamps/NP_MM/data \
#     --split-samples train:5,test:5 \
#     --shuffle
