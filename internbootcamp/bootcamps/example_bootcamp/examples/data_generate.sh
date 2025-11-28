#! /bin/bash

python -m internbootcamp.utils.data_generation \
    --instruction-config internbootcamp/bootcamps/example_bootcamp/configs/example_instruction_config.yaml \
    --output-dir internbootcamp/bootcamps/example_bootcamp/data/example_arithmetic/ \
    --split-samples train:0,test:3 \
    --shuffle