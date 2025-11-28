python -m internbootcamp.utils.tool_server.cli \
  --tools_yaml_path internbootcamp/bootcamps/example_bootcamp/configs/example_tool_config.yaml \
  --port 20000 \
  --mode unified \
  --num_workers 16 \
  --keep_running \
  --timeout_per_query 60