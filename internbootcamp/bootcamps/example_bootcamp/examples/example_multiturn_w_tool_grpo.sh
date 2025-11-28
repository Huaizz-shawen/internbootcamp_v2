# run on 8xH100
# make sure your current working directory is the root of the project

# set -x

ulimit -n 65535

# 当前项目根目录
PROJECT_DIR="$(pwd)"  
# 模型权重路径
MODEL_PATH="/mnt/shared-storage-user/songdemin/user/lipeiji/models/models--Qwen--Qwen3-30B-A3B-Instruct-2507-modify-template/snapshots/3d729a084f14c9502775d59d95c71385293f5518"  
# 配置文件目录
CONFIG_PATH="${PROJECT_DIR}/internbootcamp/bootcamps/example_bootcamp/configs/"
# 配置文件名
CONFIG_NAME="example_multiturn_w_tool_grpo"  
# 工具配置文件路径
## 远程工具配置文件路径
TOOL_CONFIG_PATH="internbootcamp/bootcamps/example_bootcamp/configs/example_tool_config_with_server_urls.yaml"  
## 本地工具配置文件路径
#TOOL_CONFIG_PATH="internbootcamp/bootcamps/example_bootcamp/configs/example_tool_config.yaml"  
# 训练集数据路径
DATA_TRAIN_FILE_PATH="/mnt/shared-storage-user/songdemin/user/lipeiji/verl-internbootcamp/bootcamps/internbootcamp/bootcamps/example_bootcamp/data/example_arithmetic/example_20250915031301_train.parquet"  
# 验证集数据路径
DATA_VAL_FILE_PATH="/mnt/shared-storage-user/songdemin/user/lipeiji/verl-internbootcamp/bootcamps/internbootcamp/bootcamps/example_bootcamp/data/example_arithmetic/example_20250915031301_test.parquet"  
# 项目名称
PROJECT_NAME="example_bootcamp"  
# 实验名称
EXPERIMENT_NAME="example_bootcamp_arithmetic_grpo_Qwen3-30B-test-0915"
# 生成时间戳
TIMESTAMP=$(date +%Y-%m-%d)

# 设置外网服务器代理(wandb)并确保不影响内网ip
export http_proxy=http://zhangjin:QiL6Vklber0wH6cYRB7TvP0opZrQBq4QjoXhg6NFyYBT1ysc6EiNbgPeaSJZ@aliyun-proxy.pjlab.org.cn:13128 && export https_proxy=https://zhangjin:QiL6Vklber0wH6cYRB7TvP0opZrQBq4QjoXhg6NFyYBT1ysc6EiNbgPeaSJZ@aliyun-proxy.pjlab.org.cn:13128 && export HTTP_PROXY=https://zhangjin:QiL6Vklber0wH6cYRB7TvP0opZrQBq4QjoXhg6NFyYBT1ysc6EiNbgPeaSJZ@aliyun-proxy.pjlab.org.cn:13128 && export HTTPS_PROXY=https://zhangjin:QiL6Vklber0wH6cYRB7TvP0opZrQBq4QjoXhg6NFyYBT1ysc6EiNbgPeaSJZ@aliyun-proxy.pjlab.org.cn:13128

export no_proxy="\
localhost,\
127.0.0.1,\
::1,\
*.local,\
*.pjlab.org.cn,\
10.130.128.0/17,\
10.0.0.0/8,\
192.168.0.0/16,\
172.16.0.0/12\
"

export no_proxy=$(echo $no_proxy | tr -s ' ' | tr -d '\n' | sed 's/ //g')

export VERL_PPO_LOGGING_LEVEL=DEBUG
export HYDRA_FULL_ERROR=1
export WANDB_API_KEY="350272d1b727788703b1f5d46518d3c1f41b274f"

python -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name="$CONFIG_NAME" \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=$PROJECT_DIR/ckpts/$EXPERIMENT_NAME\_$TIMESTAMP \
    trainer.rollout_data_dir=$PROJECT_DIR/ckpts/$EXPERIMENT_NAME\_$TIMESTAMP/rollout \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=8 \
    data.max_prompt_length=1024 \
    data.max_response_length=2048 \
    actor_rollout_ref.rollout.prompt_length=1400 \
    actor_rollout_ref.rollout.response_length=4096 \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=16 \
    data.truncation='left' \
    data.return_raw_chat=True \
    actor_rollout_ref.rollout.multi_turn.tokenization_sanity_check_mode=ignore_strippable \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=8 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.rollout.top_p=0.9 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=20 \
    trainer.val_before_train=False \
    data.train_files=$DATA_TRAIN_FILE_PATH \
    data.val_files=$DATA_VAL_FILE_PATH \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=$TOOL_CONFIG_PATH \
    trainer.total_epochs=15 $@

