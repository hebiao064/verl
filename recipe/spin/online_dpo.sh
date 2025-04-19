#!/bin/bash

set -e
set -x

export HYDRA_FULL_ERROR=1 
export HF_HUB_OFFLINE=1
export HF_HOME=/home/jobuser/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_METRICS_CACHE=$HF_HOME/metrics
export SGLANG_HOST_IP=127.0.0.1
export HOST_IP=127.0.0.1
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=WARN
export WANDB_API_KEY=25c95cfb8dfe322ae6d944a369d2ae63b65d9ece


# MODEL_NAME="Qwen/Qwen2-7B-Instruct"
MODEL_NAME="/shared/public/elr-models/Qwen/Qwen2.5-7B-Instruct/52e20a6f5f475e5c8f6a8ebda4ae5fa6b1ea22ac"
BASE_MODEL_PATH=$MODEL_NAME
REWARD_MODEL_PATH="/shared/public/sharing/bhe/qwen/Qwen2-0.5B-Reward"

TRAIN_DATA_PATH="/shared/user/bhe/data/verl/math/train.parquet"
VAL_DATA_PATH="/shared/user/bhe/data/verl/math/test.parquet"
TRAIN_PROMPT_FILES="['${TRAIN_DATA_PATH}']"
VAL_PROMPT_FILES="['${VAL_DATA_PATH}']"

PROJECT_NAME="online_dpo_gsm_rm"
EXPERIMENT_NAME="gsm_dpo_run_$(date +%Y%m%d_%H%M%S)"
CHECKPOINT_DIR="./dpo_checkpoints/${EXPERIMENT_NAME}"
LOG_FILE="${CHECKPOINT_DIR}/dpo_run.log"

NNODES=1
N_GPUS_PER_NODE=4
VISIBLE_DEVICES="4,5,6,7"
TRAIN_BATCH_SIZE=512
VAL_BATCH_SIZE=32
MAX_PROMPT_LENGTH=1024
MAX_RESPONSE_LENGTH=1024
LEARNING_RATE=1e-7
DPO_BETA=0.05
DPO_LOSS_TYPE="sigmoid"
TOTAL_EPOCHS=15
SAVE_FREQ=-1
TEST_FREQ=1
LOG_FREQ=1

MICRO_BATCH_SIZE_PER_GPU=2
RM_MICRO_BATCH_SIZE_PER_GPU=2
ENABLE_GC=True
PARAM_OFFLOAD=False
OPTIM_OFFLOAD=False
MODEL_DTYPE="bf16"

ROLLOUT_BACKEND="vllm"
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=2
VLLM_TP_SIZE=2
VLLM_GPU_MEM_UTIL=0.6

PYTHON_MODULE="recipe.spin.main_dpo"

COMMAND="CUDA_VISIBLE_DEVICES=${VISIBLE_DEVICES} PYTHONUNBUFFERED=1 python3 -m ${PYTHON_MODULE} \
    ++data.train_files=\"${TRAIN_PROMPT_FILES}\" \
    ++data.val_files=\"${VAL_PROMPT_FILES}\" \
    ++data.train_batch_size=${TRAIN_BATCH_SIZE} \
    ++data.val_batch_size=${VAL_BATCH_SIZE} \
    ++data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    ++data.max_response_length=${MAX_RESPONSE_LENGTH} \
    ++actor_rollout_ref.model.path=\"${BASE_MODEL_PATH}\" \
    ++critic.model.path=\"${BASE_MODEL_PATH}\" \
    ++actor_rollout_ref.model.enable_gradient_checkpointing=${ENABLE_GC} \
    ++actor_rollout_ref.model.fsdp_config.model_dtype=${MODEL_DTYPE} \
    ++actor_rollout_ref.actor.optim.lr=${LEARNING_RATE} \
    ++actor_rollout_ref.actor.fsdp_config.param_offload=${PARAM_OFFLOAD} \
    ++actor_rollout_ref.actor.fsdp_config.optimizer_offload=${OPTIM_OFFLOAD} \
    ++actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${MICRO_BATCH_SIZE_PER_GPU} \
    ++actor_rollout_ref.rollout.name=${ROLLOUT_BACKEND} \
    ++actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU} \
    ++actor_rollout_ref.rollout.tensor_model_parallel_size=${VLLM_TP_SIZE} \
    ++actor_rollout_ref.rollout.gpu_memory_utilization=${VLLM_GPU_MEM_UTIL} \
    ++actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU} \
    ++reward_model.model.path=\"${REWARD_MODEL_PATH}\" \
    ++reward_model.micro_batch_size_per_gpu=${RM_MICRO_BATCH_SIZE_PER_GPU} \
    ++algorithm.dpo_beta=${DPO_BETA} \
    ++algorithm.dpo_loss_type=\"${DPO_LOSS_TYPE}\" \
    ++trainer.project_name=\"${PROJECT_NAME}\" \
    ++trainer.experiment_name=\"${EXPERIMENT_NAME}\" \
    ++trainer.default_local_dir=\"${CHECKPOINT_DIR}\" \
    ++trainer.default_hdfs_dir=null \
    ++trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    ++trainer.nnodes=${NNODES} \
    ++trainer.save_freq=${SAVE_FREQ} \
    ++trainer.test_freq=${TEST_FREQ} \
    ++trainer.log_freq=${LOG_FREQ} \
    ++trainer.total_epochs=${TOTAL_EPOCHS} \
    ++trainer.val_before_train=False"

echo "Running command:"
echo "${COMMAND}"

mkdir -p "${CHECKPOINT_DIR}"

eval "${COMMAND}" 2>&1 | tee "${LOG_FILE}"

echo "Script finished. Log saved to ${LOG_FILE}"