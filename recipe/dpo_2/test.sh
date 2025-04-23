set -e
set -x
VISIBLE_DEVICES="4,5,6,7"

CUDA_VISIBLE_DEVICES=${VISIBLE_DEVICES} python3 -m recipe.dpo_2.main_ppo \
  data.train_files=$HOME/data/math/train.parquet \
  data.val_files=$HOME/data/math/test.parquet \
  data.train_batch_size=128 \
  data.val_batch_size=1312 \
  data.max_prompt_length=2048 \
  data.max_response_length=1024 \
  actor_rollout_ref.model.path=Qwen/Qwen2.5-7B-Instruct \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=64 \
  actor_rollout_ref.actor.ppo_micro_batch_size=8 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size=8 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
  actor_rollout_ref.ref.log_prob_micro_batch_size=4 \
  algorithm.kl_ctrl.kl_coef=0.001 \
  trainer.logger=['console','wandb'] \
  trainer.val_before_train=False \
  trainer.default_hdfs_dir=null \
  trainer.n_gpus_per_node=4 \
  trainer.nnodes=1 \
  trainer.save_freq=200 \
  trainer.test_freq=1 \
  +trainer.log_freq=1 \
  trainer.total_epochs=3 2>&1 | tee verl_demo.log