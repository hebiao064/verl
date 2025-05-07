# Ensure WORLD_SIZE is set (default to 2 if unset)
WORLD_SIZE=${WORLD_SIZE:-2}

if [ "$HOSTNAME" = "$MASTER_ADDR" ]; then
    ray start --head --dashboard-host=0.0.0.0
else
    ray start --address "$MASTER_ADDR:6379"
fi

# Run for loop for up to 3 minutes (180 seconds) to wait for Ray to start
for i in {1..180}; do
    # Capture node list output, handle errors gracefully
    node_output=$(ray list nodes 2>/dev/null || echo "")
    # Extract node count, default to 0 if not found
    node_count=$(echo "$node_output" | grep -E "^Total: " | awk '{print $2}' || echo "0")

    if [ "$node_count" -eq "$WORLD_SIZE" ]; then
        echo "Ray started and has $WORLD_SIZE nodes"
        break
    fi

    sleep 1
    echo "Number of Ray nodes: $node_count"
done

#!/bin/bash
# Only the head node is submitting the job to avoid duplicating job submission
if [ `hostname` == $MASTER_ADDR ]; then
    ray job submit -- \
    python -m verl.trainer.main_ppo \
        data.train_files=/shared/public/data/gsm8k/train.parquet \
        data.val_files=/shared/public/data/gsm8k/test.parquet \
        data.train_batch_size=256 \
        data.max_prompt_length=512 \
        data.max_response_length=256 \
        actor_rollout_ref.model.path=/shared/public/elr-models/Qwen/Qwen2.5-0.5B-Instruct/a8b602d9dafd3a75d382e62757d83d89fca3be54 \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.actor.ppo_mini_batch_size=64 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
        critic.optim.lr=1e-5 \
        critic.model.path=/shared/public/elr-models/Qwen/Qwen2.5-0.5B-Instruct/a8b602d9dafd3a75d382e62757d83d89fca3be54 \
        critic.ppo_micro_batch_size_per_gpu=4 \
        algorithm.kl_ctrl.kl_coef=0.001 \
        trainer.logger=['console'] \
        trainer.default_hdfs_dir=null \
        trainer.n_gpus_per_node=$GPUS_PER_NODE \
        trainer.nnodes=$WORLD_SIZE \
        trainer.save_freq=10 \
        trainer.test_freq=10 \
        trainer.total_epochs=1
    # shutdown cluster when job finishes
    ray stop
else
    # worker nodes should busy waiting until cluster shutdown
    while true; do
        sleep 10
        if ray health-check --address "$MASTER_ADDR:6379" > /dev/null 2>&1; then
            echo "Ray cluster is active. Keeping worker node active..."
        else
            echo "Ray cluster is shutting down. Worker node exiting..."
            exit 0
        fi
    done
fi