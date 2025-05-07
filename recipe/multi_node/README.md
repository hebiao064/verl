# Multi Node PPO on Kubernetes Cluster


Context:

- In some companies, MLE are not allowed to acquire mutliple nodes directly and run multi-node training like VERL's wiki showed: https://verl.readthedocs.io/en/latest/start/multinode.html
- The official way to run it is through Kuberay + Ray Operator to start a Ray Cluster Job
- We've seen huge burden on introducing Kuberay and maintaining it
- Given that VERL just need Ray Core and simply start a Ray Cluster and submit the job, we can simply hack it using a customized shell script

# Usage

1. Start a Pytorch Job (multi-node) on K8s cluster
2. Run `multi_node_ppo.sh` on all nodes assigned to the job, e.g: 1 master node + 3 worker nodes can run RL with 4 * 8 = 32 GPUs