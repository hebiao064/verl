++ CUDA_VISIBLE_DEVICES=4,5,6,7
++ PYTHONUNBUFFERED=1
++ python3 -m recipe.spin.main_dpo '++data.train_files=['\''/shared/user/bhe/data/verl/math/train.parquet'\'']' '++data.val_files=['\''/shared/user/bhe/data/verl/math/test.parquet'\'']' ++data.train_batch_size=512 ++data.val_batch_size=32 ++data.max_prompt_length=1024 ++data.max_response_length=1024 ++actor_rollout_ref.model.path=/shared/public/elr-models/Qwen/Qwen2.5-7B-Instruct/52e20a6f5f475e5c8f6a8ebda4ae5fa6b1ea22ac ++critic.model.path=/shared/public/elr-models/Qwen/Qwen2.5-7B-Instruct/52e20a6f5f475e5c8f6a8ebda4ae5fa6b1ea22ac ++actor_rollout_ref.model.enable_gradient_checkpointing=True ++actor_rollout_ref.model.fsdp_config.model_dtype=bf16 ++actor_rollout_ref.actor.optim.lr=1e-7 ++actor_rollout_ref.actor.fsdp_config.param_offload=False ++actor_rollout_ref.actor.fsdp_config.optimizer_offload=False ++actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 ++actor_rollout_ref.rollout.name=sglang ++actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 ++actor_rollout_ref.rollout.tensor_model_parallel_size=2 ++actor_rollout_ref.rollout.gpu_memory_utilization=0.6 ++actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 ++reward_model.model.path=/shared/public/sharing/bhe/qwen/Qwen2-0.5B-Reward/snapshots/80fb188cabd3e854c2fb983f23a153c0e58a69e0 ++reward_model.micro_batch_size_per_gpu=2 ++algorithm.dpo_beta=0.05 ++algorithm.dpo_loss_type=sigmoid ++trainer.project_name=online_dpo_gsm_rm ++trainer.experiment_name=gsm_dpo_run_20250419_200541 ++trainer.default_local_dir=./dpo_checkpoints/gsm_dpo_run_20250419_200541 ++trainer.default_hdfs_dir=null ++trainer.n_gpus_per_node=4 ++trainer.nnodes=1 ++trainer.save_freq=-1 ++trainer.test_freq=1 ++trainer.log_freq=1 ++trainer.total_epochs=15 ++trainer.val_before_train=False
/home/jobuser/sglang/venv/lib/python3.10/site-packages/transformers/utils/hub.py:105: FutureWarning: Using `TRANSFORMERS_CACHE` is deprecated and will be removed in v5 of Transformers. Use `HF_HOME` instead.
  warnings.warn(
2025-04-19 20:05:50,607	INFO worker.py:1832 -- Started a local Ray instance. View the dashboard at [1m[32m127.0.0.1:8265 [39m[22m
[36m(pid=133975)[0m /home/jobuser/sglang/venv/lib/python3.10/site-packages/transformers/utils/hub.py:105: FutureWarning: Using `TRANSFORMERS_CACHE` is deprecated and will be removed in v5 of Transformers. Use `HF_HOME` instead.
[36m(pid=133975)[0m   warnings.warn(
[36m(TaskRunner pid=133975)[0m {'actor_rollout_ref': {'actor': {'checkpoint': {'contents': ['model',
[36m(TaskRunner pid=133975)[0m                                                              'optimizer',
[36m(TaskRunner pid=133975)[0m                                                              'extra']},
[36m(TaskRunner pid=133975)[0m                                  'clip_ratio': 0.2,
[36m(TaskRunner pid=133975)[0m                                  'clip_ratio_c': 3.0,
[36m(TaskRunner pid=133975)[0m                                  'clip_ratio_high': 0.2,
[36m(TaskRunner pid=133975)[0m                                  'clip_ratio_low': 0.2,
[36m(TaskRunner pid=133975)[0m                                  'entropy_coeff': 0.001,
[36m(TaskRunner pid=133975)[0m                                  'fsdp_config': {'fsdp_size': -1,
[36m(TaskRunner pid=133975)[0m                                                  'optimizer_offload': False,
[36m(TaskRunner pid=133975)[0m                                                  'param_offload': False,
[36m(TaskRunner pid=133975)[0m                                                  'wrap_policy': {'min_num_params': 0}},
[36m(TaskRunner pid=133975)[0m                                  'grad_clip': 1.0,
[36m(TaskRunner pid=133975)[0m                                  'kl_loss_coef': 0.001,
[36m(TaskRunner pid=133975)[0m                                  'kl_loss_type': 'low_var_kl',
[36m(TaskRunner pid=133975)[0m                                  'loss_agg_mode': 'token-mean',
[36m(TaskRunner pid=133975)[0m                                  'optim': {'lr': 1e-07,
[36m(TaskRunner pid=133975)[0m                                            'lr_warmup_steps': -1,
[36m(TaskRunner pid=133975)[0m                                            'lr_warmup_steps_ratio': 0.0,
[36m(TaskRunner pid=133975)[0m                                            'min_lr_ratio': None,
[36m(TaskRunner pid=133975)[0m                                            'total_training_steps': -1,
[36m(TaskRunner pid=133975)[0m                                            'warmup_style': 'constant',
[36m(TaskRunner pid=133975)[0m                                            'weight_decay': 0.01},
[36m(TaskRunner pid=133975)[0m                                  'ppo_epochs': 1,
[36m(TaskRunner pid=133975)[0m                                  'ppo_max_token_len_per_gpu': 16384,
[36m(TaskRunner pid=133975)[0m                                  'ppo_micro_batch_size': None,
[36m(TaskRunner pid=133975)[0m                                  'ppo_micro_batch_size_per_gpu': 2,
[36m(TaskRunner pid=133975)[0m                                  'ppo_mini_batch_size': 512,
[36m(TaskRunner pid=133975)[0m                                  'shuffle': False,
[36m(TaskRunner pid=133975)[0m                                  'strategy': 'fsdp',
[36m(TaskRunner pid=133975)[0m                                  'ulysses_sequence_parallel_size': 1,
[36m(TaskRunner pid=133975)[0m                                  'use_dynamic_bsz': False,
[36m(TaskRunner pid=133975)[0m                                  'use_kl_loss': False,
[36m(TaskRunner pid=133975)[0m                                  'use_torch_compile': True},
[36m(TaskRunner pid=133975)[0m                        'hybrid_engine': True,
[36m(TaskRunner pid=133975)[0m                        'model': {'enable_gradient_checkpointing': True,
[36m(TaskRunner pid=133975)[0m                                  'external_lib': None,
[36m(TaskRunner pid=133975)[0m                                  'fsdp_config': {'model_dtype': 'bf16'},
[36m(TaskRunner pid=133975)[0m                                  'override_config': {},
[36m(TaskRunner pid=133975)[0m                                  'path': '/shared/public/elr-models/Qwen/Qwen2.5-7B-Instruct/52e20a6f5f475e5c8f6a8ebda4ae5fa6b1ea22ac',
[36m(TaskRunner pid=133975)[0m                                  'use_remove_padding': False},
[36m(TaskRunner pid=133975)[0m                        'ref': {'fsdp_config': {'param_offload': False,
[36m(TaskRunner pid=133975)[0m                                                'wrap_policy': {'min_num_params': 0}},
[36m(TaskRunner pid=133975)[0m                                'log_prob_max_token_len_per_gpu': 16384,
[36m(TaskRunner pid=133975)[0m                                'log_prob_micro_batch_size': None,
[36m(TaskRunner pid=133975)[0m                                'log_prob_micro_batch_size_per_gpu': 2,
[36m(TaskRunner pid=133975)[0m                                'log_prob_use_dynamic_bsz': False,
[36m(TaskRunner pid=133975)[0m                                'ulysses_sequence_parallel_size': 1},
[36m(TaskRunner pid=133975)[0m                        'rollout': {'disable_log_stats': True,
[36m(TaskRunner pid=133975)[0m                                    'do_sample': True,
[36m(TaskRunner pid=133975)[0m                                    'dtype': 'bfloat16',
[36m(TaskRunner pid=133975)[0m                                    'enable_chunked_prefill': True,
[36m(TaskRunner pid=133975)[0m                                    'enforce_eager': True,
[36m(TaskRunner pid=133975)[0m                                    'free_cache_engine': True,
[36m(TaskRunner pid=133975)[0m                                    'gpu_memory_utilization': 0.6,
[36m(TaskRunner pid=133975)[0m                                    'ignore_eos': False,
[36m(TaskRunner pid=133975)[0m                                    'load_format': 'dummy_dtensor',
[36m(TaskRunner pid=133975)[0m                                    'log_prob_max_token_len_per_gpu': 16384,
[36m(TaskRunner pid=133975)[0m                                    'log_prob_micro_batch_size': None,
[36m(TaskRunner pid=133975)[0m                                    'log_prob_micro_batch_size_per_gpu': 2,
[36m(TaskRunner pid=133975)[0m                                    'log_prob_use_dynamic_bsz': False,
[36m(TaskRunner pid=133975)[0m                                    'max_model_len': None,
[36m(TaskRunner pid=133975)[0m                                    'max_num_batched_tokens': 8192,
[36m(TaskRunner pid=133975)[0m                                    'max_num_seqs': 1024,
[36m(TaskRunner pid=133975)[0m                                    'n': 1,
[36m(TaskRunner pid=133975)[0m                                    'name': 'sglang',
[36m(TaskRunner pid=133975)[0m                                    'prompt_length': 1024,
[36m(TaskRunner pid=133975)[0m                                    'response_length': 1024,
[36m(TaskRunner pid=133975)[0m                                    'temperature': 1.0,
[36m(TaskRunner pid=133975)[0m                                    'tensor_model_parallel_size': 2,
[36m(TaskRunner pid=133975)[0m                                    'top_k': -1,
[36m(TaskRunner pid=133975)[0m                                    'top_p': 1,
[36m(TaskRunner pid=133975)[0m                                    'use_fire_sampling': False,
[36m(TaskRunner pid=133975)[0m                                    'val_kwargs': {'do_sample': True,
[36m(TaskRunner pid=133975)[0m                                                   'n': 1,
[36m(TaskRunner pid=133975)[0m                                                   'temperature': 0.7,
[36m(TaskRunner pid=133975)[0m                                                   'top_k': -1,
[36m(TaskRunner pid=133975)[0m                                                   'top_p': 1.0}}},
[36m(TaskRunner pid=133975)[0m  'algorithm': {'adv_estimator': 'gae',
[36m(TaskRunner pid=133975)[0m                'dpo_beta': 0.05,
[36m(TaskRunner pid=133975)[0m                'dpo_loss_type': 'sigmoid',
[36m(TaskRunner pid=133975)[0m                'gamma': 1.0,
[36m(TaskRunner pid=133975)[0m                'kl_ctrl': {'horizon': 10000,
[36m(TaskRunner pid=133975)[0m                            'kl_coef': 0.001,
[36m(TaskRunner pid=133975)[0m                            'target_kl': 0.1,
[36m(TaskRunner pid=133975)[0m                            'type': 'fixed'},
[36m(TaskRunner pid=133975)[0m                'kl_penalty': 'kl',
[36m(TaskRunner pid=133975)[0m                'lam': 1.0,
[36m(TaskRunner pid=133975)[0m                'use_kl_in_reward': False},
[36m(TaskRunner pid=133975)[0m  'critic': {'checkpoint': {'contents': ['model', 'optimizer', 'extra']},
[36m(TaskRunner pid=133975)[0m             'cliprange_value': 0.5,
[36m(TaskRunner pid=133975)[0m             'forward_max_token_len_per_gpu': 32768,
[36m(TaskRunner pid=133975)[0m             'forward_micro_batch_size': 4,
[36m(TaskRunner pid=133975)[0m             'forward_micro_batch_size_per_gpu': None,
[36m(TaskRunner pid=133975)[0m             'grad_clip': 1.0,
[36m(TaskRunner pid=133975)[0m             'model': {'enable_gradient_checkpointing': True,
[36m(TaskRunner pid=133975)[0m                       'external_lib': None,
[36m(TaskRunner pid=133975)[0m                       'fsdp_config': {'fsdp_size': -1,
[36m(TaskRunner pid=133975)[0m                                       'optimizer_offload': False,
[36m(TaskRunner pid=133975)[0m                                       'param_offload': False,
[36m(TaskRunner pid=133975)[0m                                       'wrap_policy': {'min_num_params': 0}},
[36m(TaskRunner pid=133975)[0m                       'override_config': {},
[36m(TaskRunner pid=133975)[0m                       'path': '/shared/public/elr-models/Qwen/Qwen2.5-7B-Instruct/52e20a6f5f475e5c8f6a8ebda4ae5fa6b1ea22ac',
[36m(TaskRunner pid=133975)[0m                       'tokenizer_path': '/shared/public/elr-models/Qwen/Qwen2.5-7B-Instruct/52e20a6f5f475e5c8f6a8ebda4ae5fa6b1ea22ac',
[36m(TaskRunner pid=133975)[0m                       'use_remove_padding': False},
[36m(TaskRunner pid=133975)[0m             'optim': {'lr': 1e-05,
[36m(TaskRunner pid=133975)[0m                       'lr_warmup_steps_ratio': 0.0,
[36m(TaskRunner pid=133975)[0m                       'min_lr_ratio': None,
[36m(TaskRunner pid=133975)[0m                       'total_training_steps': -1,
[36m(TaskRunner pid=133975)[0m                       'warmup_style': 'constant',
[36m(TaskRunner pid=133975)[0m                       'weight_decay': 0.01},
[36m(TaskRunner pid=133975)[0m             'ppo_epochs': 1,
[36m(TaskRunner pid=133975)[0m             'ppo_max_token_len_per_gpu': 32768,
[36m(TaskRunner pid=133975)[0m             'ppo_micro_batch_size': 4,
[36m(TaskRunner pid=133975)[0m             'ppo_micro_batch_size_per_gpu': None,
[36m(TaskRunner pid=133975)[0m             'ppo_mini_batch_size': 512,
[36m(TaskRunner pid=133975)[0m             'rollout_n': 1,
[36m(TaskRunner pid=133975)[0m             'shuffle': False,
[36m(TaskRunner pid=133975)[0m             'strategy': 'fsdp',
[36m(TaskRunner pid=133975)[0m             'ulysses_sequence_parallel_size': 1,
[36m(TaskRunner pid=133975)[0m             'use_dynamic_bsz': False},
[36m(TaskRunner pid=133975)[0m  'custom_reward_function': {'name': 'compute_score', 'path': None},
[36m(TaskRunner pid=133975)[0m  'data': {'filter_overlong_prompts': False,
[36m(TaskRunner pid=133975)[0m           'filter_overlong_prompts_workers': 1,
[36m(TaskRunner pid=133975)[0m           'image_key': 'images',
[36m(TaskRunner pid=133975)[0m           'max_prompt_length': 1024,
[36m(TaskRunner pid=133975)[0m           'max_response_length': 1024,
[36m(TaskRunner pid=133975)[0m           'prompt_key': 'prompt',
[36m(TaskRunner pid=133975)[0m           'return_raw_chat': True,
[36m(TaskRunner pid=133975)[0m           'return_raw_input_ids': True,
[36m(TaskRunner pid=133975)[0m           'reward_fn_key': 'data_source',
[36m(TaskRunner pid=133975)[0m           'shuffle': True,
[36m(TaskRunner pid=133975)[0m           'tokenizer': None,
[36m(TaskRunner pid=133975)[0m           'train_batch_size': 512,
[36m(TaskRunner pid=133975)[0m           'train_files': ['/shared/user/bhe/data/verl/math/train.parquet'],
[36m(TaskRunner pid=133975)[0m           'truncation': 'error',
[36m(TaskRunner pid=133975)[0m           'val_batch_size': 32,
[36m(TaskRunner pid=133975)[0m           'val_files': ['/shared/user/bhe/data/verl/math/test.parquet']},
[36m(TaskRunner pid=133975)[0m  'reward_model': {'enable': True,
[36m(TaskRunner pid=133975)[0m                   'forward_max_token_len_per_gpu': 32768,
[36m(TaskRunner pid=133975)[0m                   'max_length': None,
[36m(TaskRunner pid=133975)[0m                   'micro_batch_size': None,
[36m(TaskRunner pid=133975)[0m                   'micro_batch_size_per_gpu': 2,
[36m(TaskRunner pid=133975)[0m                   'model': {'external_lib': None,
[36m(TaskRunner pid=133975)[0m                             'fsdp_config': {'fsdp_size': -1,
[36m(TaskRunner pid=133975)[0m                                             'param_offload': False,
[36m(TaskRunner pid=133975)[0m                                             'wrap_policy': {'min_num_params': 0}},
[36m(TaskRunner pid=133975)[0m                             'input_tokenizer': None,
[36m(TaskRunner pid=133975)[0m                             'path': '/shared/public/sharing/bhe/qwen/Qwen2-0.5B-Reward/snapshots/80fb188cabd3e854c2fb983f23a153c0e58a69e0',
[36m(TaskRunner pid=133975)[0m                             'use_remove_padding': False},
[36m(TaskRunner pid=133975)[0m                   'reward_manager': 'naive',
[36m(TaskRunner pid=133975)[0m                   'strategy': 'fsdp',
[36m(TaskRunner pid=133975)[0m                   'ulysses_sequence_parallel_size': 1,
[36m(TaskRunner pid=133975)[0m                   'use_dynamic_bsz': False},
[36m(TaskRunner pid=133975)[0m  'trainer': {'balance_batch': True,
[36m(TaskRunner pid=133975)[0m              'critic_warmup': 0,
[36m(TaskRunner pid=133975)[0m              'default_hdfs_dir': None,
[36m(TaskRunner pid=133975)[0m              'default_local_dir': './dpo_checkpoints/gsm_dpo_run_20250419_200541',
[36m(TaskRunner pid=133975)[0m              'del_local_ckpt_after_load': False,
[36m(TaskRunner pid=133975)[0m              'experiment_name': 'gsm_dpo_run_20250419_200541',
[36m(TaskRunner pid=133975)[0m              'log_freq': 1,
[36m(TaskRunner pid=133975)[0m              'log_val_generations': 0,
[36m(TaskRunner pid=133975)[0m              'logger': ['console', 'mlflow'],
[36m(TaskRunner pid=133975)[0m              'max_actor_ckpt_to_keep': None,
[36m(TaskRunner pid=133975)[0m              'max_critic_ckpt_to_keep': None,
[36m(TaskRunner pid=133975)[0m              'n_gpus_per_node': 4,
[36m(TaskRunner pid=133975)[0m              'nnodes': 1,
[36m(TaskRunner pid=133975)[0m              'project_name': 'online_dpo_gsm_rm',
[36m(TaskRunner pid=133975)[0m              'ray_wait_register_center_timeout': 300,
[36m(TaskRunner pid=133975)[0m              'resume_from_path': None,
[36m(TaskRunner pid=133975)[0m              'resume_mode': 'auto',
[36m(TaskRunner pid=133975)[0m              'save_freq': -1,
[36m(TaskRunner pid=133975)[0m              'test_freq': 1,
[36m(TaskRunner pid=133975)[0m              'total_epochs': 15,
[36m(TaskRunner pid=133975)[0m              'total_training_steps': None,
[36m(TaskRunner pid=133975)[0m              'val_before_train': False}}
[36m(TaskRunner pid=133975)[0m Biao local_path: /shared/public/elr-models/Qwen/Qwen2.5-7B-Instruct/52e20a6f5f475e5c8f6a8ebda4ae5fa6b1ea22ac
[33m(raylet)[0m [2025-04-19 20:06:00,489 E 120465 120495] (raylet) file_system_monitor.cc:116: /tmp/ray/session_2025-04-19_20-05-49_215536_119897 is over 95% full, available space: 26.4552 GB; capacity: 763.865 GB. Object creation will fail if spilling is required.
[36m(TaskRunner pid=133975)[0m WARNING: val_batch_size is deprecated. Validation datasets are sent to inference engines as a whole batch, which will schedule the memory themselves.
[36m(TaskRunner pid=133975)[0m [validate_config] All configuration checks passed successfully!
[36m(TaskRunner pid=133975)[0m dataset len: 7500
[36m(TaskRunner pid=133975)[0m dataset len: 5000
[36m(TaskRunner pid=133975)[0m Size of train dataloader: 14
[36m(TaskRunner pid=133975)[0m Total training steps: 210
[36m(TaskRunner pid=133975)[0m DeprecationWarning: `ray.state.available_resources_per_node` is a private attribute and access will be removed in a future Ray version.
[36m(TaskRunner pid=133975)[0m WARNING:2025-04-19 20:06:03,829:Waiting for register center actor T6sSV8_register_center to be ready. Elapsed time: 0 seconds out of 300 seconds.
[36m(pid=134850)[0m /home/jobuser/sglang/venv/lib/python3.10/site-packages/transformers/utils/hub.py:105: FutureWarning: Using `TRANSFORMERS_CACHE` is deprecated and will be removed in v5 of Transformers. Use `HF_HOME` instead.
[36m(pid=134850)[0m   warnings.warn(
[33m(raylet)[0m [2025-04-19 20:06:10,501 E 120465 120495] (raylet) file_system_monitor.cc:116: /tmp/ray/session_2025-04-19_20-05-49_215536_119897 is over 95% full, available space: 26.4612 GB; capacity: 763.865 GB. Object creation will fail if spilling is required.
[36m(TaskRunner pid=133975)[0m Biao critic_wg: <verl.single_controller.ray.base.RayWorkerGroup object at 0x718054221510>
[36m(pid=150266)[0m /home/jobuser/sglang/venv/lib/python3.10/site-packages/transformers/utils/hub.py:105: FutureWarning: Using `TRANSFORMERS_CACHE` is deprecated and will be removed in v5 of Transformers. Use `HF_HOME` instead.
[36m(pid=150266)[0m   warnings.warn(
[36m(pid=150265)[0m /home/jobuser/sglang/venv/lib/python3.10/site-packages/transformers/utils/hub.py:105: FutureWarning: Using `TRANSFORMERS_CACHE` is deprecated and will be removed in v5 of Transformers. Use `HF_HOME` instead.
[36m(pid=150265)[0m   warnings.warn(
[33m(raylet)[0m [2025-04-19 20:06:20,513 E 120465 120495] (raylet) file_system_monitor.cc:116: /tmp/ray/session_2025-04-19_20-05-49_215536_119897 is over 95% full, available space: 26.4611 GB; capacity: 763.865 GB. Object creation will fail if spilling is required.
[36m(WorkerDict pid=134850)[0m Critic overriding config {'bos_token_id': None, 'eos_token_id': 151645, 'pad_token_id': 151643}
[36m(WorkerDict pid=134850)[0m Biao's local_path for critic model: /shared/public/elr-models/Qwen/Qwen2.5-7B-Instruct/52e20a6f5f475e5c8f6a8ebda4ae5fa6b1ea22ac
[36m(WorkerDict pid=150266)[0m Flash Attention 2.0 only supports torch.float16 and torch.bfloat16 dtypes, but the current dype in Qwen2ForTokenClassification is torch.float32. You should run training or inference using Automatic Mixed-Precision via the `with torch.autocast(device_type='torch_device'):` decorator, or load the model with the `torch_dtype` argument. Example: `model = AutoModel.from_pretrained("openai/whisper-tiny", attn_implementation="flash_attention_2", torch_dtype=torch.float16)`
[36m(WorkerDict pid=150266)[0m You are attempting to use Flash Attention 2.0 with a model not initialized on GPU. Make sure to move the model to GPU after initializing it on CPU with `model.to('cuda')`.
[36m(WorkerDict pid=150267)[0m 
Loading checkpoint shards:   0%|          | 0/4 [00:00<?, ?it/s]
[36m(WorkerDict pid=134850)[0m 
Loading checkpoint shards:  25%|██▌       | 1/4 [00:03<00:09,  3.11s/it]
[36m(pid=150267)[0m /home/jobuser/sglang/venv/lib/python3.10/site-packages/transformers/utils/hub.py:105: FutureWarning: Using `TRANSFORMERS_CACHE` is deprecated and will be removed in v5 of Transformers. Use `HF_HOME` instead.
[36m(pid=150267)[0m   warnings.warn(
[36m(WorkerDict pid=134850)[0m Flash Attention 2.0 only supports torch.float16 and torch.bfloat16 dtypes, but the current dype in Qwen2ForTokenClassification is torch.float32. You should run training or inference using Automatic Mixed-Precision via the `with torch.autocast(device_type='torch_device'):` decorator, or load the model with the `torch_dtype` argument. Example: `model = AutoModel.from_pretrained("openai/whisper-tiny", attn_implementation="flash_attention_2", torch_dtype=torch.float16)`[32m [repeated 3x across cluster] (Ray deduplicates logs by default. Set RAY_DEDUP_LOGS=0 to disable log deduplication, or see https://docs.ray.io/en/master/ray-observability/user-guides/configure-logging.html#log-deduplication for more options.)[0m
[36m(WorkerDict pid=134850)[0m You are attempting to use Flash Attention 2.0 with a model not initialized on GPU. Make sure to move the model to GPU after initializing it on CPU with `model.to('cuda')`.[32m [repeated 3x across cluster][0m
[36m(WorkerDict pid=150265)[0m 
Loading checkpoint shards:   0%|          | 0/4 [00:00<?, ?it/s][32m [repeated 3x across cluster][0m
[33m(raylet)[0m [2025-04-19 20:06:30,525 E 120465 120495] (raylet) file_system_monitor.cc:116: /tmp/ray/session_2025-04-19_20-05-49_215536_119897 is over 95% full, available space: 26.4611 GB; capacity: 763.865 GB. Object creation will fail if spilling is required.
[36m(WorkerDict pid=150267)[0m 
Loading checkpoint shards:  50%|█████     | 2/4 [00:07<00:07,  3.55s/it][32m [repeated 7x across cluster][0m
[36m(WorkerDict pid=134850)[0m Qwen2ForTokenClassification contains 7.07B parameters
[36m(WorkerDict pid=134850)[0m Before critic FSDP, memory allocated (GB): 0.0, memory reserved (GB): 0.0
[36m(WorkerDict pid=150266)[0m Biao's local_path for critic model: /shared/public/elr-models/Qwen/Qwen2.5-7B-Instruct/52e20a6f5f475e5c8f6a8ebda4ae5fa6b1ea22ac[32m [repeated 3x across cluster][0m
[36m(WorkerDict pid=134850)[0m 
Loading checkpoint shards: 100%|██████████| 4/4 [00:11<00:00,  2.68s/it]
Loading checkpoint shards: 100%|██████████| 4/4 [00:11<00:00,  2.84s/it]
[36m(WorkerDict pid=134850)[0m Some weights of Qwen2ForTokenClassification were not initialized from the model checkpoint at /shared/public/elr-models/Qwen/Qwen2.5-7B-Instruct/52e20a6f5f475e5c8f6a8ebda4ae5fa6b1ea22ac and are newly initialized: ['score.bias', 'score.weight']
[36m(WorkerDict pid=134850)[0m You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
[36m(WorkerDict pid=134850)[0m NCCL version 2.21.5+cuda12.4
[33m(raylet)[0m [2025-04-19 20:06:40,537 E 120465 120495] (raylet) file_system_monitor.cc:116: /tmp/ray/session_2025-04-19_20-05-49_215536_119897 is over 95% full, available space: 26.4611 GB; capacity: 763.865 GB. Object creation will fail if spilling is required.
[36m(WorkerDict pid=150265)[0m 
Loading checkpoint shards:  75%|███████▌  | 3/4 [00:10<00:03,  3.56s/it][32m [repeated 4x across cluster][0m
[36m(WorkerDict pid=150265)[0m 
Loading checkpoint shards: 100%|██████████| 4/4 [00:13<00:00,  3.09s/it]
Loading checkpoint shards: 100%|██████████| 4/4 [00:13<00:00,  3.27s/it][32m [repeated 3x across cluster][0m
[36m(WorkerDict pid=150265)[0m Some weights of Qwen2ForTokenClassification were not initialized from the model checkpoint at /shared/public/elr-models/Qwen/Qwen2.5-7B-Instruct/52e20a6f5f475e5c8f6a8ebda4ae5fa6b1ea22ac and are newly initialized: ['score.bias', 'score.weight'][32m [repeated 3x across cluster][0m
[36m(WorkerDict pid=150265)[0m You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.[32m [repeated 3x across cluster][0m
[36m(WorkerDict pid=150265)[0m Total steps: 210, num_warmup_steps: 0
[36m(WorkerDict pid=150265)[0m Critic use_remove_padding=False
[36m(WorkerDict pid=134850)[0m After critic FSDP, memory allocated (GB): 6.585031509399414, memory reserved (GB): 14.33203125
[36m(WorkerDict pid=134850)[0m Model config after override: Qwen2Config {
[36m(WorkerDict pid=134850)[0m   "architectures": [
[36m(WorkerDict pid=134850)[0m     "Qwen2ForCausalLM"
[36m(WorkerDict pid=134850)[0m   ],
[36m(WorkerDict pid=134850)[0m   "attention_dropout": 0.0,
[36m(WorkerDict pid=134850)[0m   "eos_token_id": 151645,
[36m(WorkerDict pid=134850)[0m   "hidden_act": "silu",
[36m(WorkerDict pid=134850)[0m   "hidden_size": 3584,
[36m(WorkerDict pid=134850)[0m   "initializer_range": 0.02,
[36m(WorkerDict pid=134850)[0m   "intermediate_size": 18944,
[36m(WorkerDict pid=134850)[0m   "max_position_embeddings": 32768,
[36m(WorkerDict pid=134850)[0m   "max_window_layers": 28,
[36m(WorkerDict pid=134850)[0m   "model_type": "qwen2",
[36m(WorkerDict pid=134850)[0m   "num_attention_heads": 28,
[36m(WorkerDict pid=134850)[0m   "num_hidden_layers": 28,
[36m(WorkerDict pid=134850)[0m   "num_key_value_heads": 4,
[36m(WorkerDict pid=134850)[0m   "pad_token_id": 151643,
[36m(WorkerDict pid=134850)[0m   "rms_norm_eps": 1e-06,
[36m(WorkerDict pid=134850)[0m   "rope_scaling": null,
[36m(WorkerDict pid=134850)[0m   "rope_theta": 1000000.0,
[36m(WorkerDict pid=134850)[0m   "sliding_window": 131072,
[36m(WorkerDict pid=134850)[0m   "tie_word_embeddings": false,
[36m(WorkerDict pid=134850)[0m   "torch_dtype": "bfloat16",
[36m(WorkerDict pid=134850)[0m   "transformers_version": "4.51.3",
[36m(WorkerDict pid=134850)[0m   "use_cache": true,
[36m(WorkerDict pid=134850)[0m   "use_sliding_window": false,
[36m(WorkerDict pid=134850)[0m   "vocab_size": 152064
[36m(WorkerDict pid=134850)[0m }
[36m(WorkerDict pid=134850)[0m 
[36m(WorkerDict pid=134850)[0m 
Loading checkpoint shards:   0%|          | 0/4 [00:00<?, ?it/s]
[36m(WorkerDict pid=150267)[0m 
Loading checkpoint shards: 100%|██████████| 4/4 [00:00<00:00, 64.20it/s]
[36m(WorkerDict pid=134850)[0m Qwen2ForCausalLM contains 7.62B parameters
[36m(WorkerDict pid=134850)[0m wrap_policy: functools.partial(<function _or_policy at 0x713ba852bd00>, policies=[functools.partial(<function transformer_auto_wrap_policy at 0x713ba852bbe0>, transformer_layer_cls={<class 'transformers.models.qwen2.modeling_qwen2.Qwen2DecoderLayer'>})])
[36m(WorkerDict pid=134850)[0m Actor use_remove_padding=False
[36m(WorkerDict pid=134850)[0m Some weights of Qwen2ForTokenClassification were not initialized from the model checkpoint at /shared/public/sharing/bhe/qwen/Qwen2-0.5B-Reward/snapshots/80fb188cabd3e854c2fb983f23a153c0e58a69e0 and are newly initialized: ['score.bias']
[36m(WorkerDict pid=134850)[0m You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
[36m(WorkerDict pid=150265)[0m You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
[36m(WorkerDict pid=134850)[0m Model config after override: Qwen2Config {
[36m(WorkerDict pid=134850)[0m   "architectures": [
[36m(WorkerDict pid=134850)[0m     "Qwen2ForCausalLM"
[36m(WorkerDict pid=134850)[0m   ],
[36m(WorkerDict pid=134850)[0m   "attention_dropout": 0.0,
[36m(WorkerDict pid=134850)[0m   "eos_token_id": 151645,
[36m(WorkerDict pid=134850)[0m   "hidden_act": "silu",
[36m(WorkerDict pid=134850)[0m   "hidden_size": 3584,
[36m(WorkerDict pid=134850)[0m   "initializer_range": 0.02,
[36m(WorkerDict pid=134850)[0m   "intermediate_size": 18944,
[36m(WorkerDict pid=134850)[0m   "max_position_embeddings": 32768,
[36m(WorkerDict pid=134850)[0m   "max_window_layers": 28,
[36m(WorkerDict pid=134850)[0m   "model_type": "qwen2",
[36m(WorkerDict pid=134850)[0m   "num_attention_heads": 28,
[36m(WorkerDict pid=134850)[0m   "num_hidden_layers": 28,
[36m(WorkerDict pid=134850)[0m   "num_key_value_heads": 4,
[36m(WorkerDict pid=134850)[0m   "pad_token_id": 151643,
[36m(WorkerDict pid=134850)[0m   "rms_norm_eps": 1e-06,
[36m(WorkerDict pid=134850)[0m   "rope_scaling": null,
[36m(WorkerDict pid=134850)[0m   "rope_theta": 1000000.0,
[36m(WorkerDict pid=134850)[0m   "sliding_window": 131072,
[36m(WorkerDict pid=134850)[0m   "tie_word_embeddings": false,
[36m(WorkerDict pid=134850)[0m   "torch_dtype": "bfloat16",
[36m(WorkerDict pid=134850)[0m   "transformers_version": "4.51.3",
[36m(WorkerDict pid=134850)[0m   "use_cache": true,
[36m(WorkerDict pid=134850)[0m   "use_sliding_window": false,
[36m(WorkerDict pid=134850)[0m   "vocab_size": 152064
[36m(WorkerDict pid=134850)[0m }
[36m(WorkerDict pid=134850)[0m 
[36m(WorkerDict pid=134850)[0m Total steps: 210, num_warmup_steps: 0[32m [repeated 3x across cluster][0m
[36m(WorkerDict pid=134850)[0m Critic use_remove_padding=False[32m [repeated 3x across cluster][0m
[36m(WorkerDict pid=150266)[0m wrap_policy: functools.partial(<function _or_policy at 0x741f1721bd00>, policies=[functools.partial(<function transformer_auto_wrap_policy at 0x741f1721bbe0>, transformer_layer_cls={<class 'transformers.models.qwen2.modeling_qwen2.Qwen2DecoderLayer'>})])[32m [repeated 3x across cluster][0m
[36m(WorkerDict pid=134850)[0m Flash Attention 2.0 only supports torch.float16 and torch.bfloat16 dtypes, but the current dype in Qwen2ForCausalLM is torch.float32. You should run training or inference using Automatic Mixed-Precision via the `with torch.autocast(device_type='torch_device'):` decorator, or load the model with the `torch_dtype` argument. Example: `model = AutoModel.from_pretrained("openai/whisper-tiny", attn_implementation="flash_attention_2", torch_dtype=torch.float16)`
[36m(WorkerDict pid=134850)[0m 
Loading checkpoint shards:   0%|          | 0/4 [00:00<?, ?it/s][32m [repeated 4x across cluster][0m
[36m(WorkerDict pid=150266)[0m 
Loading checkpoint shards: 100%|██████████| 4/4 [00:00<00:00, 65.14it/s][32m [repeated 3x across cluster][0m
[33m(raylet)[0m [2025-04-19 20:06:50,548 E 120465 120495] (raylet) file_system_monitor.cc:116: /tmp/ray/session_2025-04-19_20-05-49_215536_119897 is over 95% full, available space: 26.461 GB; capacity: 763.865 GB. Object creation will fail if spilling is required.
[36m(WorkerDict pid=134850)[0m 
Loading checkpoint shards:  25%|██▌       | 1/4 [00:03<00:09,  3.13s/it]
[36m(WorkerDict pid=150267)[0m Some weights of Qwen2ForTokenClassification were not initialized from the model checkpoint at /shared/public/sharing/bhe/qwen/Qwen2-0.5B-Reward/snapshots/80fb188cabd3e854c2fb983f23a153c0e58a69e0 and are newly initialized: ['score.bias'][32m [repeated 3x across cluster][0m
[36m(WorkerDict pid=150267)[0m You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.[32m [repeated 2x across cluster][0m
[36m(WorkerDict pid=150266)[0m Flash Attention 2.0 only supports torch.float16 and torch.bfloat16 dtypes, but the current dype in Qwen2ForCausalLM is torch.float32. You should run training or inference using Automatic Mixed-Precision via the `with torch.autocast(device_type='torch_device'):` decorator, or load the model with the `torch_dtype` argument. Example: `model = AutoModel.from_pretrained("openai/whisper-tiny", attn_implementation="flash_attention_2", torch_dtype=torch.float16)`[32m [repeated 3x across cluster][0m
[36m(WorkerDict pid=150266)[0m 
Loading checkpoint shards:   0%|          | 0/4 [00:00<?, ?it/s][32m [repeated 3x across cluster][0m
[36m(WorkerDict pid=150267)[0m 
Loading checkpoint shards:  50%|█████     | 2/4 [00:08<00:08,  4.33s/it][32m [repeated 7x across cluster][0m
[33m(raylet)[0m [2025-04-19 20:07:00,559 E 120465 120495] (raylet) file_system_monitor.cc:116: /tmp/ray/session_2025-04-19_20-05-49_215536_119897 is over 95% full, available space: 26.4601 GB; capacity: 763.865 GB. Object creation will fail if spilling is required.
[36m(WorkerDict pid=134850)[0m 
Loading checkpoint shards: 100%|██████████| 4/4 [00:12<00:00,  3.06s/it]
Loading checkpoint shards: 100%|██████████| 4/4 [00:12<00:00,  3.09s/it]
[36m(WorkerDict pid=150267)[0m 
Loading checkpoint shards:  75%|███████▌  | 3/4 [00:12<00:04,  4.30s/it][32m [repeated 4x across cluster][0m
[36m(WorkerDict pid=134850)[0m Qwen2ForCausalLM contains 7.62B parameters
[36m(WorkerDict pid=150266)[0m Actor use_remove_padding=False[32m [repeated 3x across cluster][0m
[36m(WorkerDict pid=134850)[0m wrap_policy: functools.partial(<function _or_policy at 0x713ba852bd00>, policies=[functools.partial(<function transformer_auto_wrap_policy at 0x713ba852bbe0>, transformer_layer_cls={<class 'transformers.models.qwen2.modeling_qwen2.Qwen2DecoderLayer'>})])
[36m(WorkerDict pid=150265)[0m wrap_policy: functools.partial(<function _or_policy at 0x7867d8923d00>, policies=[functools.partial(<function transformer_auto_wrap_policy at 0x7867d8923be0>, transformer_layer_cls={<class 'transformers.models.qwen2.modeling_qwen2.Qwen2DecoderLayer'>})])
[36m(WorkerDict pid=150267)[0m Total steps: 210, num_warmup_steps: 0
[36m(WorkerDict pid=150267)[0m Building rollout for sglang
[36m(WorkerDict pid=150266)[0m NCCL version 2.21.5+cuda12.4
[36m(WorkerDict pid=134850)[0m Before building sglang rollout, memory allocated (GB): 13.67811393737793, memory reserved (GB): 28.48046875
Error executing job with overrides: ["++data.train_files=['/shared/user/bhe/data/verl/math/train.parquet']", "++data.val_files=['/shared/user/bhe/data/verl/math/test.parquet']", '++data.train_batch_size=512', '++data.val_batch_size=32', '++data.max_prompt_length=1024', '++data.max_response_length=1024', '++actor_rollout_ref.model.path=/shared/public/elr-models/Qwen/Qwen2.5-7B-Instruct/52e20a6f5f475e5c8f6a8ebda4ae5fa6b1ea22ac', '++critic.model.path=/shared/public/elr-models/Qwen/Qwen2.5-7B-Instruct/52e20a6f5f475e5c8f6a8ebda4ae5fa6b1ea22ac', '++actor_rollout_ref.model.enable_gradient_checkpointing=True', '++actor_rollout_ref.model.fsdp_config.model_dtype=bf16', '++actor_rollout_ref.actor.optim.lr=1e-7', '++actor_rollout_ref.actor.fsdp_config.param_offload=False', '++actor_rollout_ref.actor.fsdp_config.optimizer_offload=False', '++actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2', '++actor_rollout_ref.rollout.name=sglang', '++actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2', '++actor_rollout_ref.rollout.tensor_model_parallel_size=2', '++actor_rollout_ref.rollout.gpu_memory_utilization=0.6', '++actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2', '++reward_model.model.path=/shared/public/sharing/bhe/qwen/Qwen2-0.5B-Reward/snapshots/80fb188cabd3e854c2fb983f23a153c0e58a69e0', '++reward_model.micro_batch_size_per_gpu=2', '++algorithm.dpo_beta=0.05', '++algorithm.dpo_loss_type=sigmoid', '++trainer.project_name=online_dpo_gsm_rm', '++trainer.experiment_name=gsm_dpo_run_20250419_200541', '++trainer.default_local_dir=./dpo_checkpoints/gsm_dpo_run_20250419_200541', '++trainer.default_hdfs_dir=null', '++trainer.n_gpus_per_node=4', '++trainer.nnodes=1', '++trainer.save_freq=-1', '++trainer.test_freq=1', '++trainer.log_freq=1', '++trainer.total_epochs=15', '++trainer.val_before_train=False']
Traceback (most recent call last):
  File "/export/apps/python/3.10/lib/python3.10/runpy.py", line 196, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "/export/apps/python/3.10/lib/python3.10/runpy.py", line 86, in _run_code
    exec(code, run_globals)
  File "/home/jobuser/spin/verl/recipe/spin/main_dpo.py", line 200, in <module>
    main()
  File "/home/jobuser/sglang/venv/lib/python3.10/site-packages/hydra/main.py", line 94, in decorated_main
    _run_hydra(
  File "/home/jobuser/sglang/venv/lib/python3.10/site-packages/hydra/_internal/utils.py", line 394, in _run_hydra
    _run_app(
  File "/home/jobuser/sglang/venv/lib/python3.10/site-packages/hydra/_internal/utils.py", line 457, in _run_app
    run_and_report(
  File "/home/jobuser/sglang/venv/lib/python3.10/site-packages/hydra/_internal/utils.py", line 223, in run_and_report
    raise ex
  File "/home/jobuser/sglang/venv/lib/python3.10/site-packages/hydra/_internal/utils.py", line 220, in run_and_report
    return func()
  File "/home/jobuser/sglang/venv/lib/python3.10/site-packages/hydra/_internal/utils.py", line 458, in <lambda>
    lambda: hydra.run(
  File "/home/jobuser/sglang/venv/lib/python3.10/site-packages/hydra/_internal/hydra.py", line 132, in run
    _ = ret.return_value
  File "/home/jobuser/sglang/venv/lib/python3.10/site-packages/hydra/core/utils.py", line 260, in return_value
    raise self._return_value
  File "/home/jobuser/sglang/venv/lib/python3.10/site-packages/hydra/core/utils.py", line 186, in run_job
    ret.return_value = task_function(task_cfg)
  File "/home/jobuser/spin/verl/recipe/spin/main_dpo.py", line 59, in main
    run_ppo(config)
  File "/home/jobuser/spin/verl/recipe/spin/main_dpo.py", line 77, in run_ppo
    ray.get(runner.run.remote(config))
  File "/home/jobuser/sglang/venv/lib/python3.10/site-packages/ray/_private/auto_init_hook.py", line 21, in auto_init_wrapper
    return fn(*args, **kwargs)
  File "/home/jobuser/sglang/venv/lib/python3.10/site-packages/ray/_private/client_mode_hook.py", line 103, in wrapper
    return func(*args, **kwargs)
  File "/home/jobuser/sglang/venv/lib/python3.10/site-packages/ray/_private/worker.py", line 2771, in get
    values, debugger_breakpoint = worker.get_objects(object_refs, timeout=timeout)
  File "/home/jobuser/sglang/venv/lib/python3.10/site-packages/ray/_private/worker.py", line 919, in get_objects
    raise value.as_instanceof_cause()
ray.exceptions.RayTaskError(RuntimeError): [36mray::TaskRunner.run()[39m (pid=133975, ip=100.96.252.211, actor_id=973645b1307ede5cc2d5a4b601000000, repr=<main_dpo.TaskRunner object at 0x71b10d57b520>)
  File "/home/jobuser/spin/verl/recipe/spin/main_dpo.py", line 194, in run
    trainer.init_workers()
  File "/home/jobuser/spin/verl/recipe/spin/ray_trainer.py", line 620, in init_workers
    self.actor_rollout_wg.init_model()
  File "/home/jobuser/spin/verl/verl/single_controller/ray/base.py", line 43, in func
    output = ray.get(output)
ray.exceptions.RayTaskError(RuntimeError): [36mray::WorkerDict.actor_rollout_init_model()[39m (pid=150267, ip=100.96.252.211, actor_id=074e1886e8279250bc26618401000000, repr=<verl.single_controller.ray.base.WorkerDict object at 0x7a84ac3a5cc0>)
  File "/home/jobuser/spin/verl/verl/single_controller/ray/base.py", line 439, in func
    return getattr(self.worker_dict[key], name)(*args, **kwargs)
  File "/home/jobuser/spin/verl/verl/single_controller/base/decorator.py", line 409, in inner
    return func(*args, **kwargs)
  File "/home/jobuser/spin/verl/recipe/spin/fsdp_workers.py", line 423, in init_model
    self.rollout, self.rollout_sharding_manager = self._build_rollout(
  File "/home/jobuser/spin/verl/recipe/spin/fsdp_workers.py", line 359, in _build_rollout
    rollout = SGLangRollout(actor_module=self.config.model.path,
  File "/home/jobuser/spin/verl/verl/workers/rollout/sglang_rollout/sglang_rollout.py", line 152, in __init__
    [ip, port_args] = broadcast_pyobj([ip, port_args],
  File "/home/jobuser/sglang/venv/lib/python3.10/site-packages/sglang/srt/utils.py", line 868, in broadcast_pyobj
    dist.broadcast(tensor_size, src=src, group=dist_group)
  File "/home/jobuser/sglang/venv/lib/python3.10/site-packages/torch/distributed/c10d_logger.py", line 81, in wrapper
    return func(*args, **kwargs)
  File "/home/jobuser/sglang/venv/lib/python3.10/site-packages/torch/distributed/distributed_c10d.py", line 2726, in broadcast
    work = group.broadcast([tensor], opts)
RuntimeError: No backend type associated with device type cpu
[36m(TaskRunner pid=133975)[0m Unhandled error (suppress with 'RAY_IGNORE_UNHANDLED_ERRORS=1'): [36mray::WorkerDict.actor_rollout_init_model()[39m (pid=150266, ip=100.96.252.211, actor_id=559fba35d4847535d08f581b01000000, repr=<verl.single_controller.ray.base.WorkerDict object at 0x741eec22dd50>)
[36m(TaskRunner pid=133975)[0m   File "/home/jobuser/spin/verl/verl/single_controller/ray/base.py", line 439, in func
[36m(TaskRunner pid=133975)[0m     return getattr(self.worker_dict[key], name)(*args, **kwargs)
[36m(TaskRunner pid=133975)[0m   File "/home/jobuser/spin/verl/verl/single_controller/base/decorator.py", line 409, in inner
[36m(TaskRunner pid=133975)[0m     return func(*args, **kwargs)
[36m(TaskRunner pid=133975)[0m   File "/home/jobuser/spin/verl/recipe/spin/fsdp_workers.py", line 423, in init_model
[36m(TaskRunner pid=133975)[0m     self.rollout, self.rollout_sharding_manager = self._build_rollout(
[36m(TaskRunner pid=133975)[0m   File "/home/jobuser/spin/verl/recipe/spin/fsdp_workers.py", line 359, in _build_rollout
[36m(TaskRunner pid=133975)[0m     rollout = SGLangRollout(actor_module=self.config.model.path,
[36m(TaskRunner pid=133975)[0m   File "/home/jobuser/spin/verl/verl/workers/rollout/sglang_rollout/sglang_rollout.py", line 152, in __init__
[36m(TaskRunner pid=133975)[0m     [ip, port_args] = broadcast_pyobj([ip, port_args],
[36m(TaskRunner pid=133975)[0m   File "/home/jobuser/sglang/venv/lib/python3.10/site-packages/sglang/srt/utils.py", line 863, in broadcast_pyobj
[36m(TaskRunner pid=133975)[0m     dist.broadcast(tensor_size, src=src, group=dist_group)
[36m(TaskRunner pid=133975)[0m   File "/home/jobuser/sglang/venv/lib/python3.10/site-packages/torch/distributed/c10d_logger.py", line 81, in wrapper
[36m(TaskRunner pid=133975)[0m     return func(*args, **kwargs)
[36m(TaskRunner pid=133975)[0m   File "/home/jobuser/sglang/venv/lib/python3.10/site-packages/torch/distributed/distributed_c10d.py", line 2726, in broadcast
[36m(TaskRunner pid=133975)[0m     work = group.broadcast([tensor], opts)
[36m(TaskRunner pid=133975)[0m RuntimeError: No backend type associated with device type cpu
[36m(WorkerDict pid=150267)[0m 
Loading checkpoint shards: 100%|██████████| 4/4 [00:16<00:00,  4.09s/it]
Loading checkpoint shards: 100%|██████████| 4/4 [00:16<00:00,  4.18s/it][32m [repeated 3x across cluster][0m
[36m(WorkerDict pid=150266)[0m /home/jobuser/sglang/venv/lib/python3.10/site-packages/sglang/srt/utils.py:858: UserWarning: The given NumPy array is not writable, and PyTorch does not support non-writable tensors. This means writing to this tensor will result in undefined behavior. You may want to copy the array to protect its data or make it writable before converting it to a tensor. This type of warning will be suppressed for the rest of this program. (Triggered internally at /pytorch/torch/csrc/utils/tensor_numpy.cpp:203.)
[36m(WorkerDict pid=150266)[0m   tensor_data = torch.ByteTensor(
[36m(WorkerDict pid=134850)[0m Actor use_remove_padding=False[32m [repeated 4x across cluster][0m
[36m(WorkerDict pid=150266)[0m wrap_policy: functools.partial(<function _or_policy at 0x741f1721bd00>, policies=[functools.partial(<function transformer_auto_wrap_policy at 0x741f1721bbe0>, transformer_layer_cls={<class 'transformers.models.qwen2.modeling_qwen2.Qwen2DecoderLayer'>})])[32m [repeated 2x across cluster][0m
[36m(WorkerDict pid=134850)[0m Total steps: 210, num_warmup_steps: 0[32m [repeated 3x across cluster][0m
[36m(WorkerDict pid=134850)[0m Building rollout for sglang[32m [repeated 3x across cluster][0m
