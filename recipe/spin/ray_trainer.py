"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""
import os
import uuid
import torch.nn.functional as F
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Type, Dict
from copy import deepcopy
from collections import defaultdict
from functools import partial
from tqdm import tqdm
import ray
import numpy as np
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.metric_utils import compute_data_metrics, compute_throughout_metrics, compute_timing_metrics, reduce_metrics, bootstrap_metric, calc_maj_val, process_validation_metrics
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
from verl.utils.tracking import ValidationGenerationsLogger
from torch.utils.data import RandomSampler, SequentialSampler, DataLoader
from torchdata.stateful_dataloader import StatefulDataLoader
from tensordict import TensorDict

WorkerType = Type[Worker]

class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """
    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6

class AdvantageEstimator(str, Enum):
    """
    Using an enumeration class to avoid spelling errors in adv_estimator
    """
    GAE = 'gae'
    GRPO = 'grpo'
    REINFORCE_PLUS_PLUS = 'reinforce_plus_plus'
    REINFORCE_PLUS_PLUS_BASELINE = 'reinforce_plus_plus_baseline'
    REMAX = 'remax'
    RLOO = 'rloo'

@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    Mapping
    """
    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)
    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes,
                                            use_gpu=True,
                                            max_colocate_count=1,
                                            name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool
        self._check_resource_available()
    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]
    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])
    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray.state.available_resources_per_node()
        node_available_gpus = {node: node_info.get('GPU', 0) for node, node_info in node_available_resources.items()}
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}")
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            num_gpus, num_nodes = process_on_nodes[0], len(process_on_nodes)
            for node, available_gpus in node_available_gpus.items():
                if available_gpus >= num_gpus:
                    node_available_gpus[node] -= num_gpus
                    num_nodes -= 1
                    if num_nodes == 0:
                        break
            if num_nodes > 0:
                raise ValueError(
                    f"Resource pool {resource_pool_name}: {num_gpus}*{num_nodes} cannot be satisfied in this ray cluster"
                )
import torch
from verl.utils.torch_functional import masked_mean
def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty='kl'):
    responses = data.batch['responses']
    response_length = responses.size(1)
    token_level_scores = data.batch['token_level_scores']
    batch_size = data.batch.batch_size[0]
    attention_mask = data.batch['attention_mask']
    response_mask = attention_mask[:, -response_length:]
    kld = core_algos.kl_penalty(data.batch['old_log_probs'], data.batch['ref_log_prob'],
                                kl_penalty=kl_penalty)
    kld = kld * response_mask
    beta = kl_ctrl.value
    token_level_rewards = token_level_scores - beta * kld
    current_kl = masked_mean(kld, mask=response_mask, axis=-1)
    current_kl = torch.mean(current_kl, dim=0).item()
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch['token_level_rewards'] = token_level_rewards
    metrics = {'actor/reward_kl_penalty': current_kl, 'actor/reward_kl_penalty_coeff': beta}
    return data, metrics
def compute_response_mask(data: DataProto):
    responses = data.batch['responses']
    response_length = responses.size(1)
    attention_mask = data.batch['attention_mask']
    return attention_mask[:, -response_length:]
def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1):
    if "response_mask" not in data.batch.keys():
        data.batch['response_mask'] = compute_response_mask(data)
    if adv_estimator == AdvantageEstimator.GAE:
        values = data.batch['values']
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch['token_level_rewards'],
            values=data.batch['values'],
            response_mask=data.batch['response_mask'],
            gamma=gamma,
            lam=lam)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.GRPO:
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch['token_level_rewards'],
            response_mask=data.batch['response_mask'],
            index=data.non_tensor_batch['uid'])
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE:
        advantages, returns = core_algos.compute_reinforce_plus_plus_baseline_outcome_advantage(
            token_level_rewards=data.batch['token_level_rewards'],
            response_mask=data.batch['response_mask'],
            index=data.non_tensor_batch['uid'])
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS:
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards=data.batch['token_level_rewards'],
            response_mask=data.batch['response_mask'],
            gamma=gamma)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.REMAX:
        advantages, returns = core_algos.compute_remax_outcome_advantage(
            token_level_rewards=data.batch['token_level_rewards'],
            reward_baselines=data.batch['reward_baselines'],
            response_mask=data.batch['response_mask'])
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.RLOO:
        advantages, returns = core_algos.compute_rloo_outcome_advantage(
            token_level_rewards=data.batch['token_level_rewards'],
            response_mask=data.batch['response_mask'],
            index=data.non_tensor_batch['uid'])
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    else:
        raise NotImplementedError
    return data
@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    timing_raw[name] = timer.last
class RayPPOTrainer(object):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """
    def __init__(self,
                 config,
                 tokenizer,
                 role_worker_mapping: dict[Role, WorkerType],
                 resource_pool_manager: ResourcePoolManager,
                 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
                 processor=None,
                 reward_fn=None,
                 val_reward_fn=None):
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn
        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, 'Currently, only support hybrid engine'
        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f'{role_worker_mapping.keys()=}'
        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.validation_generations_logger = ValidationGenerationsLogger()
        if config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(config.algorithm.kl_ctrl)
        if self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in [
                AdvantageEstimator.GRPO, AdvantageEstimator.REINFORCE_PLUS_PLUS, AdvantageEstimator.REMAX,
                AdvantageEstimator.RLOO, AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE
        ]:
            self.use_critic = False
        else:
            raise NotImplementedError
        self._validate_config()
        self._create_dataloader()
    
    def _validate_config(self):
        config = self.config
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % n_gpus == 0, \
            f"real_train_batch_size ({real_train_batch_size}) must be divisible by total n_gpus ({n_gpus})."
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            settings = {
                "actor_rollout_ref.actor": "micro_batch_size",
                "critic": "micro_batch_size",
                "reward_model": "micro_batch_size",
                "actor_rollout_ref.ref": "log_prob_micro_batch_size",
                "actor_rollout_ref.rollout": "log_prob_micro_batch_size",
            }
            if name in settings:
                param = settings[name]
                param_per_gpu = f"{param}_per_gpu"
                if mbs is None and mbs_per_gpu is None:
                    raise ValueError(
                        f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'.")
                if mbs is not None and mbs_per_gpu is not None:
                    raise ValueError(
                        f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. "
                        f"Please remove '{name}.{param}' because only '*_{param_per_gpu}' is supported (the former is deprecated)."
                    )
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            check_mutually_exclusive(config.actor_rollout_ref.actor.ppo_micro_batch_size,
                                     config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                                     "actor_rollout_ref.actor")
            if self.use_reference_policy:
                check_mutually_exclusive(config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                                         config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                                         "actor_rollout_ref.ref")
            check_mutually_exclusive(config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                                     config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                                     "actor_rollout_ref.rollout")
        if self.use_critic and not config.critic.use_dynamic_bsz:
            check_mutually_exclusive(config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu,
                                     "critic")
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu,
                                     "reward_model")
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.actor_rollout_ref.actor.ppo_mini_batch_size
            sp_size = config.actor_rollout_ref.actor.get('ulysses_sequence_parallel_size', 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert config.actor_rollout_ref.actor.ppo_mini_batch_size % config.actor_rollout_ref.actor.ppo_micro_batch_size == 0
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus
        assert config.actor_rollout_ref.actor.loss_agg_mode in [
            "token-mean", "seq-mean-token-sum", "seq-mean-token-mean"
        ], f"Invalid loss_agg_mode: {config.actor_rollout_ref.actor.loss_agg_mode}"
        if config.algorithm.use_kl_in_reward and config.actor_rollout_ref.actor.use_kl_loss:
            print(f"NOTICE: You have both enabled in-reward kl and kl loss.")
        if self.use_critic and not config.critic.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.critic.ppo_mini_batch_size
            sp_size = config.critic.get('ulysses_sequence_parallel_size', 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus
        if config.actor_rollout_ref.actor.strategy == 'fsdp':
            if config.actor_rollout_ref.actor.get('ulysses_sequence_parallel_size', 1) > 1 or \
                    config.actor_rollout_ref.ref.get('ulysses_sequence_parallel_size', 1) > 1:
                assert config.actor_rollout_ref.model.use_remove_padding, \
                    "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."
        if self.use_critic and config.critic.strategy == 'fsdp':
            if config.critic.get('ulysses_sequence_parallel_size', 1) > 1:
                assert config.critic.model.use_remove_padding, \
                    "When using sequence parallelism for critic, you must enable `use_remove_padding`."
        if config.data.get('val_batch_size', None) is not None:
            print(
                f"WARNING: val_batch_size is deprecated. Validation datasets are sent to inference engines as a whole batch, which will schedule the memory themselves."
            )
        if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
            assert config.actor_rollout_ref.rollout.temperature > 0, \
                "validation gen temperature should be greater than 0 when enabling do_sample"
        print("[validate_config] All configuration checks passed successfully!")
    
    # def _create_dataloader(self):
    #     self.train_dataset = RLHFDataset(parquet_files=self.config.data.train_files,
    #                                      tokenizer=self.tokenizer,
    #                                      processor=self.processor,
    #                                      prompt_key=self.config.data.prompt_key,
    #                                      image_key=self.config.data.get('image_key', 'images'),
    #                                      max_prompt_length=self.config.data.max_prompt_length,
    #                                      return_raw_chat=self.config.data.get('return_raw_chat', False),
    #                                      truncation=self.config.data.get('truncation', 'error'),
    #                                      filter_overlong_prompts=self.config.data.filter_overlong_prompts,
    #                                      num_workers=self.config.data.get('filter_overlong_prompts_workers', None))
    #     assert self.train_dataset.truncation == self.config.data.get(
    #         'truncation', 'error'
    #     ), f'dataset truncation {self.train_dataset.truncation} must be the same as config {self.config.data.get("truncation", "error")}'
    #     if self.config.data.shuffle:
    #         train_dataloader_generator = torch.Generator()
    #         train_dataloader_generator.manual_seed(self.config.data.get('seed', 1))
    #         sampler = RandomSampler(data_source=self.train_dataset, generator=train_dataloader_generator)
    #     else:
    #         sampler = SequentialSampler(data_source=self.train_dataset)
    #     self.train_dataloader = StatefulDataLoader(dataset=self.train_dataset,
    #                                                batch_size=self.config.data.get('gen_batch_size',
    #                                                                                self.config.data.train_batch_size),
    #                                                num_workers=8,
    #                                                drop_last=True,
    #                                                collate_fn=collate_fn,
    #                                                sampler=sampler)
    #     self.val_dataset = RLHFDataset(parquet_files=self.config.data.val_files,
    #                                    tokenizer=self.tokenizer,
    #                                    processor=self.processor,
    #                                    prompt_key=self.config.data.prompt_key,
    #                                    image_key=self.config.data.get('image_key', 'images'),
    #                                    max_prompt_length=self.config.data.max_prompt_length,
    #                                    return_raw_chat=self.config.data.get('return_raw_chat', False),
    #                                    truncation=self.config.data.get('truncation', 'error'),
    #                                    filter_overlong_prompts=self.config.data.filter_overlong_prompts,
    #                                    num_workers=self.config.data.get('filter_overlong_prompts_workers', None))
    #     assert self.val_dataset.truncation == self.config.data.get(
    #         'truncation', 'error'
    #     ), f'dataset truncation {self.val_dataset.truncation} must be the same as config {self.config.data.get("truncation", "error")}'
    #     self.val_dataloader = StatefulDataLoader(
    #         dataset=self.val_dataset,
    #         batch_size=len(self.val_dataset),
    #         num_workers=8,
    #         shuffle=False,
    #         drop_last=False,
    #         collate_fn=collate_fn)
    #     assert len(self.train_dataloader) >= 1
    #     assert len(
    #         self.val_dataloader
    #     ) == 1, "Validation dataloader must have a single batch, which inference engines will schedule the memory themselves."
    #     print(f'Size of train dataloader: {len(self.train_dataloader)}')
    #     total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs
    #     if self.config.trainer.total_training_steps is not None:
    #         total_training_steps = self.config.trainer.total_training_steps
    #     self.total_training_steps = total_training_steps
    #     print(f'Total training steps: {self.total_training_steps}')
    #     OmegaConf.set_struct(self.config, True)
    #     with open_dict(self.config):
    #         self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
    #         self.config.critic.optim.total_training_steps = total_training_steps
    def _create_dataloader(self):
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.utils.import_utils import load_extern_type
        if "custom_cls" in self.config.data and self.config.data.custom_cls.get("path", None) is not None:
            dataset_cls = load_extern_type(self.config.data.custom_cls.path, self.config.data.custom_cls.name)
            if not issubclass(dataset_cls, Dataset):
                raise TypeError(f"The custom dataset class '{self.config.data.custom_cls.name}' from "
                                f"'{self.config.data.custom_cls.path}' must inherit from torch.utils.data.Dataset")
        else:
            dataset_cls = RLHFDataset

        self.train_dataset = dataset_cls(
            data_files=self.config.data.train_files,
            tokenizer=self.tokenizer,
            processor=self.processor,
            config=self.config.data,
        )

        # use sampler for better ckpt resume
        if self.config.data.shuffle:
            train_dataloader_generator = torch.Generator()
            train_dataloader_generator.manual_seed(self.config.data.get('seed', 1))
            sampler = RandomSampler(data_source=self.train_dataset, generator=train_dataloader_generator)
        else:
            sampler = SequentialSampler(data_source=self.train_dataset)

        self.train_dataloader = StatefulDataLoader(dataset=self.train_dataset,
                                                   batch_size=self.config.data.get('gen_batch_size',
                                                                                   self.config.data.train_batch_size),
                                                   num_workers=8,
                                                   drop_last=True,
                                                   collate_fn=collate_fn,
                                                   sampler=sampler)

        self.val_dataset = dataset_cls(
            data_files=self.config.data.val_files,
            tokenizer=self.tokenizer,
            processor=self.processor,
            config=self.config.data,
        )
        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            # Validation datasets are sent to inference engines as a whole batch,
            # which will schedule the memory themselves.
            batch_size=len(self.val_dataset),
            num_workers=8,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn)

        assert len(self.train_dataloader) >= 1
        assert len(
            self.val_dataloader
        ) == 1, "Validation dataloader must have a single batch, which inference engines will schedule the memory themselves."

        print(f'Size of train dataloader: {len(self.train_dataloader)}')

        # inject total_training_steps to actor/critic optim_config. This is hacky.
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f'Total training steps: {self.total_training_steps}')

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""
        generations_to_log = self.config.trainer.log_val_generations
        if generations_to_log == 0:
            return
        import numpy as np
        samples = list(zip(inputs, outputs, scores))
        samples.sort(key=lambda x: x[0])
        rng = np.random.RandomState(42)
        rng.shuffle(samples)
        samples = samples[:generations_to_log]
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)
    
    # def _validate(self):
    #     data_source_lst = []
    #     reward_extra_infos_dict: dict[str, list] = defaultdict(list)
    #     sample_inputs = []
    #     sample_outputs = []
    #     sample_scores = []
    #     reward_scores_dict: dict[str, list] = defaultdict(list)
    #     for test_data in self.val_dataloader:
    #         test_batch = DataProto.from_single_dict(test_data)
    #         original_prompt_ids = test_batch.batch['input_ids']
    #         input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in original_prompt_ids]
    #         sample_inputs.extend(input_texts)
    #         original_non_tensor_data = test_batch.non_tensor_batch
    #         if self.config.reward_model.enable and test_batch[0].non_tensor_batch['reward_model']['style'] == 'model':
    #             return {}
    #         input_ids = test_batch.batch['input_ids']
    #         input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
    #         sample_inputs.extend(input_texts)
    #         if 'multi_modal_inputs' in test_batch.non_tensor_batch.keys():
    #             test_gen_batch = test_batch.pop(
    #                 batch_keys=['input_ids', 'attention_mask', 'position_ids'],
    #                 non_tensor_batch_keys=['raw_prompt_ids', 'multi_modal_data', 'multi_modal_inputs'],
    #             )
    #         else:
    #             test_gen_batch = test_batch.pop(
    #                 batch_keys=['input_ids', 'attention_mask', 'position_ids'],
    #                 non_tensor_batch_keys=['raw_prompt_ids'],
    #             )
    #         test_gen_batch.meta_info = {
    #             'eos_token_id': self.tokenizer.eos_token_id,
    #             'pad_token_id': self.tokenizer.pad_token_id,
    #             'recompute_log_prob': False,
    #             'do_sample': self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
    #             'validate': True,
    #         }
    #         print(f'test_gen_batch meta info: {test_gen_batch.meta_info}')
    #         test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
    #         print("going to generate sequences")
    #         test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
    
    #         test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            
    #         print('validation generation end')
    #         output_ids = test_output_gen_batch.batch['responses']
    #         output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
    #         sample_outputs.extend(output_texts)
    #         if self.config.reward_model.enable:
    #             rm_tensors = {
    #                  'input_ids': test_output_gen_batch.batch['input_ids'],
    #                  'attention_mask': test_output_gen_batch.batch['attention_mask'],
    #                  'responses': test_output_gen_batch.batch['responses'],
    #                  'position_ids': test_output_gen_batch.batch.get('position_ids'),
    #             }
    #             rm_tensors = {k: v for k, v in rm_tensors.items() if v is not None}
    #             rm_input_proto = DataProto.from_dict(tensors=rm_tensors)
    #             rm_non_tensor_batch_data = {}
    #             if 'raw_prompt_ids' in original_non_tensor_data:
    #                  rm_non_tensor_batch_data['raw_prompt'] = original_non_tensor_data['raw_prompt_ids']
    #             if 'data_source' in original_non_tensor_data:
    #                  rm_non_tensor_batch_data['data_source'] = original_non_tensor_data['data_source']
    #             rm_input_proto.non_tensor_batch = rm_non_tensor_batch_data
    #             rm_input_proto_padded, rm_pad_size = pad_dataproto_to_divisor(rm_input_proto, self.rm_wg.world_size)
    #             rm_scores_proto_padded = self.rm_wg.compute_rm_score(rm_input_proto_padded)
    #             rm_scores_proto = unpad_dataproto(rm_scores_proto_padded, pad_size=rm_pad_size)
    #             token_level_rm_scores = rm_scores_proto.batch['rm_scores']
    #             rm_scalar_scores = token_level_rm_scores.sum(dim=1)
    #             scores = rm_scalar_scores.cpu().tolist()
    #             sample_scores.extend(scores)
    #             reward_scores_dict["reward"].extend(scores)
    #             if 'data_source' in original_non_tensor_data:
    #                 data_source_lst.append(original_non_tensor_data.get('data_source', ['unknown'] * len(scores)))
    #             else:
    #                 ds_name = self.config.data.train_files[0]
    #                 if 'tldr' in ds_name: ds_name='trl-lib/tldr'
    #                 elif 'gsm8k' in ds_name: ds_name='openai/gsm8k'
    #                 else: ds_name = 'unknown'
    #                 data_source_lst.append([ds_name] * len(scores))
    #         else:
    #              print("Warning: Reward model is disabled, cannot compute validation scores.")
    #              scores = [0.0] * len(output_texts)
    #              sample_scores.extend(scores)
    #              reward_scores_dict["reward"].extend(scores)
    #              data_source_lst.append(['unknown'] * len(scores))
    #     self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)
    #     for key_info, lst in reward_extra_infos_dict.items():
    #         assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"
    #     if not data_source_lst:
    #              print("Warning: No validation data processed.")
    #              return {}
    #     data_sources = np.concatenate(data_source_lst, axis=0) if data_source_lst else np.array([])
    #     if 'reward' in reward_scores_dict:
    #         reward_list = reward_scores_dict['reward']
    #         if reward_list:
    #             has_nan = np.isnan(reward_list).any()
    #             has_inf = np.isinf(reward_list).any()
    #     data_src2var2metric2val = process_validation_metrics(data_sources, sample_inputs, reward_scores_dict)
    #     metric_dict = {}
    #     for data_source, var2metric2val in data_src2var2metric2val.items():
    #         core_var = "reward"
    #         for var_name, metric2val in var2metric2val.items():
    #             if not metric2val:
    #                 continue
    #             n_max = 1
    #             try:
    #                 keys_with_at = [k for k in metric2val.keys() if "@" in k]
    #                 if keys_with_at:
    #                     n_max = max([int(name.split("@")[-1].split("/")[0]) for name in keys_with_at])
    #             except Exception as e_nmax:
    #                 print(f"DEBUG: Error calculating n_max: {e_nmax}")
    #             for metric_name, metric_val in metric2val.items():
    #                 if var_name == core_var and any(metric_name.startswith(pfx) for pfx in ["mean", "std", "maj", "best"]):
    #                     metric_sec = "val-core"
    #                 else:
    #                     metric_sec = "val-aux"
    #                 pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
    #                 metric_dict[pfx] = metric_val
    #     print(f"DEBUG: _validate returning metrics keys: {metric_dict.keys()}")
    #     return metric_dict
    # def _validate(self):
    #     data_source_lst = []
    #     reward_extra_infos_dict: dict[str, list] = defaultdict(list)

    #     # Lists to collect samples for the table
    #     sample_inputs = []
    #     sample_outputs = []
    #     sample_scores = []

    #     for test_data in self.val_dataloader:
    #         test_batch = DataProto.from_single_dict(test_data)

    #         # repeat test batch
    #         test_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n,
    #                                        interleave=True)

    #         # we only do validation on rule-based rm
    #         if self.config.reward_model.enable and test_batch[0].non_tensor_batch['reward_model']['style'] == 'model':
    #             return {}

    #         # Store original inputs
    #         input_ids = test_batch.batch['input_ids']
    #         # TODO: Can we keep special tokens except for padding tokens?
    #         input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
    #         sample_inputs.extend(input_texts)

    #         if 'multi_modal_inputs' in test_batch.non_tensor_batch.keys():
    #             test_gen_batch = test_batch.pop(
    #                 batch_keys=['input_ids', 'attention_mask', 'position_ids'],
    #                 non_tensor_batch_keys=['raw_prompt_ids', 'multi_modal_data', 'multi_modal_inputs'],
    #             )
    #         else:
    #             test_gen_batch = test_batch.pop(
    #                 batch_keys=['input_ids', 'attention_mask', 'position_ids'],
    #                 non_tensor_batch_keys=['raw_prompt_ids'],
    #             )

    #         test_gen_batch.meta_info = {
    #             'eos_token_id': self.tokenizer.eos_token_id,
    #             'pad_token_id': self.tokenizer.pad_token_id,
    #             'recompute_log_prob': False,
    #             'do_sample': self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
    #             'validate': True,
    #         }
    #         print(f'test_gen_batch meta info: {test_gen_batch.meta_info}')

    #         # pad to be divisible by dp_size
    #         test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
    #         test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)

    #         # unpad
    #         test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
    #         print('validation generation end')

    #         # Store generated outputs
    #         output_ids = test_output_gen_batch.batch['responses']
    #         output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
    #         sample_outputs.extend(output_texts)

    #         test_batch = test_batch.union(test_output_gen_batch)

    #         # evaluate using reward_function
    #         result = self.val_reward_fn(test_batch, return_dict=True)
    #         reward_tensor = result["reward_tensor"]
    #         scores = reward_tensor.sum(-1).cpu().tolist()
    #         sample_scores.extend(scores)

    #         reward_extra_infos_dict["reward"].extend(scores)
    #         if "reward_extra_info" in result:
    #             for key, lst in result["reward_extra_info"].items():
    #                 reward_extra_infos_dict[key].extend(lst)

    #         data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))

    #     self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

    #     for key_info, lst in reward_extra_infos_dict.items():
    #         assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

    #     data_sources = np.concatenate(data_source_lst, axis=0)

    #     data_src2var2metric2val = process_validation_metrics(data_sources, sample_inputs, reward_extra_infos_dict)
    #     metric_dict = {}
    #     for data_source, var2metric2val in data_src2var2metric2val.items():
    #         core_var = "acc" if "acc" in var2metric2val else "reward"
    #         for var_name, metric2val in var2metric2val.items():
    #             n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
    #             for metric_name, metric_val in metric2val.items():
    #                 if (var_name == core_var) and any(
    #                         metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"]) and (f"@{n_max}"
    #                                                                                              in metric_name):
    #                     metric_sec = "val-core"
    #                 else:
    #                     metric_sec = "val-aux"
    #                 pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
    #                 metric_dict[pfx] = metric_val

    #     return metric_dict
    
    def _validate(self):
        # --- Add global_steps access if needed for logging ---
        current_step = getattr(self, 'global_steps', 'N/A')
        print(f"\nRunning validation at DPO step {current_step}...")

        # --- Keep track of results across batches ---
        all_input_texts = []
        all_output_texts = []
        all_rm_scores = []
        data_source_lst = [] # Collect data sources per batch
        reward_scores_dict: dict[str, list] = defaultdict(list) # To pass to process_validation_metrics
        reward_extra_infos_dict: dict[str, list] = defaultdict(list) # To pass to process_validation_metrics

        # --- Define a reasonable validation batch size ---
        # ---> SET YOUR SMALL BATCH SIZE HERE <---
        # Start very small (e.g., 8, 16, 32). 16 is a safe start based on previous logs.
        validation_batch_size = 16
        # ---> END SETTING BATCH SIZE <---
        print(f"Processing validation set in batches of {validation_batch_size}...")

        # Ensure self.val_dataset is accessible (should be created in _create_dataloader)
        if not hasattr(self, 'val_dataset') or self.val_dataset is None:
             print("ERROR: self.val_dataset not found in trainer. Cannot run validation.")
             return {}

        # --- Create a *new* DataLoader specifically for batched validation inference ---
        val_sampler = SequentialSampler(self.val_dataset) # Process in order
        # Use the existing collate_fn
        val_dataloader_for_inference = DataLoader(
            self.val_dataset,
            batch_size=validation_batch_size, # Use the small batch size
            sampler=val_sampler,
            collate_fn=collate_fn,
            num_workers=self.config.data.get("num_workers", 4), # Use config value or default
            drop_last=False # Process all validation samples
        )
        # --- End DataLoader Creation ---

        # --- Loop through small validation batches ---
        for batch_idx, val_batch_dict in enumerate(tqdm(val_dataloader_for_inference, desc="Validation Generation Batches")):
            try:
                test_batch = DataProto.from_single_dict(val_batch_dict)
                current_batch_size = test_batch.batch.batch_size[0] # Get actual batch size

                # Repeat test batch if needed (from your previous version)
                # Note: Repeating might increase memory usage if 'n' > 1
                n_repeat = self.config.actor_rollout_ref.rollout.val_kwargs.get('n', 1)
                if n_repeat > 1:
                    # Careful: Repeating non-tensor data needs manual handling if process_validation_metrics needs it later
                    print(f"Warning: Repeating validation batch {n_repeat} times. Ensure non-tensor data is handled if needed.")
                    test_batch = test_batch.repeat(repeat_times=n_repeat, interleave=True)
                    # Update batch size for tracking
                    current_batch_size *= n_repeat


                # Store original inputs (do this *before* repeating if you only want unique inputs logged)
                # Let's assume we want inputs corresponding to each generated output
                original_prompt_ids = test_batch.batch['input_ids'] # Use potentially repeated ids
                input_texts_batch = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in original_prompt_ids]
                all_input_texts.extend(input_texts_batch) # Accumulate

                original_non_tensor_data = test_batch.non_tensor_batch # Store non-tensor for RM step

                # --- Prepare batch for generation worker ---
                # Adjust pop keys based on your actual data structure
                pop_batch_keys=['input_ids', 'attention_mask']
                if 'position_ids' in test_batch.batch: pop_batch_keys.append('position_ids')
                # Only pop keys needed for generation, keep original non-tensor data if needed later by RM
                pop_non_tensor_keys = []
                if 'multi_modal_inputs' in test_batch.non_tensor_batch:
                    pop_non_tensor_keys.extend(['multi_modal_data', 'multi_modal_inputs'])

                # Use deepcopy to avoid modifying the original test_batch needed for RM
                gen_batch_for_worker = deepcopy(test_batch).pop(
                    batch_keys=pop_batch_keys,
                    non_tensor_batch_keys=pop_non_tensor_keys,
                )

                # --- Set Meta Info for Generation ---
                gen_meta = {
                    'eos_token_id': self.tokenizer.eos_token_id,
                    'pad_token_id': self.tokenizer.pad_token_id,
                    'recompute_log_prob': False,
                    'validate': True,
                }
                gen_meta.update(self.config.actor_rollout_ref.rollout.val_kwargs) # Add kwargs like do_sample, temp, etc.
                gen_meta['n'] = n_repeat # Pass n value
                gen_batch_for_worker.meta_info = gen_meta
                # print(f'DEBUG: test_gen_batch meta info: {gen_batch_for_worker.meta_info}') # Debug

                # --- Generate sequences for the small batch ---
                gen_batch_padded, pad_size = pad_dataproto_to_divisor(gen_batch_for_worker, self.actor_rollout_wg.world_size)
                # print(f"DEBUG: Going to generate sequences for batch {batch_idx}") # Debug
                output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(gen_batch_padded)
                test_output_gen_batch = unpad_dataproto(output_gen_batch_padded, pad_size=pad_size)
                # print(f"DEBUG: validation generation batch {batch_idx} end") # Debug

                # --- Process results for this batch ---
                output_ids = test_output_gen_batch.batch.get('responses')
                if output_ids is None:
                     print(f"\nWarning: No 'responses' found in generation output for validation batch {batch_idx}. Skipping.")
                     # Append dummy data to keep lists aligned
                     all_output_texts.extend(["<GENERATION FAILED>"] * current_batch_size)
                     all_rm_scores.extend([0.0] * current_batch_size)
                     data_source_lst.extend(['unknown_gen_failed'] * current_batch_size)
                     reward_scores_dict["reward"].extend([0.0] * current_batch_size)
                     continue # Skip rest of loop for this batch

                output_texts_batch = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
                all_output_texts.extend(output_texts_batch) # Accumulate

                # --- Get RM scores / Use val_reward_fn ---
                if self.val_reward_fn:
                    # Reconstruct the batch with prompts and responses for reward function
                    # Use the generated output which contains the full sequence
                    # --- Corrected block ---
                    rm_tensors_dict = {
                        'input_ids': test_output_gen_batch.batch['input_ids'], # Prompt + Response IDs from generator
                        'attention_mask': test_output_gen_batch.batch['attention_mask'], # Mask for prompt + response
                        'responses': output_ids, # Just the response IDs
                        'position_ids': test_output_gen_batch.batch.get('position_ids'), # Position IDs for prompt + response
                    }
                    # Filter out None tensors BEFORE creating DataProto
                    rm_tensors_dict_filtered = {k: v for k,v in rm_tensors_dict.items() if v is not None}

                    # Create DataProto using only the tensors argument
                    batch_for_reward = DataProto.from_dict(tensors=rm_tensors_dict_filtered)

                    # Assign non_tensor_batch attribute AFTER creation
                    batch_for_reward.non_tensor_batch = original_non_tensor_data # Use original non-tensor data

                    # Ensure the batch attribute has the correct batch_size set if needed by TensorDict operations
                    if hasattr(batch_for_reward, 'batch') and isinstance(batch_for_reward.batch, TensorDict):
                        batch_for_reward.batch._batch_size = torch.Size([current_batch_size]) # Explicitly set batch size if needed
                    # --- End Corrected block ---



                    # --- DETAILED DEBUG LOGGING BEFORE RM CALL ---
                    print(f"\n---- DEBUG: Checking Inputs Sent to compute_rm_score (Batch {batch_idx}) ----")
                    try:
                        # Assuming RM vocab size is same as actor/tokenizer for now
                        rm_vocab_size = self.tokenizer.vocab_size
                        print(f"Assumed RM Vocab Size: {rm_vocab_size}")

                        # Check tensors within the batch that will be sent (before padding)
                        tensors_to_check_rm = batch_for_reward.batch
                        for key, tensor in tensors_to_check_rm.items():
                            print(f"  Tensor '{key}':")
                            if tensor is None: print("    Tensor is None!"); continue
                            print(f"    Shape: {tensor.shape}")
                            print(f"    DType: {tensor.dtype}")
                            if torch.is_floating_point(tensor):
                                if torch.isnan(tensor).any(): print(f"    ERROR: NaN found in {key}!")
                                if torch.isinf(tensor).any(): print(f"    ERROR: Inf found in {key}!")
                            if tensor.numel() == 0: print("    Tensor is empty!"); continue

                            try:
                                min_val, max_val = tensor.min().item(), tensor.max().item()
                                print(f"    Min Value: {min_val}")
                                print(f"    Max Value: {max_val}")
                                if 'input_ids' in key or 'responses' in key: # Check token ids
                                    if min_val < 0: print(f"    ERROR: Negative id in {key}!")
                                    if max_val >= rm_vocab_size: print(f"    ERROR: id >= vocab_size in {key}!")
                                elif 'attention_mask' in key:
                                    if not torch.all((tensor.float() == 0.0) | (tensor.float() == 1.0)): print(f"    ERROR: {key} has values other than 0 or 1!")
                                elif 'position_ids' in key:
                                    seq_len = tensor.shape[1]
                                    if min_val < 0: print(f"    ERROR: Negative position_id!")
                                    if max_val >= seq_len: print(f"    ERROR: position_id {max_val} >= seq len {seq_len}!")
                            except Exception as e_check: print(f"    ERROR checking min/max for {key}: {e_check}")
                        print("    Non-Tensor Keys:", list(batch_for_reward.non_tensor_batch.keys()))
                    except Exception as e_debug:
                        print(f"ERROR during pre-RM debug checks: {e_debug}")
                    print("-----------------------------------------------------------")
                    # --- END DETAILED DEBUG LOGGING ---


                    try: # Add try-except around RM scoring per batch
                         # Pad the batch for the RM worker group size
                         rm_input_proto_padded, rm_pad_size = pad_dataproto_to_divisor(batch_for_reward, self.rm_wg.world_size)
                         # Call the reward function / RM worker
                         rm_scores_proto_padded = self.rm_wg.compute_rm_score(rm_input_proto_padded)
                         rm_scores_proto = unpad_dataproto(rm_scores_proto_padded, pad_size=rm_pad_size)

                         # Process reward function output (adapted from your original code)
                         token_level_rm_scores = rm_scores_proto.batch.get('rm_scores')
                         if token_level_rm_scores is not None:
                             rm_scalar_scores = token_level_rm_scores.sum(dim=1)
                             batch_scores = rm_scalar_scores.cpu().tolist()
                         else:
                             print(f"\nWarning: No 'rm_scores' found in RM output for validation batch {batch_idx}. Using val_reward_fn directly if possible.")
                             # Fallback maybe? Depends on val_reward_fn structure
                             # This part might need adjustment based on how your val_reward_fn provides scores
                             batch_scores = [0.0] * current_batch_size # Default to 0 if no scores

                         # Accumulate standard reward scores
                         all_rm_scores.extend(batch_scores)
                         reward_scores_dict["reward"].extend(batch_scores)

                         # Accumulate extra info if reward_fn provides it (e.g., in rm_scores_proto.meta_info)
                         # This part needs to align with how your rm_worker / val_reward_fn returns extra info
                         if 'reward_extra_info' in rm_scores_proto.meta_info:
                             for key, lst in rm_scores_proto.meta_info["reward_extra_info"].items():
                                 if len(lst) == current_batch_size:
                                     reward_extra_infos_dict[key].extend(lst)
                                 else:
                                     print(f"Warning: Length mismatch for reward_extra_info '{key}' in validation batch.")
                                     reward_extra_infos_dict[key].extend([None]*current_batch_size)

                         # Determine data source for this batch
                         if 'data_source' in original_non_tensor_data:
                              ds_batch = original_non_tensor_data.get('data_source', ['unknown'] * current_batch_size)
                              # Handle potential repetition if n_repeat > 1
                              if n_repeat > 1 and len(ds_batch) == current_batch_size // n_repeat:
                                  ds_batch = [item for item in ds_batch for _ in range(n_repeat)]
                              data_source_lst.extend(ds_batch[:current_batch_size]) # Ensure correct length
                         else: # Fallback
                              ds_name = self.config.data.val_files[0] if self.config.data.val_files else 'unknown_val'
                              # Simple name derivation (you might need better logic)
                              if 'tldr' in ds_name: ds_name='trl-lib/tldr'
                              elif 'gsm8k' in ds_name: ds_name='openai/gsm8k'
                              data_source_lst.extend([ds_name] * current_batch_size)

                    except Exception as e:
                         print(f"\nERROR during validation RM scoring batch {batch_idx}: {e}. Appending zeros for batch.")
                         all_rm_scores.extend([0.0] * current_batch_size)
                         reward_scores_dict["reward"].extend([0.0] * current_batch_size)
                         data_source_lst.extend(['unknown_reward_error'] * current_batch_size)
                         # Also add placeholders for extra info if needed for consistent length
                         for key in reward_extra_infos_dict: reward_extra_infos_dict[key].extend([None]*current_batch_size)
                else:
                     # Append dummy scores if no reward function
                     print("Warning: No validation reward function (self.val_reward_fn) provided.")
                     all_rm_scores.extend([0.0] * current_batch_size)
                     reward_scores_dict["reward"].extend([0.0] * current_batch_size)
                     data_source_lst.extend(['unknown_no_reward_fn'] * current_batch_size)


            except Exception as batch_processing_error:
                print(f"\nERROR processing validation batch {batch_idx}: {batch_processing_error}. Skipping batch.")
                # Estimate batch size if possible, otherwise use planned size
                batch_size_in_dict = len(val_batch_dict.get('input_ids', [])) if val_batch_dict else validation_batch_size
                # Append placeholders to maintain list alignment
                all_input_texts.extend(["<INPUT ERROR>"] * batch_size_in_dict)
                all_output_texts.extend(["<PROCESSING ERROR>"] * batch_size_in_dict)
                all_rm_scores.extend([0.0] * batch_size_in_dict)
                data_source_lst.extend(['unknown_batch_error'] * batch_size_in_dict)
                reward_scores_dict["reward"].extend([0.0] * batch_size_in_dict)
                for key in reward_extra_infos_dict: reward_extra_infos_dict[key].extend([None]*batch_size_in_dict)
                continue # Skip to next batch

        # --- End Batch Loop ---
        print("\nValidation generation and scoring complete.")

        # --- Calculate final metrics using accumulated results ---
        self._maybe_log_val_generations(inputs=all_input_texts, outputs=all_output_texts, scores=all_rm_scores)

        # Check if any results were collected
        num_processed = len(all_input_texts)
        if not data_source_lst or not reward_scores_dict.get("reward") or num_processed == 0:
            print("Warning: No valid validation results collected after processing batches.")
            return {}

        # Ensure data_sources is correctly formed relative to processed items
        # This assumes the order is preserved by SequentialSampler and DataLoader
        if len(data_source_lst) != num_processed:
             print(f"Warning: Length mismatch between collected data_source_lst ({len(data_source_lst)}) and processed items ({num_processed}). Trying recovery.")
             try:
                 data_sources = np.array([item.get(self.config.data.reward_fn_key, 'unknown')
                                          for item in self.val_dataset])[:num_processed] # Slice to match processed
                 if len(data_sources) != num_processed:
                      raise ValueError(f"Length mismatch even after slicing dataset sources ({len(data_sources)} vs {num_processed})")
             except Exception as e:
                 print(f"ERROR: Could not accurately determine data sources for metrics: {e}.")
                 # Use collected list only if it matches the reward list length (fallback)
                 if len(data_source_lst) == len(reward_scores_dict["reward"]):
                     print("Using collected data_source_lst as fallback.")
                     data_sources = np.array(data_source_lst)
                 else:
                     print("ERROR: Cannot proceed with metric calculation due to length mismatch.")
                     return {"val-core/error": 1}
        else:
             data_sources = np.array(data_source_lst)

        # Final length check before processing metrics
        final_reward_scores = reward_scores_dict["reward"]
        if len(data_sources) != num_processed or len(final_reward_scores) != num_processed:
            print(f"ERROR: Final length mismatch before metric processing. "
                  f"DataSources: {len(data_sources)}, Scores: {len(final_reward_scores)}. Should be {num_processed}. Cannot compute metrics.")
            return {"val-core/error": 1}

        # Process final metrics using accumulated dictionaries
        try:
            # Combine standard reward with any extra info for processing
            reward_info_for_metrics = reward_scores_dict.copy()
            # Make sure extra info lists have same length as standard reward list
            for k, v_list in reward_extra_infos_dict.items():
                if len(v_list) == len(final_reward_scores):
                     reward_info_for_metrics[k] = v_list
                else:
                     print(f"Warning: Length mismatch for extra info '{k}' ({len(v_list)} vs {len(final_reward_scores)}). Skipping.")

            # Use the combined dict for processing
            data_src2var2metric2val = process_validation_metrics(data_sources, all_input_texts, reward_info_for_metrics)
            metric_dict = {}
            for data_source, var2metric2val in data_src2var2metric2val.items():
                 # Use logic from your provided code to determine core_var etc.
                 core_var = "acc" if "acc" in var2metric2val else "reward"
                 for var_name, metric2val in var2metric2val.items():
                     if not metric2val: continue
                     try: # Safer n_max calculation
                         n_max_keys = [int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys() if "@" in name]
                         n_max = max(n_max_keys) if n_max_keys else 1
                     except ValueError: n_max = 1 # Handle cases where parsing fails

                     for metric_name, metric_val in metric2val.items():
                         is_core_metric = (var_name == core_var) and \
                                          any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"]) and \
                                          (f"@{n_max}" in metric_name or "@" not in metric_name)

                         metric_sec = "val-core" if is_core_metric else "val-aux"
                         pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                         metric_dict[pfx] = metric_val
        except Exception as e:
            print(f"ERROR processing final validation metrics: {e}")
            # Fallback metric if processing fails
            metric_dict = {"val-core/overall/reward/mean": np.mean(all_rm_scores) if all_rm_scores else 0.0}

        print(f"DEBUG: _validate returning metrics keys: {metric_dict.keys()}")
        return metric_dict

    
    def init_workers(self):
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.ActorRollout],
                                                     config=self.config.actor_rollout_ref,
                                                     role='actor_rollout')
            self.resource_pool_to_cls[resource_pool]['actor_rollout'] = actor_rollout_cls
        else:
            raise NotImplementedError
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]['critic'] = critic_cls
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy],
                                                  config=self.config.actor_rollout_ref,
                                                  role='ref')
            self.resource_pool_to_cls[resource_pool]['ref'] = ref_policy_cls
        if self.use_rm:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]['rm'] = rm_cls
        all_wg = {}
        self.wg_dicts = []
        wg_kwargs = {}
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool,
                                                ray_cls_with_init=worker_dict_cls,
                                                **wg_kwargs)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            self.wg_dicts.append(wg_dict)
        if self.use_critic:
            self.critic_wg = all_wg['critic']
            self.critic_wg.init_model()
        if self.use_reference_policy:
            self.ref_policy_wg = all_wg['ref']
            self.ref_policy_wg.init_model()
        if self.use_rm:
            self.rm_wg = all_wg['rm']
            self.rm_wg.init_model()
        self.actor_rollout_wg = all_wg['actor_rollout']
        self.actor_rollout_wg.init_model()
    
    def _save_checkpoint(self):
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir,
                                                f'global_step_{self.global_steps}')
        print(f'local_global_step_folder: {local_global_step_folder}')
        actor_local_path = os.path.join(local_global_step_folder, 'actor')
        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
            self.config.trainer.default_hdfs_dir, f'global_step_{self.global_steps}', 'actor')
        remove_previous_ckpt_in_save = self.config.trainer.get('remove_previous_ckpt_in_save', False)
        if remove_previous_ckpt_in_save:
            print(
                'Warning: remove_previous_ckpt_in_save is deprecated, set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead'
            )
        max_actor_ckpt_to_keep = self.config.trainer.get('max_actor_ckpt_to_keep',
                                                         None) if not remove_previous_ckpt_in_save else 1
        max_critic_ckpt_to_keep = self.config.trainer.get('max_critic_ckpt_to_keep',
                                                          None) if not remove_previous_ckpt_in_save else 1
        self.actor_rollout_wg.save_checkpoint(actor_local_path,
                                              actor_remote_path,
                                              self.global_steps,
                                              max_ckpt_to_keep=max_actor_ckpt_to_keep)
        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, 'critic')
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
                self.config.trainer.default_hdfs_dir, f'global_step_{self.global_steps}', 'critic')
            self.critic_wg.save_checkpoint(critic_local_path,
                                           critic_remote_path,
                                           self.global_steps,
                                           max_ckpt_to_keep=max_critic_ckpt_to_keep)
        dataloader_local_path = os.path.join(local_global_step_folder, 'data.pt')
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)
        local_latest_checkpointed_iteration = os.path.join(self.config.trainer.default_local_dir,
                                                           'latest_checkpointed_iteration.txt')
        with open(local_latest_checkpointed_iteration, 'w') as f:
            f.write(str(self.global_steps))
    
    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == 'disable':
            return 0
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError('load from hdfs is not implemented yet')
        else:
            checkpoint_folder = self.config.trainer.default_local_dir
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)
        if self.config.trainer.resume_mode == 'auto':
            if global_step_folder is None:
                print('Training from scratch')
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert 'global_step_' in self.config.trainer.resume_from_path, "resume ckpt must specify the global_steps"
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f'Load from checkpoint folder: {global_step_folder}')
        self.global_steps = int(global_step_folder.split('global_step_')[-1])
        print(f'Setting global step to {self.global_steps}')
        print(f'Resuming from {global_step_folder}')
        actor_path = os.path.join(global_step_folder, 'actor')
        critic_path = os.path.join(global_step_folder, 'critic')
        self.actor_rollout_wg.load_checkpoint(actor_path,
                                              del_local_after_load=self.config.trainer.del_local_ckpt_after_load)
        if self.use_critic:
            self.critic_wg.load_checkpoint(critic_path,
                                           del_local_after_load=self.config.trainer.del_local_ckpt_after_load)
        dataloader_local_path = os.path.join(global_step_folder, 'data.pt')
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")
    
    def _balance_batch(self, batch: DataProto, metrics, logging_prefix='global_seqlen'):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch['attention_mask']
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch['attention_mask'].view(batch_size, -1).sum(-1).tolist()
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst,
                                                              k_partitions=world_size,
                                                              equal_size=True)
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst,
                                                    partitions=global_partition_lst,
                                                    prefix=logging_prefix)
        metrics.update(global_balance_stats)
    
    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from verl.utils.tracking import Tracking
        from omegaconf import OmegaConf
        logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True))
        self.global_steps = 0
        self._load_checkpoint()
        if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True):
            val_metrics = self._validate()
            pprint(f'Initial validation metrics: {val_metrics}')
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get('val_only', False):
                return
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")
        self.global_steps += 1
        last_val_metrics = None
        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                if 'multi_modal_inputs' in batch.non_tensor_batch.keys():
                    gen_batch = batch.pop(
                        batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                        non_tensor_batch_keys=['raw_prompt_ids', 'multi_modal_data', 'multi_modal_inputs'],
                    )
                else:
                    gen_batch = batch.pop(
                        batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                        non_tensor_batch_keys=['raw_prompt_ids'],
                    )
                is_last_step = self.global_steps >= self.total_training_steps
                with _timer('step', timing_raw):
                    with _timer('gen', timing_raw):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with _timer('gen_max', timing_raw):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info['do_sample'] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)
                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))
                            batch.batch['reward_baselines'] = reward_baseline_tensor
                            del gen_baseline_batch, gen_baseline_output
                    batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                                                             dtype=object)
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)
                    batch.batch['response_mask'] = compute_response_mask(batch)
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)
                    batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()
                    with _timer('old_log_prob', timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        batch = batch.union(old_log_prob)
                    if self.use_reference_policy:
                        with _timer('ref', timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)
                    if self.use_critic:
                        with _timer('values', timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)
                    with _timer('adv', timing_raw):
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)
                        reward_extra_infos_dict: dict[str, list]
                        try:
                            reward_result = self.reward_fn(batch, return_dict=True)
                            reward_tensor = reward_result['reward_tensor']
                            reward_extra_infos_dict = reward_result['reward_extra_info']
                        except Exception as e:
                            print(f'Error in reward_fn: {e}')
                            reward_tensor = self.reward_fn(batch)
                            reward_extra_infos_dict = {}
                        batch.batch['token_level_scores'] = reward_tensor
                        print(f'{list(reward_extra_infos_dict.keys())=}')
                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(batch,
                                                                 kl_ctrl=self.kl_ctrl_in_reward,
                                                                 kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch['token_level_rewards'] = batch.batch['token_level_scores']
                        batch = compute_advantage(batch,
                                                  adv_estimator=self.config.algorithm.adv_estimator,
                                                  gamma=self.config.algorithm.gamma,
                                                  lam=self.config.algorithm.lam,
                                                  num_repeat=self.config.actor_rollout_ref.rollout.n)
                    if self.use_critic:
                        with _timer('update_critic', timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                        metrics.update(critic_output_metrics)
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        with _timer('update_actor', timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                        metrics.update(actor_output_metrics)
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
                        (is_last_step or  self.global_steps % self.config.trainer.test_freq == 0):
                        with _timer('testing', timing_raw):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)
                    if self.config.trainer.save_freq > 0 and ( is_last_step or \
                            self.global_steps % self.config.trainer.save_freq == 0):
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                logger.log(data=metrics, step=self.global_steps)
                if is_last_step:
                    pprint(f'Final validation metrics: {last_val_metrics}')
                    progress_bar.close()
                    return
                progress_bar.update(1)
                self.global_steps += 1
    
    def _aggregate_metrics(self, result_proto: DataProto) -> Dict:
        """
        Extracts the metrics dictionary from the DataProto returned by a worker.
        Uses try-except for robust access.
        """
        metrics = {}
        if not isinstance(result_proto, DataProto):
            print(f"Warning [_aggregate_metrics]: Received object is not DataProto: {type(result_proto)}")
            return metrics
        try:
            meta_info_dict = getattr(result_proto, 'meta_info', None)
            if isinstance(meta_info_dict, dict):
                worker_metrics_dict = meta_info_dict.get('metrics', None)
                if isinstance(worker_metrics_dict, dict):
                    metrics = worker_metrics_dict
                elif worker_metrics_dict is not None:
                    print(f"Warning [_aggregate_metrics]: 'metrics' key found in meta_info but value is not a dict: {type(worker_metrics_dict)}")
            elif meta_info_dict is not None:
                print(f"Warning [_aggregate_metrics]: meta_info attribute exists but is not a dict: {type(meta_info_dict)}")
        except Exception as e:
            print(f"Error in _aggregate_metrics extracting metrics: {e}")
        return metrics if isinstance(metrics, dict) else {}
    
    def fit_dpo2(self):
        """
        The training loop for Online DPO using self.val_reward_fn for preference.
        Calls generate_sequences twice, computes preference scores using val_reward_fn,
        calculates log probabilities, prepares DPO batch, and updates actor.
        """
        if not hasattr(self, 'actor_rollout_wg') or not self.actor_rollout_wg:
             raise RuntimeError("Actor worker group must be initialized before fit_dpo2.")
        if not hasattr(self, 'train_dataloader') or not self.train_dataloader:
             raise RuntimeError("Train dataloader must be initialized before fit_dpo2.")
        if self.use_reference_policy and (not hasattr(self, 'ref_policy_wg') or not self.ref_policy_wg):
             print("Warning: use_reference_policy is True, but ref_policy_wg is not initialized. DPO loss will assume reference_free=True.")
             self.use_reference_policy = False
        if not self.val_reward_fn:
             raise ValueError("self.val_reward_fn must be provided to determine DPO preference in this modified loop.")
        
        from verl.utils.tracking import Tracking
        from omegaconf import OmegaConf
        logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True, throw_on_missing=False))
        print("Starting Online DPO training (using val_reward_fn for preference)...")
        
        loaded_step = self._load_checkpoint()
        self.global_steps = loaded_step + 1 if loaded_step > 0 else 1
        actor_wg = self.actor_rollout_wg
        ref_wg = self.ref_policy_wg if self.use_reference_policy and hasattr(self, 'ref_policy_wg') else None
        if self.val_reward_fn and OmegaConf.select(self.config.trainer, "val_before_train", default=True):
            print("Running validation before DPO training...")
            val_metrics = self._validate()
            if val_metrics and logger:
                 try: logger.log(data=val_metrics, step=max(0, self.global_steps - 1))
                 except Exception as e: print(f"[Step {max(0, self.global_steps - 1)} Pre-Val Metrics Log Error]: {e} | Metrics: {val_metrics}")
            if OmegaConf.select(self.config.trainer, "val_only", default=False):
                 print("Validation only mode enabled. Exiting DPO training.")
                 if logger and hasattr(logger, 'finish'): logger.finish()
                 return
        
        self.total_training_steps = OmegaConf.select(self.config.trainer, "total_training_steps", default=None)
        if self.total_training_steps is None:
             try: train_len = len(self.train_dataloader)
             except TypeError:
                  print("Warning: Train dataloader has no length. Set trainer.total_training_steps manually.")
                  train_len = int(1e6)
             self.total_training_steps = train_len * self.config.trainer.total_epochs
        
        print(f"Starting DPO training from global step {self.global_steps}. Total steps: {self.total_training_steps}")
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="DPO Training", position=0, leave=True)
        should_stop = False
        for epoch in range(self.config.trainer.total_epochs):
            if should_stop: break
            
            print(f"--- Starting DPO Epoch {epoch} ---")
            try:
                 train_iterator = iter(self.train_dataloader)
            except TypeError:
                 print("Warning: Dataloader is not iterable. Loop might not work as expected.")
                 train_iterator = self.train_dataloader
            for batch_idx, batch_dict in enumerate(train_iterator):
                if self.global_steps > self.total_training_steps:
                    should_stop = True; break
                metrics = {}; timing_raw = {}
                step_timer = Timer(logger=None)
                with _timer('step', timing_raw):
                    step_timer.start()
                    try:
                         initial_batch_proto = DataProto.from_single_dict(batch_dict)
                         initial_batch_size = initial_batch_proto.batch['input_ids'].shape[0]
                         if initial_batch_size == 0: raise ValueError("Empty batch received from dataloader")
                         original_prompt_tensors_for_batch = initial_batch_proto.select(batch_keys=['input_ids', 'attention_mask', 'position_ids'])
                         original_prompt_ids_only = initial_batch_proto.select(batch_keys=['input_ids']).batch['input_ids']
                         original_prompt_non_tensor = initial_batch_proto.non_tensor_batch
                    except Exception as e:
                         print(f"ERROR preparing initial batch: {e}. Skipping.");
                         continue
                    
                    gen_input_proto = initial_batch_proto
                    pop_batch_keys=['input_ids','attention_mask','position_ids' ]; pop_non_tensor_keys=[]
                    
                    if 'multi_modal_inputs' in gen_input_proto.non_tensor_batch:
                         pop_non_tensor_keys.extend(['multi_modal_data','multi_modal_inputs'])
                    else:
                        pop_non_tensor_keys.append('raw_prompt_ids')
                        
                    gen_batch_for_worker = gen_input_proto.pop(batch_keys=pop_batch_keys, non_tensor_batch_keys=pop_non_tensor_keys)
                    
                        
                    try:
                        # with _timer('gen_1', timing_raw): gen_output_1_proto = actor_wg.generate_sequences(deepcopy(gen_batch_for_worker))
                        # with _timer('gen_2', timing_raw): gen_output_2_proto = actor_wg.generate_sequences(deepcopy(gen_batch_for_worker))
                        print(f"[Step {self.global_steps}] Calling generate_sequences (Call 1)")
                        gen_output_1_proto = self.actor_rollout_wg.generate_sequences(deepcopy(gen_batch_for_worker))
                        print(f"[Step {self.global_steps}] Calling generate_sequences (Call 2)")
                        gen_output_2_proto = self.actor_rollout_wg.generate_sequences(deepcopy(gen_batch_for_worker)) # Error likely occurs here or during Call 1
                        print(f"[Step {self.global_steps}] Both generation calls finished.")
                    except Exception as e:
                        print(f"ERROR during generation step {self.global_steps}: {e}. Skipping batch.")
                        step_timer.stop(); continue
                    paired_responses, paired_input_ids, paired_attn_mask, paired_pos_ids = None, None, None, None
                    with _timer('process_paired_data', timing_raw):
                         try:
                             responses1 = gen_output_1_proto.batch.get('responses')
                             responses2 = gen_output_2_proto.batch.get('responses')
                             input_ids1 = gen_output_1_proto.batch.get('input_ids')
                             input_ids2 = gen_output_2_proto.batch.get('input_ids')
                             attn_mask1 = gen_output_1_proto.batch.get('attention_mask')
                             attn_mask2 = gen_output_2_proto.batch.get('attention_mask')
                             pos_ids1 = gen_output_1_proto.batch.get('position_ids')
                             pos_ids2 = gen_output_2_proto.batch.get('position_ids')
                             if any(t is None for t in [responses1, responses2, input_ids1, input_ids2, attn_mask1, attn_mask2]):
                                 raise ValueError("Missing essential tensors from generation.")
                             target_len_resp = max(responses1.shape[1], responses2.shape[1])
                             responses1_padded = F.pad(responses1, (0, target_len_resp - responses1.shape[1]), value=self.tokenizer.pad_token_id)
                             responses2_padded = F.pad(responses2, (0, target_len_resp - responses2.shape[1]), value=self.tokenizer.pad_token_id)
                             paired_responses = torch.stack([responses1_padded, responses2_padded], dim=1).view(initial_batch_size * 2, -1)
                             target_len_full = max(input_ids1.shape[1], input_ids2.shape[1])
                             input_ids1_padded = F.pad(input_ids1, (0, target_len_full - input_ids1.shape[1]), value=self.tokenizer.pad_token_id)
                             input_ids2_padded = F.pad(input_ids2, (0, target_len_full - input_ids2.shape[1]), value=self.tokenizer.pad_token_id)
                             paired_input_ids = torch.stack([input_ids1_padded, input_ids2_padded], dim=1).view(initial_batch_size * 2, -1)
                             attn_mask1_padded = F.pad(attn_mask1, (0, target_len_full - attn_mask1.shape[1]), value=0)
                             attn_mask2_padded = F.pad(attn_mask2, (0, target_len_full - attn_mask2.shape[1]), value=0)
                             paired_attn_mask = torch.stack([attn_mask1_padded, attn_mask2_padded], dim=1).view(initial_batch_size * 2, -1)
                             paired_pos_ids = None
                             if pos_ids1 is not None and pos_ids2 is not None:
                                 pos_ids1_padded = F.pad(pos_ids1, (0, target_len_full - pos_ids1.shape[1]), value=0)
                                 pos_ids2_padded = F.pad(pos_ids2, (0, target_len_full - pos_ids2.shape[1]), value=0)
                                 paired_pos_ids = torch.stack([pos_ids1_padded, pos_ids2_padded], dim=1).view(initial_batch_size * 2, -1)
                         except Exception as e:
                              print(f"ERROR processing paired data at step {self.global_steps}: {e}. Skipping batch.")
                              step_timer.stop(); continue
                    
                    print(f"[Step {self.global_steps}] Paired data processed successfully.")
                    
                    preference_data = []
                    chosen_scores_list = []
                    rejected_scores_list = []
                    with _timer('compute_rm_scores_and_preference', timing_raw):
                        try:
                            rm_tensors = {'input_ids': paired_input_ids, 'attention_mask': paired_attn_mask}
                            if paired_pos_ids is not None: rm_tensors['position_ids'] = paired_pos_ids
                            if paired_responses is not None:
                                rm_tensors['responses'] = paired_responses
                            else:
                                raise ValueError("paired_responses is None, cannot proceed with RM scoring.")
                            rm_non_tensor_batch = {}
                            if 'raw_prompt' in original_prompt_non_tensor:
                                original_raw_prompts = original_prompt_non_tensor['raw_prompt']
                                if len(original_raw_prompts) == initial_batch_size:
                                    paired_raw_prompts_list = [prompt for prompt in original_raw_prompts for _ in range(2)]
                                    try:
                                        rm_non_tensor_batch['raw_prompt'] = np.array(paired_raw_prompts_list, dtype=object)
                                        for key, val in original_prompt_non_tensor.items():
                                            if key != 'raw_prompt':
                                                duplicated_val = [item for item in val for _ in range(2)]
                                                try:
                                                    rm_non_tensor_batch[key] = np.array(duplicated_val, dtype=object)
                                                except Exception:
                                                    rm_non_tensor_batch[key] = duplicated_val
                                    except Exception as np_e:
                                        print(f"Warning: Failed processing non-tensor data for RM: {np_e}")
                                else:
                                    print(f"Warning: Found 'raw_prompt' but length mismatch. Skipping non-tensor for RM.")
                            else:
                                print(f"Warning: 'raw_prompt' key not found for RM. Template switching might fail.")
                            rm_input_proto = DataProto.from_dict(tensors=rm_tensors)
                            if rm_non_tensor_batch:
                                rm_input_proto.non_tensor_batch = rm_non_tensor_batch
                            rm_input_proto_padded, rm_pad_size = pad_dataproto_to_divisor(rm_input_proto, self.rm_wg.world_size)
                            rm_scores_proto_padded = self.rm_wg.compute_rm_score(rm_input_proto_padded)
                            rm_scores_proto = unpad_dataproto(rm_scores_proto_padded, pad_size=rm_pad_size)
                            token_level_rm_scores = rm_scores_proto.batch['rm_scores']
                            all_rm_scores = token_level_rm_scores.sum(dim=1)
                            if all_rm_scores.shape[0] != initial_batch_size * 2:
                                raise ValueError(f"Unexpected number of RM scores. Expected {initial_batch_size * 2}, Got {all_rm_scores.shape[0]}.")
                            scores_gen1 = all_rm_scores[0::2]
                            scores_gen2 = all_rm_scores[1::2]
                            for i in range(initial_batch_size):
                                s1 = scores_gen1[i].item()
                                s2 = scores_gen2[i].item()
                                if s1 >= s2:
                                    preference_data.append({'winner_idx': 0, 'loser_idx': 1})
                                    chosen_scores_list.append(s1)
                                    rejected_scores_list.append(s2)
                                else:
                                    preference_data.append({'winner_idx': 1, 'loser_idx': 0})
                                    chosen_scores_list.append(s2)
                                    rejected_scores_list.append(s1)
                            if chosen_scores_list:
                                metrics['reward/rm_score_chosen'] = np.mean(chosen_scores_list)
                                metrics['reward/rm_score_rejected'] = np.mean(rejected_scores_list)
                                metrics['reward/rm_score_margin'] = np.mean(np.array(chosen_scores_list) - np.array(rejected_scores_list))
                        except Exception as e:
                            step_timer.stop(); continue
                    ref_logps_pairs_token_level = None
                    try:
                        with _timer('compute_log_probs_pairs', timing_raw):
                             logprob_tensors = {'input_ids': paired_input_ids, 'attention_mask': paired_attn_mask, 'responses': paired_responses}
                             if paired_pos_ids is not None: logprob_tensors['position_ids'] = paired_pos_ids
                             logprob_input_proto = DataProto.from_dict(tensors=logprob_tensors)
                             if self.use_reference_policy:
                                 if ref_wg is None: raise RuntimeError("use_reference_policy is True but ref_wg is None/uninitialized")
                                 ref_logp_result_proto = ref_wg.compute_ref_log_prob(logprob_input_proto)
                                 ref_logps_pairs_token_level = ref_logp_result_proto.batch['ref_log_prob']
                             else:
                                 print("Warning: No reference policy worker. DPO loss will assume reference_free=True.")
                                 ref_logps_pairs_token_level = None
                    except Exception as e:
                         print(f"ERROR during log probability calculation at step {self.global_steps}: {e}. Skipping DPO step for this batch.")
                         step_timer.stop(); continue
                    
                    print(f"[Step {self.global_steps}] Log probabilities calculated successfully.")
                    
                    dpo_update_batch_proto = None
                    with _timer('prepare_dpo_batch', timing_raw):
                        try:
                             chosen_ids_list, chosen_mask_list, chosen_labels_list, chosen_pos_list = [], [], [], []
                             rejected_ids_list, rejected_mask_list, rejected_labels_list, rejected_pos_list = [], [], [], []
                             ref_chosen_logps_list, ref_rejected_logps_list = [], []
                             prompt_lengths = [mask.sum().item() for mask in original_prompt_tensors_for_batch.batch['attention_mask']]
                             for i in range(initial_batch_size):
                                 pref = preference_data[i]
                                 win_pair_idx, lose_pair_idx = i * 2 + pref['winner_idx'], i * 2 + pref['loser_idx']
                                 prompt_len = prompt_lengths[i]
                                 device = paired_input_ids.device
                                 chosen_ids_list.append(paired_input_ids[win_pair_idx])
                                 chosen_mask_list.append(paired_attn_mask[win_pair_idx])
                                 full_len_c = paired_attn_mask[win_pair_idx].sum().item()
                                 labels_c = torch.full_like(paired_input_ids[win_pair_idx], -100)
                                 if prompt_len < labels_c.shape[0]:
                                     labels_c[prompt_len:full_len_c] = paired_input_ids[win_pair_idx, prompt_len:full_len_c]
                                 chosen_labels_list.append(labels_c)
                                 if ref_logps_pairs_token_level is not None:
                                     ref_chosen_logps_list.append(ref_logps_pairs_token_level[win_pair_idx])
                                 if paired_pos_ids is not None: chosen_pos_list.append(paired_pos_ids[win_pair_idx])
                                 rejected_ids_list.append(paired_input_ids[lose_pair_idx])
                                 rejected_mask_list.append(paired_attn_mask[lose_pair_idx])
                                 full_len_r = paired_attn_mask[lose_pair_idx].sum().item()
                                 labels_r = torch.full_like(paired_input_ids[lose_pair_idx], -100)
                                 if prompt_len < labels_r.shape[0]:
                                     labels_r[prompt_len:full_len_r] = paired_input_ids[lose_pair_idx, prompt_len:full_len_r]
                                 rejected_labels_list.append(labels_r)
                                 if ref_logps_pairs_token_level is not None:
                                     ref_rejected_logps_list.append(ref_logps_pairs_token_level[lose_pair_idx])
                                 if paired_pos_ids is not None: rejected_pos_list.append(paired_pos_ids[lose_pair_idx])
                             dpo_tensors = {
                                 'chosen_input_ids': torch.stack(chosen_ids_list), 'chosen_attention_mask': torch.stack(chosen_mask_list), 'chosen_labels': torch.stack(chosen_labels_list),
                                 'rejected_input_ids': torch.stack(rejected_ids_list), 'rejected_attention_mask': torch.stack(rejected_mask_list), 'rejected_labels': torch.stack(rejected_labels_list),
                             }
                             if ref_chosen_logps_list:
                                 dpo_tensors['reference_chosen_logps'] = torch.stack(ref_chosen_logps_list)
                                 dpo_tensors['reference_rejected_logps'] = torch.stack(ref_rejected_logps_list)
                             if chosen_pos_list:
                                 dpo_tensors['chosen_position_ids'] = torch.stack(chosen_pos_list)
                                 dpo_tensors['rejected_position_ids'] = torch.stack(rejected_pos_list)
                             dpo_meta = {
                                 'dpo_beta': OmegaConf.select(self.config.algorithm, "dpo_beta", default=0.1),
                                 'dpo_loss_type': OmegaConf.select(self.config.algorithm, "dpo_loss_type", default='sigmoid'),
                                 'dpo_label_smoothing': OmegaConf.select(self.config.algorithm, "dpo_label_smoothing", default=0.0),
                                 'use_reference_policy': self.use_reference_policy,
                                 'reference_free': not self.use_reference_policy
                             }
                             dpo_update_batch_proto = DataProto.from_dict(tensors=dpo_tensors, meta_info=dpo_meta)
                        except Exception as e:
                             print(f"ERROR preparing DPO batch at step {self.global_steps}: {e}. Skipping DPO step for this batch.")
                             step_timer.stop(); continue
                    if dpo_update_batch_proto is not None:
                         try:
                             with _timer('update_actor_dpo', timing_raw):
                                 dpo_result_proto = actor_wg.update_actor_dpo(dpo_update_batch_proto)
                                 worker_metrics = self._aggregate_metrics(dpo_result_proto)
                                 print(f"[Step {self.global_steps}] Worker Metrics Received (after aggregate): {worker_metrics}")
                                 if worker_metrics:
                                     metrics.update(worker_metrics)
                                 else:
                                     print(f"[Step {self.global_steps}] Warning: No metrics received from DPO worker.")
                         except Exception as e:
                             print(f"ERROR during DPO update step {self.global_steps}: {e}. Model update may have failed.")
                             step_timer.stop(); continue
                    else:
                         print(f"Warning: Skipping DPO update at step {self.global_steps} because batch preparation failed earlier.")
                         step_timer.stop(); continue
                    step_timer.stop()
                    metrics['time/step'] = step_timer.last
                
                if logger and self.global_steps % self.config.trainer.log_freq == 0:
                     log_payload = metrics.copy()
                     print(f"[Step {self.global_steps}] Logging Step Payload: {log_payload}")
                     try: logger.log(data=log_payload, step=self.global_steps)
                     except Exception as e: print(f"Logging failed at step {self.global_steps}: {e}")
                postfix_metrics = {k: f"{v:.3f}" if isinstance(v, float) else v for k, v in metrics.items() if isinstance(v, (int, float))}
                progress_bar.set_postfix(postfix_metrics)
                save_freq = OmegaConf.select(self.config.trainer, "save_freq", default = -1)
                
                if save_freq > 0 and (self.global_steps % save_freq == 0):
                    print(f"\nSaving DPO checkpoint at step {self.global_steps}...")
                    with _timer('save_checkpoint', timing_raw): self._save_checkpoint()
                    metrics.update({'time/save_checkpoint': timing_raw.get('save_checkpoint', 0)})
                test_freq = OmegaConf.select(self.config.trainer, "test_freq", default = -1)
                
                if test_freq > 0 and (self.global_steps % test_freq == 0):
                     print(f"\nRunning validation at DPO step {self.global_steps}...")
                     val_timing_raw = {}
                     with _timer('testing', val_timing_raw):
                          val_metrics = self._validate()
                     if val_metrics and logger:
                          val_metrics['time/validation'] = val_timing_raw.get('testing', 0)
                          print(f"[Step {self.global_steps}] Logging Validation Payload: {val_metrics}")
                          try: logger.log(data=val_metrics, step=self.global_steps)
                          except Exception as e: print(f"[Step {self.global_steps} Val Metrics Log Error]: {e} | Metrics: {val_metrics}")
                     metrics.update({'time/validation_run': val_timing_raw.get('testing', 0)})
                self.global_steps += 1
                progress_bar.update(1)
            
            print(f"--- Finished DPO Epoch {epoch} ---")
            if hasattr(self.train_dataloader, 'reset'):
                try: self.train_dataloader.reset()
                except Exception as e: print(f"Warning: Failed to reset train dataloader state: {e}")
        
        progress_bar.close()
        print(f"DPO Training finished at step {self.global_steps-1}.")
        print(f"Saving final DPO checkpoint at step {self.global_steps-1}...")
        self._save_checkpoint()
        if self.val_reward_fn:
             print("Running final validation...")
             final_val_metrics = self._validate()
             if final_val_metrics and logger:
                 final_val_metrics['final_validation'] = True
                 try: logger.log(data=final_val_metrics, step=self.global_steps - 1)
                 except Exception as e: print(f"[Final Val Metrics Log Error]: {e} | Metrics: {final_val_metrics}")
        if logger and hasattr(logger, 'finish'): logger.finish()
        print("DPO Training Run Complete.")