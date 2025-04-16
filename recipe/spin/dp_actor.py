# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor
"""

import itertools
from typing import Iterable, Tuple
import math
import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl import DataProto
from verl.trainer.ppo.core_algos import compute_policy_loss, kl_penalty, agg_loss, compute_online_dpo_loss, get_batch_logps
from verl.workers.actor import BasePPOActor
from verl.utils.py_functional import append_to_dict
from verl.utils.torch_functional import logprobs_from_logits, masked_mean
from verl.utils.ulysses import ulysses_pad_and_slice_inputs, gather_outpus_and_unpad
from verl.utils.seqlen_balancing import rearrange_micro_batches, get_reverse_idx
import verl.utils.torch_functional as verl_F

from typing import Iterable, Tuple, Dict, Any

from flash_attn.bert_padding import pad_input, unpad_input, rearrange, index_first_axis

__all__ = ['DataParallelPPOActor']


class DataParallelPPOActor(BasePPOActor):

    def __init__(
        self,
        config,
        actor_module: nn.Module,
        actor_optimizer: torch.optim.Optimizer = None,
    ):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.use_remove_padding = self.config.get('use_remove_padding', False)
        print(f'Actor use_remove_padding={self.use_remove_padding}')
        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.compute_entropy_from_logits = (
            torch.compile(verl_F.entropy_from_logits, dynamic=True)
            if self.config.get('use_torch_compile', True)  #  use torch compile by default
            else verl_F.entropy_from_logits)

    def _forward_micro_batch(self, micro_batch, temperature) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns: 
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch['responses'].size(-1)
        multi_modal_inputs = {}
        if 'multi_modal_inputs' in micro_batch:
            for key in micro_batch['multi_modal_inputs'][0].keys():
                multi_modal_inputs[key] = torch.cat([inputs[key] for inputs in micro_batch['multi_modal_inputs']],
                                                    dim=0)

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            input_ids = micro_batch['input_ids']
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch['attention_mask']
            position_ids = micro_batch['position_ids']
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1),
                                                           attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."),
                                                          indices).transpose(0, 1).unsqueeze(
                                                              1)  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                                                          indices).transpose(0, 1)

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(input_ids_rmpad, \
                                                                                                position_ids_rmpad, \
                                                                                                sp_size=self.ulysses_sequence_parallel_size)
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(input_ids_rmpad_rolled, None,
                                                                                self.ulysses_sequence_parallel_size)

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                output = self.actor_module(input_ids=input_ids_rmpad,
                                           attention_mask=None,
                                           position_ids=position_ids_rmpad,
                                           **multi_modal_inputs,
                                           use_cache=False)  # prevent model thinks we are generating
                logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)

                logits_rmpad.div_(temperature)

                # compute entropy
                entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)

                # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                log_probs = logprobs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                    entropy_rmpad = gather_outpus_and_unpad(entropy_rmpad,
                                                            gather_dim=0,
                                                            unpad_dim=0,
                                                            padding_size=pad_size)
                # pad back to (bsz, seqlen)
                full_entropy = pad_input(hidden_states=entropy_rmpad.unsqueeze(-1),
                                         indices=indices,
                                         batch=batch_size,
                                         seqlen=seqlen)
                full_log_probs = pad_input(hidden_states=log_probs.unsqueeze(-1),
                                           indices=indices,
                                           batch=batch_size,
                                           seqlen=seqlen)

                # only return response part:
                entropy = full_entropy.squeeze(-1)[:, -response_length - 1:-1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1:-1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                output = self.actor_module(input_ids=input_ids,
                                           attention_mask=attention_mask,
                                           position_ids=position_ids,
                                           **multi_modal_inputs,
                                           use_cache=False)  # prevent model thinks we are generating
                logits = output.logits
                print(f"Reference Worker produced logits shape: {logits.shape}") # Add this line

                logits.div_(temperature)
                logits = logits[:, -response_length - 1:-1, :]  # (bsz, response_length, vocab_size)
                log_probs = logprobs_from_logits(logits, micro_batch['responses'])
                entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
            print(f"Reference Worker _forward_micro_batch returning log_probs shape: {log_probs.shape}") # Add this

            return entropy, log_probs
    
    def _forward_micro_batch(self, micro_batch, temperature) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates token-level entropy and log probabilities for a micro-batch.

        Args:
            micro_batch (dict): Dictionary containing tensors like 'input_ids',
                                'attention_mask', 'position_ids', 'responses'.
            temperature (float): Temperature for scaling logits.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - entropy: Token-level entropy. Shape: (batch_size, sequence_length - 1)
                           Corresponds to entropy of predicting token t+1 given tokens up to t.
                - log_probs: Token-level log probabilities. Shape: (batch_size, sequence_length - 1)
                             Corresponds to log P(token t+1 | tokens up to t).
        """
        # response_length = micro_batch['responses'].size(-1) # Not needed for slicing anymore
        multi_modal_inputs = {}
        if 'multi_modal_inputs' in micro_batch:
            for key in micro_batch['multi_modal_inputs'][0].keys():
                multi_modal_inputs[key] = torch.cat([inputs[key] for inputs in micro_batch['multi_modal_inputs']],
                                                    dim=0)

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            input_ids = micro_batch['input_ids']
            batch_size, seqlen = input_ids.shape
            print(f"[_forward_micro_batch] Input seqlen: {seqlen}") # Log input seqlen
            attention_mask = micro_batch['attention_mask']
            position_ids = micro_batch['position_ids']
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

            # Prepare labels for log probability calculation (shift input_ids)
            # Shape: (batch_size, sequence_length)
            labels = torch.roll(input_ids, shifts=-1, dims=1)
            labels[:, -1] = -100 # Or your label_pad_token_id

            if self.use_remove_padding:
                print("[_forward_micro_batch] Using rmpad path.") # Log path
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1),
                                                           attention_mask)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)

                if position_ids.dim() == 3:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."),
                                                          indices).transpose(0, 1).unsqueeze(1)
                else:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                                                          indices).transpose(0, 1)

                labels_rmpad = index_first_axis(rearrange(labels.unsqueeze(-1), "b s ... -> (b s) ..."),
                                                indices).transpose(0, 1)

                if self.use_ulysses_sp:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(input_ids_rmpad, \
                                                                                                position_ids_rmpad, \
                                                                                                sp_size=self.ulysses_sequence_parallel_size)
                    labels_rmpad, _, _ = ulysses_pad_and_slice_inputs(labels_rmpad, None,
                                                                      self.ulysses_sequence_parallel_size)

                labels_rmpad_squeezed = labels_rmpad.squeeze(0)

                # Forward pass
                output = self.actor_module(input_ids=input_ids_rmpad,
                                           attention_mask=None,
                                           position_ids=position_ids_rmpad,
                                           **multi_modal_inputs,
                                           use_cache=False)
                logits_rmpad = output.logits.squeeze(0)
                print(f"[_forward_micro_batch] rmpad logits_rmpad shape: {logits_rmpad.shape}") # Log shape

                logits_rmpad.div_(temperature)

                entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)

                # [* ADDED PRINT *] Log shapes before logprobs_from_logits
                print(f"[_forward_micro_batch] rmpad calling logprobs_from_logits with logits shape: {logits_rmpad.shape}, labels shape: {labels_rmpad_squeezed.shape}")
                log_probs_rmpad = logprobs_from_logits(logits=logits_rmpad, labels=labels_rmpad_squeezed)
                # [* ADDED PRINT *] Log shape after logprobs_from_logits
                print(f"[_forward_micro_batch] rmpad log_probs_rmpad shape: {log_probs_rmpad.shape}")


                if self.use_ulysses_sp:
                    log_probs_rmpad = gather_outpus_and_unpad(log_probs_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                    entropy_rmpad = gather_outpus_and_unpad(entropy_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                    print(f"[_forward_micro_batch] rmpad log_probs_rmpad shape after gather: {log_probs_rmpad.shape}") # Log shape

                # [* ADDED PRINT *] Log shapes before pad_input
                print(f"[_forward_micro_batch] rmpad calling pad_input with log_probs shape: {log_probs_rmpad.unsqueeze(-1).shape}, indices len: {len(indices)}, batch: {batch_size}, seqlen: {seqlen}")
                full_log_probs = pad_input(hidden_states=log_probs_rmpad.unsqueeze(-1),
                                           indices=indices,
                                           batch=batch_size,
                                           seqlen=seqlen) # Make sure seqlen is correct (e.g., 8192)
                # [* ADDED PRINT *] Log shape after pad_input
                print(f"[_forward_micro_batch] rmpad full_log_probs shape after pad_input: {full_log_probs.shape}")

                full_entropy = pad_input(hidden_states=entropy_rmpad.unsqueeze(-1),
                                         indices=indices,
                                         batch=batch_size,
                                         seqlen=seqlen)

                log_probs = full_log_probs.squeeze(-1)[:, :-1]
                entropy = full_entropy.squeeze(-1)[:, :-1]


            else:  # not using rmpad and no ulysses sp
                print("[_forward_micro_batch] Using non-rmpad path.") # Log path
                # Forward pass
                output = self.actor_module(input_ids=input_ids,
                                           attention_mask=attention_mask,
                                           position_ids=position_ids,
                                           **multi_modal_inputs,
                                           use_cache=False)
                logits = output.logits
                print(f"[_forward_micro_batch] non-rmpad logits shape: {logits.shape}") # Log shape

                logits.div_(temperature)

                shift_logits = logits[:, :-1, :]
                shift_labels = labels[:, :-1]

                # [* ADDED PRINT *] Log shapes before logprobs_from_logits
                print(f"[_forward_micro_batch] non-rmpad calling logprobs_from_logits with logits shape: {shift_logits.shape}, labels shape: {shift_labels.shape}")
                log_probs = logprobs_from_logits(shift_logits, shift_labels)
                # [* ADDED PRINT *] Log shape after logprobs_from_logits
                print(f"[_forward_micro_batch] non-rmpad log_probs shape: {log_probs.shape}")

                entropy = verl_F.entropy_from_logits(shift_logits)

        # [* ADDED PRINT *] Log final shape before returning
        print(f"Reference Worker _forward_micro_batch returning log_probs shape: {log_probs.shape}")
        return entropy, log_probs

    def compute_log_prob(self, data: DataProto) -> torch.Tensor:
        """
        Compute the TOKEN-LEVEL log probability for the full sequence.
        This is used by the reference worker.

        Args:
            data (DataProto): Input data containing tensors like 'input_ids', 'attention_mask', etc.

        Returns:
            torch.Tensor: Token-level log probabilities for the sequence.
                          Shape: (batch_size, sequence_length - 1)
        """
        # [* ADDED PRINT *] Identify when this specific function is called
        print(f"DataParallelPPOActor: compute_log_prob called.")
        # Add print statement for input shape verification
        print(f"Reference Worker received input_ids shape: {data.batch['input_ids'].shape}")
        self.actor_module.eval()

        micro_batch_size = data.meta_info['micro_batch_size']
        temperature = data.meta_info['temperature']
        use_dynamic_bsz = data.meta_info['use_dynamic_bsz']

        select_keys = ['input_ids', 'attention_mask', 'position_ids', 'responses'] # Include responses if needed by _forward_micro_batch, though it's unused after the fix
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = 'multi_modal_inputs' in data.non_tensor_batch.keys()

        if has_multi_modal_inputs:
            num_micro_batches = data.batch.batch_size[0] // micro_batch_size
            non_tensor_select_keys = ['multi_modal_inputs']
            micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
        elif use_dynamic_bsz:
            max_token_len = data.meta_info['max_token_len'] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        log_probs_lst = []
        for micro_batch_proto in micro_batches:
            if isinstance(micro_batch_proto, DataProto):
                micro_batch_dict = {**micro_batch_proto.batch, **micro_batch_proto.non_tensor_batch}
            else: # It's a TensorDict
                 micro_batch_dict = micro_batch_proto.to_dict()


            with torch.no_grad():
                # _forward_micro_batch now returns full sequence token-level logprobs
                _, log_probs_full_seq = self._forward_micro_batch(micro_batch_dict, temperature=temperature)
                # Shape: (current_micro_batch_size, sequence_length - 1)
            log_probs_lst.append(log_probs_full_seq)

        # Concatenate results from all micro-batches
        # Shape: (total_batch_size, sequence_length - 1)
        log_probs_all = torch.concat(log_probs_lst, dim=0)

        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs_all.size(0), f"{len(indices)} vs. {log_probs_all.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs_all = log_probs_all[revert_indices]

        # [* ADDED PRINT *] Log final shape returned by compute_log_prob
        print(f"DataParallelPPOActor: compute_log_prob returning shape: {log_probs_all.shape}")
        # Return the token-level log probabilities for the full sequence length (-1)
        return log_probs_all

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    # def compute_log_prob(self, data: DataProto) -> torch.Tensor:
    #     """Compute the log probability of the responses given input_ids, attention_mask and position_ids

    #     Args:
    #         data (DataProto): a DataProto containing keys

    #             ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
    #             concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

    #             ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

    #             ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

    #             ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

    #     Returns:
    #         torch.Tensor: the log_prob tensor
    #     """
    #     # set to eval
    #     print(f"Reference Worker received input_ids shape: {data.batch['input_ids'].shape}") # Add this line

    #     self.actor_module.eval()

    #     micro_batch_size = data.meta_info['micro_batch_size']
    #     temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error
    #     use_dynamic_bsz = data.meta_info['use_dynamic_bsz']

    #     select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids']
    #     batch = data.select(batch_keys=select_keys).batch
    #     has_multi_modal_inputs = 'multi_modal_inputs' in data.non_tensor_batch.keys()

    #     if has_multi_modal_inputs:
    #         num_micro_batches = data.batch.batch_size[0] // micro_batch_size
    #         non_tensor_select_keys = ['multi_modal_inputs']
    #         micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
    #     elif use_dynamic_bsz:
    #         # split using dynamic bsz
    #         max_token_len = data.meta_info['max_token_len'] * self.ulysses_sequence_parallel_size
    #         micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
    #     else:
    #         micro_batches = batch.split(micro_batch_size)

    #     log_probs_lst = []
    #     for micro_batch in micro_batches:
    #         if isinstance(micro_batch, DataProto):
    #             micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}

    #         with torch.no_grad():
    #             _, log_probs = self._forward_micro_batch(micro_batch, temperature=temperature)
    #         log_probs_lst.append(log_probs)
    #     log_probs = torch.concat(log_probs_lst, dim=0)

    #     if use_dynamic_bsz:
    #         indices = list(itertools.chain.from_iterable(indices))
    #         assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
    #         revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
    #         log_probs = log_probs[revert_indices]

    #     return log_probs

    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error

        select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids', 'old_log_probs', 'advantages']
        if self.config.use_kl_loss:
            select_keys.append('ref_log_prob')
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = 'multi_modal_inputs' in data.non_tensor_batch.keys()

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        if has_multi_modal_inputs:
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            non_tensor_select_keys = ['multi_modal_inputs']
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        else:
            dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        for epoch in range(self.config.ppo_epochs):
            for batch_idx, data in enumerate(dataloader):
                # split batch into micro_batches
                mini_batch = data
                if has_multi_modal_inputs:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    num_micro_batches = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
                elif self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    # split batch into micro_batches
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for data in micro_batches:
                    # Support all hardwares
                    if isinstance(data, DataProto):
                        data = {**data.batch.to(torch.cuda.current_device()), **data.non_tensor_batch}
                    else:
                        data = data.to(torch.cuda.current_device())  # actor device is cpu when using offload
                    responses = data['responses']
                    response_length = responses.size(1)
                    attention_mask = data['attention_mask']
                    response_mask = attention_mask[:, -response_length:]
                    old_log_prob = data['old_log_probs']
                    advantages = data['advantages']

                    clip_ratio = self.config.clip_ratio
                    clip_ratio_low = self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                    clip_ratio_high = self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                    clip_ratio_c = self.config.get('clip_ratio_c', 3.0)
                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    # all return: (bsz, response_length)
                    entropy, log_prob = self._forward_micro_batch(micro_batch=data, temperature=temperature)

                    pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        cliprange=clip_ratio,
                        cliprange_low=clip_ratio_low,
                        cliprange_high=clip_ratio_high,
                        clip_ratio_c=clip_ratio_c)
                    # compute entropy loss from entropy
                    entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                    # compute policy loss
                    policy_loss = pg_loss - entropy_loss * entropy_coeff

                    if self.config.use_kl_loss:
                        ref_log_prob = data['ref_log_prob']
                        # compute kl loss
                        kld = kl_penalty(logprob=log_prob,
                                         ref_logprob=ref_log_prob,
                                         kl_penalty=self.config.kl_loss_type)
                        kl_loss = agg_loss(loss_mat=kld,
                                           loss_mask=response_mask,
                                           loss_agg_mode=self.config.loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics['actor/kl_loss'] = kl_loss.detach().item()
                        metrics['actor/kl_coef'] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * (len(data) / self.config.ppo_mini_batch_size)
                    else:
                        loss = policy_loss / self.gradient_accumulation
                    loss.backward()

                    data = {
                        'actor/entropy': entropy_loss.detach().item(),
                        'actor/pg_loss': pg_loss.detach().item(),
                        'actor/pg_clipfrac': pg_clipfrac.detach().item(),
                        'actor/ppo_kl': ppo_kl.detach().item(),
                        'actor/pg_clipfrac_lower': pg_clipfrac_lower.detach().item(),
                    }
                    append_to_dict(metrics, data)

                grad_norm = self._optimizer_step()
                data = {'actor/grad_norm': grad_norm.detach().item()}
            append_to_dict(metrics, data)
        self.actor_optimizer.zero_grad()
        return metrics
    s
    def update_actor_dpo(self, data: DataProto):
        """Performs the Online DPO update step with improved reference logprob handling."""
        if self.actor_optimizer is None:
            raise RuntimeError("Optimizer not provided to DataParallelPPOActor for DPO update.")
        
        self.actor_module.train()

        # Extract DPO parameters
        beta = data.meta_info.get('dpo_beta', 0.1)
        loss_type = data.meta_info.get('dpo_loss_type', 'sigmoid')
        label_smoothing = 0
        use_reference_policy_flag = data.meta_info.get('use_reference_policy', True)

        micro_batch_size = self.config.get('ppo_micro_batch_size_per_gpu')
        if micro_batch_size is None: 
            raise ValueError("config 'actor.ppo_micro_batch_size_per_gpu' must be set")

        batch_tensors = data.batch
        non_tensor_batch_data = data.non_tensor_batch

        try: 
            total_batch_size = batch_tensors['chosen_input_ids'].shape[0]
        except KeyError: 
            print("ERROR: 'chosen_input_ids' missing.")
            return {'metrics': {}}
        if total_batch_size == 0: 
            print("Warning: Empty batch.")
            return {'metrics': {}}

        num_micro_batches = math.ceil(total_batch_size / micro_batch_size)
        gradient_accumulation_steps = num_micro_batches

        total_loss = 0.0
        total_chosen_rewards = 0.0
        total_rejected_rewards = 0.0
        total_logprob_diff = 0.0
        num_pairs = 0
        metrics = {}

        self.actor_optimizer.zero_grad()

        for i in range(num_micro_batches):
            start_idx = i * micro_batch_size
            end_idx = min(start_idx + micro_batch_size, total_batch_size)
            current_micro_batch_size = end_idx - start_idx

            micro_batch_tensors = {key: tensor[start_idx:end_idx] for key, tensor in batch_tensors.items()}
            # Handle non-tensor slicing if needed
            micro_batch_non_tensor = {}  # Populate if needed

            autocast_dtype = torch.bfloat16  # Default
            fsdp_cfg = self.config.get('fsdp_config', {})
            if fsdp_cfg and fsdp_cfg.get('mixed_precision'):
                param_dtype_str = fsdp_cfg.mixed_precision.get('param_dtype', 'bf16')
                autocast_dtype = verl_F.PrecisionType.to_dtype(param_dtype_str)

            with torch.autocast(device_type='cuda', dtype=autocast_dtype):
                # --- Policy Forward & Logprobs ---
                chosen_mm_inputs = micro_batch_non_tensor.get('chosen_multi_modal_inputs', {})
                rejected_mm_inputs = micro_batch_non_tensor.get('rejected_multi_modal_inputs', {})
                
                # Policy outputs for chosen responses
                policy_chosen_outputs = self.actor_module(
                    input_ids=micro_batch_tensors['chosen_input_ids'], 
                    attention_mask=micro_batch_tensors['chosen_attention_mask'],
                    position_ids=micro_batch_tensors.get('chosen_position_ids'), 
                    **chosen_mm_inputs, 
                    use_cache=False)
                
                # Policy outputs for rejected responses
                policy_rejected_outputs = self.actor_module(
                    input_ids=micro_batch_tensors['rejected_input_ids'], 
                    attention_mask=micro_batch_tensors['rejected_attention_mask'],
                    position_ids=micro_batch_tensors.get('rejected_position_ids'), 
                    **rejected_mm_inputs, 
                    use_cache=False)
                
                # Get sequence-level logprobs for policy model
                policy_chosen_logps = get_batch_logps(
                    policy_chosen_outputs.logits, 
                    micro_batch_tensors['chosen_labels'], 
                    average_log_prob=False)
                
                policy_rejected_logps = get_batch_logps(
                    policy_rejected_outputs.logits, 
                    micro_batch_tensors['rejected_labels'], 
                    average_log_prob=False)

                # --- Handle Reference Log Probs ---
                # Get reference logprobs from batch
                if 'reference_chosen_logps' in micro_batch_tensors and 'reference_rejected_logps' in micro_batch_tensors:
                    reference_chosen_logps_token_level = micro_batch_tensors['reference_chosen_logps']
                    reference_rejected_logps_token_level = micro_batch_tensors['reference_rejected_logps']
                    
                    # IMPROVED HANDLING: Convert token-level to sequence-level logprobs consistently
                    if reference_chosen_logps_token_level.ndim > 1:  # It's token-level
                        chosen_tokens_mask = (micro_batch_tensors['chosen_labels'] != -100)[:, 1:].to(reference_chosen_logps_token_level.dtype)
                        
                        # Ensure compatible shapes for masking
                        if reference_chosen_logps_token_level.shape[1] != chosen_tokens_mask.shape[1]:
                            # Trim to smaller size
                            min_len = min(reference_chosen_logps_token_level.shape[1], chosen_tokens_mask.shape[1])
                            reference_chosen_logps = (reference_chosen_logps_token_level[:, :min_len] * 
                                                    chosen_tokens_mask[:, :min_len]).sum(dim=1)
                        else:
                            reference_chosen_logps = (reference_chosen_logps_token_level * chosen_tokens_mask).sum(dim=1)
                    else:
                        # Already sequence-level
                        reference_chosen_logps = reference_chosen_logps_token_level
                    
                    # Same for rejected
                    if reference_rejected_logps_token_level.ndim > 1:
                        rejected_tokens_mask = (micro_batch_tensors['rejected_labels'] != -100)[:, 1:].to(reference_rejected_logps_token_level.dtype)
                        
                        if reference_rejected_logps_token_level.shape[1] != rejected_tokens_mask.shape[1]:
                            min_len = min(reference_rejected_logps_token_level.shape[1], rejected_tokens_mask.shape[1])
                            reference_rejected_logps = (reference_rejected_logps_token_level[:, :min_len] * 
                                                    rejected_tokens_mask[:, :min_len]).sum(dim=1)
                        else:
                            reference_rejected_logps = (reference_rejected_logps_token_level * rejected_tokens_mask).sum(dim=1)
                    else:
                        reference_rejected_logps = reference_rejected_logps_token_level
                else:
                    # No reference logprobs provided, use zeros (reference-free mode)
                    reference_chosen_logps = torch.zeros_like(policy_chosen_logps)
                    reference_rejected_logps = torch.zeros_like(policy_rejected_logps)
                    use_reference_policy_flag = False  # Force reference-free mode
                
                # Add debugging information
                print(f"[DPO Debug] Reference chosen logprobs shape: {reference_chosen_logps.shape}")
                print(f"[DPO Debug] Reference rejected logprobs shape: {reference_rejected_logps.shape}")
                print(f"[DPO Debug] Policy chosen logprobs shape: {policy_chosen_logps.shape}")
                print(f"[DPO Debug] Policy rejected logprobs shape: {policy_rejected_logps.shape}")
                print(f"[DPO Debug] Mean ref logprob diff: {(reference_chosen_logps - reference_rejected_logps).mean().item():.4f}")
                print(f"[DPO Debug] Mean policy logprob diff: {(policy_chosen_logps - policy_rejected_logps).mean().item():.4f}")

                # --- Compute DPO Loss ---
                loss = compute_online_dpo_loss(
                    policy_chosen_logps, 
                    policy_rejected_logps,
                    reference_chosen_logps, 
                    reference_rejected_logps,
                    beta, 
                    label_smoothing, 
                    loss_type,
                    reference_free=(not use_reference_policy_flag)
                )

                scaled_loss = loss / gradient_accumulation_steps

                # --- Accumulate Metrics ---
                total_loss += loss.item()
                num_pairs += current_micro_batch_size
                
                with torch.no_grad():
                    # Calculate reward metrics
                    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps)
                    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps)
                    
                    total_chosen_rewards += chosen_rewards.sum().item()
                    total_rejected_rewards += rejected_rewards.sum().item()
                    
                    pi_logratios = policy_chosen_logps - policy_rejected_logps
                    ref_logratios = reference_chosen_logps - reference_rejected_logps
                    logits = pi_logratios - ref_logratios
                    total_logprob_diff += logits.sum().item()

            # --- Backward Pass ---
            scaled_loss.backward()

        # --- Optimizer Step ---
        grad_norm = self._optimizer_step()

        # --- Populate Final Metrics ---
        if num_pairs > 0:
            metrics['actor/dpo_loss'] = total_loss / num_micro_batches
            metrics['actor/grad_norm'] = grad_norm.item() if torch.isfinite(grad_norm) else float('inf')
            metrics['actor/rewards_chosen'] = total_chosen_rewards / num_pairs
            metrics['actor/rewards_rejected'] = total_rejected_rewards / num_pairs
            metrics['actor/rewards_accuracies'] = (total_chosen_rewards > total_rejected_rewards).float().mean().item() if torch.is_tensor(total_chosen_rewards) else float(total_chosen_rewards > total_rejected_rewards)
            metrics['actor/rewards_margins'] = (total_chosen_rewards - total_rejected_rewards) / num_pairs
            metrics['actor/logits'] = total_logprob_diff / num_pairs
        else: 
            metrics['actor/dpo_loss'] = 0.0
            metrics['actor/grad_norm'] = 0.0

        # Zero gradients to prevent accumulation across batches
        self.actor_optimizer.zero_grad(set_to_none=True)
        return metrics