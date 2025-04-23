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

from verl import DataProto
from verl.utils.reward_score import _default_compute_score
import torch
from collections import defaultdict


class NaiveRewardManager:
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key='data_source') -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score
        self.reward_fn_key = reward_fn_key

    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch['rm_scores']}
            else:
                return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            data_source = data_item.non_tensor_batch[self.reward_fn_key]

            extra_info = data_item.non_tensor_batch.get('extra_info', None)

            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )

            if isinstance(score, dict):
                reward = score["score"]
                # Store the information including original reward
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score

            reward_tensor[i, valid_response_length - 1] = reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print(f"[score]", score)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
    
    # def __call__(self, data: DataProto, return_dict=False):
    #     """We will expand this function gradually based on the available datasets"""
    #     print(f"\n---- [DEBUG RewardManager] Entering __call__ for batch ----") # Added

    #     # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
    #     if 'rm_scores' in data.batch.keys():
    #         print(f"---- [DEBUG RewardManager] Found 'rm_scores' in batch, returning directly. ----") # Added
    #         if return_dict:
    #             return {"reward_tensor": data.batch['rm_scores']}
    #         else:
    #             return data.batch['rm_scores']

    #     reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
    #     reward_extra_info = defaultdict(list)
    #     already_print_data_sources = {}
    #     batch_size = len(data) # Get batch size
    #     print(f"---- [DEBUG RewardManager] Batch size: {batch_size} ----") # Added

    #     for i in range(batch_size):
    #         print(f"\n---- [DEBUG RewardManager] Processing item {i} ----") # Added
    #         data_item = data[i]  # DataProtoItem

    #         try: # Added try-except block for safer debugging
    #             # --- Prompt Decoding ---
    #             prompt_ids = data_item.batch['prompts']
    #             prompt_length = prompt_ids.shape[-1]
    #             valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
    #             valid_prompt_ids = prompt_ids[-valid_prompt_length:]
    #             prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
    #             # print(f"  [DEBUG] Decoded Prompt (sample): {prompt_str[:100]}...") # Optional: Can be very verbose

    #             # --- Response Decoding ---
    #             response_ids = data_item.batch['responses']
    #             valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum().item() # Ensure it's an int
    #             valid_response_ids = response_ids[:valid_response_length]
    #             response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
    #             print(f"  [DEBUG] Decoded Response (sample): {response_str[:100]}...") # Added

    #             # --- Ground Truth Extraction ---
    #             ground_truth = None # Default
    #             try:
    #                 # This path depends on how your data is structured
    #                 ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
    #                 print(f"  [DEBUG] Ground Truth: {ground_truth}") # Added
    #             except KeyError:
    #                 print(f"  [DEBUG] WARNING: Key 'reward_model' or 'ground_truth' not found in non_tensor_batch for item {i}!") # Added
    #             except Exception as e_gt:
    #                  print(f"  [DEBUG] ERROR accessing ground_truth for item {i}: {e_gt}") # Added


    #             # --- Data Source Extraction ---
    #             data_source = None # Default
    #             try:
    #                 data_source = data_item.non_tensor_batch[self.reward_fn_key]
    #                 print(f"  [DEBUG] Data Source: {data_source}") # Added
    #             except KeyError:
    #                  print(f"  [DEBUG] WARNING: Reward function key '{self.reward_fn_key}' not found in non_tensor_batch for item {i}!") # Added
    #             except Exception as e_ds:
    #                   print(f"  [DEBUG] ERROR accessing data_source for item {i}: {e_ds}") # Added


    #             # --- Extra Info Extraction ---
    #             extra_info = data_item.non_tensor_batch.get('extra_info', None)
    #             # print(f"  [DEBUG] Extra Info: {extra_info}") # Optional: Can be verbose

    #             # --- Call compute_score ---
    #             score = 0.0 # Default score if compute fails
    #             if self.compute_score and data_source: # Check inputs are valid
    #                 print(f"  [DEBUG] Calling self.compute_score ('{self.compute_score.__name__}') with data_source='{data_source}'...") # Added
    #                 try:
    #                     score = self.compute_score(
    #                         data_source=data_source,
    #                         solution_str=response_str,
    #                         ground_truth=ground_truth,
    #                         extra_info=extra_info,
    #                     )
    #                     print(f"  [DEBUG] self.compute_score returned: {score}") # Added
    #                 except Exception as e_compute:
    #                      print(f"  [DEBUG] ERROR during self.compute_score call: {e_compute}") # Added
    #                      # Keep score as 0.0
    #             else:
    #                 print(f"  [DEBUG] Skipping self.compute_score due to missing data_source/ground_truth or compute_score is None.") # Added

    #             # --- Process Score and Assign Reward ---
    #             if isinstance(score, dict):
    #                 reward = score.get("score", 0.0) # Default to 0.0 if 'score' key missing
    #                 print(f"  [DEBUG] Extracted reward '{reward}' from score dict.") # Added
    #                 for key, value in score.items():
    #                     reward_extra_info[key].append(value)
    #             else:
    #                 reward = float(score) # Ensure it's a float
    #                 print(f"  [DEBUG] Using scalar score '{reward}' as reward.") # Added

    #             # Assign reward only to the last valid token position
    #             if valid_response_length > 0:
    #                  reward_tensor[i, valid_response_length - 1] = reward
    #                  print(f"  [DEBUG] Assigned reward {reward} to reward_tensor index [{i}, {valid_response_length - 1}]") # Added
    #             else:
    #                  print(f"  [DEBUG] Skipping reward assignment as valid_response_length is 0.") # Added


    #             # --- Console Printing Logic (Optional) ---
    #             if self.num_examine > 0:
    #                 if data_source not in already_print_data_sources:
    #                     already_print_data_sources[data_source] = 0
    #                 if already_print_data_sources.get(data_source, 0) < self.num_examine:
    #                     already_print_data_sources[data_source] += 1
    #                     print(f"--- Example {already_print_data_sources[data_source]}/{self.num_examine} for {data_source} ---")
    #                     print("[prompt]", prompt_str)
    #                     print("[response]", response_str)
    #                     print("[ground_truth]", ground_truth)
    #                     if isinstance(score, dict):
    #                         for key, value in score.items(): print(f"[{key}]", value)
    #                     else: print(f"[score]", score)
    #                     print("-" * 20)


    #         except Exception as e_item:
    #             print(f"---- [DEBUG RewardManager] ERROR processing item {i}: {e_item} ----") # Added
    #             import traceback
    #             traceback.print_exc()
    #             # Continue to the next item

    #     print(f"---- [DEBUG RewardManager] Finished processing batch. Returning reward tensor and extra info. ----") # Added
    #     if return_dict:
    #         return {
    #             "reward_tensor": reward_tensor,
    #             "reward_extra_info": reward_extra_info,
    #         }
    #     else:
    #         return reward_tensor
