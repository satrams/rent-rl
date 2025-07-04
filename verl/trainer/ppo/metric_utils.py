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
Metrics related to the PPO trainer.
"""

from collections import defaultdict
from functools import partial
from typing import Any, Callable, Dict, List

import numpy as np
import torch

from verl import DataProto


def reduce_metrics(metrics: Dict[str, List[Any]]) -> Dict[str, Any]:
    for key, val in metrics.items():
        metrics[key] = np.mean(val)
    return metrics


def _compute_response_info(batch: DataProto) -> Dict[str, Any]:
    response_length = batch.batch["responses"].shape[-1]

    prompt_mask = batch.batch["attention_mask"][:, :-response_length]
    response_mask = batch.batch["attention_mask"][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)

    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
    )

def compute_entropy_metrics(batch: DataProto, base_entropy: float) -> Dict[str, Any]:
    actual_reward_tensor = batch.batch["actual_reward_tensor"].sum(-1)
    extracted_entropys = batch.batch["extracted_entropys"]
    if "extracted_entropys_hard" in batch.batch:
        extracted_entropys_hard = batch.batch["extracted_entropys_hard"]
    else:
        extracted_entropys_hard = None
    
    extracted_answers = batch.batch["extracted_answers"]
    metrics = {}
    # Calculate number of correct answers (actual reward is 1) given extracted entropy as base_entropy
    correct_answers_with_base_entropy = 0
    total_answers_with_base_entropy = 0
    
    for i in range(len(extracted_entropys)):
        # Check if entropy is equal to base_entropy
        if extracted_entropys[i] == base_entropy:
            total_answers_with_base_entropy += 1
            # Check if the answer is correct (reward is 1)
            if actual_reward_tensor[i] == 1.0:
                correct_answers_with_base_entropy += 1
    if total_answers_with_base_entropy > 0:
        correct_answers_with_base_entropy_ratio = correct_answers_with_base_entropy / total_answers_with_base_entropy
    else:
        correct_answers_with_base_entropy_ratio = 0
    num_base_entropy = len([entropy for entropy in extracted_entropys if entropy == base_entropy])
    if len(extracted_entropys) > 0:
        base_entropy_ratio = num_base_entropy / len(extracted_entropys)
    else:
        base_entropy_ratio = 0
    metrics["entropy/correct_answers_with_base_entropy_ratio"] = correct_answers_with_base_entropy_ratio
    metrics["entropy/base_entropy_ratio"] = base_entropy_ratio
    metrics["entropy/correct_answers_with_base_entropy"] = correct_answers_with_base_entropy

    valid_answers =len([answer for answer in extracted_answers if answer is not None])
    valid_answer_indices = [i for i, answer in enumerate(extracted_answers) if answer is not None]
    
    
    invalid_answers = len(extracted_answers) - valid_answers
    valid_ratio = valid_answers / len(extracted_answers)
    valid_entropy = extracted_entropys[valid_answer_indices]
    valid_entropy_mean = torch.mean(valid_entropy).item()
    valid_entropy_std = torch.std(valid_entropy).item()
    
    all_entropy_mean = torch.mean(extracted_entropys).item()
    all_entropy_std = torch.std(extracted_entropys).item()
    
    # Calculate correlation between actual_reward and extracted entropy
    # First for non-base entropy values
    
    # Calculate correlations for soft entropy
    if len(valid_answer_indices) > 1:  # Need at least 2 points for correlation
        filtered_neg_entropys = [-1* extracted_entropys[i] for i in valid_answer_indices]
        filtered_rewards = [actual_reward_tensor[i].item() for i in valid_answer_indices]
        
        # Calculate correlation coefficient if we have valid data
        if len(filtered_neg_entropys) > 1 and np.std(filtered_neg_entropys) > 0 and np.std(filtered_rewards) > 0:
            valid_neg_entropy_reward_correlation = np.corrcoef(filtered_neg_entropys, filtered_rewards)[0, 1]
        else:
            valid_neg_entropy_reward_correlation = float('nan')
    else:
        valid_neg_entropy_reward_correlation = float('nan')

    all_neg_entropys = [-1*entropy for entropy in extracted_entropys]
    all_rewards = [reward.item() for reward in actual_reward_tensor]
    if len(all_neg_entropys) > 1 and np.std(all_neg_entropys) > 0 and np.std(all_rewards) > 0:
        all_neg_entropy_reward_correlation = np.corrcoef(all_neg_entropys, all_rewards)[0, 1]
    else:
        all_neg_entropy_reward_correlation = float('nan')

    # Calculate correlations for hard entropy if it exists
    if extracted_entropys_hard is not None:
        if len(valid_answer_indices) > 1:
            filtered_neg_hard_entropys = [-1* extracted_entropys_hard[i] for i in valid_answer_indices]
            if len(filtered_neg_hard_entropys) > 1 and np.std(filtered_neg_hard_entropys) > 0 and np.std(filtered_rewards) > 0:
                valid_neg_hard_entropy_reward_correlation = np.corrcoef(filtered_neg_hard_entropys, filtered_rewards)[0, 1]
            else:
                valid_neg_hard_entropy_reward_correlation = float('nan')
        else:
            valid_neg_hard_entropy_reward_correlation = float('nan')

        all_neg_hard_entropys = [-1*entropy for entropy in extracted_entropys_hard]
        if len(all_neg_hard_entropys) > 1 and np.std(all_neg_hard_entropys) > 0 and np.std(all_rewards) > 0:
            all_neg_hard_entropy_reward_correlation = np.corrcoef(all_neg_hard_entropys, all_rewards)[0, 1]
        else:
            all_neg_hard_entropy_reward_correlation = float('nan')

    metrics.update({
        "entropy/valid_ratio": valid_ratio,
        "entropy/num_valid": valid_answers,
        "entropy/valid_entropy_mean": valid_entropy_mean,
        "entropy/valid_entropy_std": valid_entropy_std,
        "entropy/valid_negative_entropy_reward_correlation": valid_neg_entropy_reward_correlation,
        "entropy/all_entropy_mean": all_entropy_mean,
        "entropy/all_entropy_std": all_entropy_std,
        "entropy/all_negative_entropy_reward_correlation": all_neg_entropy_reward_correlation,
    })

    if extracted_entropys_hard is not None:
        metrics.update({
            "entropy/valid_negative_hard_entropy_reward_correlation": valid_neg_hard_entropy_reward_correlation,
            "entropy/all_negative_hard_entropy_reward_correlation": all_neg_hard_entropy_reward_correlation,
        })
    return metrics
    
    
    
    
def compute_data_metrics(batch: DataProto, use_critic: bool = True) -> Dict[str, Any]:
    # TODO: add response length
    sequence_score = batch.batch["token_level_scores"].sum(-1)
    sequence_reward = batch.batch["token_level_rewards"].sum(-1)

    advantages = batch.batch["advantages"]
    returns = batch.batch["returns"]

    max_response_length = batch.batch["responses"].shape[-1]

    prompt_mask = batch.batch["attention_mask"][:, :-max_response_length].bool()
    response_mask = batch.batch["attention_mask"][:, -max_response_length:].bool()

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info["prompt_length"]
    response_length = response_info["response_length"]

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch["values"]
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    metrics = {
        # score
        "critic/score/mean": torch.mean(sequence_score).detach().item(),
        "critic/score/max": torch.max(sequence_score).detach().item(),
        "critic/score/min": torch.min(sequence_score).detach().item(),
        # reward
        "critic/rewards/mean": torch.mean(sequence_reward).detach().item(),
        "critic/rewards/max": torch.max(sequence_reward).detach().item(),
        "critic/rewards/min": torch.min(sequence_reward).detach().item(),
        # adv
        "critic/advantages/mean": torch.mean(valid_adv).detach().item(),
        "critic/advantages/max": torch.max(valid_adv).detach().item(),
        "critic/advantages/min": torch.min(valid_adv).detach().item(),
        # returns
        "critic/returns/mean": torch.mean(valid_returns).detach().item(),
        "critic/returns/max": torch.max(valid_returns).detach().item(),
        "critic/returns/min": torch.min(valid_returns).detach().item(),
        **(
            {
                # values
                "critic/values/mean": torch.mean(valid_values).detach().item(),
                "critic/values/max": torch.max(valid_values).detach().item(),
                "critic/values/min": torch.min(valid_values).detach().item(),
                # vf explained var
                "critic/vf_explained_var": (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
            }
            if use_critic
            else {}
        ),
        # response length
        "response_length/mean": torch.mean(response_length).detach().item(),
        "response_length/max": torch.max(response_length).detach().item(),
        "response_length/min": torch.min(response_length).detach().item(),
        "response_length/clip_ratio": torch.mean(torch.eq(response_length, max_response_length).float()).detach().item(),
        # prompt length
        "prompt_length/mean": torch.mean(prompt_length).detach().item(),
        "prompt_length/max": torch.max(prompt_length).detach().item(),
        "prompt_length/min": torch.min(prompt_length).detach().item(),
        "prompt_length/clip_ratio": torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    }
    return metrics


def compute_timing_metrics(batch: DataProto, timing_raw: Dict[str, float]) -> Dict[str, Any]:
    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info["prompt_length"]).item()
    num_response_tokens = torch.sum(response_info["response_length"]).item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    num_tokens_of_section = {
        "gen": num_response_tokens,
        **{name: num_overall_tokens for name in ["ref", "values", "adv", "update_critic", "update_actor"]},
    }

    return {
        **{f"timing_s/{name}": value for name, value in timing_raw.items()},
        **{f"timing_per_token_ms/{name}": timing_raw[name] * 1000 / num_tokens_of_section[name] for name in set(num_tokens_of_section.keys()) & set(timing_raw.keys())},
    }


def compute_throughout_metrics(batch: DataProto, timing_raw: Dict[str, float], n_gpus: int) -> Dict[str, Any]:
    total_num_tokens = sum(batch.meta_info["global_token_num"])
    time = timing_raw["step"]
    # estimated_flops, promised_flops = flops_function.estimate_flops(num_tokens, time)
    # f'Actual TFLOPs/s/GPU​': estimated_flops/(n_gpus),
    # f'Theoretical TFLOPs/s/GPU​': promised_flops,
    return {
        "perf/total_num_tokens": total_num_tokens,
        "perf/time_per_step": time,
        "perf/throughput": total_num_tokens / (time * n_gpus),
    }


def bootstrap_metric(
    data: list[Any],
    subset_size: int,
    reduce_fns: list[Callable[[np.ndarray], float]],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> list[tuple[float, float]]:
    np.random.seed(seed)

    bootstrap_metric_lsts = [[] for _ in range(len(reduce_fns))]
    for _ in range(n_bootstrap):
        bootstrap_idxs = np.random.choice(len(data), size=subset_size, replace=True)
        bootstrap_data = [data[i] for i in bootstrap_idxs]
        for i, reduce_fn in enumerate(reduce_fns):
            bootstrap_metric_lsts[i].append(reduce_fn(bootstrap_data))
    return [(np.mean(lst), np.std(lst)) for lst in bootstrap_metric_lsts]


def calc_maj_val(data: list[dict[str, Any]], vote_key: str, val_key: str) -> float:
    """
    Calculate the majority voting metric
    """
    vote2vals = defaultdict(list)
    for d in data:
        vote2vals[d[vote_key]].append(d[val_key])

    vote2cnt = {k: len(v) for k, v in vote2vals.items()}
    maj_vote = max(vote2cnt, key=vote2cnt.get)

    maj_val = vote2vals[maj_vote][0]

    return maj_val


def process_validation_metrics(data_sources: list[str], sample_inputs: list[str], infos_dict: dict[str, list[Any]], seed: int = 42) -> dict[str, dict[str, dict[str, float]]]:
    """Process validation metrics into a structured format.

    Args:
        data_sources: Array of data source identifiers for each sample
        sample_inputs: List of input prompts
        infos_dict: variable name -> list of values for each sample

    Returns:
        dict[str, dict[str, dict[str, float]]]: data source -> variable name -> metric value
    """
    # Group metrics by data source, prompt and variable
    data_src2prompt2var2vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for sample_idx, data_source in enumerate(data_sources):
        prompt = sample_inputs[sample_idx]
        var2vals = data_src2prompt2var2vals[data_source][prompt]
        for var_name, var_vals in infos_dict.items():
            var2vals[var_name].append(var_vals[sample_idx])
    # Calculate metrics for each group
    data_src2prompt2var2metric = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for data_source, prompt2var2vals in data_src2prompt2var2vals.items():
        for prompt, var2vals in prompt2var2vals.items():
            for var_name, var_vals in var2vals.items():
                if isinstance(var_vals[0], str):
                    continue
                metric = {}
                n_resps = len(var_vals)
                metric[f"mean@{n_resps}"] = np.mean(var_vals)
                metric[f"std@{n_resps}"] = np.std(var_vals)

                ns = []
                n = 2
                while n < n_resps:
                    ns.append(n)
                    n *= 2
                ns.append(n_resps)

                # If there are multiple responses, we can compute the best/worst-of-N metrics
                # If not, they are the same as the single response metrics
                if n_resps > 1:
                    for n in ns:
                        if n == n_resps:
                            # Non-bootstrapped
                            metric[f"best@{n}/mean"] = np.max(var_vals)
                            metric[f"worst@{n}/mean"] = np.min(var_vals)
                            if var2vals.get("pred", None) is not None:
                                vote_data = [{"val": val, "pred": pred} for val, pred in zip(var_vals, var2vals["pred"])]
                                metric[f"maj@{n}/mean"] = calc_maj_val(vote_data, vote_key="pred", val_key="val")
                        else:
                            # Bootstrapped
                            [(bon_mean, bon_std), (won_mean, won_std)] = bootstrap_metric(data=var_vals, subset_size=n, reduce_fns=[np.max, np.min], seed=seed)
                            metric[f"best@{n}/mean"], metric[f"best@{n}/std"] = bon_mean, bon_std
                            metric[f"worst@{n}/mean"], metric[f"worst@{n}/std"] = won_mean, won_std
                            if var2vals.get("pred", None) is not None:
                                vote_data = [{"val": val, "pred": pred} for val, pred in zip(var_vals, var2vals["pred"])]
                                [(maj_n_mean, maj_n_std)] = bootstrap_metric(
                                    data=vote_data,
                                    subset_size=n,
                                    reduce_fns=[partial(calc_maj_val, vote_key="pred", val_key="val")],
                                    seed=seed,
                                )
                                metric[f"maj@{n}/mean"], metric[f"maj@{n}/std"] = maj_n_mean, maj_n_std

                data_src2prompt2var2metric[data_source][prompt][var_name] = metric
    # Aggregate metrics across prompts
    data_src2var2metric2prompt_vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for data_source, prompt2var2metric in data_src2prompt2var2metric.items():
        for prompt, var2metric in prompt2var2metric.items():
            for var_name, metric in var2metric.items():
                for metric_name, metric_val in metric.items():
                    data_src2var2metric2prompt_vals[data_source][var_name][metric_name].append(metric_val)
    data_src2var2metric2val = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    for data_source, var2metric2prompt_vals in data_src2var2metric2prompt_vals.items():
        for var_name, metric2prompt_vals in var2metric2prompt_vals.items():
            for metric_name, prompt_vals in metric2prompt_vals.items():
                if metric_name.startswith("mean@"):  # aggregate means then compute std of means
                    data_src2var2metric2val[data_source][var_name][metric_name] = np.mean(prompt_vals)
                    
                    # now std over those rollout-means:
                    n = metric_name.split("@")[1]   # number of rollouts
                    n = int(n)
                    run_means = []
                    for run_idx in range(n):
                        # gather the run_idx'th score from each prompt
                        vals_this_run = [
                            var2vals[var_name][run_idx]
                            for var2vals in data_src2prompt2var2vals[data_source].values()
                        ]
                        run_means.append(np.mean(vals_this_run))

                    # now std over those rollout-means:
                    data_src2var2metric2val[data_source][var_name][f"std@{n}"] = np.std(run_means)
                elif metric_name.startswith("std@"):  # skip original per-prompt stds
                    continue
                else:  # average other metrics
                    data_src2var2metric2val[data_source][var_name][metric_name] = np.mean(prompt_vals)

    return data_src2var2metric2val
