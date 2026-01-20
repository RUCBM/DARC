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
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import uuid
from collections import defaultdict, Counter
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum, IntEnum, auto
from typing import Any, Dict, List, Optional, Type
import random
import numpy as np
import ray
import torch
from ray.experimental.tqdm_ray import tqdm
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizer, ProcessorMixin

from ..protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto
from ..utils import torch_functional as VF
import re
from ..single_controller.base import Worker
from ..single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from ..single_controller.ray.base import create_colocated_worker_cls
from ..utils.checkpoint import CHECKPOINT_TRACKER, remove_obsolete_ckpt
from ..utils.logger import Tracker
import json
from mathruler.grader import extract_boxed_content
from ..utils.py_functional import convert_dict_to_str, timer
from ..utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from ..workers.fsdp_workers import FSDPWorker
from ..workers.reward import FunctionRewardManager
from . import core_algos
from .config import PPOConfig
from .metrics import compute_data_metrics, compute_throughout_metrics, compute_timing_metrics, reduce_metrics

def _is_invalid_candidate(ans: object) -> bool:
    """Filter out solver 'non-answers' like \\boxed{None} from voting/scoring."""
    if not isinstance(ans, str):
        return True
    s = ans.strip()
    if not s:
        return True
    return s.lower() == "none"


class Role(IntEnum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = auto()
    Rollout = auto()
    ActorRollout = auto()
    Critic = auto()
    RefPolicy = auto()
    RewardModel = auto()
    ActorRolloutRef = auto()


class AdvantageEstimator(str, Enum):
    """
    Using an enumeration class to avoid spelling errors in adv_estimator
    """

    GAE = "gae"
    GRPO = "grpo"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REMAX = "remax"
    RLOO = "rloo"


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker."""
        return self.resource_pool_dict[self.mapping[role]]

    def get_num_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        gpus_available = ray.available_resources().get("GPU", 0)
        gpus_required = self.get_num_gpus()
        if gpus_available < gpus_required:
            raise ValueError(f"Total available GPUs {gpus_available} is less than total desired GPUs {gpus_required}.")


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.KLController, kl_penalty="kl"):
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]
    response_mask = data.batch["response_mask"]

    # compute kl between ref_policy and current policy
    kld = core_algos.compute_kl(data.batch["old_log_probs"], data.batch["ref_log_probs"], kl_penalty=kl_penalty)
    kld = kld * response_mask  # (batch_size, response_length)

    data.batch["token_level_rewards"] = token_level_scores - kl_ctrl.kl_coef * kld

    current_kl = VF.masked_mean(kld, mask=response_mask, dim=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()
    metrics = {"critic/kl": current_kl, "critic/kl_coef": kl_ctrl.kl_coef}

    # According to https://github.com/huggingface/trl/blob/v0.11.0/trl/trainer/ppo_trainer.py#L880
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    return data, metrics


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    norm_adv_by_std_in_grpo: bool = True,
):
    token_level_rewards = data.batch["token_level_rewards"]
    response_mask = data.batch["response_mask"]
    index = data.non_tensor_batch["uid"]
    if adv_estimator == AdvantageEstimator.GAE:
        values = data.batch["values"]
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards, values, response_mask, gamma, lam
        )
    elif adv_estimator == AdvantageEstimator.GRPO:
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards,
            response_mask,
            index,
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS:
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards, response_mask, gamma
        )
    elif adv_estimator == AdvantageEstimator.REMAX:
        reward_baselines = data.batch["reward_baselines"]
        advantages, returns = core_algos.compute_remax_outcome_advantage(
            token_level_rewards, reward_baselines, response_mask
        )
    elif adv_estimator == AdvantageEstimator.RLOO:
        advantages, returns = core_algos.compute_rloo_outcome_advantage(token_level_rewards, response_mask, index)
    else:
        raise NotImplementedError

    data.batch["advantages"] = advantages
    data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    def __init__(
        self,
        config: PPOConfig,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        train_dataloader: StatefulDataLoader,
        val_dataloader: StatefulDataLoader,
        role_worker_mapping: dict[Role, Type[Worker]],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: Type[RayWorkerGroup] = RayWorkerGroup,
        reward_fn: Optional[FunctionRewardManager] = None,
        val_reward_fn: Optional[FunctionRewardManager] = None,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.worker.hybrid_engine
        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, (
                f"ActorRollout should be included in {role_worker_mapping.keys()}."
            )
        else:
            raise NotImplementedError

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reward_model = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        # define KL control
        if Role.RefPolicy in role_worker_mapping and not config.algorithm.disable_kl:
            self.use_reference_policy = True
            self.kl_ctrl = core_algos.get_kl_controller(config.algorithm)
        else:
            self.use_reference_policy = False
            self.kl_ctrl = core_algos.FixedKLController(init_kl_coef=0.0)
            print("KL is disabled, no KL metrics will be logged. Please set `kl_coef=0` to log KL metrics.")

        if config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        else:
            self.use_critic = False

        if config.algorithm.adv_estimator not in list(AdvantageEstimator):
            raise NotImplementedError(f"Unknown advantage estimator: {config.algorithm.adv_estimator}.")

        # Sanity checks on batch sizes and rollout settings
        if (
            self.config.data.rollout_batch_size % self.config.worker.actor.global_batch_size != 0
        ):
            raise ValueError("Rollout batch size must be divisible by actor global batch size.")

        if (
            self.config.data.rollout_batch_size * self.config.worker.rollout.n
        ) % self.config.worker.actor.micro_batch_size_per_device_for_experience != 0:
            raise ValueError(
                "Rollout batch size * rollout.n must be divisible by actor micro batch size for experience."
            )

        if self.use_critic:
            if (
                self.config.data.rollout_batch_size % self.config.worker.critic.global_batch_size != 0
            ):
                raise ValueError("Rollout batch size must be divisible by critic global batch size.")

            if (
                self.config.data.rollout_batch_size * self.config.worker.rollout.n
            ) % self.config.worker.critic.micro_batch_size_per_device_for_experience != 0:
                raise ValueError(
                    "Rollout batch size * rollout.n must be divisible by critic micro batch size for experience."
                )

        if (
            self.config.algorithm.adv_estimator in (AdvantageEstimator.GRPO, AdvantageEstimator.RLOO)
            and self.config.worker.rollout.n == 1
        ):
            raise ValueError("GRPO and RLOO algorithm need `config.worker.rollout.n > 1`.")

        # Determine training steps
        if self.config.trainer.max_steps is not None:
            self.training_steps = self.config.trainer.max_steps
        else:
            self.training_steps = len(train_dataloader) * self.config.trainer.total_epochs

        self.config.worker.actor.optim.training_steps = self.training_steps
        self.config.worker.critic.optim.training_steps = self.training_steps
        print(f"Total training steps: {self.training_steps}")

    # ------------------------
    # Pseudo-label generation
    # ------------------------
    def _maybe_label_from_text_prompt(self, batch: DataProto) -> DataProto:
        """
        Optionally perform non-symmetric self-distillation by generating labels
        from a separate text prompt column.

        Controlled by config.worker.reward.reward_function_kwargs, with keys:
          - solver_label_mode: 'rule' | 'self_vote' | 'auto'
              'self_vote' -> build label from text_prompt via majority vote
              'rule'      -> do nothing (use existing ground_truth)
          - label_prompt_key: dataset column to read prompts from (default 'text_prompt')
          - label_n: optional int to override sampling n (default rollout.n)
          - label_temperature/top_p/etc: optional sampling overrides
        """
        try:
            kwargs = getattr(self.config.worker.reward, "reward_function_kwargs", {}) or {}
            mode = str(kwargs.get("solver_label_mode", os.getenv("SOLVER_LABEL_MODE", "rule"))).lower()
            if mode not in ("self_vote", "auto"):
                return batch

            label_prompt_key = str(kwargs.get("label_prompt_key", os.getenv("LABEL_PROMPT_KEY", "text_prompt")))
            # collect prompts for labeling
            ntb = batch.non_tensor_batch if batch.non_tensor_batch is not None else {}
            label_src = ntb.get(label_prompt_key)
            if label_src is None:
                return batch

            # normalize to list
            try:
                arr = label_src.tolist()
            except Exception:
                arr = label_src
            if not isinstance(arr, (list, np.ndarray)):
                return batch

            prompts_text: List[str] = []
            batch_to_label_idx: List[int] = []  # map from batch index -> prompts_text index
            for bi, msg in enumerate(arr):
                # msg can be a list[dict] or a str (json)
                if isinstance(msg, str):
                    m = msg.strip()
                    if m.startswith("{") or m.startswith("["):
                        try:
                            msg = json.loads(m)
                        except Exception:
                            msg = []
                if not isinstance(msg, list):
                    continue  # skip samples without a valid chat list
                if len(msg) == 0:
                    continue
                if self.tokenizer.chat_template:
                    pr = self.tokenizer.apply_chat_template(msg, add_generation_prompt=True, tokenize=False)
                else:
                    sys_text = ""
                    usr_text = ""
                    for m in msg:
                        if isinstance(m, dict) and m.get("role") == "system":
                            sys_text = str(m.get("content", ""))
                        if isinstance(m, dict) and m.get("role") == "user":
                            usr_text = str(m.get("content", ""))
                    pr = "system: " + sys_text + "\n" + "user: " + usr_text
                prompts_text.append(pr)
                batch_to_label_idx.append(bi)

            # Tokenize and build DataProto
            input_ids_list = []
            attention_mask_list = []
            position_ids_list = []
            for pr in prompts_text:
                model_inputs = self.tokenizer([pr], add_special_tokens=False, return_tensors="pt")
                input_ids = model_inputs.pop("input_ids")[0]
                attention_mask = model_inputs.pop("attention_mask")[0]
                position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)
                # NOTE： maybe bug here，导致问题被裁剪掉了，从而有一些样本从始至终利用不到！
                input_ids, attention_mask, position_ids = VF.postprocess_data(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    max_length=self.config.data.max_prompt_length,
                    pad_token_id=self.tokenizer.pad_token_id,
                    left_pad=True,
                    truncation="right",
                )
                input_ids_list.append(input_ids)
                attention_mask_list.append(attention_mask)
                position_ids_list.append(position_ids)

            if len(input_ids_list) == 0:
                return batch

            input_ids = torch.stack(input_ids_list, dim=0)
            attention_mask = torch.stack(attention_mask_list, dim=0)
            position_ids = torch.stack(position_ids_list, dim=0)

            label_batch = DataProto.from_single_dict({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "raw_prompt_ids": np.array([self.tokenizer.encode(pr, add_special_tokens=False)[: self.config.data.max_prompt_length] for pr in prompts_text], dtype=object),
            })
            # set rollout overrides if needed (reuse default n)
            label_batch.meta_info.update({
                "min_pixels": self.config.data.min_pixels,
                "max_pixels": self.config.data.max_pixels,
            })
            # sampling overrides
            try:
                label_n = int(kwargs.get("label_n", os.getenv("LABEL_N", "0")))
            except Exception:
                label_n = 0
            if label_n and label_n > 0:
                label_batch.meta_info["n"] = label_n
            # optional overrides
            for k in ["temperature", "top_p", "top_k"]:
                v = kwargs.get(f"label_{k}", os.getenv(f"LABEL_{k.upper()}", None))
                if v is not None:
                    try:
                        label_batch.meta_info[k] = type(getattr(self.config.worker.rollout, k))(v) if hasattr(self.config.worker.rollout, k) else float(v)
                    except Exception:
                        pass
            label_batch, pad_size = pad_dataproto_to_divisor(label_batch, self.actor_rollout_wg.world_size)
            label_out = self.actor_rollout_wg.generate_sequences(label_batch)
            label_out = unpad_dataproto(label_out, pad_size=pad_size)

            # Decode and majority vote per original sample
            resp_ids = label_out.batch["responses"]  # shape: [m*n, L]
            total = resp_ids.shape[0]
            m = len(prompts_text)
            # infer n from ratio
            n = max(total // max(m, 1), 1)

            def _extract_answer(txt: str) -> str:
                try:
                    ans = extract_boxed_content(txt)
                    if isinstance(ans, str) and ans.strip():
                        return ans.strip()
                except Exception:
                    pass
                pat = re.compile(r"\\boxed\{(.*?)\}")
                mm = list(pat.finditer(txt))
                if mm:
                    return mm[-1].group(1).strip()
                return ""

            answers_per_item: List[str] = [""] * m
            vote_frac_per_item: List[float] = [0.0] * m
            for idx in range(m):
                group = resp_ids[idx * n : (idx + 1) * n]
                texts = [self.tokenizer.decode(t.tolist(), skip_special_tokens=True) for t in group]
                ans = [_extract_answer(t) for t in texts]

                # 去掉不可能的候选
                ans = [sample for sample in ans if not _is_invalid_candidate(sample)]
                valid_count = len(ans)
                cnt = Counter(a for a in ans if a)
                if not cnt:
                    answers_per_item[idx] = ""
                    vote_frac_per_item[idx] = 0.0
                else:
                    top = cnt.most_common()
                    winners = {a for a, c in top if c == top[0][1]}
                    chosen = ""
                    for a in ans:
                        if a in winners:
                            chosen = a
                            break
                    # Treat "None" (e.g., \boxed{None}) as an invalid self-vote label. 题有问题！！
                    # This avoids rewarding/anchoring the solver on emitting "None".
                    chosen_norm = str(chosen).strip().lower()
                    # unwrap common LaTeX wrappers like \text{None}
                    chosen_norm = re.sub(r"\\(?:text|mathrm|mathbf|bf|rm)\s*\{\s*(.*?)\s*\}", r"\1", chosen_norm)
                    chosen_norm = chosen_norm.strip().strip(".,;:!\"'`()[]{}<>")
                    if chosen_norm == "none" or chosen_norm == "":
                        answers_per_item[idx] = ""
                        vote_frac_per_item[idx] = 0.0
                        continue

                    answers_per_item[idx] = chosen
                    try:
                        denom = n if self.config.trainer.self_vote_ratio_use_all else valid_count
                        vote_frac_per_item[idx] = float(top[0][1]) / max(denom, 1)
                    except Exception:
                        vote_frac_per_item[idx] = 0.0

            # Write back into ground_truth as rule gold
            gts = ntb.get("ground_truth")
            if gts is not None:
                new_gts = gts.copy()
                for local_idx, ans in enumerate(answers_per_item):
                    bi = batch_to_label_idx[local_idx]
                    try:
                        item = new_gts[bi]
                        if isinstance(item, dict):
                            # preserve extra fields (e.g., extra_info), force style to rule
                            item = item.copy()
                            item["ground_truth"] = str(ans)
                            item["style"] = "rule"
                            new_gts[bi] = item
                        else:
                            new_gts[bi] = str(ans)
                    except Exception:
                        new_gts[bi] = str(ans)
                batch.non_tensor_batch["ground_truth"] = new_gts

                if random.random() > 0.99:
                    print('batch.non_tensor_batch:', batch.non_tensor_batch)

            # attach per-sample vote fraction for optional masking later
            try:
                # initialize with 1.0 so non-labeled items are not masked out by default
                ratios = np.ones((len(batch),), dtype=np.float32)
                for local_idx, frac in enumerate(vote_frac_per_item):
                    bi = batch_to_label_idx[local_idx]
                    ratios[bi] = float(frac)
                batch.non_tensor_batch["self_vote_ratio"] = ratios
            except Exception:
                pass

        except Exception as e:
            # be robust; on failure, leave ground_truth unchanged
            print(f"[warn] self-vote text_prompt labeling failed: {e}")
        return batch

    def _maybe_log_val_generations(
        self, inputs: List[str], outputs: List[str], labels: List[str], scores: List[float]
    ) -> None:
        """Log a table of validation samples"""
        if self.config.trainer.val_generations_to_log <= 0:
            return

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, labels, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        samples = samples[: self.config.trainer.val_generations_to_log]
        self.logger.log_generation(samples, self.global_step)

    def _validate(self) -> Dict[str, Any]:
        reward_tensor_lst = []
        # Lists to collect samples for the table
        sample_inputs, sample_outputs, sample_labels, sample_scores = [], [], [], []
        reward_metrics_lst = defaultdict(list)
        for batch_dict in self.val_dataloader:
            test_batch = DataProto.from_single_dict(batch_dict)
            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            if "multi_modal_data" in test_batch.non_tensor_batch.keys():
                test_gen_batch = test_batch.pop(
                    batch_keys=["input_ids", "attention_mask", "position_ids"],
                    non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
                )
            else:
                test_gen_batch = test_batch.pop(
                    batch_keys=["input_ids", "attention_mask", "position_ids"],
                    non_tensor_batch_keys=["raw_prompt_ids"],
                )

            test_gen_batch.meta_info = self.config.worker.rollout.val_override_config
            test_gen_batch.meta_info.update({
                "min_pixels": self.config.data.min_pixels,
                "max_pixels": self.config.data.max_pixels,
            })
            test_gen_batch, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            test_output_gen_batch = self.actor_rollout_wg.generate_sequences(test_gen_batch)
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch, pad_size=pad_size)

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)
            sample_labels.extend(test_batch.non_tensor_batch["ground_truth"].tolist())
            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            reward_tensor, reward_metrics = ray.get(self.val_reward_fn.compute_reward.remote(test_batch))

            # Store scores
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_tensor_lst.append(reward_tensor)
            for key, value in reward_metrics.items():
                reward_metrics_lst[key].extend(value)

        self._maybe_log_val_generations(sample_inputs, sample_outputs, sample_labels, sample_scores)
        reward_score = torch.cat(reward_tensor_lst, dim=0).sum(-1).mean().item()
        val_reward_metrics = {f"val/{key}_reward": value for key, value in reduce_metrics(reward_metrics_lst).items()}
        return {"val/reward_score": reward_score, **val_reward_metrics}

    def init_workers(self) -> None:
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout], config=self.config.worker, role="actor_rollout"
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.Critic], config=self.config.worker, role="critic"
            )
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy], config=self.config.worker, role="ref"
            )
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_reward_model:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.RewardModel], config=self.config.worker, role="reward"
            )
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg: Dict[str, FSDPWorker] = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_reward_model:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

    def _save_checkpoint(self) -> None:
        # path: {save_checkpoint_path}/global_step_{global_step}/{actor,critic}
        remove_obsolete_ckpt(
            self.config.trainer.save_checkpoint_path, self.global_step, self.config.trainer.save_limit
        )
        folder_path = os.path.join(self.config.trainer.save_checkpoint_path, f"global_step_{self.global_step}")
        actor_path = os.path.join(folder_path, "actor")
        self.actor_rollout_wg.save_checkpoint(actor_path)

        if self.use_critic:
            critic_path = os.path.join(folder_path, "critic")
            self.critic_wg.save_checkpoint(critic_path)

        dataloader_path = os.path.join(folder_path, "dataloader.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_path)

        last_global_step_path = os.path.join(self.config.trainer.save_checkpoint_path, CHECKPOINT_TRACKER)
        with open(last_global_step_path, "w") as f:
            f.write(str(self.global_step))

    def _load_checkpoint(self) -> None:
        if self.config.trainer.load_checkpoint_path is None:
            return

        if "global_step_" not in self.config.trainer.load_checkpoint_path.strip(os.path.sep).split(os.path.sep)[-1]:
            raise ValueError("`load_checkpoint_path` should end with `global_step_*`.")

        print(f"Load from checkpoint: {self.config.trainer.load_checkpoint_path}.")
        self.global_step = int(self.config.trainer.load_checkpoint_path.strip(os.path.sep).split("global_step_")[-1])
        actor_path = os.path.join(self.config.trainer.load_checkpoint_path, "actor")
        self.actor_rollout_wg.load_checkpoint(actor_path)
        if self.use_critic:
            critic_path = os.path.join(self.config.trainer.load_checkpoint_path, "critic")
            self.critic_wg.load_checkpoint(critic_path)

        dataloader_path = os.path.join(self.config.trainer.load_checkpoint_path, "dataloader.pt")
        if os.path.exists(dataloader_path):
            dataloader_state_dict = torch.load(dataloader_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"No dataloader state found at {dataloader_path}, will start from scratch.")

    def _balance_batch(self, batch: DataProto, metrics: Dict[str, Any], logging_prefix: str = "global_seqlen") -> None:
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def fit(self):

        import os
        if os.getenv("VERL_DEBUG_MAIN") == "1":
            import pydevd_pycharm
            pydevd_pycharm.settrace(
                'localhost',
                port=6067,
                stdoutToServer=True,
                stderrToServer=True,
                suspend=False,  # 用断点控制停哪
            )
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        self.logger = Tracker(loggers=self.config.trainer.logger, config=self.config.to_dict())
        self.global_step = 0
        val_metrics: Optional[Dict[str, Any]] = None

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.val_before_train:
            val_metrics = self._validate()
            self.logger.log(data=val_metrics, step=self.global_step)
            if self.config.trainer.val_only:
                return

        for _ in tqdm(range(self.config.trainer.total_epochs), desc="Epoch", position=0):
            for batch_dict in tqdm(self.train_dataloader, desc="Running step", position=1):
                self.global_step += 1
                if self.global_step > self.training_steps:
                    break

                metrics, timing_raw = {}, {}
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                # pop those keys for generation
                if "multi_modal_data" in batch.non_tensor_batch.keys():
                    gen_batch = batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
                    )
                    gen_batch.meta_info.update({
                        "min_pixels": self.config.data.min_pixels,
                        "max_pixels": self.config.data.max_pixels,
                    })
                else:
                    gen_batch = batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids"],
                    )

                with timer("step", timing_raw):
                    # generate a batch
                    with timer("gen", timing_raw):  # wg: worker group
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

                    if self.config.algorithm.adv_estimator == "remax":
                        with timer("gen_max", timing_raw):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["temperature"] = 0
                            gen_baseline_batch.meta_info["n"] = 1
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor, _ = ray.get(self.reward_fn.compute_reward.remote(batch))
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))
                            batch.batch["reward_baselines"] = reward_baseline_tensor
                            del gen_baseline_batch, gen_baseline_output

                    # Non-symmetric distillation: label from text_prompt if requested
                    batch = self._maybe_label_from_text_prompt(batch)

                    batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                    )
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.worker.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # compute reward
                    with timer("reward", timing_raw):
                        reward_ref = self.reward_fn.compute_reward.remote(batch)

                    # recompute old_log_probs
                    with timer("old", timing_raw):
                        old_log_probs = self.actor_rollout_wg.compute_log_probs(batch)
                        batch = batch.union(old_log_probs)

                    # compute ref_log_probs
                    if self.use_reference_policy:
                        with timer("ref", timing_raw):
                            ref_log_probs = self.ref_policy_wg.compute_ref_log_probs(batch)
                            batch = batch.union(ref_log_probs)

                    # compute values
                    if self.use_critic:
                        with timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with timer("adv", timing_raw):
                        # get token level scores
                        reward_tensor, reward_metrics = ray.get(reward_ref)
                        batch.batch["token_level_scores"] = reward_tensor
                        reward_metrics = {f"reward/{k}": v for k, v in reduce_metrics(reward_metrics).items()}
                        metrics.update(reward_metrics)

                        # apply kl penalty if available
                        if not self.config.algorithm.use_kl_loss and self.use_reference_policy:
                            # apply kl penalty to reward
                            batch, kl_metrics = apply_kl_penalty(batch, self.kl_ctrl, self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # compute advantages, executed on the driver process
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            norm_adv_by_std_in_grpo=self.config.algorithm.norm_adv_by_std_in_grpo,
                        )

                        # Optionally mask out samples with low self-vote confidence by zeroing advantages
                        try:
                            kwargs = getattr(self.config.worker.reward, "reward_function_kwargs", {}) or {}
                            thr = float(kwargs.get("label_vote_threshold", os.getenv("LABEL_VOTE_THRESHOLD", "0") or 0))
                        except Exception:
                            thr = 0.0
                        if thr and "self_vote_ratio" in batch.non_tensor_batch:
                            try:
                                ratios_np = batch.non_tensor_batch["self_vote_ratio"]
                                # ensure numeric array
                                ratios = torch.as_tensor(ratios_np, dtype=batch.batch["advantages"].dtype, device=batch.batch["advantages"].device)
                                mask = (ratios >= thr).to(batch.batch["advantages"].dtype).unsqueeze(-1)
                                batch.batch["advantages"] = batch.batch["advantages"] * mask
                                if "returns" in batch.batch.keys():
                                    batch.batch["returns"] = batch.batch["returns"] * mask
                                # log proportion kept
                                try:
                                    metrics["grpo/self_vote_keep_frac"] = float((ratios >= thr).float().mean().item())
                                except Exception:
                                    pass
                            except Exception:
                                pass

                        # GRPO group-level stats: count kept prompts and active prompts (std>0), and std stats
                        try:
                            if self.config.algorithm.adv_estimator == AdvantageEstimator.GRPO:
                                uids = batch.non_tensor_batch.get("uid")
                                if uids is not None:
                                    # per-sample rewards (sum over tokens) as outcome scores
                                    scores = batch.batch["token_level_rewards"].sum(dim=-1).detach().float().cpu().tolist()
                                    # whether sample is kept by mask threshold; default True if no ratio or thr<=0
                                    kept_flags = None
                                    if thr and "self_vote_ratio" in batch.non_tensor_batch:
                                        try:
                                            kept_flags = (np.asarray(batch.non_tensor_batch["self_vote_ratio"]) >= float(thr)).tolist()
                                        except Exception:
                                            kept_flags = None
                                    if kept_flags is None:
                                        kept_flags = [True] * len(scores)

                                    # group by uid
                                    try:
                                        uid_list = uids.tolist() if hasattr(uids, "tolist") else list(uids)
                                    except Exception:
                                        uid_list = list(uids)

                                    group_scores = {}
                                    group_any_keep = {}
                                    for i, g in enumerate(uid_list):
                                        group_scores.setdefault(g, []).append(scores[i])
                                        if kept_flags[i]:
                                            group_any_keep[g] = True
                                        elif g not in group_any_keep:
                                            group_any_keep[g] = False

                                    stds_active = []
                                    kept_count = 0
                                    active_count = 0
                                    for g, lst in group_scores.items():
                                        # std over n samples; if n<=1 -> 0
                                        s = float(np.std(np.asarray(lst), ddof=0)) if len(lst) > 1 else 0.0
                                        if group_any_keep.get(g, True):
                                            kept_count += 1
                                            if s > 0.0:
                                                active_count += 1
                                                stds_active.append(s)

                                    # stats over active groups
                                    if len(stds_active) > 0:
                                        metrics["grpo/std_max"] = float(np.max(stds_active))
                                        metrics["grpo/std_min"] = float(np.min(stds_active))
                                        metrics["grpo/std_mean"] = float(np.mean(stds_active))
                                    else:
                                        metrics["grpo/std_max"] = 0.0
                                        metrics["grpo/std_min"] = 0.0
                                        metrics["grpo/std_mean"] = 0.0
                                    metrics["grpo/kept_prompts"] = int(kept_count)
                                    metrics["grpo/active_prompts"] = int(active_count)
                        except Exception:
                            pass

                    # Optionally dump rollout samples for debugging
                    try:
                        self._maybe_dump_rollouts(batch)
                    except Exception as _dump_exc:
                        # be robust; don't crash training due to dump errors
                        print(f"[warn] dump_rollouts failed at step {self.global_step}: {_dump_exc}")

                    # update critic
                    if self.use_critic:
                        with timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)

                        critic_metrics = reduce_metrics(critic_output.non_tensor_batch)
                        metrics.update(critic_metrics)

                    # update actor
                    if self.config.trainer.critic_warmup <= self.global_step:
                        with timer("update_actor", timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch)

                        actor_metrics = reduce_metrics(actor_output.non_tensor_batch)
                        metrics.update(actor_metrics)

                    # validate
                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.val_freq > 0
                        and self.global_step % self.config.trainer.val_freq == 0
                    ):
                        with timer("validation", timing_raw):
                            val_metrics = self._validate()

                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and self.global_step % self.config.trainer.save_freq == 0:
                        with timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                # collect metrics
                num_gpus = self.resource_pool_manager.get_num_gpus()
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, num_gpus=num_gpus))

                self.logger.log(data=metrics, step=self.global_step)

        # perform validation after training
        if self.val_reward_fn is not None:
            if (
                val_metrics is None
                or self.config.trainer.val_freq <= 0
                or self.global_step % self.config.trainer.val_freq != 0
            ):
                val_metrics = self._validate()
                self.logger.log(data=val_metrics, step=self.global_step)

            print(f"Final validation metrics: {convert_dict_to_str(val_metrics)}")

        if self.config.trainer.save_freq <= 0 or self.global_step % self.config.trainer.save_freq != 0:
            self._save_checkpoint()

    def _maybe_dump_rollouts(self, batch: DataProto) -> None:
        cfg = self.config.trainer
        n = int(getattr(cfg, "dump_rollout_n", 0) or 0)
        if n <= 0:
            return
        every = int(getattr(cfg, "dump_rollout_every", 0) or 0)
        if every > 0 and (self.global_step % every) != 0:
            return

        path = getattr(cfg, "dump_rollout_path", None)
        if not path:
            return

        # Collect data
        prompts = batch.batch.get("prompts")
        responses = batch.batch.get("responses")
        if prompts is None or responses is None:
            return

        # Optional fields
        uids = batch.non_tensor_batch.get("uid") if batch.non_tensor_batch is not None else None
        gts = batch.non_tensor_batch.get("ground_truth") if batch.non_tensor_batch is not None else None
        token_scores = batch.batch.get("token_level_scores")

        bs = prompts.shape[0]
        limit = min(bs, n)

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            for i in range(limit):
                pr_ids = prompts[i].tolist()
                rs_ids = responses[i].tolist()
                pr_txt = self.tokenizer.decode(pr_ids, skip_special_tokens=True)
                rs_txt = self.tokenizer.decode(rs_ids, skip_special_tokens=True)
                uid = None
                if uids is not None:
                    try:
                        uid = str(uids[i])
                    except Exception:
                        uid = None
                gt_val = None
                if gts is not None:
                    try:
                        gt_val = gts[i]
                        # best-effort to keep it JSON-serializable
                        if hasattr(gt_val, "item"):
                            gt_val = gt_val.item()
                    except Exception:
                        gt_val = None
                reward_sum = None
                if token_scores is not None:
                    try:
                        reward_sum = float(token_scores[i].sum().item())
                    except Exception:
                        reward_sum = None

                record = {
                    "step": int(self.global_step),
                    "uid": uid,
                    "prompt": pr_txt,
                    "response": rs_txt,
                    "ground_truth": gt_val,
                    "reward_sum": reward_sum,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
