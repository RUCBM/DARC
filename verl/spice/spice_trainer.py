from __future__ import annotations

import os
import random
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from tqdm import tqdm
import numpy as np
import ray
import torch
from tensordict import TensorDict
from transformers import PreTrainedTokenizer

from ..protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto
from ..utils import torch_functional as VF
from ..utils.checkpoint import CHECKPOINT_TRACKER, remove_obsolete_ckpt
from ..utils.logger import Tracker
from ..utils.py_functional import timer
from ..trainer import core_algos
from ..trainer.metrics import reduce_metrics
from ..trainer.ray_trainer import AdvantageEstimator, ResourcePoolManager, Role
from ..single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from ..single_controller.ray.base import create_colocated_worker_cls
from ..workers.fsdp_workers import FSDPWorker

from .config import SpiceConfig
from .corpus import MixedCorpusSampler, ParquetTextCorpus
from .parsing import ParsedQA, parse_challenger_output
from .prompting import build_challenger_messages, build_reasoner_messages
from .related import generate_relation_results
from .verify import AnswerVerifier


def _dataproto_gather(data: DataProto, indices: List[int]) -> DataProto:
    idx_t = torch.as_tensor(indices, dtype=torch.long)
    idx_np = np.asarray(indices, dtype=np.int64)
    batch = data.batch[idx_t]
    non_tensor = {k: v[idx_np] for k, v in data.non_tensor_batch.items()}
    return DataProto(batch=batch, non_tensor_batch=non_tensor, meta_info=data.meta_info.copy())


def _build_prompt_dataproto(
    tokenizer: PreTrainedTokenizer,
    messages_list: List[List[Dict[str, Any]]],
    *,
    max_prompt_length: int,
    truncation: str,
    left_pad: bool = True,
) -> Tuple[DataProto, List[str]]:
    prompts_text: List[str] = []
    input_ids_lst: List[torch.Tensor] = []
    attention_mask_lst: List[torch.Tensor] = []
    position_ids_lst: List[torch.Tensor] = []
    raw_prompt_ids: List[List[int]] = []

    for messages in messages_list:
        if tokenizer.chat_template:
            prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        else:
            # minimal fallback; most base models in this repo use chat templates
            prompt = "\n".join([f"{m.get('role','user')}: {m.get('content','')}" for m in messages])
        prompts_text.append(prompt)

        model_inputs = tokenizer([prompt], add_special_tokens=False, return_tensors="pt")
        ids = model_inputs["input_ids"][0]
        mask = model_inputs["attention_mask"][0]
        pos = torch.clip(mask.cumsum(dim=0) - 1, min=0, max=None)

        ids, mask, pos = VF.postprocess_data(
            input_ids=ids,
            attention_mask=mask,
            position_ids=pos,
            max_length=max_prompt_length,
            pad_token_id=tokenizer.pad_token_id,
            left_pad=left_pad,
            truncation=truncation,
        )
        input_ids_lst.append(ids)
        attention_mask_lst.append(mask)
        position_ids_lst.append(pos)

        rp = tokenizer.encode(prompt, add_special_tokens=False)
        if len(rp) > max_prompt_length:
            if truncation == "left":
                rp = rp[-max_prompt_length:]
            elif truncation == "right":
                rp = rp[:max_prompt_length]
            else:
                rp = rp[-max_prompt_length:]
        raw_prompt_ids.append(rp)

    batch = TensorDict(
        {
            "input_ids": torch.stack(input_ids_lst, dim=0),
            "attention_mask": torch.stack(attention_mask_lst, dim=0),
            "position_ids": torch.stack(position_ids_lst, dim=0),
        },
        batch_size=len(messages_list),
    )
    non_tensor = {"raw_prompt_ids": np.array(raw_prompt_ids, dtype=object)}
    return DataProto(batch=batch, non_tensor_batch=non_tensor), prompts_text


def _outcome_rewards_to_token_rewards(response_mask: torch.Tensor, rewards: torch.Tensor) -> torch.Tensor:
    # reward only on final valid token of each response
    token_rewards = torch.zeros_like(response_mask, dtype=torch.float32)
    lengths = response_mask.sum(dim=-1).to(torch.long)
    for i in range(response_mask.size(0)):
        if lengths[i].item() > 0:
            token_rewards[i, lengths[i].item() - 1] = rewards[i]
    return token_rewards


def _gaussian(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return np.zeros_like(x, dtype=np.float32)
    return np.exp(-((x - mu) ** 2) / (2 * (sigma**2))).astype(np.float32)


@dataclass
class SpiceTrainer:
    tokenizer: PreTrainedTokenizer
    config_ppo: Any
    config_spice: SpiceConfig
    role_worker_mapping: dict[Role, Any]
    resource_pool_manager: ResourcePoolManager
    ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup

    def __post_init__(self) -> None:
        # We use GRPO for DrGRPO-style group advantages (mean subtraction).
        if self.config_ppo.algorithm.adv_estimator != AdvantageEstimator.GRPO:
            raise ValueError("SPICE trainer expects ppo.algorithm.adv_estimator=grpo")
        if self.config_ppo.worker.rollout.n != 1:
            raise ValueError("For SPICE, set ppo.worker.rollout.n=1 and override n per generation call.")

        loop = self.config_spice.loop
        if loop.group_size_g < 2:
            raise ValueError("SPICE group_size_g must be >= 2 (DrGRPO needs group size > 1).")
        if loop.challenger_attempts_n < loop.group_size_g:
            raise ValueError("SPICE challenger_attempts_n must be >= group_size_g.")

        if loop.min_valid_tasks_per_step < 0:
            raise ValueError("min_valid_tasks_per_step must be >= 0")
        if loop.max_challenger_resample_rounds < 1:
            raise ValueError("max_challenger_resample_rounds must be >= 1")

        # FSDPWorker will set actor.global_batch_size_per_device ~= actor.global_batch_size / dp_size.
        # Prefer the actual resource-pool world size for ActorRollout (can differ from ppo.trainer.*).
        try:
            actor_pool_name = self.resource_pool_manager.mapping[Role.ActorRollout]
            world_size = int(sum(self.resource_pool_manager.resource_pool_spec[actor_pool_name]))
        except Exception:
            world_size = int(self.config_ppo.trainer.n_gpus_per_node) * int(self.config_ppo.trainer.nnodes)
        sp = int(self.config_ppo.worker.actor.ulysses_sequence_parallel_size)
        if sp <= 0 or world_size <= 0 or world_size % sp != 0:
            raise ValueError(f"Invalid parallel sizes: world_size={world_size}, ulysses_sequence_parallel_size={sp}")
        dp = world_size // sp
        gb = int(self.config_ppo.worker.actor.global_batch_size)
        gbpd = gb // dp
        if gbpd <= 0:
            raise ValueError(
                f"actor.global_batch_size too small: global_batch_size={gb}, dp_size={dp} -> per_device={gbpd}"
            )
        # Challenger + reasoner updates are combined into one per SPICE step.
        c_bsz = int(loop.batch_size_b) * int(loop.group_size_g)
        # Reasoner trains on one selected question per doc, each with G samples -> ~B*G total samples.
        r_bsz = int(loop.batch_size_b) * int(loop.group_size_g)
        if c_bsz % gbpd != 0:
            raise ValueError(
                f"Challenger batch size B*G={c_bsz} must be divisible by actor.global_batch_size_per_device~{gbpd} "
                f"(set ppo.worker.actor.global_batch_size accordingly)."
            )
        if r_bsz % gbpd != 0:
            raise ValueError(
                f"Reasoner batch size B*G={r_bsz} must be divisible by actor.global_batch_size_per_device~{gbpd} "
                f"(set ppo.worker.actor.global_batch_size accordingly)."
            )
        # Ray dispatch (DataProto.chunk) requires the batch to be divisible by the worker-group world size.
        # We pad for *generation* (B prompts), but training updates must satisfy this exactly.
        if c_bsz % world_size != 0:
            raise ValueError(
                f"Challenger update batch size B*G={c_bsz} must be divisible by world_size={world_size} "
                f"(adjust ppo.trainer.n_gpus_per_node/nnodes or spice.loop.batch_size_b/group_size_g)."
            )
        if r_bsz % world_size != 0:
            raise ValueError(
                f"Reasoner update batch size B*G={r_bsz} must be divisible by world_size={world_size} "
                f"(adjust ppo.trainer.n_gpus_per_node/nnodes or spice.loop.batch_size_b/group_size_g)."
            )

        self.verifier = AnswerVerifier()

        # Reference-policy KL (optional; mirrors RayPPOTrainer behavior)
        if Role.RefPolicy in self.role_worker_mapping and not self.config_ppo.algorithm.disable_kl:
            self.use_reference_policy = True
            self.kl_ctrl = core_algos.get_kl_controller(self.config_ppo.algorithm)
        else:
            self.use_reference_policy = False
            self.kl_ctrl = core_algos.FixedKLController(init_kl_coef=0.0)

        # Corpus sampler
        corpus_cfg = self.config_spice.corpus
        general = (
            ParquetTextCorpus(
                corpus_cfg.general_parquet,
                text_key=corpus_cfg.text_key,
                max_rows=int(corpus_cfg.max_doc_per_parquet),
            )
            if corpus_cfg.general_parquet
            else None
        )
        math = (
            ParquetTextCorpus(
                corpus_cfg.math_parquet,
                text_key=corpus_cfg.text_key,
                max_rows=int(corpus_cfg.max_doc_per_parquet),
            )
            if corpus_cfg.math_parquet
            else None
        )
        self.corpus = MixedCorpusSampler(
            general=general,
            math=math,
            mix_math_ratio=corpus_cfg.mix_math_ratio,
            seed=corpus_cfg.seed,
            max_doc_per_parquet=corpus_cfg.max_doc_per_parquet,
        )
        self.rng = random.Random(loop.seed)
        self.debug_rng = random.Random(int(loop.seed) + 9973)
        self.debug_sample_rate = 0.05

        self.logger = Tracker(loggers=self.config_ppo.trainer.logger, config=self.config_ppo.to_dict())
        self.global_step = 0

    def init_workers(self) -> None:
        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
        actor_rollout_cls = RayClassWithInitArgs(
            cls=self.role_worker_mapping[Role.ActorRollout], config=self.config_ppo.worker, role="actor_rollout"
        )
        self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls

        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.RefPolicy], config=self.config_ppo.worker, role="ref"
            )
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        all_wg: Dict[str, FSDPWorker] = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            self.wg_dicts.append(wg_dict)

        self.actor_rollout_wg = all_wg["actor_rollout"]
        if self.use_reference_policy:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()
        # Init rollout last so vLLM can estimate KV-cache memory with other models already placed.
        self.actor_rollout_wg.init_model()

    def _load_checkpoint(self) -> None:
        path = self.config_ppo.trainer.load_checkpoint_path
        if path is None:
            return

        # Match existing trainer semantics: load_checkpoint_path ends with global_step_*
        tail = path.strip(os.path.sep).split(os.path.sep)[-1]
        if "global_step_" not in tail:
            raise ValueError("`ppo.trainer.load_checkpoint_path` should end with `global_step_*`.")

        step_str = tail.split("global_step_")[-1]
        self.global_step = int(step_str)

        actor_path = os.path.join(path, "actor")
        self.actor_rollout_wg.load_checkpoint(actor_path)

    def _maybe_apply_kl_penalty(self, batch: DataProto, metrics: Dict[str, Any], prefix: str) -> DataProto:
        if not self.use_reference_policy:
            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]
            return batch

        # If KL loss is enabled, the actor update will incorporate KL against ref_log_probs;
        # we keep rewards unchanged (outcome-only), matching RayPPOTrainer behavior.
        if self.config_ppo.algorithm.use_kl_loss:
            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]
            return batch

        # Otherwise, apply KL penalty directly to rewards.
        response_mask = batch.batch["response_mask"]
        kld = core_algos.compute_kl(
            log_probs=batch.batch["old_log_probs"],
            ref_log_probs=batch.batch["ref_log_probs"],
            kl_penalty=self.config_ppo.algorithm.kl_penalty,
        )
        kld = kld * response_mask
        batch.batch["token_level_rewards"] = batch.batch["token_level_scores"] - self.kl_ctrl.kl_coef * kld

        current_kl = VF.masked_mean(kld, mask=response_mask, dim=-1)
        current_kl = torch.mean(current_kl, dim=0).item()
        metrics[f"{prefix}/kl"] = float(current_kl)
        metrics[f"{prefix}/kl_coef"] = float(self.kl_ctrl.kl_coef)

        # Update controller (fixed or adaptive)
        self.kl_ctrl.update(current_kl=current_kl, n_steps=response_mask.size(0))
        return batch

    def _save_checkpoint(self) -> None:
        remove_obsolete_ckpt(self.config_ppo.trainer.save_checkpoint_path, self.global_step, self.config_ppo.trainer.save_limit)
        folder_path = os.path.join(self.config_ppo.trainer.save_checkpoint_path, f"global_step_{self.global_step}")
        actor_path = os.path.join(folder_path, "actor")
        self.actor_rollout_wg.save_checkpoint(actor_path)
        last_global_step_path = os.path.join(self.config_ppo.trainer.save_checkpoint_path, CHECKPOINT_TRACKER)
        with open(last_global_step_path, "w") as f:
            f.write(str(self.global_step))

    def _decode_responses(self, response_ids: torch.Tensor, response_mask: torch.Tensor) -> List[str]:
        out: List[str] = []
        lengths = response_mask.sum(dim=-1).to(torch.long)
        for i in range(response_ids.size(0)):
            ids = response_ids[i][: lengths[i].item()]
            out.append(self.tokenizer.decode(ids, skip_special_tokens=True))
        return out

    @staticmethod
    def _truncate_for_log(text: str, max_len: int = 3000) -> str:
        if not isinstance(text, str):
            text = str(text)
        text = text.replace("\n", "\\n")
        if len(text) > max_len:
            return text[:max_len] + "..."
        return text

    def _maybe_log_challenger_samples(self, chal_sel: DataProto, rewards: np.ndarray) -> None:
        if self.debug_sample_rate <= 0:
            return
        texts = self._decode_responses(chal_sel.batch["responses"], chal_sel.batch["response_mask"])
        valid = chal_sel.non_tensor_batch.get("spice_valid")
        reasons = chal_sel.non_tensor_batch.get("spice_invalid_reason")
        questions = chal_sel.non_tensor_batch.get("spice_question")
        answers = chal_sel.non_tensor_batch.get("spice_answer")
        n = min(len(texts), len(rewards))
        for i in range(n):
            if self.debug_rng.random() >= self.debug_sample_rate:
                continue
            reward = float(rewards[i])
            v = bool(valid[i]) if valid is not None else False
            reason = str(reasons[i]) if reasons is not None else ""
            q = str(questions[i]) if questions is not None else ""
            a = str(answers[i]) if answers is not None else ""
            pred = self._truncate_for_log(texts[i])
            q = self._truncate_for_log(q)
            a = self._truncate_for_log(a)
            reason = self._truncate_for_log(reason, max_len=120)
            print(
                f"[spice][debug][challenger] reward={reward:.4f} valid={v} "
                f"reason={reason} pred={pred} question={q} answer={a}"
            )

    def _maybe_log_reasoner_samples(self, r_train: DataProto, correct: np.ndarray) -> None:
        if self.debug_sample_rate <= 0:
            return
        texts = self._decode_responses(r_train.batch["responses"], r_train.batch["response_mask"])
        golds = r_train.non_tensor_batch.get("spice_gold_answer")
        group_uids = r_train.non_tensor_batch.get("spice_group_uid")
        n = min(len(texts), len(correct))
        for i in range(n):
            if self.debug_rng.random() >= self.debug_sample_rate:
                continue
            score = float(correct[i])
            gold = str(golds[i]) if golds is not None else ""
            group_uid = str(group_uids[i]) if group_uids is not None else ""
            pred = self._truncate_for_log(texts[i])
            gold = self._truncate_for_log(gold)
            print(
                f"[spice][debug][reasoner] score={score:.4f} group={group_uid} pred={pred} gold={gold}"
            )

    def _generate_sequences_padded(self, prompts: DataProto, *, n: int) -> DataProto:
        prompts, pad_size = pad_dataproto_to_divisor(prompts, self.actor_rollout_wg.world_size)
        out: DataProto = self.actor_rollout_wg.generate_sequences(prompts)
        if pad_size:
            out = unpad_dataproto(out, pad_size=pad_size * int(n))
        return out

    @staticmethod
    def _load_hf_or_parquet_dataset(data_path: str):
        from datasets import load_dataset  # local import; optional for SPICE-only runs

        if "@" in data_path:
            data_path, data_split = data_path.split("@", 1)
        else:
            data_split = "train"

        if os.path.isdir(data_path):
            return load_dataset("parquet", data_dir=data_path, split="train")
        if os.path.isfile(data_path):
            return load_dataset("parquet", data_files=data_path, split="train")
        return load_dataset(data_path, split=data_split)

    def _validate_reasoner(self) -> Dict[str, Any]:
        data_cfg = self.config_ppo.data
        trainer_cfg = self.config_ppo.trainer

        val_files = str(data_cfg.val_files or "").strip()
        if not val_files:
            return {"val/enabled": 0.0}

        prompt_key = str(data_cfg.val_prompt_key or data_cfg.prompt_key or "prompt")
        answer_key = str(data_cfg.val_answer_key or data_cfg.answer_key or "answer")
        ds = self._load_hf_or_parquet_dataset(val_files)
        if prompt_key not in ds.column_names:
            raise KeyError(f"Validation dataset missing prompt key `{prompt_key}` (columns={ds.column_names}).")
        if answer_key not in ds.column_names:
            raise KeyError(f"Validation dataset missing answer key `{answer_key}` (columns={ds.column_names}).")

        n_questions = int(len(ds))
        if n_questions <= 0:
            return {"val/enabled": 1.0, "val/acc": 0.0, "val/questions": 0.0}

        # Treat ppo.data.val_batch_size as the number of prompts per validation generation call.
        if int(data_cfg.val_batch_size) == -1:
            chunk_size = n_questions
        else:
            chunk_size = max(1, int(data_cfg.val_batch_size))

        # Use val overrides if present; otherwise default to deterministic single-sample validation.
        meta = dict(self.config_ppo.worker.rollout.val_override_config or {})
        meta.setdefault("n", 1)
        # NOTE： 应该设置为0， 但是为了保持一致我们这里先设置了1.0
        meta.setdefault("temperature", 0.0)
        meta.setdefault("top_p", 1.0)
        n = int(meta.get("n", 1))
        if n <= 0:
            n = 1
            meta["n"] = 1

        total_correct = 0
        total_outputs = 0
        total_any_correct = 0

        for start in range(0, n_questions, chunk_size):
            end = min(start + chunk_size, n_questions)
            batch = ds[start:end]
            problems = [str(x) for x in batch[prompt_key]]
            golds = [str(x) for x in batch[answer_key]]

            messages_list = [build_reasoner_messages(q) for q in problems]
            prompts, _ = _build_prompt_dataproto(
                self.tokenizer,
                messages_list,
                max_prompt_length=data_cfg.max_prompt_length,
                truncation=self.config_spice.prompts.reasoner_truncation,
                left_pad=True,
            )
            prompts.meta_info.update(meta)

            prompts, pad_size = pad_dataproto_to_divisor(prompts, self.actor_rollout_wg.world_size)
            out: DataProto = self.actor_rollout_wg.generate_sequences(prompts)
            if pad_size:
                out = unpad_dataproto(out, pad_size=pad_size * n)

            texts = self._decode_responses(out.batch["responses"], out.batch["response_mask"])
            correct_flags = np.zeros((len(texts),), dtype=np.int32)
            for i, pred in enumerate(texts):
                gold = golds[i // n]
                correct_flags[i] = 1 if self.verifier.is_correct(pred, gold) else 0

            total_correct += int(correct_flags.sum())
            total_outputs += int(len(correct_flags))
            # Per-question "any-correct" among n generations (useful if n>1).
            if n == 1:
                total_any_correct += int(correct_flags.sum())
            else:
                for qi in range(len(golds)):
                    group = correct_flags[qi * n : (qi + 1) * n]
                    total_any_correct += 1 if int(group.sum()) > 0 else 0

        acc_mean = float(total_correct / max(total_outputs, 1))
        acc_any = float(total_any_correct / max(n_questions, 1))
        # Keep metric names short and consistent with existing logger conventions.
        return {
            "val/enabled": 1.0,
            "val/acc_mean": acc_mean,
            "val/acc_any": acc_any,
            "val/questions": float(n_questions),
            "val/outputs": float(total_outputs),
            "val/n": float(n),
            "val/freq": float(trainer_cfg.val_freq),
        }

    def _make_challenger_batch(self) -> Tuple[DataProto, Dict[str, Any]]:
        loop = self.config_spice.loop
        prompts_cfg = self.config_spice.prompts

        last_chal_sel: Optional[DataProto] = None
        last_stats: Dict[str, Any] = {}

        for round_idx in range(loop.max_challenger_resample_rounds):
            doc_texts, sources = self.corpus.sample_batch(loop.batch_size_b)
            doc_texts = [t[: prompts_cfg.max_doc_chars] if prompts_cfg.max_doc_chars else t for t in doc_texts]
            doc_texts_arr = np.array(doc_texts, dtype=object)

            # Per-doc answer type:
            # - general -> categorical (multiple choice)
            # - math -> integer w.p. 0.6, float w.p. 0.4
            doc_answer_types: List[str] = []
            for src in sources:
                s = str(src).strip().lower()
                if s == "general":
                    doc_answer_types.append("categorical")
                elif s == "math":
                    doc_answer_types.append("integer" if self.rng.random() < 0.6 else "float")
                else:
                    doc_answer_types.append(str(prompts_cfg.challenger_answer_type or "integer").strip().lower())

            group_ids = [f"c_{uuid.uuid4().hex}" for _ in range(loop.batch_size_b)]
            messages_list = [
                build_challenger_messages(d, answer_type=at) for d, at in zip(doc_texts, doc_answer_types)
            ]
            chal_prompts, _ = _build_prompt_dataproto(
                self.tokenizer,
                messages_list,
                max_prompt_length=self.config_ppo.data.max_prompt_length,
                truncation=prompts_cfg.challenger_truncation,
                left_pad=True,
            )
            chal_prompts.meta_info.update(
                {
                    "n": loop.challenger_attempts_n,
                    "temperature": self.config_ppo.worker.rollout.temperature,
                    "top_p": self.config_ppo.worker.rollout.top_p,
                }
            )

            chal_out = self._generate_sequences_padded(chal_prompts, n=int(loop.challenger_attempts_n))

            # Attach per-sample metadata aligned with sampling_n
            n = loop.challenger_attempts_n
            chal_out.non_tensor_batch["spice_doc_text"] = np.repeat(doc_texts_arr, repeats=n, axis=0)
            chal_out.non_tensor_batch["spice_group_uid"] = np.repeat(np.array(group_ids, dtype=object), repeats=n, axis=0)
            chal_out.non_tensor_batch["spice_doc_source"] = np.repeat(np.array(sources, dtype=object), repeats=n, axis=0)
            chal_out.non_tensor_batch["spice_answer_type"] = np.repeat(
                np.array(doc_answer_types, dtype=object), repeats=n, axis=0
            )
            chal_out.check_consistency()

            chal_texts = self._decode_responses(chal_out.batch["responses"], chal_out.batch["response_mask"])
            expected_types = chal_out.non_tensor_batch["spice_answer_type"].tolist()
            parsed: List[ParsedQA] = [
                parse_challenger_output(
                    t,
                    expected_answer_type=expected_types[i],
                    max_question_chars=prompts_cfg.max_question_chars,
                )
                for i, t in enumerate(chal_texts)
            ]
            questions = np.array([p.question for p in parsed], dtype=object)
            answers = np.array([p.answer for p in parsed], dtype=object)
            valid = np.array([bool(p.valid) for p in parsed], dtype=bool)
            reasons = np.array([p.reason for p in parsed], dtype=object)
            chal_out.non_tensor_batch["spice_question"] = questions
            chal_out.non_tensor_batch["spice_answer"] = answers
            chal_out.non_tensor_batch["spice_valid"] = valid
            chal_out.non_tensor_batch["spice_invalid_reason"] = reasons

            # Subsample G per original doc-group (stratified by valid/invalid as best-effort)
            selected_indices: List[int] = []
            g = loop.group_size_g
            for bi in range(loop.batch_size_b):
                base = bi * n
                idxs = list(range(base, base + n))
                valid_idxs = [i for i in idxs if bool(valid[i])]
                invalid_idxs = [i for i in idxs if not bool(valid[i])]
                k_valid = int(round(g * (len(valid_idxs) / max(n, 1))))
                k_valid = max(0, min(k_valid, len(valid_idxs)))
                k_invalid = g - k_valid
                k_invalid = max(0, min(k_invalid, len(invalid_idxs)))
                while k_valid + k_invalid < g and (k_valid < len(valid_idxs) or k_invalid < len(invalid_idxs)):
                    if k_valid < len(valid_idxs):
                        k_valid += 1
                    elif k_invalid < len(invalid_idxs):
                        k_invalid += 1
                take = []
                if k_valid:
                    take.extend(self.rng.sample(valid_idxs, k=k_valid))
                if k_invalid:
                    take.extend(self.rng.sample(invalid_idxs, k=k_invalid))
                if len(take) < g:
                    remaining = [i for i in idxs if i not in take]
                    if remaining:
                        take.extend(self.rng.sample(remaining, k=min(g - len(take), len(remaining))))
                selected_indices.extend(take[:g])

            chal_sel = _dataproto_gather(chal_out, selected_indices)
            valid_count = int(np.sum(chal_sel.non_tensor_batch["spice_valid"].astype(np.int32)))
            last_chal_sel = chal_sel
            last_stats = {
                "challenger/valid_frac": float(np.mean(chal_sel.non_tensor_batch["spice_valid"].astype(np.float32))),
                "challenger/valid_count": valid_count,
                "challenger/selected": int(len(chal_sel)),
                "challenger/resample_round": int(round_idx + 1),
            }
            if valid_count >= int(loop.min_valid_tasks_per_step):
                break

        assert last_chal_sel is not None
        return last_chal_sel, last_stats

    def _make_reasoner_rollouts_for_challenger(
        self, chal_sel: DataProto
    ) -> Tuple[Optional[DataProto], np.ndarray, Dict[str, Any]]:
        loop = self.config_spice.loop

        valid_mask = chal_sel.non_tensor_batch["spice_valid"].astype(bool)
        if not np.any(valid_mask):
            return None, np.zeros((len(chal_sel),), dtype=np.float32), {"reasoner/has_valid": 0.0}

        valid_indices = np.where(valid_mask)[0].tolist()
        questions = [str(chal_sel.non_tensor_batch["spice_question"][i]) for i in valid_indices]
        gold_answers = [str(chal_sel.non_tensor_batch["spice_answer"][i]) for i in valid_indices]

        q_group_ids = [f"r_{uuid.uuid4().hex}" for _ in questions]
        messages_list = [build_reasoner_messages(q) for q in questions]
        r_prompts, _ = _build_prompt_dataproto(
            self.tokenizer,
            messages_list,
            max_prompt_length=self.config_ppo.data.max_prompt_length,
            truncation=self.config_spice.prompts.reasoner_truncation,
            left_pad=True,
        )
        r_prompts.meta_info.update(
            {
                "n": loop.group_size_g,
                "temperature": self.config_ppo.worker.rollout.temperature,
                "top_p": self.config_ppo.worker.rollout.top_p,
            }
        )
        r_out = self._generate_sequences_padded(r_prompts, n=int(loop.group_size_g))

        g = loop.group_size_g
        r_out.non_tensor_batch["spice_group_uid"] = np.repeat(np.array(q_group_ids, dtype=object), repeats=g, axis=0)
        r_out.non_tensor_batch["spice_gold_answer"] = np.repeat(np.array(gold_answers, dtype=object), repeats=g, axis=0)
        r_out.check_consistency()

        r_texts = self._decode_responses(r_out.batch["responses"], r_out.batch["response_mask"])
        correct = np.zeros((len(r_texts),), dtype=np.int32)
        for i, (pred, gold) in enumerate(zip(r_texts, r_out.non_tensor_batch["spice_gold_answer"].tolist())):
            correct[i] = 1 if self.verifier.is_correct(pred, str(gold)) else 0

        # per-question variance of correctness among G samples
        qid_to_var: Dict[str, float] = {}
        qid_to_acc: Dict[str, float] = {}
        for qi, qid in enumerate(q_group_ids):
            group = correct[qi * g : (qi + 1) * g].astype(np.float32)
            p = float(np.mean(group))
            var = p * (1.0 - p)
            qid_to_var[qid] = var
            qid_to_acc[qid] = p

        # map variance back to challenger-selected samples
        var_per_chal = np.zeros((len(chal_sel),), dtype=np.float32)
        # attach question group id per valid challenger sample, else empty
        chal_q_group = np.array(["" for _ in range(len(chal_sel))], dtype=object)
        for local_idx, qid in zip(valid_indices, q_group_ids):
            chal_q_group[int(local_idx)] = qid
            var_per_chal[int(local_idx)] = float(qid_to_var[qid])
        chal_sel.non_tensor_batch["spice_reasoner_group_uid"] = chal_q_group

        # Train reasoner on one randomly chosen *valid* question per original doc-group.
        # This yields ~B questions, each with G rollouts, so ~B*G training samples.
        doc_uids = chal_sel.non_tensor_batch["spice_group_uid"].astype(object)
        qids_per_doc: Dict[str, List[str]] = {}
        for idx in valid_indices:
            doc_uid = str(doc_uids[idx])
            qid = str(chal_q_group[idx])
            if not qid:
                continue
            qids_per_doc.setdefault(doc_uid, []).append(qid)

        # Keep doc order stable across steps (deterministic given chal_sel ordering + rng seed).
        doc_order: List[str] = []
        seen_docs = set()
        for i in range(len(chal_sel)):
            duid = str(doc_uids[i])
            if duid not in seen_docs:
                seen_docs.add(duid)
                doc_order.append(duid)

        selected_qids: List[str] = []
        selected_doc_uids: List[str] = []
        for duid in doc_order:
            cands = qids_per_doc.get(duid, [])
            if not cands:
                continue
            selected_doc_uids.append(duid)
            selected_qids.append(self.rng.choice(cands))

        if not selected_qids:
            return None, var_per_chal, {"reasoner/has_valid": 0.0, "reasoner/valid_tasks": int(len(valid_indices))}

        # To satisfy Ray dispatch + actor minibatch splitting, ensure selected docs count is divisible by world size.
        dropped_docs = 0
        world_size = int(self.actor_rollout_wg.world_size)
        if world_size > 0 and (len(selected_qids) % world_size) != 0:
            keep = len(selected_qids) - (len(selected_qids) % world_size)
            if keep <= 0:
                return None, var_per_chal, {
                    "reasoner/has_valid": 0.0,
                    "reasoner/valid_tasks": int(len(valid_indices)),
                    "reasoner/selected_docs": 0.0,
                }
            drop_n = len(selected_qids) - keep
            drop_idx = set(self.rng.sample(range(len(selected_qids)), k=drop_n))
            selected_qids = [q for j, q in enumerate(selected_qids) if j not in drop_idx]
            selected_doc_uids = [d for j, d in enumerate(selected_doc_uids) if j not in drop_idx]
            dropped_docs = int(drop_n)

        qid_to_doc = {qid: duid for qid, duid in zip(selected_qids, selected_doc_uids)}
        selected_indices: List[int] = []
        for qid in selected_qids:
            q_mask = (r_out.non_tensor_batch["spice_group_uid"] == qid)
            q_indices = np.where(q_mask)[0].tolist()
            selected_indices.extend(q_indices)

        r_train = _dataproto_gather(r_out, selected_indices)
        # Extra bookkeeping for debugging/analysis.
        try:
            r_train.non_tensor_batch["spice_doc_group_uid"] = np.array(
                [qid_to_doc.get(str(qid), "") for qid in r_train.non_tensor_batch["spice_group_uid"].tolist()],
                dtype=object,
            )
        except Exception:
            pass

        group_accs = [float(qid_to_acc.get(qid, 0.0)) for qid in selected_qids]
        stats = {
            "reasoner/has_valid": 1.0,
            "reasoner/valid_tasks": int(len(valid_indices)),
            "reasoner/selected_docs": int(len(selected_qids)),
            "reasoner/dropped_docs": dropped_docs,
            # Keep the old metric name, now meaning "mean acc of selected groups".
            "reasoner/chosen_group_acc": float(np.mean(group_accs)) if group_accs else 0.0,
        }
        return r_train, var_per_chal, stats

    def fit(self) -> None:
        loop = self.config_spice.loop

        # Resume if requested
        self._load_checkpoint()

        # Optional pre-train validation (mirrors `verl.trainer.ray_trainer.RayPPOTrainer` behavior).
        if self.config_ppo.trainer.val_before_train:
            with timer("val", {}):
                val_metrics = self._validate_reasoner()
            self.logger.log(data=val_metrics, step=self.global_step)
            if self.config_ppo.trainer.val_only:
                return

        for _ in tqdm(range(loop.total_steps), desc="train"):
            self.global_step += 1
            metrics: Dict[str, Any] = {"spice/step": self.global_step}
            timing_raw: Dict[str, float] = {}

            with timer("step", timing_raw):
                with timer("gen_c", timing_raw):
                    chal_sel, chal_stats = self._make_challenger_batch()
                metrics.update(chal_stats)

                with timer("gen_r_for_c", timing_raw):
                    r_train, var_per_chal, r_stats = self._make_reasoner_rollouts_for_challenger(chal_sel)
                metrics.update(r_stats)

                # Challenger rewards
                valid_mask = chal_sel.non_tensor_batch["spice_valid"].astype(bool)
                c_rewards = np.full((len(chal_sel),), fill_value=float(loop.invalid_penalty_rho), dtype=np.float32)
                if np.any(valid_mask):
                    gauss = _gaussian(
                        var_per_chal[valid_mask].astype(np.float32),
                        mu=float(self.config_spice.reward.challenger_target_variance),
                        sigma=float(self.config_spice.reward.challenger_sigma),
                    )
                    c_rewards[valid_mask] = gauss
                    relation_payload: List[Dict[str, str]] = []
                    relation_indices: List[int] = []
                    doc_texts = chal_sel.non_tensor_batch.get("spice_doc_text")
                    questions = chal_sel.non_tensor_batch.get("spice_question")
                    if doc_texts is not None and questions is not None:
                        for idx in np.where(valid_mask)[0].tolist():
                            relation_payload.append(
                                {
                                    "text": str(doc_texts[idx]),
                                    "question": str(questions[idx]),
                                }
                            )
                            relation_indices.append(int(idx))
                    if relation_payload:
                        relation_results = generate_relation_results(relation_payload)
                        related_flags: List[bool] = []
                        for sample_idx, rel in zip(relation_indices, relation_results):
                            is_related = bool(rel.get("related", False))
                            related_flags.append(is_related)
                            if not is_related:
                                c_rewards[sample_idx] = 0.0
                        if related_flags:
                            metrics["challenger/related_frac"] = float(np.mean(related_flags))
                            metrics["challenger/unrelated_count"] = int(len(related_flags) - sum(related_flags))

                metrics["challenger/reward_mean"] = float(np.mean(c_rewards))
                metrics["challenger/reward_valid_mean"] = float(np.mean(c_rewards[valid_mask])) if np.any(valid_mask) else 0.0
                self._maybe_log_challenger_samples(chal_sel, c_rewards)

                # Build challenger training batch with outcome rewards
                c_reward_t = torch.as_tensor(c_rewards, dtype=torch.float32)
                c_token_rewards = _outcome_rewards_to_token_rewards(chal_sel.batch["response_mask"], c_reward_t)
                chal_sel.batch["token_level_scores"] = c_token_rewards

                # Build reasoner training batch (G samples) with binary reward
                if r_train is not None:
                    r_texts = self._decode_responses(r_train.batch["responses"], r_train.batch["response_mask"])
                    correct = np.zeros((len(r_texts),), dtype=np.float32)
                    for i, (pred, gold) in enumerate(zip(r_texts, r_train.non_tensor_batch["spice_gold_answer"].tolist())):
                        correct[i] = 1.0 if self.verifier.is_correct(pred, str(gold)) else 0.0
                    metrics["reasoner/acc_mean"] = float(np.mean(correct))
                    self._maybe_log_reasoner_samples(r_train, correct)
                    r_reward_t = torch.as_tensor(correct, dtype=torch.float32)
                    r_token_rewards = _outcome_rewards_to_token_rewards(r_train.batch["response_mask"], r_reward_t)
                    r_train.batch["token_level_scores"] = r_token_rewards
                else:
                    metrics["reasoner/acc_mean"] = 0.0

                # Combine challenger + reasoner into a single on-policy update (DrGRPO).
                train_batches = [chal_sel]
                if r_train is not None:
                    train_batches.append(r_train)

                keep_keys = ["spice_group_uid"]
                if all("multi_modal_inputs" in b.non_tensor_batch for b in train_batches):
                    keep_keys.append("multi_modal_inputs")

                def _strip_non_tensor(dp: DataProto) -> DataProto:
                    keep = {k: dp.non_tensor_batch[k] for k in keep_keys if k in dp.non_tensor_batch}
                    return DataProto(batch=dp.batch, non_tensor_batch=keep, meta_info=dp.meta_info)

                train_batches = [_strip_non_tensor(b) for b in train_batches]
                train = DataProto.concat(train_batches) if len(train_batches) > 1 else train_batches[0]

                train.meta_info["global_token_num"] = torch.sum(train.batch["attention_mask"], dim=-1).tolist()
                with timer("old", timing_raw):
                    old_lp = self.actor_rollout_wg.compute_log_probs(train)
                train = train.union(old_lp)
                if self.use_reference_policy:
                    with timer("ref", timing_raw):
                        ref_lp = self.ref_policy_wg.compute_ref_log_probs(train)
                    train = train.union(ref_lp)
                train = self._maybe_apply_kl_penalty(train, metrics, prefix="challenger")

                uid = train.non_tensor_batch["spice_group_uid"]
                adv, ret = core_algos.compute_grpo_outcome_advantage(
                    token_level_rewards=train.batch["token_level_rewards"],
                    response_mask=train.batch["response_mask"],
                    index=uid,
                    norm_adv_by_std_in_grpo=False,
                )
                train.batch["advantages"] = adv
                train.batch["returns"] = ret
                with timer("update_c", timing_raw):
                    actor_out = self.actor_rollout_wg.update_actor(train)
                actor_metrics = reduce_metrics(actor_out.non_tensor_batch)
                metrics.update({f"challenger_update/{k}": v for k, v in actor_metrics.items()})

                # Checkpointing
                if self.config_ppo.trainer.save_freq > 0 and self.global_step % self.config_ppo.trainer.save_freq == 0:
                    with timer("save", timing_raw):
                        self._save_checkpoint()

                # Validation
                if self.config_ppo.trainer.val_freq > 0 and self.global_step % self.config_ppo.trainer.val_freq == 0:
                    with timer("val", timing_raw):
                        metrics.update(self._validate_reasoner())

            metrics.update({f"timing_s/{k}": v for k, v in timing_raw.items()})
            self.logger.log(data=metrics, step=self.global_step)

        save_freq = int(self.config_ppo.trainer.save_freq or 0)
        if save_freq <= 0 or self.global_step % save_freq != 0:
            self._save_checkpoint()
