from __future__ import annotations

import json
import os

import ray
from omegaconf import OmegaConf

from ..single_controller.ray import RayWorkerGroup
from ..utils.tokenizer import get_tokenizer
from ..workers.fsdp_workers import FSDPWorker
from ..trainer.ray_trainer import ResourcePoolManager, Role

from .config import SpiceRunConfig
from .spice_trainer import SpiceTrainer


@ray.remote(num_cpus=1)
class Runner:
    def run(self, config: SpiceRunConfig):
        print(json.dumps(OmegaConf.to_container(OmegaConf.structured(config), resolve=True), indent=2))

        ppo = config.ppo
        tokenizer = get_tokenizer(
            (config.tokenizer_path or ppo.worker.actor.model.model_path),
            override_chat_template=ppo.data.override_chat_template,
            trust_remote_code=ppo.worker.actor.model.trust_remote_code,
            use_fast=True,
        )

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(FSDPWorker),
            Role.Critic: ray.remote(FSDPWorker),
            Role.RefPolicy: ray.remote(FSDPWorker),
        }

        global_pool_id = "global_pool"
        resource_pool_spec = {global_pool_id: [ppo.trainer.n_gpus_per_node] * ppo.trainer.nnodes}
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
            Role.RefPolicy: global_pool_id,
        }
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        trainer = SpiceTrainer(
            tokenizer=tokenizer,
            config_ppo=ppo,
            config_spice=config.spice,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=RayWorkerGroup,
        )
        trainer.init_workers()
        trainer.fit()


def main():
    cli_args = OmegaConf.from_cli()
    default_config = OmegaConf.structured(SpiceRunConfig())

    # Allow `verl.trainer.main`-style overrides (data/worker/trainer/algorithm) as aliases for ppo.*.
    # This keeps SPICE CLI ergonomics consistent with existing training scripts.
    def _merge_alias_into_ppo(field_name: str) -> None:
        if field_name not in cli_args:
            return
        alias_cfg = cli_args[field_name]
        if alias_cfg is None:
            del cli_args[field_name]
            return
        if "ppo" not in cli_args or cli_args["ppo"] is None:
            cli_args["ppo"] = {}
        ppo_cfg = cli_args["ppo"].get(field_name, None)
        cli_args["ppo"][field_name] = OmegaConf.merge(ppo_cfg or {}, alias_cfg)
        del cli_args[field_name]

    for _f in ("data", "worker", "trainer", "algorithm"):
        _merge_alias_into_ppo(_f)

    # tokens.json is optional for SPICE (wandb/hf), but keep parity with existing entrypoint
    tokens_path = os.getenv("TOKENS_FILE", "tokens.json")
    if os.path.exists(tokens_path):
        with open(tokens_path, "r") as f:
            tokens = json.load(f)
        if isinstance(tokens, dict):
            if "huggingface" in tokens and not os.getenv("HF_TOKEN"):
                os.environ["HF_TOKEN"] = tokens["huggingface"]
            if "wandb" in tokens and not os.getenv("WANDB_API_KEY"):
                os.environ["WANDB_API_KEY"] = tokens["wandb"]

    if hasattr(cli_args, "config"):
        config_path = cli_args.pop("config", None)
        file_config = OmegaConf.load(config_path)
        default_config = OmegaConf.merge(default_config, file_config)

    merged = OmegaConf.merge(default_config, cli_args)
    cfg: SpiceRunConfig = OmegaConf.to_object(merged)
    cfg.deep_post_init()

    if not ray.is_initialized():
        tokenizer_parallel = os.environ.get("TOKENIZERS_PARALLELISM", "true")
        runtime_env = {
            "env_vars": {
                "TOKENIZERS_PARALLELISM": tokenizer_parallel,
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "WARN",
                "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False",
                "PYTHONUNBUFFERED": "1",
            }
        }
        ray.init(runtime_env=runtime_env, num_cpus=16)

    runner = Runner.remote()
    ray.get(runner.run.remote(cfg))


if __name__ == "__main__":
    main()
