# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
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

import asyncio
import atexit
import copy
import importlib.resources as pkg_resources
import inspect
import math
import os
import sys
import textwrap
import time
import warnings
from collections import defaultdict, deque
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
import torch
import torch.utils.data
import transformers
from accelerate.logging import get_logger
from accelerate.utils import gather, gather_object, is_peft_model, set_seed
from datasets import Dataset, IterableDataset
from huggingface_hub import CommitScheduler, DatasetCard, DatasetCardData, create_repo
from packaging.version import Version
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import Sampler
from transformers import (
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    TrainerCallback,
    is_trackio_available,
    is_wandb_available,
)
from transformers.utils import is_peft_available, is_rich_available
from trl.chat_template_utils import (
    add_response_schema,
    get_training_chat_template,
    parse_response,
)
from trl.data_utils import (
    apply_chat_template,
    is_conversational,
    prepare_multimodal_messages,
)
from trl.extras.profiling import profiling_context, profiling_decorator
from trl.generation.vllm_generation import VLLMGeneration
from trl.import_utils import is_jmespath_available, is_liger_kernel_available
from trl.models import prepare_deepspeed, prepare_fsdp, unwrap_model_for_generation
from trl.models.utils import _ForwardRedirection, disable_gradient_checkpointing
from trl.trainer.base_trainer import _BaseTrainer
from trl.trainer.callbacks import SyncRefModelCallback
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import (
    RepeatSampler,
    create_model_from_path,
    disable_dropout_in_model,
    entropy_from_logits,
    get_config_model_id,
    identity,
    nanmax,
    nanmin,
    nanstd,
    pad,
    print_prompt_completions_sample,
    selective_log_softmax,
    shuffle_sequence_dict,
    shutdown_event_loop_in_daemon,
    split_pixel_values_by_grid,
    split_tensor_dict,
    start_event_loop_in_daemon,
    unsplit_pixel_values_by_grid,
    use_adapter,
)

if is_peft_available():
    from peft import PeftConfig, PeftModel, get_peft_model

if is_liger_kernel_available():
    from liger_kernel.chunked_loss import LigerFusedLinearGRPOLoss


if is_wandb_available():
    import wandb

if is_trackio_available():
    import trackio

logger = get_logger(__name__)

# A reward function can be a string, interpreted as a model ID and loaded as a pretrained model, a pretrained model, or
# a callable that returns a list of floats (the rewards). The callable receives prompts, completions, and additional
# arguments from the trainer (refer to the trainer's source for details). To ensure forward compatibility, it should
# accept **kwargs.
RewardFunc = str | PreTrainedModel | Callable[..., list[float | None]]

# What we call a rollout function is a callable that takes prompts (list) and the trainer instance as parameters and
# returns a dict of generation results. Those results must include "prompt_ids", "completion_ids", and "logprobs"
# fields. Any extra fields (per-completion) are forwarded to the reward functions.
RolloutFunc = Callable[[list[str], "GRPOTrainer"], dict[str, Any]]


class _SupportsReset(Protocol):
    def reset(self, **kwargs) -> str | None: ...


EnvironmentFactory = Callable[[], _SupportsReset]


@dataclass
class GrailConfigV2(GRPOConfig):
    use_grail: bool = field(
        default=False,
        metadata={"help": "Whether to use GRAIL."},
    )
    grail_std: float = field(
        default=None,
        metadata={
            "help": "The standard deviation parameter (spread) for GRAIL Z-scaling."
        },
    )
    grail_mean: float = field(
        default=1.0,
        metadata={
            "help": "The mean baseline weight parameter for neutral tokens in GRAIL Z-scaling."
        },
    )
    grail_w_min: float = field(
        default=0.1,
        metadata={"help": "The minimum weight for GRAIL."},
    )
    grail_w_max: float = field(
        default=5.0,
        metadata={"help": "The maximum weight for GRAIL."},
    )
    grail_leaf_source: str = field(
        default="embeddings", metadata={"help": "Use hidden states as leaf for GRAIL"}
    )
    grail_rollout_symmetry: str = field(
        default="both",
        metadata={
            "help": "Whether to apply GRAIL symmetrically to both the correct and wrong rollouts, or only to the correct or wrong rollouts."
        },
    )
    use_oar_g: bool = field(
        default=False,
        metadata={"help": "Whether to use OAR-G."},
    )
    oar_g_tau: float = field(
        default=0.4,
        metadata={"help": "Bi-level threshold tau for OAR-G."},
    )
    oar_g_beta: float = field(
        default=2.0,
        metadata={"help": "Boosting coefficient beta for OAR-G."},
    )
    oar_g_eps: float = field(
        default=1e-12,
        metadata={"help": "Smoothing constant epsilon for OAR-G."},
    )
    oar_g_noise_std: float = field(
        default=1e-3,
        metadata={
            "help": "Standard deviation of noise injected into embeddings for OAR-G."
        },
    )


class GrailTrainerV2(_BaseTrainer):
    """
    GRAIL Trainer for GRPO training with GRAIL and OAR-G support.
    """

    _tag_names = ["trl", "grpo", "grail"]
    _name = "grail"
    _paper = {}

    def __init__(
        self,
        model: "str | PreTrainedModel | PeftModel",
        reward_funcs: RewardFunc | list[RewardFunc],
        args: GRPOConfig | None = None,
        train_dataset: Dataset | IterableDataset | None = None,
        eval_dataset: (
            Dataset | IterableDataset | dict[str, Dataset | IterableDataset] | None
        ) = None,
        processing_class: PreTrainedTokenizerBase | ProcessorMixin | None = None,
        reward_processing_classes: (
            PreTrainedTokenizerBase | list[PreTrainedTokenizerBase] | None
        ) = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[
            torch.optim.Optimizer | None, torch.optim.lr_scheduler.LambdaLR | None
        ] = (None, None),
        peft_config: "PeftConfig | None" = None,
        tools: list[Callable] | None = None,
        rollout_func: RolloutFunc | None = None,
        environment_factory: EnvironmentFactory | None = None,
    ):
        # Args
        if args is None:
            model_name = (
                model if isinstance(model, str) else get_config_model_id(model.config)
            )
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        # Model
        if isinstance(model, str):
            model_init_kwargs = args.model_init_kwargs or {}
            # Distributed training requires device_map=None ("auto" fails)
            if args.distributed_state.distributed_type in ["MULTI_GPU", "DEEPSPEED"]:
                model_init_kwargs["device_map"] = None
            model = create_model_from_path(model, **model_init_kwargs)
        else:
            if args.model_init_kwargs is not None:
                logger.warning(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "The `model_init_kwargs` will be ignored."
                )

        # Some models (SmolVLM/Idefics3) don't support `logits_to_keep` argument and error out if we pass it
        # Inspect the forward method before we wrap the model with PEFT
        self.model_kwarg_keys = (
            inspect.signature(model.forward).parameters.keys()
            if not hasattr(model, "get_base_model")
            else inspect.signature(model.get_base_model().forward).parameters.keys()
        )

        # Processing class
        if processing_class is None:
            processing_class = AutoProcessor.from_pretrained(
                get_config_model_id(model.config),
                truncation_side="left",
                padding_side="left",
            )

        # Handle pad token for processors or tokenizers
        if isinstance(processing_class, ProcessorMixin):
            tokenizer = processing_class.tokenizer
        elif isinstance(processing_class, PreTrainedTokenizerBase):
            tokenizer = processing_class
        else:
            raise TypeError(
                "The `processing_class` must be either a `PreTrainedTokenizerBase` or a `ProcessorMixin`"
            )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        self.pad_token = tokenizer.pad_token
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id

        if is_peft_available() and is_peft_model(model) and peft_config is not None:
            raise ValueError(
                "You passed a `PeftModel` instance together with a `peft_config` to the trainer. Please first merge "
                "and unload the existing adapter, save the resulting base model, and then pass that base model along "
                "with the new `peft_config` to the trainer."
            )

        if is_peft_available() and is_peft_model(model) and args.beta != 0.0:
            # If the model is a PEFT model with a pretrained adapter, we need to create a "ref" adapter that is a copy
            # of the "default" adapter, so that we can use it as the reference model during GRPO training.
            model.add_adapter("ref", model.peft_config["default"])
            for name, param in model.named_parameters():
                if ".default." in name:
                    ref_name = name.replace(".default.", ".ref.")
                    ref_param = model.get_parameter(ref_name)
                    ref_param.data.copy_(param.data)

        # Create PEFT model
        if peft_config is not None:
            model = get_peft_model(model, peft_config)

        # When using gradient checkpointing with PEFT, we need to enable input gradients. transformers.Trainer normally
        # handles this, but a bug currently prevents it; see https://github.com/huggingface/transformers/issues/42489
        if is_peft_available() and is_peft_model(model) and args.gradient_checkpointing:
            model.enable_input_require_grads()

        # When using QLoRA, the PEFT adapter weights are converted to bf16 to follow the recommendations from the
        # original paper (see https://huggingface.co/papers/2305.14314, paragraph 3). Normally, this can be done by
        # passing `autocast_adapter_dtype=False` to `get_peft_model`, but this option is not yet supported for
        # quantized models. See: https://github.com/huggingface/peft/issues/2889
        # Non-quantized models do not have the `is_loaded_in_{8,4}bit` attributes, whereas quantized models do
        if getattr(model, "is_loaded_in_4bit", False) or getattr(
            model, "is_loaded_in_8bit", False
        ):
            for param in model.parameters():
                if param.requires_grad:
                    param.data = param.data.to(torch.bfloat16)

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        self.reward_func_names = []
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                model_init_kwargs = args.model_init_kwargs or {}
                # Distributed training requires device_map=None ("auto" fails)
                if args.distributed_state.distributed_type in [
                    "MULTI_GPU",
                    "DEEPSPEED",
                ]:
                    model_init_kwargs["device_map"] = None
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
            if isinstance(
                reward_funcs[i], nn.Module
            ):  # Use Module over PretrainedModel for compat w/ compiled models
                self.reward_func_names.append(
                    get_config_model_id(reward_funcs[i].config).split("/")[-1]
                )
            else:
                self.reward_func_names.append(reward_funcs[i].__name__)
        self.reward_funcs = reward_funcs

        # Reward weights
        if args.reward_weights is not None:
            if len(args.reward_weights) != len(reward_funcs):
                raise ValueError(
                    f"Number of reward weights ({len(args.reward_weights)}) must match number of reward "
                    f"functions ({len(reward_funcs)})"
                )
            self.reward_weights = torch.tensor(args.reward_weights, dtype=torch.float32)
        else:
            self.reward_weights = torch.ones(len(reward_funcs), dtype=torch.float32)

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        if len(reward_processing_classes) != len(reward_funcs):
            raise ValueError(
                f"The number of reward processing classes ({len(reward_processing_classes)}) must match the number of "
                f"reward functions ({len(reward_funcs)})."
            )

        for i, (reward_processing_class, reward_func) in enumerate(
            zip(reward_processing_classes, reward_funcs, strict=True)
        ):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(
                        get_config_model_id(reward_func.config)
                    )
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = (
                        reward_processing_class.eos_token
                    )
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class

        self.reward_processing_classes = reward_processing_classes

        # Rollout function
        if (
            rollout_func is not None
            and os.environ.get("TRL_EXPERIMENTAL_SILENCE", "0") != "1"
        ):
            warnings.warn(
                "You are using 'rollout_func', which is an experimental feature. This API may change or be removed at "
                "any time without prior notice. Silence this warning by setting environment variable "
                "TRL_EXPERIMENTAL_SILENCE=1.",
                UserWarning,
                stacklevel=2,
            )
        self.rollout_func = rollout_func
        if (
            environment_factory is not None
            and os.environ.get("TRL_EXPERIMENTAL_SILENCE", "0") != "1"
        ):
            warnings.warn(
                "You are using 'environment_factory', which is an experimental feature. This API may change or be "
                "removed at any time without prior notice. Silence this warning by setting environment variable "
                "TRL_EXPERIMENTAL_SILENCE=1.",
                UserWarning,
                stacklevel=2,
            )

        # Tools
        if tools:
            if not Version(transformers.__version__) >= Version("5.0.0"):
                raise ImportError(
                    "Using tools with GRPOTrainer requires transformers version 5.0.0 or higher. Please upgrade "
                    "transformers with `pip install --upgrade transformers` to use this feature."
                )
        if environment_factory:
            if not Version(transformers.__version__) >= Version("5.2.0"):
                raise ImportError(
                    "Using `environment_factory` with GRPOTrainer requires transformers version 5.2.0 or higher. "
                    "Please install transformers from the main branch with `pip install "
                    "git+https://github.com/huggingface/transformers.git@main` to use this feature."
                )
        if tools or environment_factory:
            if not is_jmespath_available():
                raise ImportError(
                    "Using tools with GRPOTrainer requires the jmespath library for response parsing. Please install "
                    "it with `pip install jmespath` to use this feature."
                )

        # Create the environments and extract their methods to be used as tools. We create one environment per rollout
        generation_batch_size = (
            args.per_device_train_batch_size * args.steps_per_generation
        )
        if environment_factory is not None:
            self.environments = [
                environment_factory() for _ in range(generation_batch_size)
            ]
            environment_methods = [[] for _ in range(generation_batch_size)]
            for i, environment in enumerate(self.environments):
                has_reset = False
                for name, member in inspect.getmembers(
                    environment, predicate=inspect.ismethod
                ):
                    if name == "reset":
                        has_reset = True
                    elif not name.startswith("_"):
                        environment_methods[i].append(member)
                if not has_reset:
                    raise ValueError(
                        "Each environment instance returned by `environment_factory` must define a callable `reset` "
                    )
        else:
            self.environments = None

        tools = tools or []
        self._sync_tool_dicts = [{} for _ in range(generation_batch_size)]
        self._async_tool_dicts = [{} for _ in range(generation_batch_size)]
        for i in range(generation_batch_size):
            for tool in tools + (
                environment_methods[i] if self.environments is not None else []
            ):
                if inspect.iscoroutinefunction(tool):
                    self._async_tool_dicts[i][tool.__name__] = tool
                else:
                    self._sync_tool_dicts[i][tool.__name__] = tool

        self.tools = tools + (
            environment_methods[0] if self.environments is not None else []
        )

        # Check for async functions to start an event loop on a daemon thread
        self._has_async_funcs = any(
            inspect.iscoroutinefunction(func) for func in self.reward_funcs + self.tools
        )

        if self._has_async_funcs:
            self.async_loop_thread, self.async_loop, self.async_loop_ready_event = (
                start_event_loop_in_daemon(name="GRPOTrainer-AsyncLoop")
            )
            # wait until the event loop is running in the daemon thread
            self.async_loop_ready_event.wait()
            atexit.register(
                shutdown_event_loop_in_daemon, self.async_loop_thread, self.async_loop
            )

        # At the time of initial implementation, most tokenizers do not have built-in support for response schemas.
        # While waiting for broader adoption, we provide this utility function to manually set the response schema for
        # known chat templates.
        # We need `getattr`` until the base class sets a default None value for response_schema
        if self.tools and not getattr(processing_class, "response_schema", None):
            processing_class = add_response_schema(processing_class)
        # In multi-turn training, the chat template *must* be prefix-preserving. If the tokenizer's original template
        # isn't, we replace it at initialization with a training-safe, prefix-preserving template.
        if self.tools:
            self.chat_template = get_training_chat_template(processing_class)
        else:
            self.chat_template = None

        # Training arguments
        self.max_completion_length = (
            args.max_completion_length
        )  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.max_tool_calling_iterations = (
            args.max_tool_calling_iterations or sys.maxsize
        )
        self.num_generations_eval = args.num_generations_eval or self.num_generations
        self.chat_template_kwargs = args.chat_template_kwargs or {}
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.top_k = args.top_k
        self.min_p = args.min_p
        self.repetition_penalty = args.repetition_penalty
        self.use_transformers_paged = args.use_transformers_paged
        self.pad_to_multiple_of = args.pad_to_multiple_of
        self.use_vllm = args.use_vllm
        self.vllm_mode = args.vllm_mode
        self.vllm_gpu_memory_utilization = (
            args.vllm_gpu_memory_utilization
        )  # only applies to colocation mode
        self.vllm_tensor_parallel_size = (
            args.vllm_tensor_parallel_size
        )  # only applies to colocation mode
        self.vllm_importance_sampling_correction = (
            args.vllm_importance_sampling_correction
        )
        self.vllm_importance_sampling_mode = args.vllm_importance_sampling_mode
        self.vllm_importance_sampling_cap = args.vllm_importance_sampling_cap
        self.use_liger_kernel = args.use_liger_kernel
        self.loss_type = args.loss_type
        self.multi_objective_aggregation = args.multi_objective_aggregation
        self.scale_rewards = args.scale_rewards
        self.importance_sampling_level = args.importance_sampling_level

        # GRAIL-specific parameters
        self.use_grail = args.use_grail
        self.grail_std = args.grail_std
        self.grail_leaf_source = args.grail_leaf_source
        self.grail_rollout_symmetry = args.grail_rollout_symmetry

        if self.grail_rollout_symmetry not in ["both", "correct", "wrong"]:
            raise ValueError(
                f"Invalid value for grail_rollout_symmetry: {self.grail_rollout_symmetry}. "
                "Possible values are 'both', 'correct', and 'wrong'."
            )
        self.grail_mean = args.grail_mean

        self.grail_w_min = args.grail_w_min
        self.grail_w_max = args.grail_w_max

        # OAR-G specific parameters
        self.use_oar_g = args.use_oar_g
        self.oar_g_tau = args.oar_g_tau
        self.oar_g_beta = args.oar_g_beta
        self.oar_g_eps = args.oar_g_eps
        self.oar_g_noise_std = args.oar_g_noise_std

        self.off_policy_mask_threshold = args.off_policy_mask_threshold
        if self.use_liger_kernel and self.off_policy_mask_threshold is not None:
            raise ValueError(
                "Liger kernel does not support off-policy sequence masking yet."
            )
        self.mask_truncated_completions = args.mask_truncated_completions
        self.top_entropy_quantile = args.top_entropy_quantile
        if self.use_liger_kernel and self.top_entropy_quantile < 1.0:
            raise NotImplementedError(
                "Liger Kernels don't currently support masking token positions based on entropy."
            )
        if self.use_liger_kernel and self.importance_sampling_level not in (
            "token",
            "sequence",
        ):
            raise ValueError(
                f"Unknown importance sampling level: {self.importance_sampling_level}. "
                "Possible values are 'token' and 'sequence'."
            )

        # Datasets
        self.shuffle_dataset = args.shuffle_dataset

        if train_dataset is None:
            raise ValueError("`train_dataset` is required")
        elif (
            isinstance(train_dataset, IterableDataset)
            or isinstance(eval_dataset, IterableDataset)
            or (
                isinstance(eval_dataset, dict)
                and any(isinstance(ds, IterableDataset) for ds in eval_dataset.values())
            )
        ):
            # See https://github.com/huggingface/trl/issues/3213
            raise NotImplementedError(
                "Iterable datasets are not yet supported in GRPOTrainer. Please use a standard dataset instead."
            )

        if args.loss_type == "luspo" and args.importance_sampling_level != "sequence":
            logger.warning(
                "When using `'luspo'` loss, `importance_sampling_level` should be set to `'sequence'` to mirror the "
                "paper's setup."
            )

        if args.loss_type == "vespo" and args.importance_sampling_level != "token":
            logger.warning(
                "VESPO computes sequence-level importance weights internally. `importance_sampling_level` should be "
                "set to `'token'` (the default)."
            )

        if (
            self.loss_type == "vespo"
            and self.use_vllm
            and self.vllm_importance_sampling_correction
        ):
            if self.vllm_importance_sampling_mode not in [
                "token_truncate",
                "token_mask",
            ]:
                raise ValueError(
                    f"VESPO loss requires `vllm_importance_sampling_mode` to be either 'token_truncate' or "
                    f"'token_mask'. Got: {self.vllm_importance_sampling_mode}."
                )

        # Multi-step
        self.num_iterations = args.num_iterations  # = 𝜇 in the GRPO paper
        self.epsilon_low = args.epsilon
        self.epsilon_high = (
            args.epsilon_high if args.epsilon_high is not None else args.epsilon
        )
        # Tracks the number of iterations (forward + backward passes), including those within a grad accum cycle
        self._step = 0
        # Buffer the batch to reuse generated outputs across multiple updates. For more details, see
        # `_get_train_sampler` and `_prepare_inputs`.
        self._buffered_inputs = None

        # Transformers explicitly set use_reentrant=True in the past to silence a PyTorch warning, but the default was
        # never updated once PyTorch switched to recommending use_reentrant=False. Until that change lands upstream
        # (see https://github.com/huggingface/transformers/pull/43203) and is released (most likely in 5.0.0), we
        # default to the recommended non-reentrant behavior here, while preserving any user-provided value.
        if args.gradient_checkpointing and Version(transformers.__version__) < Version(
            "5.0.0"
        ):
            args.gradient_checkpointing_kwargs = (
                args.gradient_checkpointing_kwargs or {}
            )
            args.gradient_checkpointing_kwargs.setdefault("use_reentrant", False)

        super().__init__(
            model=model,
            args=args,
            data_collator=identity,  # No data collation is needed in GRPO
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
            # In Trainer, `training_step` scales the loss by `gradient_accumulation_steps` only if `compute_loss_func`
            # is None. For DAPO, loss scaling instead depends on the total number of completions tokens across the
            # global accumulated batch. To control scaling ourselves, we must disable Trainer’s built-in scaling. The
            # simplest (though a bit hacky) way is to set `compute_loss_func` to any non-None value, which bypasses
            # that behavior without rewriting `training_step`.
            compute_loss_func="non-None value to disable scaling",
        )

        # Reference model
        self.beta = args.beta
        if self.beta == 0.0:
            # If beta is 0.0, the reference model is not needed
            self.ref_model = None
        elif is_peft_model(model):
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None
        else:
            # For deepspeed, fsdp or non-distributed models, create a reference model from scratch
            model_init_kwargs = args.model_init_kwargs or {}
            # Distributed training requires device_map=None ("auto" fails)
            if self.args.distributed_state.distributed_type in [
                "MULTI_GPU",
                "DEEPSPEED",
            ]:
                model_init_kwargs["device_map"] = None
            self.ref_model = create_model_from_path(
                get_config_model_id(self.model.config), **model_init_kwargs
            )

        # Disable dropout in the models
        if args.disable_dropout:
            disable_dropout_in_model(model)
            if self.ref_model is not None:
                disable_dropout_in_model(self.ref_model)

        # Cast LM Head To FP32
        if args.cast_lm_head_to_fp32:

            def _cast_lm_head_to_fp32(target_model: PreTrainedModel):
                """Cast lm_head to fp32 while preserving embedding output dtype if tied."""

                def cast_inputs_to_fp32(module, inputs):
                    # Preserve other positional args and kwargs untouched
                    if not inputs:
                        return inputs
                    return (inputs[0].to(torch.float32),) + inputs[1:]

                original_dtype_local = target_model.lm_head.weight.dtype
                target_model.lm_head = target_model.lm_head.float()
                target_model.lm_head.register_forward_pre_hook(cast_inputs_to_fp32)

                if target_model.config.tie_word_embeddings:

                    def cast_outputs_to_original_dtype(module, args, output):
                        return output.to(original_dtype_local)

                    # Only cast activations; weights are now fp32 (intentional for numerical stability of logits)
                    target_model.model.embed_tokens.register_forward_hook(
                        cast_outputs_to_original_dtype
                    )

            _cast_lm_head_to_fp32(model)
            if self.ref_model is not None:
                _cast_lm_head_to_fp32(self.ref_model)

        # Liger loss
        if self.use_liger_kernel:
            if not is_liger_kernel_available():
                raise ImportError(
                    "Liger is required to use `use_liger_kernel` as the GRPO loss. Run `pip install liger-kernel`."
                )
            # redirect the model.module forward to the model forward to ensure pre-forward hooks are called
            self._forward_redirection = _ForwardRedirection()

            self.liger_grpo_loss = LigerFusedLinearGRPOLoss(
                beta=self.beta,
                epsilon_low=self.epsilon_low,
                epsilon_high=self.epsilon_high,
                temperature=self.temperature,
                use_ref_model=self.beta != 0.0,
                loss_type=self.loss_type,
                max_completion_length=self.max_completion_length,
                importance_sampling_level=self.importance_sampling_level,
            )

        # Initialize the metrics
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self._total_train_tokens = 0
        self._current_train_step_time = 0.0
        self.log_completions = args.log_completions
        self.log_unique_prompts = args.log_unique_prompts
        self.num_completions_to_print = args.num_completions_to_print
        # Keep logs sized to the generation batch to record only outputs from the latest model update.
        self._logs = {
            "images": deque(maxlen=args.generation_batch_size),
            "prompt": deque(maxlen=args.generation_batch_size),
            "completion": deque(maxlen=args.generation_batch_size),
            "rewards": defaultdict(lambda: deque(maxlen=args.generation_batch_size)),
            "advantages": deque(maxlen=args.generation_batch_size),
            "extra": defaultdict(lambda: deque(maxlen=args.generation_batch_size)),
        }
        # Buffers for user-logged data from reward functions, flushed after gathering
        self._pending_extra_logs = defaultdict(list)
        self._pending_metrics = defaultdict(list)

        # Ensure each process receives a unique seed to prevent duplicate completions when generating with
        # transformers if num_generations exceeds per_device_train_batch_size. We could skip it if we use vLLM, but
        # it's safer to set it in all cases.
        set_seed(args.seed, device_specific=True)

        if self.use_vllm:
            # Initialize vLLM generation backend
            self.vllm_generation = VLLMGeneration(
                model=self.model,
                accelerator=self.accelerator,
                is_fsdp_enabled=self.is_fsdp_enabled,
                processing_class=self.processing_class,
                # vLLM configuration
                mode=args.vllm_mode,
                structured_outputs_regex=args.vllm_structured_outputs_regex,
                # Server mode configuration
                server_base_url=args.vllm_server_base_url,
                server_host=args.vllm_server_host,
                server_port=args.vllm_server_port,
                group_port=args.vllm_group_port,
                server_timeout=args.vllm_server_timeout,
                # Colocate mode configuration
                tensor_parallel_size=args.vllm_tensor_parallel_size,
                gpu_memory_utilization=args.vllm_gpu_memory_utilization,
                max_model_length=args.vllm_max_model_length,
                max_num_seqs=args.per_device_train_batch_size
                * args.vllm_tensor_parallel_size
                * args.steps_per_generation,
                enable_sleep_mode=args.vllm_enable_sleep_mode,
                model_impl=args.vllm_model_impl,
                # Generation configuration
                repetition_penalty=self.repetition_penalty,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                min_p=self.min_p,
                max_completion_length=self.max_completion_length,
                logprobs=0,  # we only need the generated token logprobs for the importance sampling correction
                generation_kwargs=args.generation_kwargs,
            )
            self._last_loaded_step = (
                -1
            )  # tag to avoid useless loading during grad accumulation
        else:
            generation_kwargs = {
                "max_new_tokens": self.max_completion_length,
                "do_sample": True,
                "pad_token_id": tokenizer.pad_token_id,
                "bos_token_id": tokenizer.bos_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "min_p": self.min_p,
                "repetition_penalty": self.repetition_penalty,
                "cache_implementation": args.cache_implementation,
            }
            if args.generation_kwargs is not None:
                generation_kwargs.update(args.generation_kwargs)
            self.generation_config = GenerationConfig(**generation_kwargs)
            # Keep training-specific generation kwargs to overwrite model's original generation config
            self.generation_kwargs = generation_kwargs

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        # Add tags to the model
        self.model.add_model_tags(self._tag_names)

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            elif self.is_fsdp_enabled:
                self.ref_model = prepare_fsdp(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(
                    self.ref_model, evaluation_mode=True
                )

        if args.sync_ref_model:
            if self.beta == 0.0:
                raise ValueError(
                    "You passed `sync_ref_model=True` while `beta=0.0`, which means the reference model is not used "
                    "during training. Consequently, GRPOTrainer does not create a `ref_model` instance, and there is "
                    "nothing to synchronize. Please set `sync_ref_model=False`, or set `beta` to a non-zero value."
                )
            if is_peft_model(model):
                raise NotImplementedError(
                    "You passed `sync_ref_model=True` while using a PEFT model, which is currently not supported. "
                    "With PEFT, GRPOTrainer does not keep a separate reference model in memory; instead, it recovers "
                    "reference behavior by temporarily disabling the adapter. As a result, there is no standalone "
                    "`ref_model` instance to synchronize. Use `sync_ref_model=False`, or opt for full fine-tuning if "
                    "you need a synced reference model. If you need `sync_ref_model` to work with PEFT, please open a "
                    "feature request at https://github.com/huggingface/trl/issues."
                )
            self.add_callback(
                SyncRefModelCallback(
                    ref_model=self.ref_model, accelerator=self.accelerator
                )
            )

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                if self.is_deepspeed_enabled:
                    self.reward_funcs[i] = prepare_deepspeed(
                        reward_func, self.accelerator
                    )
                else:
                    # set device placement to True to make `prepare_model` move `reward_func` to device when using fsdp
                    self.reward_funcs[i] = self.accelerator.prepare_model(
                        reward_func, evaluation_mode=True, device_placement=True
                    )

        if self.accelerator.is_main_process and self.log_completions:
            os.makedirs(
                os.path.join(self.args.output_dir, "completions"), exist_ok=True
            )
            if self.args.log_completions_hub_repo is not None:
                repo_id = self.args.log_completions_hub_repo
                create_repo(
                    repo_id,
                    private=self.args.hub_private_repo,
                    repo_type="dataset",
                    exist_ok=True,
                )
                template_path = pkg_resources.files("trl").joinpath(
                    "templates/completions_dataset_card.md"
                )
                card_data = DatasetCardData(
                    pretty_name="TRL Completion logs",
                    tags=["trl", "trl-logs", "completions"],
                )
                card = DatasetCard.from_template(
                    card_data=card_data,
                    template_path=str(template_path),
                    repo_id=repo_id,
                    hub_model_id=self.args.hub_model_id,
                )
                card.push_to_hub(repo_id)
                self.commit_scheduler = CommitScheduler(
                    repo_id=repo_id,
                    repo_type="dataset",
                    folder_path=f"{self.args.output_dir}/completions",
                    every=2,  # minutes
                    allow_patterns=["*.parquet"],
                )

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs (usually, "input_ids"
        # and "attention_mask"). In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't
        # work. Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt", "image", "images"]

    # This method overrides `Trainer.get_train_dataloader` to support our custom batching strategy.
    # Instead of returning a standard per-step batch (i.e., `per_device_batch_size), our dataloader loads an
    # *generation* batch (i.e., `per_device_batch_size × steps_per_generation`). This allows us to generate completions
    # once every steps_per_generation step—rather than once per accumulation step—which is significantly more
    # efficient. The only change from the original implementation is multiplying the batch size by
    # `steps_per_generation`. Thus, `_prepare_inputs` is called with this *generation* batch, and it handles the
    # splitting internally.
    # Maintenance note: This method is a copy-paste of the original `Trainer.get_train_dataloader` with only one line
    # modification.
    def get_train_dataloader(self):
        return self._get_dataloader(
            dataset=self.train_dataset,
            description="Training",
            batch_size=self._train_batch_size
            * self.args.steps_per_generation,  # < this is the change
            sampler_fn=self._get_train_sampler,
            is_training=True,
        )

    def _get_train_sampler(self, dataset: Dataset | None = None) -> Sampler:
        # Returns a sampler that
        # 1. ensures each prompt is repeated across multiple processes. This guarantees that identical prompts are
        #    distributed to different GPUs, allowing rewards to be computed and normalized correctly within each prompt
        #    group. Using the same seed across processes ensures consistent prompt assignment, preventing discrepancies
        #    in group formation.
        # 2. repeats the batch multiple times to allow reusing generations across multiple updates. Refer to
        #    _prepare_inputs to see how the generations are stored and reused.

        # In the following figure, the values are the prompt indices. The first row shows the first sampled batch, the
        # second row shows the second sampled batch, and so on.
        #
        #                                      |   GPU 0  |   GPU 1  |
        #
        #                 global_step   step    <-───>  num_generations=2
        #                                       <-───────> per_device_train_batch_size=3
        #  grad_accum    ▲  ▲  0          0     0   0   1   1   2   2   <- Generate for the first `steps_per_generation` (prompts 0 to 11); store the completions; use the first slice to compute the loss
        #     =2         ▼  |  0          1     3   3   4   4   5   5   <- Take the stored generations and use the second slice to compute the loss
        #                   |
        #                   |  1          2     6   6   7   7   8   8   <- Take the stored generations and use the third slice to compute the loss
        #  steps_per_gen=4  ▼  1          3     9   9  10  10  11  11   <- Take the stored generations and use the fourth slice to compute the loss
        #
        #                      2          4    12  12  13  13  14  14   <- Generate for the second `steps_per_generation` (prompts 12 to 23); store the completions; use the first slice to compute the loss
        #                      2          5    15  15  16  16  17  17   <- Take the stored generations and use the second slice to compute the loss
        #                                          ...
        if dataset is None:
            dataset = self.train_dataset
        return RepeatSampler(
            data_source=dataset,
            mini_repeat_count=self.num_generations,
            batch_size=self.args.generation_batch_size // self.num_generations,
            repeat_count=self.num_iterations * self.args.steps_per_generation,
            shuffle=self.shuffle_dataset,
            seed=self.args.seed,
        )

    def _get_eval_sampler(self, eval_dataset) -> Sampler:
        # See _get_train_sampler for an explanation of the sampler.
        return RepeatSampler(
            data_source=eval_dataset,
            mini_repeat_count=self.num_generations_eval,
            seed=self.args.seed,
        )

    @profiling_decorator
    def _get_last_hidden_state(
        self,
        unwrapped_model,
        input_ids,
        attention_mask,
        logits_to_keep,
        pixel_values=None,
        image_grid_thw=None,
        pixel_attention_mask=None,
        image_sizes=None,
    ):
        if is_peft_model(unwrapped_model):
            unwrapped_model = unwrapped_model.base_model.model

        # Build model inputs - check if the model supports logits_to_keep (some models and VLMs don't)
        model_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}

        # For Qwen models:
        if image_grid_thw is not None and pixel_values is not None:
            model_inputs["image_grid_thw"] = image_grid_thw
        # For Gemma, SmolVLM2, LLaVa-Next etc.:
        if pixel_values is not None:
            model_inputs["pixel_values"] = pixel_values
        # For SmolVLM2
        if pixel_attention_mask is not None:
            model_inputs["pixel_attention_mask"] = pixel_attention_mask
        # For LLaVa-Next
        if image_sizes is not None:
            model_inputs["image_sizes"] = image_sizes

        # Only add logits_to_keep if the model supports it
        if "logits_to_keep" in self.model_kwarg_keys:
            # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
            model_inputs["logits_to_keep"] = logits_to_keep + 1

        model_inputs["use_cache"] = (
            False  # only used in generation; set False to suppress warnings
        )

        last_hidden_state = unwrapped_model.model(**model_inputs).last_hidden_state
        # Exclude the last value: it corresponds to the next token pred
        last_hidden_state = last_hidden_state[:, :-1, :]  # (B, L-1, H)
        # Only keep the last logits_to_keep. For model that support logits_to_keep, this is a no-op.
        last_hidden_state = last_hidden_state[
            :, -logits_to_keep:, :
        ]  # (B, logits_to_keep, H)
        return last_hidden_state

    @staticmethod
    def _find_boxed_char_span(text: str) -> tuple[int, int] | None:
        """
        Find the first complete \\boxed{...} span in *text* using brace-depth
        tracking in character space (identical logic to extract_boxed_inner).

        Returns (span_start, span_end) where text[span_start:span_end] is the
        full "\\boxed{...}" string including the closing brace, or None if no
        complete span is found.
        """
        MARKER = r"\boxed{"
        marker_len = len(MARKER)
        i = 0
        while True:
            start = text.find(MARKER, i)
            if start == -1:
                return None  # no \boxed{ found at all

            # Walk forward tracking brace depth to find the matching '}'
            j = start + marker_len
            depth = 1
            while j < len(text) and depth > 0:
                if text[j] == "{":
                    depth += 1
                elif text[j] == "}":
                    depth -= 1
                j += 1

            if depth == 0:
                # j is one past the closing '}', so the span is [start, j)
                return (start, j)
            else:
                # Unbalanced braces — skip this occurrence and keep searching
                i = start + marker_len

    def _get_boxed_span_masks(
        self,
        completion_ids: torch.Tensor,  # (B, T)
        seq_len: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Derive four boolean token masks from the first \\boxed{...} span in each
        completion, covering the three sub-regions described in Section 9:

          answer_mask    — the full \\boxed{...} span (marker + content + closing brace)
          boundary_mask  — the \\boxed{ prefix tokens AND the trailing } token
          content_mask   — inner answer content: answer_mask & ~boundary_mask
          post_mask      — tokens strictly after the closing } of \\boxed{...}

        The same dummy-prefix trick as _get_answer_mask is used to recover
        token-accurate boundaries without being affected by BOS/boundary rules.

        All four masks have shape (B, T) and dtype torch.bool.
        """
        BOXED_MARKER = r"\boxed{"

        answer_mask = torch.zeros(completion_ids.shape, dtype=torch.bool, device=device)
        boundary_mask = torch.zeros(
            completion_ids.shape, dtype=torch.bool, device=device
        )
        content_mask = torch.zeros(
            completion_ids.shape, dtype=torch.bool, device=device
        )
        post_mask = torch.zeros(completion_ids.shape, dtype=torch.bool, device=device)

        dummy_str = "a"
        dummy_ids = self.processing_class.encode(dummy_str, add_special_tokens=False)
        dummy_len = len(dummy_ids)

        def _encode_mid(text: str) -> list[int]:
            combined = self.processing_class.encode(
                dummy_str + text, add_special_tokens=False
            )
            return combined[dummy_len:]

        pad_id = self.processing_class.pad_token_id

        for i in range(completion_ids.shape[0]):
            ids = completion_ids[i].tolist()

            n_leading_pads = 0
            if pad_id is not None:
                for t in ids:
                    if t == pad_id:
                        n_leading_pads += 1
                    else:
                        break

            text = self.processing_class.decode(ids, skip_special_tokens=True)
            span = self._find_boxed_char_span(text)
            if span is None:
                continue

            span_start, span_end = span  # character indices

            # Token boundaries for the full span
            prefix_toks = _encode_mid(text[:span_start])
            span_toks = _encode_mid(text[span_start:span_end])
            tok_start = n_leading_pads + len(prefix_toks)
            tok_end = min(tok_start + len(span_toks), seq_len)  # exclusive

            if tok_start >= tok_end:
                continue

            # Token boundary for the opening marker ("\boxed{" portion only)
            marker_toks = _encode_mid(BOXED_MARKER)
            marker_tok_end = min(tok_start + len(marker_toks), tok_end)

            # Full span mask
            answer_mask[i, tok_start:tok_end] = True

            # Boundary: opening marker tokens + final closing-brace token
            boundary_mask[i, tok_start:marker_tok_end] = True  # "\boxed{" tokens
            boundary_mask[i, tok_end - 1] = True  # closing "}"

            # Inner content: answer span minus boundaries
            content_mask[i, marker_tok_end : tok_end - 1] = True

            # Post-answer: everything after the closing "}"
            if tok_end < seq_len:
                post_mask[i, tok_end:] = True

        return answer_mask, boundary_mask, content_mask, post_mask

    def _get_answer_mask(
        self,
        completion_ids: torch.Tensor,  # (B, T) — integer token IDs
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Identify answer tokens in token space by working in **string space**.

        Strategy
        --------
        1. Decode each completion to a plain string.
        2. Find the first complete \\boxed{...} character span using
           brace-depth tracking (_find_boxed_char_span).
        3. Split the string at the span boundaries:
               prefix = text[:span_start]           (everything before \\boxed{)
               span   = text[span_start:span_end]   (the full \\boxed{...})
        4. Re-encode each piece with the dummy-prefix trick (neutralises
           BOS / word-boundary tokenisation rules) and count tokens:
               tok_start = len(prefix_tokens)
               tok_end   = tok_start + len(span_tokens)
        5. Mark those tokens as answer tokens.

        Encoding the span separately (rather than text[:span_end]) ensures
        that characters *after* the closing '}' cannot affect how '}' is
        tokenised — avoiding a fused-token boundary error at the right edge.

        This completely avoids token-level brace matching and all associated
        tokeniser edge cases (fused tokens, multi-character token merges, etc.).

        Returns a boolean mask of shape (B, T).
        """
        answer_mask = torch.zeros(completion_ids.shape, dtype=torch.bool, device=device)

        # Dummy prefix used to cancel beginning-of-sequence tokenisation rules.
        # We encode "a<text>" and strip the leading token(s) for "a" so that
        # the remaining tokens reflect mid-sequence tokenisation behaviour.
        dummy_str = "a"
        dummy_ids = self.processing_class.encode(dummy_str, add_special_tokens=False)
        dummy_len = len(dummy_ids)

        def _encode_mid(text: str) -> list[int]:
            """Encode *text* as if it appears mid-sequence (no BOS rules)."""
            combined = self.processing_class.encode(
                dummy_str + text, add_special_tokens=False
            )
            return combined[dummy_len:]

        pad_id = self.processing_class.pad_token_id

        for i in range(completion_ids.shape[0]):
            ids = completion_ids[i].tolist()

            # Count leading padding tokens.
            # decode(skip_special_tokens=True) silently strips them, so
            # re-encoded token counts are relative to the *unpadded* content.
            # For right-padded sequences n_leading_pads == 0 (no-op).
            # For left-padded sequences we must shift tok_start/tok_end by this
            # amount so the mask aligns with the correct positions in the tensor.
            n_leading_pads = 0
            if pad_id is not None:
                for t in ids:
                    if t == pad_id:
                        n_leading_pads += 1
                    else:
                        break

            text = self.processing_class.decode(ids, skip_special_tokens=True)

            span = self._find_boxed_char_span(text)
            if span is None:
                continue  # no complete \boxed{...} — leave mask all-False

            span_start, span_end = span

            # Recover token boundaries by encoding each piece separately.
            #
            # Encoding the span in isolation (text[span_start:span_end]) rather
            # than text[:span_end] means characters *after* the closing '}' are
            # never seen by the tokeniser here, so they cannot cause '}' to fuse
            # with the next character and shift tok_end.
            #
            # The left edge still has a minor boundary effect (the last char of
            # the prefix might tokenise differently without '\boxed{' following),
            # but '\boxed{' always starts with '\', which is almost never merged
            # leftward with whitespace or letters, making this negligible.
            prefix_tokens = _encode_mid(text[:span_start])
            span_tokens = _encode_mid(text[span_start:span_end])

            tok_start = n_leading_pads + len(prefix_tokens)
            tok_end = tok_start + len(span_tokens)  # exclusive

            # Guard: whitespace normalisation on decode/re-encode can produce a
            # slightly different total token count vs the live sequence.  Clamp.
            tok_end = min(tok_end, seq_len)
            if tok_start < tok_end:
                answer_mask[i, tok_start:tok_end] = True

        return answer_mask

    def compute_grail_weights(
        self,
        model,
        input_ids: torch.Tensor,  # (B, P+C) — full prompt + completion ids
        attention_mask: torch.Tensor,  # (B, P+C)
        logits_to_keep: int,  # number of completion tokens
        completion_ids: torch.Tensor,  # (B, T)
        mask: torch.Tensor,  # (B, T) — 1 for valid tokens, 0 for padding
        advantages: torch.Tensor = None,  # (B, 1) or (B,)
    ) -> torch.Tensor:
        """
        Compute GRAIL weights for a batch.

        Falls back to uniform weights per sequence when:
        - no \boxed{} answer tokens are detected in the sequence
        """
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"
        batch_size = completion_ids.shape[0]
        seq_len = completion_ids.shape[1]

        # ------------------------------------------------------------------
        # Phase 1: attribution forward pass — input embeddings as the leaf.
        #
        # We run a *separate* forward pass here (distinct from the GRPO logps
        # forward) with the input embeddings as an explicit autograd leaf.:
        #
        #   input_embeds_leaf  <-- leaf (ALL token positions)
        #          │
        #   full transformer body
        #          │
        #   last_hidden  (connected to leaf)
        #          │
        #   lm_head (weights detached)
        #          │
        #   logits → ans_loss  (only answer positions contribute)
        #          │
        #   autograd.grad(ans_loss, input_embeds_leaf)
        #          │
        #   grads[b, t, :]
        # ------------------------------------------------------------------
        with torch.enable_grad():
            if is_peft_model(model):
                base_model = model.base_model.model
            else:
                base_model = model

            # 1. Embed the full prompt+completion sequence.
            #    Detach from the parameter graph, then re-attach as a fresh leaf
            #    so that autograd.grad targets only this tensor and never touches
            #    any parameter .grad buffers.

            input_embeds = base_model.get_input_embeddings()(input_ids)  # (B, P+C, d)

            input_embeds_leaf = input_embeds.detach().requires_grad_(True)

            # 2. Run the full transformer body with the leaf embeddings.
            transformer_outputs = base_model.model(
                inputs_embeds=input_embeds_leaf,
                attention_mask=attention_mask,
                use_cache=False,
                output_hidden_states=True,
            )

            last_hidden = transformer_outputs.last_hidden_state  # (B, P+C, d)
            hidden_states = transformer_outputs.hidden_states

            if self.grail_leaf_source == "embeddings":
                leaf_tensor = input_embeds_leaf
            else:
                L = len(hidden_states) - 1
                if self.grail_leaf_source == "first":
                    leaf_tensor = hidden_states[1]
                elif self.grail_leaf_source == "middle":
                    leaf_tensor = hidden_states[L // 2]
                elif self.grail_leaf_source == "last":
                    leaf_tensor = hidden_states[L - 1]
                else:
                    raise ValueError(
                        f"Unknown grail_leaf_source: {self.grail_leaf_source}"
                    )

            # 3. Slice to the completion window
            last_hidden_completion = last_hidden[:, :-1, :]  # (B, P+C-1, d)
            last_hidden_completion = last_hidden_completion[
                :, -logits_to_keep:, :
            ]  # (B, T, d)

            # 4. Compute answer-token log-probs via detached lm_head weights.

            has_bias = (
                base_model.lm_head.bias is not None
                and base_model.lm_head.bias.numel() > 0
            )

            # DeepSpeed ZeRO-3 compatibility: gather weights explicitly before manual linear projection
            if hasattr(base_model.lm_head.weight, "ds_id"):
                import deepspeed

                params_to_gather = [base_model.lm_head.weight]
                if has_bias:
                    params_to_gather.append(base_model.lm_head.bias)

                with deepspeed.zero.GatheredParameters(params_to_gather):
                    lm_weight = base_model.lm_head.weight.detach()
                    lm_bias = base_model.lm_head.bias.detach() if has_bias else None
                    logits = torch.nn.functional.linear(
                        last_hidden_completion, lm_weight, lm_bias
                    )  # (B, T, V)
            else:
                lm_weight = base_model.lm_head.weight.detach()
                lm_bias = base_model.lm_head.bias.detach() if has_bias else None
                logits = torch.nn.functional.linear(
                    last_hidden_completion, lm_weight, lm_bias
                )  # (B, T, V)

            logits = logits / self.temperature
            logps = selective_log_softmax(logits, completion_ids)  # (B, T)

            answer_mask = self._get_answer_mask(completion_ids, seq_len, device)

            if not answer_mask.any():
                # No answer tokens detected: use a dummy zero loss so that
                # autograd.grad still runs (needed for distributed sync hooks).
                ans_loss = (logps * 0.0).sum()
            else:
                ans_loss = -logps[answer_mask].sum()

            grads = torch.autograd.grad(
                outputs=ans_loss,
                inputs=leaf_tensor,
            )[
                0
            ]  # (B, P+C, d)

        # Slice gradients and embeddings to the completion window.
        grads_completion = grads[:, -(logits_to_keep + 1) : -1, :].detach()  # (B, T, d)
        embeds_completion = leaf_tensor[
            :, -(logits_to_keep + 1) : -1, :
        ].detach()  # (B, T, d)

        # ------------------------------------------------------------------
        # Phase 2: Score computation
        # ------------------------------------------------------------------
        with torch.no_grad():
            scores = (grads_completion.float() * embeds_completion.float()).norm(
                dim=-1
            )  # (B, T)

            grad_norms = (
                grads_completion.float().norm(dim=-1).mean(dim=-1, keepdim=True)
            )  # (B, 1)
            no_answer = ~answer_mask.any(dim=-1, keepdim=True)  # (B, 1)
            fallback = no_answer  # (B, 1)

            if advantages is not None:
                if self.grail_rollout_symmetry == "correct":
                    symmetry_fallback = advantages <= 0
                    if symmetry_fallback.dim() == 1:
                        symmetry_fallback = symmetry_fallback.unsqueeze(-1)
                    fallback = fallback | symmetry_fallback
                elif self.grail_rollout_symmetry == "wrong":
                    symmetry_fallback = advantages > 0
                    if symmetry_fallback.dim() == 1:
                        symmetry_fallback = symmetry_fallback.unsqueeze(-1)
                    fallback = fallback | symmetry_fallback

            float_mask = mask.float()  # (B, T)

            if self.grail_std:
                # --- Log-Transform + Z-Score ---
                # Log-transform to compress extreme attention sink outliers.
                log_scores = torch.log(
                    scores + 1e-8
                )  # (B, T) — raw log, padding not yet excluded
                valid_counts = float_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)

                # Compute mean over valid tokens only (mask applied in sum, not on log_scores)
                mean_log_scores = (log_scores * float_mask).sum(
                    dim=-1, keepdim=True
                ) / valid_counts

                # Compute Std Dev of log scores over valid tokens only
                variance = ((log_scores - mean_log_scores) ** 2) * float_mask
                std_log_scores = torch.sqrt(
                    variance.sum(dim=-1, keepdim=True) / valid_counts + 1e-8
                )

                # Calculate Z-Scores
                z_scores = (log_scores - mean_log_scores) / std_log_scores

                # Map to target distribution
                raw_weights = self.grail_mean + (z_scores * self.grail_std)

                # Clip and apply fallback/padding masks
                clipped_w = raw_weights.clamp(self.grail_w_min, self.grail_w_max)

                # --- Clip Ratio Logging ---
                active_grail_mask = mask.bool() & ~fallback.bool()

                # Identify which active tokens hit the boundaries
                is_min_clipped = (raw_weights <= self.grail_w_min) & active_grail_mask
                is_max_clipped = (raw_weights >= self.grail_w_max) & active_grail_mask

                active_tokens_count = active_grail_mask.sum()

                if active_tokens_count > 0:
                    local_min_frac = is_min_clipped.sum().float() / active_tokens_count
                    local_max_frac = is_max_clipped.sum().float() / active_tokens_count
                else:
                    local_min_frac = torch.tensor(float("nan"), device=device)
                    local_max_frac = torch.tensor(float("nan"), device=device)

                gathered_min_frac = self.accelerator.gather(local_min_frac)
                gathered_max_frac = self.accelerator.gather(local_max_frac)

                if not torch.isnan(gathered_min_frac).all():
                    self._metrics[mode]["grail/clip_frac_min"].append(
                        gathered_min_frac.nanmean().item()
                    )
                    self._metrics[mode]["grail/clip_frac_max"].append(
                        gathered_max_frac.nanmean().item()
                    )

                weights = torch.where(fallback, torch.ones_like(clipped_w), clipped_w)
                weights = weights * float_mask
            else:
                raise ValueError("A non-zero GRAIL std was not enabled")

            _, boundary_mask_grail, content_mask_grail, post_mask_grail = (
                self._get_boxed_span_masks(completion_ids, seq_len, device)
            )

            weights[content_mask_grail] = self.grail_w_max

            weights[boundary_mask_grail] = self.grail_mean

            post_and_valid = post_mask_grail & mask.bool()
            weights[post_and_valid] = self.grail_mean

            # --------------------------------------------------------------
            # Logging: fallback/answer rates + weight stats
            # --------------------------------------------------------------
            fallback_rate = fallback.float().mean()
            answer_found_rate = (~no_answer).float().mean()
            valid_weights = weights[mask.bool()]

            # --- gradnorm logging ---
            valid_grad_norms = grad_norms[~no_answer]

            pad_value = -1.0
            padded_norms = self.accelerator.pad_across_processes(
                valid_grad_norms, dim=0, pad_index=pad_value
            )
            gathered_norms = self.accelerator.gather(padded_norms)

            gathered_norms = gathered_norms[gathered_norms != pad_value]

            if gathered_norms.numel() > 0:
                p10_norm = torch.quantile(gathered_norms.float(), 0.10).item()
                p90_norm = torch.quantile(gathered_norms.float(), 0.90).item()
                self._metrics[mode]["grail/grad_norm_p10"].append(p10_norm)
                self._metrics[mode]["grail/grad_norm_p90"].append(p90_norm)

            self._metrics[mode]["grail/fallback_rate"].append(
                self.accelerator.gather(fallback_rate).mean().item()
            )
            self._metrics[mode]["grail/answer_found_rate"].append(
                self.accelerator.gather(answer_found_rate).mean().item()
            )

            if valid_weights.numel() > 0:
                local_mean = valid_weights.mean()
                local_std = valid_weights.std()
            else:
                local_mean = torch.tensor(float("nan"), device=device)
                local_std = torch.tensor(float("nan"), device=device)

            gathered_means = self.accelerator.gather(local_mean)
            gathered_stds = self.accelerator.gather(local_std)

            if not torch.isnan(gathered_means).all():
                self._metrics[mode]["grail/weight_mean"].append(
                    gathered_means.nanmean().item()
                )
                self._metrics[mode]["grail/weight_std"].append(
                    gathered_stds.nanmean().item()
                )

        return weights.detach()

    def compute_oar_g_weights(
        self,
        model,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        logits_to_keep: int,
        completion_ids: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"
        batch_size = completion_ids.shape[0]
        seq_len = completion_ids.shape[1]

        with torch.enable_grad():
            # Resolve base model (handles PEFT wrapping)
            if is_peft_model(model):
                base_model = model.base_model.model
            else:
                base_model = model

            lm_weight = base_model.lm_head.weight.detach()
            lm_bias = (
                base_model.lm_head.bias.detach()
                if base_model.lm_head.bias is not None
                and base_model.lm_head.bias.numel() > 0
                else None
            )

            with torch.no_grad():
                clean_embeds = base_model.get_input_embeddings()(input_ids)
                transformer_outputs = base_model.model(
                    inputs_embeds=clean_embeds,
                    attention_mask=attention_mask,
                    use_cache=False,
                    output_hidden_states=False,
                )
                last_hidden = transformer_outputs.last_hidden_state
                last_hidden_completion = last_hidden[:, -(logits_to_keep + 1) : -1, :]
                logits_clean = torch.nn.functional.linear(
                    last_hidden_completion, lm_weight, lm_bias
                )

                answer_mask = self._get_answer_mask(completion_ids, seq_len, device)

                ans_counts = answer_mask.sum(dim=1, keepdim=True).clamp(min=1)  # (B, 1)
                masked_logits_clean = logits_clean * answer_mask.unsqueeze(
                    -1
                )  # (B, T, V)
                mean_logits_clean = (
                    masked_logits_clean.sum(dim=1) / ans_counts
                )  # (B, V)
                P_0 = torch.softmax(mean_logits_clean, dim=-1).detach()  # (B, V)

            input_embeds = base_model.get_input_embeddings()(input_ids).detach()

            noise = torch.zeros_like(input_embeds)
            completion_slice = slice(-(logits_to_keep + 1), -1)
            noise[:, completion_slice, :] = (
                torch.randn_like(noise[:, completion_slice, :]) * self.oar_g_noise_std
            )

            noisy_embeds = input_embeds + noise
            noisy_embeds.requires_grad_(True)

            transformer_outputs = base_model.model(
                inputs_embeds=noisy_embeds,
                attention_mask=attention_mask,
                use_cache=False,
                output_hidden_states=False,
            )
            last_hidden_noisy = transformer_outputs.last_hidden_state
            last_hidden_noisy_completion = last_hidden_noisy[
                :, -(logits_to_keep + 1) : -1, :
            ]

            logits_noisy = torch.nn.functional.linear(
                last_hidden_noisy_completion, lm_weight, lm_bias
            )

            masked_logits_noisy = logits_noisy * answer_mask.unsqueeze(-1)
            mean_logits_noisy = masked_logits_noisy.sum(dim=1) / ans_counts
            log_P_eps = torch.nn.functional.log_softmax(
                mean_logits_noisy, dim=-1
            )  # (B, V)

            has_answer = answer_mask.any(dim=-1)  # (B,)
            if not has_answer.any():
                kl_loss = (log_P_eps * 0.0).sum()  # dummy
            else:
                kl_per_seq = (P_0 * (torch.log(P_0 + self.oar_g_eps) - log_P_eps)).sum(
                    dim=-1
                )  # (B,)
                kl_loss = (kl_per_seq * has_answer.float()).sum()

            grads = torch.autograd.grad(
                outputs=kl_loss,
                inputs=noisy_embeds,
            )[0]

        grads_completion = grads[:, -(logits_to_keep + 1) : -1, :].detach()
        embeds_completion = noisy_embeds[:, -(logits_to_keep + 1) : -1, :].detach()

        with torch.no_grad():
            scores = (
                (grads_completion.float() * embeds_completion.float()).sum(dim=-1).abs()
            )
            float_mask = mask.float()

            bar_I = torch.log(1 + scores) * float_mask

            masked_for_max = bar_I - (1.0 - float_mask) * 1e9
            masked_for_min = bar_I + (1.0 - float_mask) * 1e9

            max_I = masked_for_max.max(dim=-1, keepdim=True)[0]
            min_I = masked_for_min.min(dim=-1, keepdim=True)[0]

            hat_I = (bar_I - min_I) / (max_I - min_I + self.oar_g_eps)

            T_valid = float_mask.sum(dim=-1, keepdim=True)
            hat_I_for_quant = hat_I.masked_fill(~mask.bool(), float("inf"))
            sorted_hat_I, _ = torch.sort(hat_I_for_quant, dim=-1)

            quant_idx = (T_valid * self.oar_g_tau).long()
            max_idx = torch.clamp(T_valid.long() - 1, min=0)
            quant_idx = torch.min(quant_idx, max_idx)
            quant_idx = torch.max(quant_idx, torch.zeros_like(quant_idx))
            tau_seq = sorted_hat_I.gather(dim=-1, index=quant_idx)  # (B, 1)

            beta = self.oar_g_beta
            eps = self.oar_g_eps

            suppress = torch.clamp(hat_I / (tau_seq + eps), min=0.1)
            boost = 1 + beta * (hat_I - tau_seq) / (1 - tau_seq + eps)

            omega = torch.where(hat_I < tau_seq, suppress, boost)
            omega = omega * float_mask

            sum_omega = omega.sum(dim=-1, keepdim=True)
            T = float_mask.sum(dim=-1, keepdim=True)

            tilde_omega = omega * T / (sum_omega + 1e-8)
            clipped_w = tilde_omega

            no_answer = ~answer_mask.any(dim=-1, keepdim=True)
            fallback = no_answer

            weights = torch.where(fallback, torch.ones_like(clipped_w), clipped_w)
            weights = weights * float_mask

            fallback_rate = fallback.float().mean()
            answer_found_rate = (~no_answer).float().mean()
            valid_weights = weights[mask.bool()]

            self._metrics[mode]["oar_g/fallback_rate"].append(
                self.accelerator.gather(fallback_rate).mean().item()
            )
            self._metrics[mode]["oar_g/answer_found_rate"].append(
                self.accelerator.gather(answer_found_rate).mean().item()
            )

            if valid_weights.numel() > 0:
                local_mean = valid_weights.mean()
                local_std = valid_weights.std()
            else:
                local_mean = torch.tensor(float("nan"), device=device)
                local_std = torch.tensor(float("nan"), device=device)

            gathered_means = self.accelerator.gather(local_mean)
            gathered_stds = self.accelerator.gather(local_std)

            if not torch.isnan(gathered_means).all():
                self._metrics[mode]["oar_g/weight_mean"].append(
                    gathered_means.nanmean().item()
                )
                self._metrics[mode]["oar_g/weight_std"].append(
                    gathered_stds.nanmean().item()
                )

        return weights.detach()

    def get_high_entropy_mask(
        self, entropies: torch.Tensor, mask: torch.Tensor, threshold: float
    ) -> torch.Tensor:
        """
        Returns a binary mask identifying tokens whose entropy exceeds a given quantile threshold.

        Args:
            entropies (`torch.Tensor`):
                Tensor of shape (batch_size, seq_len) with per-token entropy values.
            mask (`torch.Tensor`):
                Binary mask of the same shape as `entropies`, where `1` indicates valid tokens and `0` padding.
            threshold (`float`):
                Quantile threshold between `0.0` and `1.0` to select high-entropy tokens.

        Returns:
            `torch.Tensor`:
                Boolean mask of shape (batch_size, seq_len), where `True` indicates tokens with entropy >= threshold
                and `False` otherwise.
        """
        local = entropies[mask.bool()].float()

        # Use a negative pad_value as a sentinel because entropy values are always >= 0.
        # This guarantees that the sentinel cannot collide with any real entropy value.
        pad_value = -1e9

        # Pad across processes so that every rank has the same tensor length
        padded = self.accelerator.pad_across_processes(
            local, dim=0, pad_index=pad_value
        )
        gathered = self.accelerator.gather(padded)

        # Drop sentinel values (safe because no entropy can be negative)
        gathered = gathered[gathered != pad_value]

        if gathered.numel() == 0:
            return torch.zeros_like(entropies, dtype=torch.bool)

        entropy_threshold = torch.quantile(gathered, threshold)
        masked_entropies = entropies * mask.float()
        entropy_mask = masked_entropies >= entropy_threshold
        return entropy_mask & mask.bool()  # ensure padding tokens are always masked out

    @profiling_decorator
    def _get_per_token_logps_and_entropies(
        self,
        model,
        input_ids,
        attention_mask,
        logits_to_keep,
        batch_size=None,
        compute_entropy=False,
        pixel_values=None,
        image_grid_thw=None,
        num_images=None,
        pixel_attention_mask=None,
        image_sizes=None,
        token_type_ids=None,
        mm_token_type_ids=None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Compute log-probs and (optionally) entropies for each token."""
        batch_size = batch_size or input_ids.size(
            0
        )  # Chunk inputs into smaller batches to reduce memory peak
        all_logps = []
        all_entropies = []
        for start in range(0, input_ids.size(0), batch_size):
            input_ids_batch = input_ids[start : start + batch_size]
            attention_mask_batch = attention_mask[start : start + batch_size]

            # Build model inputs - check if the model supports logits_to_keep (some models and VLMs don't)
            model_inputs = {
                "input_ids": input_ids_batch,
                "attention_mask": attention_mask_batch,
            }
            if image_grid_thw is not None and pixel_values is not None:
                rows_per_image = image_grid_thw.prod(dim=-1)
                rows_per_sample = torch.split(rows_per_image, num_images)
                rows_per_sample = torch.stack([s.sum() for s in rows_per_sample])
                cum_rows = torch.cat(
                    [
                        torch.tensor([0], device=rows_per_sample.device),
                        rows_per_sample.cumsum(0),
                    ]
                )
                row_start, row_end = (
                    cum_rows[start].item(),
                    cum_rows[start + batch_size].item(),
                )
                model_inputs["pixel_values"] = pixel_values[row_start:row_end]
                cum_imgs = torch.tensor([0] + num_images).cumsum(0)
                img_start, img_end = cum_imgs[start], cum_imgs[start + batch_size]
                model_inputs["image_grid_thw"] = image_grid_thw[img_start:img_end]
            elif pixel_values is not None:
                model_inputs["pixel_values"] = pixel_values[start : start + batch_size]
            if pixel_attention_mask is not None:
                model_inputs["pixel_attention_mask"] = pixel_attention_mask[
                    start : start + batch_size
                ]
            if image_sizes is not None:
                model_inputs["image_sizes"] = image_sizes[start : start + batch_size]
            if token_type_ids is not None:
                model_inputs["token_type_ids"] = token_type_ids[
                    start : start + batch_size
                ]
            if mm_token_type_ids is not None:
                model_inputs["mm_token_type_ids"] = mm_token_type_ids[
                    start : start + batch_size
                ]

            # Only add logits_to_keep if the model supports it
            if "logits_to_keep" in self.model_kwarg_keys:
                # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
                model_inputs["logits_to_keep"] = logits_to_keep + 1

            model_inputs["use_cache"] = (
                False  # only used in generation; set False to suppress warnings
            )

            logits = model(**model_inputs).logits

            # Exclude the last value: it corresponds to the next token pred
            logits = logits[:, :-1, :]  # (B, L-1, H)
            # Only keep the last logits_to_keep. For model that support logits_to_keep, this is a no-op.
            logits = logits[:, -logits_to_keep:, :]  # (B, logits_to_keep, H)
            # Divide logits by sampling temperature.
            # See https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo#policy-training-implementation-details
            logits.div_(self.temperature)
            completion_ids = input_ids_batch[:, -logits_to_keep:]
            logps = selective_log_softmax(logits, completion_ids)  # compute logprobs
            all_logps.append(logps)

            if compute_entropy:
                with torch.no_grad():
                    entropies = entropy_from_logits(logits)
                all_entropies.append(entropies)

        logps = torch.cat(all_logps, dim=0)
        entropies = torch.cat(all_entropies, dim=0) if compute_entropy else None
        return logps, entropies

    def training_step(self, model, inputs, num_items_in_batch):
        time_before = time.perf_counter()
        output = super().training_step(model, inputs, num_items_in_batch)
        self._step += 1
        time_after = time.perf_counter()
        self._current_train_step_time += time_after - time_before
        if self._step % self.current_gradient_accumulation_steps == 0:
            self._metrics["train"]["step_time"].append(self._current_train_step_time)
            self._current_train_step_time = 0.0
        return output

    @profiling_decorator
    def _prepare_inputs(
        self, generation_batch: dict[str, torch.Tensor | Any]
    ) -> dict[str, torch.Tensor | Any]:
        # Prepares inputs for model training/evaluation by managing completion generation and batch handling.
        # During training:
        #   - Receives the local generation batch (Per-GPU batch size × steps per generation)
        #     from the modified training dataloader instead of the standard local batch
        #   - Generates completions once for the entire generation batch and splits it into batches of size
        #     `per_device_train_batch_size`
        #   - Buffers these completions and returns the appropriate slice for the current accumulation step
        #   - Optimizes by regenerating completions only periodically (every steps_per_generation * num_iterations)
        # During evaluation:
        #   - The input is treated as a standard local batch (no accumulation, no multiple iterations)
        #   - Completions are generated for each batch without buffering or reuse
        # Returns a single local batch in both cases.

        mode = "train" if self.model.training else "eval"
        if mode == "train":
            generate_every = self.args.steps_per_generation * self.num_iterations
            if self._step % generate_every == 0 or self._buffered_inputs is None:
                # self._buffered_inputs=None can occur when resuming from a checkpoint
                generation_batch = self._generate_and_score_completions(
                    generation_batch
                )
                generation_batch = split_pixel_values_by_grid(generation_batch)
                generation_batch = shuffle_sequence_dict(generation_batch)
                generation_batches = split_tensor_dict(
                    generation_batch, self.args.steps_per_generation
                )
                self._buffered_inputs = [
                    unsplit_pixel_values_by_grid(batch) for batch in generation_batches
                ]
            inputs = self._buffered_inputs[self._step % self.args.steps_per_generation]
        else:
            # In evaluation, there is neither batch grouping for generation, nor multiple iterations, hence
            # local generation batch == local eval batch
            inputs = self._generate_and_score_completions(generation_batch)
        return inputs

    def _log_completion_extra(self, column: str, values: list):
        """
        Log extra columns to the completions table. Called from reward functions via the `log_extra` kwarg.

        Args:
            column (`str`):
                Name of the column to add.
            values (`list`):
                Values for the column, one per sample in the batch.
        """
        self._pending_extra_logs[column].extend(values)

    def _log_metric(self, name: str, value: float):
        """
        Log a scalar metric from a reward function. Called via the `log_metric` kwarg. Values are averaged over each
        logging step and reported alongside built-in metrics like `kl` and `entropy`.

        Args:
            name (`str`):
                Name of the metric.
            value (`float`):
                Scalar value for this batch.
        """
        self._pending_metrics[name].append(value)

    @profiling_decorator
    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        device = self.accelerator.device
        rewards_per_func = torch.zeros(
            len(prompts), len(self.reward_funcs), device=device
        )

        # Repeat all input columns (but "prompt", "completion", and "completion_ids") to match the num of generations
        keys = [
            key
            for key in inputs[0]
            if key not in ["prompt", "completion", "completion_ids"]
        ]
        reward_kwargs = {key: [example[key] for example in inputs] for key in keys}

        # This allows for dynamic reward shaping based on training progress.
        reward_kwargs["trainer_state"] = self.state

        # Allow reward functions to log extra columns to the completions table.
        reward_kwargs["log_extra"] = self._log_completion_extra

        # Allow reward functions to log additional scalar metrics.
        reward_kwargs["log_metric"] = self._log_metric

        async_funcs_info = []  # async custom functions for asyncio.gather

        for i, (reward_func, reward_processing_class, reward_func_name) in enumerate(
            zip(
                self.reward_funcs,
                self.reward_processing_classes,
                self.reward_func_names,
                strict=True,
            )
        ):
            if isinstance(
                reward_func, nn.Module
            ):  # Module (no PretrainedModel) for compat with compiled models
                with profiling_context(self, reward_func_name):
                    if is_conversational(inputs[0]):
                        messages = [
                            {"messages": p + c}
                            for p, c in zip(prompts, completions, strict=True)
                        ]
                        texts = [
                            apply_chat_template(
                                x, reward_processing_class, **self.chat_template_kwargs
                            )["text"]
                            for x in messages
                        ]
                    else:
                        texts = [
                            p + c for p, c in zip(prompts, completions, strict=True)
                        ]
                    reward_inputs = reward_processing_class(
                        text=texts,
                        return_tensors="pt",
                        padding=True,
                        padding_side="right",
                        add_special_tokens=False,
                    )
                    reward_inputs = super()._prepare_inputs(reward_inputs)
                    with torch.inference_mode():
                        rewards_per_func[:, i] = reward_func(**reward_inputs).logits[
                            :, 0
                        ]  # Shape (B*G,)
            elif inspect.iscoroutinefunction(
                reward_func
            ):  # Separate async reward funcs to run them in parallel later
                async_funcs_info.append((i, reward_func, reward_func_name))
            else:
                # Run synchronous reward function
                with profiling_context(self, reward_func_name):
                    if self.environments is not None:
                        reward_kwargs["environments"] = self.environments
                    output_reward_func = reward_func(
                        prompts=prompts,
                        completions=completions,
                        completion_ids=completion_ids_list,
                        **reward_kwargs,
                    )
                    # Convert None values to NaN
                    output_reward_func = [
                        reward if reward is not None else torch.nan
                        for reward in output_reward_func
                    ]
                    rewards_per_func[:, i] = torch.tensor(
                        output_reward_func, dtype=torch.float32, device=device
                    )

        # Execute async custom functions in parallel using asyncio.gather
        if async_funcs_info:

            async def _invoke_async(index, func, func_name):
                with profiling_context(self, func_name):
                    output = await func(
                        prompts=prompts,
                        completions=completions,
                        completion_ids=completion_ids_list,
                        **reward_kwargs,
                    )
                    output = [r if r is not None else torch.nan for r in output]
                    return index, output

            async def _run_async_funcs():
                coros = [
                    _invoke_async(i, func, func_name)
                    for (i, func, func_name) in async_funcs_info
                ]
                return await asyncio.gather(*coros)

            async_results = asyncio.run_coroutine_threadsafe(
                _run_async_funcs(), self.async_loop
            ).result()
            for idx, output_reward_func in async_results:
                rewards_per_func[:, idx] = torch.tensor(
                    output_reward_func, dtype=torch.float32, device=device
                )

        # If all reward functions return None for a given row, issue a detailed warning
        if torch.isnan(rewards_per_func).all(dim=1).any():
            nan_row_idx = (
                torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
            )
            row_reward_kwargs = {
                key: value[nan_row_idx]
                for key, value in reward_kwargs.items()
                if key not in ("trainer_state", "log_extra", "log_metric")
            }
            row_reward_kwargs["prompt"] = prompts[nan_row_idx]
            row_reward_kwargs["completion"] = completions[nan_row_idx]
            logger.warning(
                f"All reward functions returned None for the following kwargs:\n{row_reward_kwargs}\n"
                "Please ensure that at least one reward function returns a valid reward."
            )

        # Gather the reward per function: this part is crucial, because the rewards are normalized per group and the
        # completions may be distributed across processes
        rewards_per_func = gather(rewards_per_func)
        return rewards_per_func

    def _tokenize_prompts(self, prompts: list):
        """Tokenize prompts and extract images/multimodal fields for generation."""
        if is_conversational({"prompt": prompts[0]}):
            # Extract images from messages for VLM support
            images = []
            has_images = False
            for prompt in prompts:
                prompt_images = []
                for message in prompt:
                    if isinstance(message["content"], list):
                        for part in message["content"]:
                            if part["type"] == "image":
                                prompt_images.append(part["image"])
                                has_images = True
                images.append(prompt_images if prompt_images else None)
            images = images if has_images else None

            # We pass padding=True to work around a bug introduced in transformers 5.2.0 in some processors
            # (e.g. Qwen2.5-VL) that crash on batched unpadded input. We then unpad input_ids using attention_mask.
            # See: https://github.com/huggingface/transformers/issues/44514
            tokenized = self.processing_class.apply_chat_template(
                conversation=prompts,
                tools=self.tools,
                chat_template=self.chat_template,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                padding=True,
                **self.chat_template_kwargs,
            )
            # Unpad input_ids: remove padding tokens using attention_mask to get per-sequence lists
            prompt_ids = [
                [tok for tok, m in zip(ids, mask, strict=True) if m]
                for ids, mask in zip(
                    tokenized["input_ids"], tokenized["attention_mask"], strict=True
                )
            ]
            # For VLMs, the processor returns extra multimodal fields (pixel_values, image_grid_thw, etc.)
            multimodal_fields = {
                k: v
                for k, v in tokenized.items()
                if k not in ("input_ids", "attention_mask")
            }
        else:
            prompt_ids = self.processing_class(text=prompts)["input_ids"]
            images = None
            multimodal_fields = {}
        return prompt_ids, images, multimodal_fields

    def _generate_single_turn(self, prompt_ids, images, multimodal_fields):
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        # Generate completions using either vLLM or regular generation
        if self.use_vllm:
            # Sync weights if training step changed
            if self.state.global_step != self._last_loaded_step:
                with profiling_context(self, "sync_weights"):
                    self.vllm_generation.sync_weights()
                self._last_loaded_step = self.state.global_step

            # Generate using vLLM with raw token IDs
            num_generations = (
                self.num_generations if mode == "train" else self.num_generations_eval
            )
            _, completion_ids, logprobs, _ = self.vllm_generation.generate(
                prompts=prompt_ids,
                images=images,
                num_generations=num_generations,
                profiler=profiling_context(self, "vLLM.generate"),
            )
            # vLLM returns per-token top-k logprobs; keep only the top-1 (sampled token) logprob
            logprobs = [[lp[0] for lp in seq] for seq in logprobs]

        elif self.use_transformers_paged:
            with (
                profiling_context(self, "transformers.generate_batch"),
                unwrap_model_for_generation(
                    self.model_wrapped,
                    self.accelerator,
                    gather_deepspeed3_params=self.args.ds3_gather_for_generation,
                ) as unwrapped_model,
                torch.no_grad(),
                (
                    FSDP.summon_full_params(self.model_wrapped, recurse=False)
                    if self.is_fsdp_enabled
                    else nullcontext()
                ),
            ):
                # Cast to the appropriate dtype based on training configuration
                if self.args.bf16:
                    unwrapped_model.to(torch.bfloat16)
                elif self.args.fp16:
                    unwrapped_model.to(torch.float16)
                if self.args.cast_lm_head_to_fp32:
                    unwrapped_model.lm_head.to(torch.float32)
                with torch.inference_mode():
                    # Continuous batching API expects 'inputs' arg only
                    all_outputs = unwrapped_model.generate_batch(
                        prompt_ids,
                        generation_config=self.generation_config,
                        progress_bar=False,
                    )
                    unwrapped_model.train()  # restore training mode, as generate_batch forces eval mode
            completion_ids = [
                output.generated_tokens for output in all_outputs.values()
            ]
            logprobs = None  # not used in this case

        else:
            # Regular generation path: left-pad token IDs into tensors
            prompt_tensors = [torch.tensor(ids) for ids in prompt_ids]
            padded_ids = pad(
                prompt_tensors, padding_value=self.pad_token_id, padding_side="left"
            )
            attention_mask = pad(
                [torch.ones_like(t) for t in prompt_tensors],
                padding_value=0,
                padding_side="left",
            )
            generate_inputs = {
                "input_ids": padded_ids,
                "attention_mask": attention_mask,
            }
            # For VLMs, include multimodal fields as tensors (pixel_values, image_grid_thw, etc.)
            for k, v in multimodal_fields.items():
                if isinstance(v, torch.Tensor):
                    generate_inputs[k] = v
                elif isinstance(v, list) and v and isinstance(v[0], list):
                    # Per-token field (e.g., token_type_ids): left-pad like input_ids
                    generate_inputs[k] = pad(
                        [torch.tensor(x) for x in v],
                        padding_value=0,
                        padding_side="left",
                    )
                else:
                    generate_inputs[k] = torch.tensor(np.array(v))
            generate_inputs = super()._prepare_inputs(generate_inputs)

            with (
                profiling_context(self, "transformers.generate"),
                unwrap_model_for_generation(
                    self.model_wrapped,
                    self.accelerator,
                    gather_deepspeed3_params=self.args.ds3_gather_for_generation,
                    generation_kwargs=self.generation_kwargs,  # Override model.generation_config with generation_kwargs to fix transformers#42762
                ) as unwrapped_model,
                torch.no_grad(),
                (
                    FSDP.summon_full_params(self.model_wrapped, recurse=False)
                    if self.is_fsdp_enabled
                    else nullcontext()
                ),
            ):
                prompt_completion_ids = unwrapped_model.generate(
                    **generate_inputs,
                    generation_config=self.generation_config,
                    disable_compile=True,
                )
            # Compute prompt length and extract completion ids
            prompt_length = generate_inputs["input_ids"].size(1)
            completion_ids = prompt_completion_ids[:, prompt_length:]

            # Mask everything after the first EOS token
            is_eos = completion_ids == self.eos_token_id
            eos_idx = torch.full(
                (is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device
            )
            eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
            sequence_indices = torch.arange(is_eos.size(1), device=device).expand(
                is_eos.size(0), -1
            )
            completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
            completion_ids = [
                c[m].tolist()
                for c, m in zip(
                    completion_ids.cpu(), completion_mask.bool().cpu(), strict=True
                )
            ]
            logprobs = None  # not used in this case

        return completion_ids, logprobs

    def _get_tool_suffix_ids(self, tool_messages):
        """Get token IDs for tool result formatting by using a minimal dummy conversation."""
        dummy_messages = [
            {"role": "user", "content": "dummy"},
            {"role": "assistant", "content": "dummy"},
        ]
        prefix_ids = self.processing_class.apply_chat_template(
            dummy_messages,
            add_generation_prompt=False,
            chat_template=self.chat_template,
            return_dict=False,
            **self.chat_template_kwargs,
        )
        full_ids = self.processing_class.apply_chat_template(
            dummy_messages + tool_messages,
            add_generation_prompt=True,
            chat_template=self.chat_template,
            return_dict=False,
            **self.chat_template_kwargs,
        )

        # Some chat templates (notably Qwen3/Qwen3.5) render "...<|im_end|>\n" after an assistant/tool block.
        # When we compute `suffix_ids` by slicing `full_ids`, we must align the slicing boundary to
        # EOS (not EOS + newline).
        last_eos_idx = max(
            i for i, tok_id in enumerate(prefix_ids) if tok_id == self.eos_token_id
        )
        prefix_ids = prefix_ids[: last_eos_idx + 1]

        if full_ids[: len(prefix_ids)] != prefix_ids:
            raise ValueError(
                "Unexpected tokenization: the EOS-trimmed prefix IDs are not a prefix of the full IDs."
            )

        return full_ids[len(prefix_ids) :]

    def _tool_call_loop(
        self,
        prompts,
        prompt_ids,
        completion_ids,
        completions,
        logprobs,
        images,
        multimodal_fields,
    ):
        # Tool execution loop: execute tools, then regenerate completions with tool results appended to the prompt
        tool_calls = [completion[0].get("tool_calls") for completion in completions]
        idxs_with_tool = [idx for idx, tool_call in enumerate(tool_calls) if tool_call]
        tool_calls = [tool_calls[idx] for idx in idxs_with_tool]
        tool_mask = [
            [1] * len(ids) for ids in completion_ids
        ]  # 0 for tool result tokens, 1 elsewhere
        tool_call_count = 0
        tool_failure_count = 0
        iteration_num = 0
        while idxs_with_tool and iteration_num < self.max_tool_calling_iterations:
            prompt_completion_tools = [
                prompts[i] for i in idxs_with_tool
            ]  # select only prompts that need tool calls

            # Call the tools, and build the new prompt for generation
            for idx in range(len(idxs_with_tool)):
                idx_with_tool = idxs_with_tool[idx]
                tool_call_list = tool_calls[idx]
                prompt_completion_tool = prompt_completion_tools[idx]
                sync_tool_dict = self._sync_tool_dicts[idx_with_tool]
                async_tool_dict = self._async_tool_dicts[idx_with_tool]
                # Append the last assistant message (which triggered tool_calls) to the prompt
                prompt_completion_tool.append(completions[idx_with_tool][-1])
                async_coros = []
                tool_call_results = []
                for tool_call in tool_call_list:
                    tool_call_count += 1
                    if tool_call["type"] == "function":
                        function = tool_call["function"]
                        name = function["name"]
                        try:
                            if name in sync_tool_dict:
                                tool_call_results.append(
                                    (
                                        name,
                                        sync_tool_dict[name](**function["arguments"]),
                                    )
                                )
                            elif name in async_tool_dict:
                                async_coros.append(
                                    (
                                        name,
                                        async_tool_dict[name](**function["arguments"]),
                                    )
                                )
                            else:
                                raise ValueError(f"Tool {name} not found.")
                        except Exception as e:
                            tool_failure_count += 1
                            result = {"error": str(e)}
                            tool_call_results.append((name, result))
                    else:
                        tool_failure_count += 1
                        name = tool_call.get("name", "unknown")
                        tool_call_results.append(
                            (
                                name,
                                {
                                    "error": f"Unsupported tool call type: {tool_call['type']}"
                                },
                            )
                        )

                if async_coros:

                    async def _run_async_tools(async_coros):
                        coros = [coro for _, coro in async_coros]
                        results = await asyncio.gather(*coros, return_exceptions=True)
                        return [
                            (name, result)
                            for (name, _), result in zip(
                                async_coros, results, strict=False
                            )
                        ]

                    async_results = asyncio.run_coroutine_threadsafe(
                        _run_async_tools(async_coros), self.async_loop
                    ).result()

                    for name, result in async_results:
                        if isinstance(result, Exception):
                            tool_failure_count += 1
                            tool_call_results.append((name, {"error": str(result)}))
                        else:
                            tool_call_results.append((name, result))

                for name, result in tool_call_results:
                    tool_message = {
                        "role": "tool",
                        "name": name,
                        "content": str(result),
                    }
                    prompt_completion_tool.append(tool_message)
                    completions[idx_with_tool].append(tool_message)

            # Build token IDs by concatenation: prompt + completion + tool_suffix.
            prompt_completion_tool_ids = []
            for idx in range(len(idxs_with_tool)):
                idx_with_tool = idxs_with_tool[idx]
                # Extract trailing tool messages from completions
                tool_messages = []
                for message in reversed(completions[idx_with_tool]):
                    if message["role"] == "tool":
                        tool_messages.insert(0, message)
                    else:
                        break
                suffix_ids = self._get_tool_suffix_ids(tool_messages)
                prompt_completion_tool_ids.append(
                    prompt_ids[idx_with_tool]
                    + completion_ids[idx_with_tool]
                    + suffix_ids
                )

            # Filter samples whose length exceeds max allowed length. This is important, because both
            # vLLM and transformers will error out if the input is longer than the model's max length.
            if self.use_vllm and self.vllm_mode == "colocate":
                max_model_len = (
                    self.vllm_generation.llm.llm_engine.model_config.max_model_len
                )
            elif not self.use_vllm:
                max_model_len = self.model.config.max_position_embeddings
            else:
                raise NotImplementedError(
                    f"Unsupported mode detected: use_vllm={self.use_vllm}, vllm_mode={self.vllm_mode}"
                )
            overlong = [len(pct) >= max_model_len for pct in prompt_completion_tool_ids]
            for idx in range(len(idxs_with_tool)):
                idx_with_tool = idxs_with_tool[idx]
                if overlong[idx]:
                    prompt_length = len(prompt_ids[idx_with_tool])
                    ct = prompt_completion_tool_ids[idx][
                        prompt_length : prompt_length + self.max_completion_length
                    ]
                    completion_ids[idx_with_tool] = ct
                    tool_mask[idx_with_tool] += [1] * (
                        len(ct) - len(tool_mask[idx_with_tool])
                    )
                    if logprobs is not None:
                        logprobs[idx_with_tool] += [0.0] * (
                            len(ct) - len(logprobs[idx_with_tool])
                        )
            # Keep only non-overlong items for further processing
            idxs_with_tool = [
                idx for idx, o in zip(idxs_with_tool, overlong, strict=True) if not o
            ]
            prompt_completion_tools = [
                pct
                for pct, o in zip(prompt_completion_tools, overlong, strict=True)
                if not o
            ]
            prompt_completion_tool_ids = [
                pct
                for pct, o in zip(prompt_completion_tool_ids, overlong, strict=True)
                if not o
            ]
            if not idxs_with_tool:
                break  # all overlong, exit tool loop

            # Filter images and multimodal fields to match the current subset (index into full batch)
            loop_images = [images[i] for i in idxs_with_tool] if images else None
            loop_multimodal_fields = (
                {
                    k: [v[i] for i in idxs_with_tool]
                    for k, v in multimodal_fields.items()
                }
                if multimodal_fields
                else {}
            )

            # Generate new completions after tool execution (using concatenated IDs, no re-tokenization)
            post_tool_ids, post_tool_logprobs = self._generate_single_turn(
                prompt_completion_tool_ids, loop_images, loop_multimodal_fields
            )

            # Truncate so that pct[len(prompt_ids[idx]) :] + post_tool does not exceed max_completion_length
            for idx in range(len(idxs_with_tool)):
                idx_with_tool = idxs_with_tool[idx]
                prompt_len = len(prompt_ids[idx_with_tool])
                completion_tool_ids = prompt_completion_tool_ids[idx][prompt_len:]
                excess_length = (
                    len(completion_tool_ids)
                    + len(post_tool_ids[idx])
                    - self.max_completion_length
                )
                if excess_length > 0:
                    # If exceeding max length, truncate post_tool_ids
                    post_tool_ids[idx] = post_tool_ids[idx][:-excess_length]
                    if logprobs is not None:
                        post_tool_logprobs[idx] = post_tool_logprobs[idx][
                            :-excess_length
                        ]
                    excess_length = (
                        len(completion_tool_ids)
                        + len(post_tool_ids[idx])
                        - self.max_completion_length
                    )
                    if excess_length > 0:
                        # If still exceeding max length, truncate completion_tool_ids as well
                        prompt_completion_tool_ids[idx] = prompt_completion_tool_ids[
                            idx
                        ][:-excess_length]

            # Update tool_mask: the tool result should be 0 and the post-tool 1
            for idx in range(len(idxs_with_tool)):
                idx_with_tool = idxs_with_tool[idx]
                prompt_completion_tool_length = len(prompt_completion_tool_ids[idx])
                prompt_length = len(prompt_ids[idx_with_tool])
                completion_length = len(completion_ids[idx_with_tool])
                post_tool_length = len(post_tool_ids[idx])
                tool_length = (
                    prompt_completion_tool_length - prompt_length - completion_length
                )
                tool_mask[idx_with_tool] += [0] * tool_length + [1] * post_tool_length
                if logprobs is not None:
                    logprobs[idx_with_tool] += [0.0] * tool_length + post_tool_logprobs[
                        idx
                    ]

            # Update completion_ids with the new completions (after tool execution)
            for idx in range(len(idxs_with_tool)):
                idx_with_tool = idxs_with_tool[idx]
                prompt_length = len(prompt_ids[idx_with_tool])
                pct = prompt_completion_tool_ids[idx]  # = prompt-completion-tool
                completion_ids[idx_with_tool] = pct[prompt_length:] + post_tool_ids[idx]

            # Decode post-tool completions
            post_tool_completions = [
                parse_response(self.processing_class, ids) if ids else {}
                for ids in post_tool_ids
            ]

            # Add post-tool completions to the existing completions
            for idx in range(len(idxs_with_tool)):
                idx_with_tool = idxs_with_tool[idx]
                if post_tool_completions[
                    idx
                ]:  # {} if post-tool completions completely truncated
                    completions[idx_with_tool].append(post_tool_completions[idx])

            # Check for further tool calls
            tool_calls = [
                completion.get("tool_calls") for completion in post_tool_completions
            ]
            idxs_with_tool = [
                idx
                for idx, tool_call in zip(idxs_with_tool, tool_calls, strict=True)
                if tool_call
            ]
            tool_calls = [tool_call for tool_call in tool_calls if tool_call]
            iteration_num += 1
        return (
            tool_mask,
            completions,
            completion_ids,
            logprobs,
            tool_call_count,
            tool_failure_count,
        )

    def _generate(self, prompts: list):
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        # Copy the prompts to avoid modifying the original list
        prompts = copy.deepcopy(prompts)

        if self.rollout_func is not None:
            # Keep vLLM weights in sync for custom rollouts that rely on vLLM utilities.
            if self.use_vllm and self.state.global_step != self._last_loaded_step:
                with profiling_context(self, "sync_weights"):
                    self.vllm_generation.sync_weights()
                self._last_loaded_step = self.state.global_step

            # Pass prompts to rollout_func preserving structured messages.
            # Chat templating must happen inside rollout_func, at the backend boundary, so that
            # multimodal content (images, typed content blocks) is not lost before rollout logic runs.
            output = self.rollout_func(prompts, self)
            required_keys = {"prompt_ids", "completion_ids", "logprobs"}
            missing_keys = required_keys - output.keys()
            if missing_keys:
                missing_keys_list = sorted(missing_keys)
                raise ValueError(
                    f"rollout_func must return keys {missing_keys_list} in its output dict."
                )
            extra_fields = {k: v for k, v in output.items() if k not in required_keys}
            prompt_ids, completion_ids, logprobs = (
                output["prompt_ids"],
                output["completion_ids"],
                output["logprobs"],
            )
        else:
            prompt_ids, images, multimodal_fields = self._tokenize_prompts(prompts)
            completion_ids, logprobs = self._generate_single_turn(
                prompt_ids, images, multimodal_fields
            )
            extra_fields = {}

        # Decode completions. It's important to use `parse_response` when possible, because it handles tool calls.
        if is_conversational({"prompt": prompts[0]}):
            if (
                Version(transformers.__version__)
                >= Version("5.0.0")  # parse_response added in v5
                and isinstance(
                    self.processing_class, PreTrainedTokenizerBase
                )  # doesn't work with processors
                and hasattr(
                    self.processing_class, "response_schema"
                )  # attribute not set by default for now
                and self.processing_class.response_schema
                is not None  # only works if the tokenizer has a schema
            ):
                completions = [
                    [parse_response(self.processing_class, ids)]
                    for ids in completion_ids
                ]
            else:
                contents = self.processing_class.batch_decode(
                    completion_ids, skip_special_tokens=True
                )
                completions = [
                    [{"role": "assistant", "content": content}] for content in contents
                ]
        else:
            completions = self.processing_class.batch_decode(
                completion_ids, skip_special_tokens=True
            )

        # Extract tool calls from the completions and (possibly) execute them
        if self.tools:
            (
                tool_mask,
                completions,
                completion_ids,
                logprobs,
                tool_call_count,
                tool_failure_count,
            ) = self._tool_call_loop(
                prompts,
                prompt_ids,
                completion_ids,
                completions,
                logprobs,
                images,
                multimodal_fields,
            )
        else:
            # Support custom env_mask from rollout_func (e.g., for environment feedback masking)
            # Internally treated as tool_mask - marks model tokens (1) vs external tokens (0)
            tool_mask = extra_fields.pop("env_mask", None)

        # Get completion length per sequence, used for logging
        prompt_lengths = torch.tensor([len(ids) for ids in prompt_ids], device=device)
        if tool_mask is not None:  # count only model-generated tokens (tool_mask=1)
            completion_lengths = torch.tensor(
                [sum(mask) for mask in tool_mask], device=device
            )
        else:
            completion_lengths = torch.tensor(
                [len(ids) for ids in completion_ids], device=device
            )
        agg_prompt_lengths = self.accelerator.gather(prompt_lengths)
        agg_completion_lengths = self.accelerator.gather(completion_lengths)
        total_prompt_tokens = agg_prompt_lengths.sum()
        total_completion_tokens = (
            agg_completion_lengths.sum()
        )  # = num_items_in_batch, required for the DAPO loss

        # Log the metrics
        if mode == "train":
            self.state.num_input_tokens_seen += (
                total_prompt_tokens + total_completion_tokens
            ).item()
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]

        # Log completion lengths, mean, min, max
        self._metrics[mode]["completions/mean_length"].append(
            agg_completion_lengths.float().mean().item()
        )
        self._metrics[mode]["completions/min_length"].append(
            agg_completion_lengths.float().min().item()
        )
        self._metrics[mode]["completions/max_length"].append(
            agg_completion_lengths.float().max().item()
        )

        # Identify sequences that terminated with EOS and log their lengths
        eos_and_pad = [self.eos_token_id, self.pad_token_id]
        is_truncated = torch.tensor(
            [ids[-1] not in eos_and_pad for ids in completion_ids], device=device
        )
        agg_is_truncated = self.accelerator.gather(is_truncated)
        self._metrics[mode]["completions/clipped_ratio"].append(
            agg_is_truncated.float().mean().item()
        )
        term_completion_lengths = agg_completion_lengths[~agg_is_truncated]
        if (
            len(term_completion_lengths) == 0
        ):  # edge case where no terminated sequences are found
            term_completion_lengths = torch.zeros(1, device=device)
        self._metrics[mode]["completions/mean_terminated_length"].append(
            term_completion_lengths.float().mean().item()
        )
        self._metrics[mode]["completions/min_terminated_length"].append(
            term_completion_lengths.float().min().item()
        )
        self._metrics[mode]["completions/max_terminated_length"].append(
            term_completion_lengths.float().max().item()
        )

        if self.tools:
            agg_tool_call_count = self.accelerator.gather(
                torch.tensor(tool_call_count, device=device)
            ).sum()
            tool_call_frequency = (agg_tool_call_count / len(agg_prompt_lengths)).item()
            self._metrics[mode]["tools/call_frequency"].append(tool_call_frequency)
            agg_tool_failure_count = self.accelerator.gather(
                torch.tensor(tool_failure_count, device=device)
            ).sum()
            failure_frequency = (
                (agg_tool_failure_count / agg_tool_call_count).item()
                if agg_tool_call_count > 0
                else 0.0
            )
            self._metrics[mode]["tools/failure_frequency"].append(failure_frequency)

        return (
            prompt_ids,
            completion_ids,
            tool_mask,
            completions,
            total_completion_tokens,
            logprobs,
            extra_fields,
        )

    def _generate_and_score_completions(
        self, inputs: list[dict[str, torch.Tensor | Any]]
    ) -> dict[str, torch.Tensor | Any]:
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        prompts = [x["prompt"] for x in inputs]

        if self.environments:
            for prompt, environment, reset_kwargs in zip(
                prompts, self.environments, inputs, strict=True
            ):
                observation = environment.reset(**reset_kwargs)
                if observation is None:
                    continue
                prompt[-1]["content"] += observation

        if "images" in inputs[0]:
            images = [example.get("images") for example in inputs]
        elif "image" in inputs[0]:
            images = [
                [example.get("image")] if example.get("image") is not None else None
                for example in inputs
            ]
        else:
            images = None
        # Transformers requires at least one image in the batch, otherwise it throws an error
        if images is not None and all(img_list == [] for img_list in images):
            images = None

        # If the prompts are conversational and the inputs contain images, we need to convert the prompts from
        # [{"role": "user", "content": "What color is the sky?"}] to
        # [{"role": "user", "content": [{"type": "image", "image": <Image>}, {"type": "text", "text": "What color is the sky?"}]}]
        if images is not None:
            if not is_conversational(inputs[0]):
                raise ValueError(
                    "Multimodal training requires conversational prompts. It looks like the dataset contains "
                    "non-conversational inputs, likely because a chat template was applied before passing the dataset "
                    "to the trainer. Please provide the raw conversational prompts and let the trainer apply the chat "
                    "template internally."
                )
            prompts = [
                prepare_multimodal_messages(prompt, image_list)
                for prompt, image_list in zip(prompts, images, strict=True)
            ]

        (
            prompt_ids_list,
            completion_ids_list,
            tool_mask_list,
            completions,
            num_items_in_batch,
            sampling_per_token_logps_list,
            extra_fields,
        ) = self._generate(prompts)

        # Convert lists of token IDs to padded tensors
        prompt_ids = [torch.tensor(ids) for ids in prompt_ids_list]
        prompt_mask = [torch.ones_like(ids, dtype=torch.long) for ids in prompt_ids]
        prompt_ids = pad(
            prompt_ids,
            padding_value=self.pad_token_id,
            padding_side="left",
            pad_to_multiple_of=self.pad_to_multiple_of,
        ).to(device=device)
        prompt_mask = pad(
            prompt_mask,
            padding_value=0,
            padding_side="left",
            pad_to_multiple_of=self.pad_to_multiple_of,
        ).to(device=device)
        completion_ids = [torch.tensor(ids) for ids in completion_ids_list]
        completion_mask = [
            torch.ones_like(ids, dtype=torch.long) for ids in completion_ids
        ]
        completion_ids = pad(
            completion_ids,
            padding_value=self.pad_token_id,
            padding_side="right",
            pad_to_multiple_of=self.pad_to_multiple_of,
        ).to(device=device)
        completion_mask = pad(
            completion_mask,
            padding_value=0,
            padding_side="right",
            pad_to_multiple_of=self.pad_to_multiple_of,
        ).to(device=device)
        if sampling_per_token_logps_list is not None:
            sampling_per_token_logps = [
                torch.tensor(logps) for logps in sampling_per_token_logps_list
            ]
            sampling_per_token_logps = pad(
                sampling_per_token_logps,
                padding_value=0.0,
                padding_side="right",
                pad_to_multiple_of=self.pad_to_multiple_of,
            ).to(device=device)
        else:
            sampling_per_token_logps = None
        if tool_mask_list is not None:
            tool_mask = [torch.tensor(mask) for mask in tool_mask_list]
            tool_mask = pad(
                tool_mask,
                padding_value=1,
                padding_side="right",
                pad_to_multiple_of=self.pad_to_multiple_of,
            ).to(device=device)
        else:
            tool_mask = None

        # If mask_truncated_completions is enabled, zero out truncated completions for attention and loss masking
        if self.mask_truncated_completions:
            eos_and_pad = [self.eos_token_id, self.pad_token_id]
            is_truncated = torch.tensor(
                [ids[-1] not in eos_and_pad for ids in completion_ids_list],
                device=device,
            )
            # Mask completion_mask for attention masking
            completion_mask = completion_mask * (~is_truncated).unsqueeze(1).int()
            # Also mask tool_mask for consistency in multi-turn training
            if tool_mask is not None:
                tool_mask = tool_mask * (~is_truncated).unsqueeze(1).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        prompt_completion_ids = torch.cat(
            [prompt_ids, completion_ids], dim=1
        )  # (B, P+C)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)

        logits_to_keep = completion_ids.size(
            1
        )  # we only need to compute the logits for the completion tokens
        batch_size = (
            self.args.per_device_train_batch_size
            if mode == "train"
            else self.args.per_device_eval_batch_size
        )

        num_images = (
            [len(img_list) for img_list in images] if images is not None else None
        )

        # Get forward_kwargs for models with multimodal inputs
        if images is not None:
            prompts_text = [
                apply_chat_template(
                    {"prompt": prompt},
                    self.processing_class,
                    tools=self.tools,
                    **self.chat_template_kwargs,
                )["prompt"]
                for prompt in prompts
            ]
            prompt_inputs = self.processing_class(
                images=images, text=prompts_text, padding=True, return_tensors="pt"
            )
            prompt_inputs = super()._prepare_inputs(prompt_inputs)
            forward_kwargs = {
                k: v
                for k, v in prompt_inputs.items()
                if k not in ["input_ids", "attention_mask"]
            }
        else:
            forward_kwargs = {}

        # If token_type_ids are used, extend them with zeros for the completion part
        if "token_type_ids" in forward_kwargs:
            token_type_ids = forward_kwargs["token_type_ids"]
            if self.pad_to_multiple_of is not None:
                # Needed only with pad_to_multiple_of: otherwise prompt_ids and token_type_ids must have equal len
                padding_size = prompt_ids.size(1) - token_type_ids.size(1)
                if padding_size > 0:
                    token_type_ids = torch.cat(
                        [
                            token_type_ids.new_zeros(
                                (token_type_ids.size(0), padding_size)
                            ),
                            token_type_ids,
                        ],
                        dim=1,
                    )
            forward_kwargs["token_type_ids"] = torch.cat(
                [token_type_ids, token_type_ids.new_zeros(completion_ids.shape)], dim=1
            )
        # If mm_token_type_ids are used, extend them with zeros for the completion part
        if "mm_token_type_ids" in forward_kwargs:
            mm_token_type_ids = forward_kwargs["mm_token_type_ids"]
            if self.pad_to_multiple_of is not None:
                # Needed only with pad_to_multiple_of: otherwise prompt_ids and mm_token_type_ids must have equal len
                padding_size = prompt_ids.size(1) - mm_token_type_ids.size(1)
                if padding_size > 0:
                    mm_token_type_ids = torch.cat(
                        [
                            mm_token_type_ids.new_zeros(
                                (mm_token_type_ids.size(0), padding_size)
                            ),
                            mm_token_type_ids,
                        ],
                        dim=1,
                    )
            forward_kwargs["mm_token_type_ids"] = torch.cat(
                [mm_token_type_ids, mm_token_type_ids.new_zeros(completion_ids.shape)],
                dim=1,
            )

        # When gradient checkpointing is enabled with use_reentrant=True (non default), calling the model inside a
        # torch.no_grad() block triggers a harmless PyTorch warning ("None of the inputs have requires_grad=True").
        # Temporarily disable checkpointing to avoid this warning during inference.
        with (
            torch.no_grad(),
            disable_gradient_checkpointing(
                self.model, self.args.gradient_checkpointing_kwargs
            ),
        ):
            # If the generation and optimization steps are misaligned—i.e., if generation does not occur at the end of
            # a full optimizer step (when gradient_accumulation_steps is not a multiple of generate_every)—then the
            # samples may come from an earlier version of the model. In that case, we need to track old_per_token_logps
            # for importance sampling. If the steps are aligned, importance sampling isn't necessary and we set
            # old_per_token_logps to None.
            # When using vLLM, we always compute old_per_token_logps for importance sampling, it was shown that the
            # distribution mismatch between vLLM and the training model can be large and harm the training.
            generate_every = (
                self.args.steps_per_generation * self.num_iterations
            )  # generation frequency
            if self.args.gradient_accumulation_steps % generate_every != 0 or (
                self.use_vllm and self.vllm_importance_sampling_correction
            ):
                old_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                    self.model,
                    prompt_completion_ids,
                    attention_mask,
                    logits_to_keep,
                    batch_size,
                    num_images=num_images,
                    **forward_kwargs,  # may contain pixel_values, image_grid_thw, pixel_attention_mask and image_sizes
                )
            else:
                old_per_token_logps = None

            # Compute the importance sampling ratio when using vLLM, to correct for potential distribution mismatch
            if self.use_vllm and self.vllm_importance_sampling_correction:
                mask = (
                    completion_mask
                    if tool_mask is None
                    else completion_mask * tool_mask
                )
                per_token_logps_diff = (
                    old_per_token_logps - sampling_per_token_logps
                ) * mask

                sequence_level_is = self.vllm_importance_sampling_mode in [
                    "sequence_mask",
                    "sequence_truncate",
                ]
                if sequence_level_is:
                    per_sequence_logps_diff = per_token_logps_diff.sum(
                        dim=-1, keepdim=True
                    )
                    logps_diff = per_sequence_logps_diff
                else:
                    logps_diff = per_token_logps_diff

                vllm_importance_sampling_ratio = torch.exp(logps_diff)

                # vllm_importance_sampling_ratio.shape:
                #   token_* modes:     (B, T)  (per-token ratio)
                #   sequence_* modes:  (B, 1)  (per-sequence ratio)

                if self.vllm_importance_sampling_mode in [
                    "sequence_truncate",
                    "token_truncate",
                ]:
                    vllm_importance_sampling_ratio = torch.clamp(
                        vllm_importance_sampling_ratio,
                        max=self.vllm_importance_sampling_cap,
                    )
                elif self.vllm_importance_sampling_mode in [
                    "sequence_mask",
                    "token_mask",
                ]:
                    vllm_importance_sampling_ratio = (
                        vllm_importance_sampling_ratio.masked_fill(
                            vllm_importance_sampling_ratio
                            > self.vllm_importance_sampling_cap,
                            value=0.0,
                        )
                    )
                else:
                    raise ValueError(
                        f"Unknown vLLM importance sampling level: {self.vllm_importance_sampling_mode}. Possible values are 'token_truncate', 'token_mask', 'sequence_truncate', and 'sequence_mask'."
                    )

            # Compute the per-token log probabilities for the reference model
            if self.beta != 0.0:
                if self.ref_model is not None:
                    ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                        self.ref_model,
                        prompt_completion_ids,
                        attention_mask,
                        logits_to_keep,
                        batch_size=batch_size,
                        num_images=num_images,
                        **forward_kwargs,  # may contain pixel_values, image_grid_thw, pixel_attention_mask and image_sizes
                    )
                else:
                    # When training a PEFT adapter, how we obtain the reference depends on the setup:
                    # - New adapter: disabling adapters yields the base model.
                    # - Re-training an existing adapter: an initial copy is loaded under the name "ref".
                    model = self.accelerator.unwrap_model(self.model)
                    with use_adapter(
                        model,
                        adapter_name="ref" if "ref" in model.peft_config else None,
                    ):
                        ref_per_token_logps, _ = (
                            self._get_per_token_logps_and_entropies(
                                self.model,
                                prompt_completion_ids,
                                attention_mask,
                                logits_to_keep,
                                batch_size=batch_size,
                                num_images=num_images,
                                **forward_kwargs,  # may contain pixel_values, image_grid_thw, pixel_attention_mask and image_sizes
                            )
                        )
            else:
                ref_per_token_logps = None

        # Decode
        prompts_text = self.processing_class.batch_decode(
            prompt_ids, skip_special_tokens=True
        )
        completions_text = self.processing_class.batch_decode(
            completion_ids, skip_special_tokens=True
        )

        # Merge extra_fields from rollout_func into inputs for reward functions
        if extra_fields:
            for i, inp in enumerate(inputs):
                for key, values in extra_fields.items():
                    if isinstance(values, list) and i < len(values):
                        inp[key] = values[i]
                    elif not isinstance(values, list):
                        inp[key] = values

        # Calculate rewards for each reward function. rewards_per_func aggregates rewards across all processes. This is
        # important because rewards will be normalized per group, and completions are distributed. We will later slice
        # rewards_per_func to extract each process's subset.
        rewards_per_func = self._calculate_rewards(
            inputs, prompts, completions, completion_ids_list
        )
        num_generations = (
            self.num_generations if mode == "train" else self.num_generations_eval
        )

        if self.multi_objective_aggregation == "sum_then_normalize":
            # Apply weights to each reward function's output and sum
            rewards = (
                rewards_per_func * self.reward_weights.to(device).unsqueeze(0)
            ).nansum(dim=1)
            mean_grouped_rewards = rewards.view(-1, num_generations).mean(dim=1)
            mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(
                num_generations, dim=0
            )
            if self.scale_rewards in ["group", "none"]:
                # If self.scale_rewards = "none", we'll only use std_rewards to check for zero std for logging
                if num_generations > 1:
                    std_rewards = rewards.view(-1, num_generations).std(dim=1)
                    std_rewards = std_rewards.repeat_interleave(num_generations, dim=0)
                else:  # doesn't occur during training, but could occur in eval when num_generations_eval=1
                    std_rewards = torch.zeros_like(rewards)
            elif self.scale_rewards == "batch":
                # Compute global std
                if rewards.numel() > 1:
                    std_rewards = rewards.std().expand_as(rewards)
                else:  # doesn't occur during training, but could occur in eval when num_generations_eval=batch_size=1
                    std_rewards = torch.zeros_like(rewards)
            else:
                raise ValueError(
                    f"Invalid value for scale_rewards: {self.scale_rewards}. Must be one of 'batch', 'group', or 'none'."
                )

            advantages = rewards - mean_grouped_rewards
            if self.scale_rewards != "none":
                advantages = advantages / (std_rewards + 1e-4)
            is_std_zero = torch.isclose(
                std_rewards, torch.zeros_like(std_rewards)
            )  # for logging

        elif self.multi_objective_aggregation == "normalize_then_sum":
            grouped = rewards_per_func.view(-1, num_generations, len(self.reward_funcs))
            mean_k = torch.nanmean(grouped, dim=1, keepdim=True)
            std_k = (
                nanstd(grouped, dim=1, keepdim=True)
                if num_generations > 1
                else torch.zeros_like(mean_k)
            )
            reward_k = (grouped - mean_k) / (std_k + 1e-4)
            reward_k = reward_k.view(-1, len(self.reward_funcs))
            rewards = (reward_k * self.reward_weights.to(device).unsqueeze(0)).nansum(
                dim=1
            )
            std_rewards = (
                rewards.std().expand_as(rewards)
                if rewards.numel() > 1
                else torch.zeros_like(rewards)
            )
            advantages = (rewards - rewards.mean()) / (std_rewards + 1e-4)
            is_std_zero = torch.isclose(
                std_rewards, torch.zeros_like(std_rewards)
            )  # for logging

        else:
            raise ValueError(
                f"Invalid multi_objective_aggregation: {self.multi_objective_aggregation}. Must be "
                "'sum_then_normalize' or 'normalize_then_sum'."
            )

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        all_process_advantages = (
            advantages.clone()
        )  # keep the aggregated advantages for logging
        advantages = advantages[process_slice]

        # Calculate mean reward per function, but only for samples where the function was applied (non-NaN values)
        for i, reward_func_name in enumerate(self.reward_func_names):
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(mean_rewards)
            std_func_rewards = nanstd(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/std"].append(
                std_func_rewards
            )
        rewards = rewards_per_func.nansum(dim=1)
        self._metrics[mode]["reward"].append(rewards.mean().item())
        self._metrics[mode]["reward_std"].append(rewards.std().item())
        self._metrics[mode]["frac_reward_zero_std"].append(
            is_std_zero.float().mean().item()
        )

        # Log prompt and completion texts
        self._logs["prompt"].extend(gather_object(prompts_text))
        self._logs["completion"].extend(gather_object(completions_text))
        for i, name in enumerate(self.reward_func_names):
            self._logs["rewards"][name].extend(rewards_per_func[:, i].tolist())
        self._logs["advantages"].extend(all_process_advantages.tolist())

        # Flush user-logged extra columns (from log_extra), gathering across processes.
        # Keys must be sorted so that all ranks call gather_object in the same order, otherwise values
        # get mis-attributed across columns (dict insertion order may differ between processes).
        for column in sorted(self._pending_extra_logs):
            self._logs["extra"][column].extend(
                gather_object(self._pending_extra_logs[column])
            )
        self._pending_extra_logs.clear()

        # Flush user-logged metrics (from log_metric), averaging across processes.
        # Keys must be sorted so that all ranks call accelerator.gather in the same order, otherwise values
        # get mis-attributed across metrics (dict insertion order may differ between processes).
        for name in sorted(self._pending_metrics):
            values = self._pending_metrics[name]
            local_mean = sum(values) / len(values)
            global_mean = (
                self.accelerator.gather(torch.tensor(local_mean, device=device))
                .mean()
                .item()
            )
            self._metrics[mode][name].append(global_mean)
        self._pending_metrics.clear()

        if images is not None:
            self._logs["images"].extend(gather_object(images))

        if self.use_vllm and self.vllm_importance_sampling_correction:
            delta = torch.abs(old_per_token_logps - sampling_per_token_logps)
            mask = (
                completion_mask.bool()
                if tool_mask is None
                else (completion_mask * tool_mask).bool()
            )
            delta = delta[mask]
            mean_delta = (
                torch.mean(delta)
                if delta.numel() > 0
                else torch.tensor(0.0, device=device)
            )
            max_delta = (
                torch.max(delta)
                if delta.numel() > 0
                else torch.tensor(0.0, device=device)
            )
            self._metrics[mode]["sampling/sampling_logp_difference/mean"].append(
                self.accelerator.gather(mean_delta).mean().item()
            )
            self._metrics[mode]["sampling/sampling_logp_difference/max"].append(
                self.accelerator.gather(max_delta).max().item()
            )
            if sequence_level_is:
                flat_is_ratio = vllm_importance_sampling_ratio.flatten()
            else:
                flat_is_ratio = vllm_importance_sampling_ratio[mask]

            min_importance_sampling_ratio = (
                torch.min(flat_is_ratio)
                if flat_is_ratio.numel() > 0
                else torch.tensor(0.0, device=device)
            )
            mean_importance_sampling_ratio = (
                torch.mean(flat_is_ratio)
                if flat_is_ratio.numel() > 0
                else torch.tensor(0.0, device=device)
            )
            max_importance_sampling_ratio = (
                torch.max(flat_is_ratio)
                if flat_is_ratio.numel() > 0
                else torch.tensor(0.0, device=device)
            )
            self._metrics[mode]["sampling/importance_sampling_ratio/min"].append(
                nanmin(self.accelerator.gather(min_importance_sampling_ratio)).item()
            )
            self._metrics[mode]["sampling/importance_sampling_ratio/mean"].append(
                self.accelerator.gather(mean_importance_sampling_ratio).nanmean().item()
            )
            self._metrics[mode]["sampling/importance_sampling_ratio/max"].append(
                nanmax(self.accelerator.gather(max_importance_sampling_ratio)).item()
            )

        output = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "num_items_in_batch": num_items_in_batch,
        }
        if old_per_token_logps is not None:
            output["old_per_token_logps"] = old_per_token_logps
        if self.use_vllm and self.vllm_importance_sampling_correction:
            output["importance_sampling_ratio"] = vllm_importance_sampling_ratio
        if sampling_per_token_logps is not None:
            output["sampling_per_token_logps"] = sampling_per_token_logps
        if ref_per_token_logps is not None:
            output["ref_per_token_logps"] = ref_per_token_logps
        if "pixel_values" in forward_kwargs:
            output["pixel_values"] = forward_kwargs["pixel_values"]
        if "image_grid_thw" in forward_kwargs:
            output["image_grid_thw"] = forward_kwargs["image_grid_thw"]
        if "pixel_attention_mask" in forward_kwargs:
            output["pixel_attention_mask"] = forward_kwargs["pixel_attention_mask"]
        if "image_sizes" in forward_kwargs:
            output["image_sizes"] = forward_kwargs["image_sizes"]
        if "token_type_ids" in forward_kwargs:
            output["token_type_ids"] = forward_kwargs["token_type_ids"]
        if "mm_token_type_ids" in forward_kwargs:
            output["mm_token_type_ids"] = forward_kwargs["mm_token_type_ids"]
        if images is not None:
            output["num_images"] = num_images
        if tool_mask is not None:
            output["tool_mask"] = tool_mask
        return output

    def compute_liger_loss(self, unwrapped_model, inputs):
        # Compute the per-token log probabilities for the model
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = (
            inputs["completion_ids"],
            inputs["completion_mask"],
        )
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(
            1
        )  # we only need to compute the logits for the completion tokens

        # Get the last hidden state of the model
        last_hidden_state = self._get_last_hidden_state(
            unwrapped_model,
            input_ids,
            attention_mask,
            logits_to_keep,
            inputs.get("pixel_values"),
            inputs.get("image_grid_thw"),
            inputs.get("pixel_attention_mask"),
            inputs.get("image_sizes"),
        )

        # Apply tool_mask (from env_mask) for loss computation in multi-turn training scenarios
        loss_mask = (
            completion_mask
            if "tool_mask" not in inputs
            else completion_mask * inputs["tool_mask"]
        )
        # Compute loss and metrics using liger grpo loss
        loss, metrics = self.liger_grpo_loss(
            _input=last_hidden_state,
            lin_weight=unwrapped_model.lm_head.weight,
            selected_token_ids=completion_ids,
            # The attention_mask parameter in liger loss is actually used as a loss mask (not model attention)
            attention_mask=loss_mask,
            advantages=inputs["advantages"],
            bias=unwrapped_model.lm_head.bias,
            old_per_token_logps=inputs.get("old_per_token_logps"),
            ref_per_token_logps=inputs.get("ref_per_token_logps"),
            vllm_is_ratio=inputs.get("importance_sampling_ratio"),
        )
        # Extract metrics from the liger_grpo_loss output
        # KL divergence is the first metric when beta is non-zero
        mean_kl = metrics[0] if self.beta != 0.0 else None
        clip_ratio = metrics[-1]

        mode = "train" if self.model.training else "eval"
        if self.beta != 0.0:
            self._metrics[mode]["kl"].append(
                self.accelerator.gather(mean_kl).mean().item()
            )
        self._metrics[mode]["clip_ratio"].append(
            self.accelerator.gather(clip_ratio).mean().item()
        )
        normalizer = (
            self.current_gradient_accumulation_steps if mode == "train" else 1.0
        )  # no accum in eval
        return loss / normalizer

    @profiling_decorator
    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        if self.use_liger_kernel:
            # Compute the loss using the liger grpo loss
            unwrapped_model = self.accelerator.unwrap_model(model)
            return self._forward_redirection(
                model, unwrapped_model, self.compute_liger_loss, unwrapped_model, inputs
            )
        else:
            return self._compute_loss(model, inputs)

    @staticmethod
    def get_off_policy_mask(
        advantages: torch.Tensor,
        per_token_logps: torch.Tensor,
        sampling_per_token_logps: torch.Tensor,
        mask: torch.Tensor,
        off_policy_threshold: float,
    ) -> torch.Tensor:
        """
        Computes the Off-Policy Sequence Mask from DeepSeek-V3.2 paper. Returns a (B, 1) tensor where 1.0 indicates
        "Keep" and 0.0 indicates "Drop".
        """
        # forward KL div: log(pi_old) - log(pi_theta)
        kl_div = sampling_per_token_logps - per_token_logps.detach()
        # Sequence-level Mean KL (ignoring prompt+padding)
        seq_kl_sum = (kl_div * mask).sum(dim=1, keepdim=True)
        avg_seq_kl = seq_kl_sum / mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        # Keep if (Advantage >= 0) OR (KL <= delta)
        is_pos_adv = advantages >= 0
        is_low_kl = avg_seq_kl <= off_policy_threshold
        return (is_pos_adv | is_low_kl).to(dtype=mask.dtype)  # (B, 1)

    @staticmethod
    @torch.no_grad()
    def get_gamma_weights(
        advantages: torch.Tensor,
        log_ratio_per_token: torch.Tensor,
        mask: torch.Tensor,
        importance_sampling_ratio: torch.Tensor | None,  # (B, T)
        k_pos: float = 2.0,
        lambda_pos: float = 3.0,
        k_neg: float = 3.0,
        lambda_neg: float = 2.0,
    ) -> torch.Tensor:
        """
        Computes the Gamma weights for the VESPO loss. For reference:
            φ(w) = e^λ × w^k × e^{-λw} is the gamma weighting (normalized so φ(1)=1)
                with w = sequence-level importance sampling ratio
        note: we will compute φ(w) in log space

        φ(w) is detached via @torch.no_grad(), only acts as gradient scaling coefficient

        VESPO loss = -φ(w) × A × log_prob, gradient naturally gives φ(w) × A × ∇log π
        """
        # reducing clamp range directly to log(1e-8) ~ -18.42, to avoid recomputing log_w=log(w.clamp(min=1e-8)) later
        # This is solely for matching truthfully the original implementation, otherwise keeping -20 could be fine.
        lower_clamp = math.log(1e-8)

        # Sequence-level log ratio Σ log(π_θ/π_old) (not a mean like for `log_importance_weights`)
        log_ratio_clamped = torch.clamp(log_ratio_per_token, -20.0, 20.0)
        seq_log_ratio = torch.sum(
            log_ratio_clamped * mask, dim=-1, keepdim=True
        )  # (B, 1)

        # Apply token-level TIS or MIS correction (in log space)
        if importance_sampling_ratio is not None:
            log_is_ratio = torch.clamp(
                torch.log(importance_sampling_ratio), lower_clamp, 20.0
            )
            # log(w) = log(π_θ/π_old) + log(π_old/π_sampler)
            seq_log_ratio += torch.sum(log_is_ratio, dim=-1, keepdim=True)

        log_w_seq = torch.clamp(seq_log_ratio, lower_clamp, 20.0)
        w_seq = torch.exp(log_w_seq)

        # compute k and lambda based on advantage sign
        is_nonneg_adv = advantages >= 0
        k_seq = torch.where(is_nonneg_adv, k_pos, k_neg)
        lambda_seq = torch.where(is_nonneg_adv, lambda_pos, lambda_neg).clamp(min=1e-4)

        # log(φ(w)) = λ + k × log(w) - λ × w
        log_phi = lambda_seq + k_seq * log_w_seq - lambda_seq * w_seq
        phi_seq = torch.exp(log_phi).nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)

        return phi_seq  # (B, 1)

    def _compute_loss(self, model, inputs):
        # Compute the per-token log probabilities for the model
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = (
            inputs["completion_ids"],
            inputs["completion_mask"],
        )
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(
            1
        )  # we only need to compute the logits for the completion tokens
        mask = (
            completion_mask
            if "tool_mask" not in inputs
            else completion_mask * inputs["tool_mask"]
        )

        # Compute the per_token_logps and the entropy at each position in the completion
        per_token_logps, entropies = self._get_per_token_logps_and_entropies(
            model,
            input_ids,
            attention_mask,
            logits_to_keep,
            compute_entropy=True,
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            num_images=inputs.get("num_images"),
            pixel_attention_mask=inputs.get("pixel_attention_mask"),
            image_sizes=inputs.get("image_sizes"),
            token_type_ids=inputs.get("token_type_ids"),
            mm_token_type_ids=inputs.get("mm_token_type_ids"),
        )

        if self.top_entropy_quantile < 1.0:
            entropy_mask = self.get_high_entropy_mask(
                entropies, mask, 1 - self.top_entropy_quantile
            )
        else:
            entropy_mask = None

        # Compute the loss
        advantages = inputs["advantages"]
        # In the base GRPO implementation, advantages are expected to have shape (B,). To support subclasses that
        # provide advantages with shape (B, T) (e.g., MiniLLM), we *conditionally* unsqueeze the tensor.
        if advantages.dim() == 1:
            advantages = advantages.unsqueeze(1)
        # When num_iterations == 1 and steps_per_generation <= gradient_accumulation_steps,
        # old_per_token_logps == per_token_logps. In this case we can skip its computation
        # (see _generate_and_score_completions) and instead use per_token_logps.detach().
        # The exception is when using vLLM, where we always compute old_per_token_logps
        # for importance sampling
        old_per_token_logps = inputs.get("old_per_token_logps")
        old_per_token_logps = (
            per_token_logps.detach()
            if old_per_token_logps is None
            else old_per_token_logps
        )

        if self.off_policy_mask_threshold is not None:
            # OPSM should use inference-time logprobs to detect both sources of off-policyness:
            # 1. Drift from gradient updates (always present)
            # 2. Drift from training-inference mismatch (when using vLLM)
            # When using vLLM, prioritize sampling_per_token_logps, otherwise use old_per_token_logps
            sampling_per_token_logps = inputs.get(
                "sampling_per_token_logps", old_per_token_logps
            )

            off_policy_mask = self.get_off_policy_mask(
                advantages=advantages,
                per_token_logps=per_token_logps,
                sampling_per_token_logps=sampling_per_token_logps,
                mask=mask,
                off_policy_threshold=self.off_policy_mask_threshold,
            )

        log_ratio = per_token_logps - old_per_token_logps
        if self.importance_sampling_level == "token":
            log_importance_weights = log_ratio
        elif self.importance_sampling_level == "sequence":
            log_importance_weights = (log_ratio * mask).sum(-1) / mask.sum(-1).clamp(
                min=1.0
            )
            log_importance_weights = log_importance_weights.unsqueeze(-1)
        else:
            raise ValueError(
                f"Unknown importance sampling level: {self.importance_sampling_level}. Possible values are 'token' "
                "and 'sequence'."
            )

        coef_1 = torch.exp(log_importance_weights)

        # Compute the KL divergence between the model and the reference model
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps)
                - (ref_per_token_logps - per_token_logps)
                - 1
            )
            # Importance sampling correction for the KL divergence
            if self.args.use_bias_correction_kl:
                per_token_kl = per_token_kl * coef_1

        # --- OAR-G---
        if self.use_oar_g:
            oar_g_weights = self.compute_oar_g_weights(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                logits_to_keep=logits_to_keep,
                completion_ids=completion_ids,
                mask=mask,
            )
            advantages = advantages * oar_g_weights

        # --- GRAIL---
        if self.use_grail:
            unwrapped_for_grail = self.accelerator.unwrap_model(model)
            grail_weights = self.compute_grail_weights(
                model=unwrapped_for_grail,
                input_ids=input_ids,
                attention_mask=attention_mask,
                logits_to_keep=logits_to_keep,
                completion_ids=completion_ids,
                mask=mask,
                advantages=advantages,
            )
            advantages = advantages * grail_weights
            ## --- GRAIL advantage scale logging ---
            mode = "train" if self.model.training else "eval"

            raw_adv = (
                inputs["advantages"].squeeze(-1)
                if inputs["advantages"].dim() > 1
                else inputs["advantages"]
            )

            weighted_adv_valid = (advantages * mask).abs()
            valid_count = mask.sum().clamp(min=1.0)

            raw_abs_mean = raw_adv.abs().mean()
            raw_abs_max = raw_adv.abs().max()
            wtd_abs_mean = weighted_adv_valid.sum() / valid_count
            wtd_abs_max = (advantages * mask).abs().max()

            # Scale ratio: how much larger is the effective advantage after GRAIL?
            scale_ratio = wtd_abs_mean / (raw_abs_mean + 1e-8)

            self._metrics[mode]["grail/advantages_raw_abs_mean"].append(
                self.accelerator.gather(raw_abs_mean).mean().item()
            )
            self._metrics[mode]["grail/advantages_raw_abs_max"].append(
                self.accelerator.gather(raw_abs_max).max().item()
            )
            self._metrics[mode]["grail/advantages_weighted_abs_mean"].append(
                self.accelerator.gather(wtd_abs_mean).mean().item()
            )
            self._metrics[mode]["grail/advantages_weighted_abs_max"].append(
                self.accelerator.gather(wtd_abs_max).max().item()
            )
            self._metrics[mode]["grail/advantages_scale_ratio"].append(
                self.accelerator.gather(scale_ratio).mean().item()
            )

        # From here, log_importance_weights (and all subsequent tensors, coef_1, coef_2, etc.) shape depends on
        # importance_sampling_level: "token" level: (B, T); "sequence" level: (B, 1)
        if self.loss_type == "cispo":
            clamped_ratios = torch.clamp(coef_1, max=self.epsilon_high).detach()
            per_token_loss = -clamped_ratios * advantages * per_token_logps
        elif self.loss_type in ["grpo", "bnpo", "dr_grpo", "dapo", "luspo"]:
            coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
            # Two-sided clipping
            if self.args.delta is not None:
                coef_1 = torch.clamp(coef_1, max=self.args.delta)

            per_token_loss1 = coef_1 * advantages
            per_token_loss2 = coef_2 * advantages
            per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        elif self.loss_type == "sapo":
            temperatures = torch.where(
                advantages > 0,
                self.args.sapo_temperature_pos,
                self.args.sapo_temperature_neg,
            )
            soft_coef_1 = torch.sigmoid(temperatures * (coef_1 - 1)) * 4 / temperatures
            per_token_loss = -soft_coef_1 * advantages
        elif self.loss_type == "vespo":
            phi_seq = self.get_gamma_weights(
                advantages=advantages,
                log_ratio_per_token=log_ratio,
                mask=mask,
                importance_sampling_ratio=inputs.get("importance_sampling_ratio"),
                k_pos=self.args.vespo_k_pos,
                lambda_pos=self.args.vespo_lambda_pos,
                k_neg=self.args.vespo_k_neg,
                lambda_neg=self.args.vespo_lambda_neg,
            )
            per_token_loss = -phi_seq * advantages * per_token_logps
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        if self.off_policy_mask_threshold is not None:
            per_token_loss = per_token_loss * off_policy_mask

        if entropy_mask is not None:
            per_token_loss = per_token_loss * entropy_mask

        if (
            self.use_vllm
            and self.vllm_importance_sampling_correction
            and self.loss_type != "vespo"
        ):
            per_token_loss = per_token_loss * inputs["importance_sampling_ratio"]

        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        mode = "train" if self.model.training else "eval"
        if self.loss_type in ["grpo", "sapo"]:
            loss = (
                (per_token_loss * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)
            ).mean()
            normalizer = (
                self.current_gradient_accumulation_steps if mode == "train" else 1.0
            )  # no accum in eval
            loss = loss / normalizer
        elif self.loss_type == "bnpo":
            loss = (per_token_loss * mask).sum() / mask.sum().clamp(min=1.0)
            normalizer = (
                self.current_gradient_accumulation_steps if mode == "train" else 1.0
            )  # no accum in eval
            loss = loss / normalizer
        elif self.loss_type == "dr_grpo":
            loss = (per_token_loss * mask).sum() / (
                per_token_loss.size(0) * self.max_completion_length
            )
            normalizer = (
                self.current_gradient_accumulation_steps if mode == "train" else 1.0
            )  # no accum in eval
            loss = loss / normalizer
        elif self.loss_type in ["cispo", "dapo", "vespo"]:
            normalizer = inputs["num_items_in_batch"] / self.accelerator.num_processes
            loss = (per_token_loss * mask).sum() / normalizer
        elif self.loss_type == "luspo":
            # Unless importance_sampling_level="token" (not recommended here), per_token_loss is expected to be (B, 1)
            loss = (per_token_loss * mask.sum(1, keepdim=True)).mean()
            normalizer = (
                self.current_gradient_accumulation_steps if mode == "train" else 1.0
            )
            loss = loss / normalizer
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # Log the metrics
        completion_token_count = mask.sum().clamp(min=1.0)

        def masked_batch_mean(x):
            if x.shape[1] == 1:  # when importance_sampling_level == "sequence"
                return x.mean()
            else:
                return (x * mask).sum() / completion_token_count

        if self.beta != 0.0:
            mean_kl = masked_batch_mean(per_token_kl)
            self._metrics[mode]["kl"].append(
                self.accelerator.gather(mean_kl).nanmean().item()
            )

        mean_entropy = masked_batch_mean(entropies)
        self._metrics[mode]["entropy"].append(
            self.accelerator.gather(mean_entropy).nanmean().item()
        )

        if self.loss_type in ["grpo", "bnpo", "dr_grpo", "dapo", "luspo"]:
            # Compute the clipped probability ratios
            is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages < 0)
            is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (advantages > 0)
            is_region_clipped = is_low_clipped | is_high_clipped

            low_clip = masked_batch_mean(is_low_clipped.float())
            high_clip = masked_batch_mean(is_high_clipped.float())
            clip_ratio = masked_batch_mean(is_region_clipped.float())

            gathered_low_clip = self.accelerator.gather(low_clip)
            self._metrics[mode]["clip_ratio/low_mean"].append(
                gathered_low_clip.nanmean().item()
            )
            self._metrics[mode]["clip_ratio/low_min"].append(
                nanmin(gathered_low_clip).item()
            )
            gathered_high_clip = self.accelerator.gather(high_clip)
            self._metrics[mode]["clip_ratio/high_mean"].append(
                gathered_high_clip.nanmean().item()
            )
            self._metrics[mode]["clip_ratio/high_max"].append(
                nanmax(gathered_high_clip).item()
            )
            gathered_clip_ratio = self.accelerator.gather(clip_ratio)
            self._metrics[mode]["clip_ratio/region_mean"].append(
                gathered_clip_ratio.nanmean().item()
            )
        elif self.loss_type == "cispo":
            is_cispo_clipped = (coef_1 > self.epsilon_high) & (advantages > 0)
            cispo_clip_ratio = masked_batch_mean(is_cispo_clipped.float())
            gathered_cispo_clip_ratio = self.accelerator.gather(cispo_clip_ratio)
            self._metrics[mode]["cispo_clip_ratio"].append(
                gathered_cispo_clip_ratio.nanmean().item()
            )
        elif self.loss_type == "vespo":
            gathered_phi_seq = self.accelerator.gather(phi_seq)
            self._metrics[mode]["vespo/phi_seq_mean"].append(
                gathered_phi_seq.nanmean().item()
            )

        return loss

    # During eval, Trainer calls prediction_step. If no labels are present in the inputs, it only runs forward and
    # returns logits. We override prediction_step to force compute_loss, because this trainer doesn't involve labels.
    def prediction_step(
        self, model, inputs, prediction_loss_only, ignore_keys: list[str] | None = None
    ):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            loss = loss.mean().detach()
        return loss, None, None

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        mode = "train" if self.model.training else "eval"
        metrics = {
            key: sum(val) / len(val) for key, val in self._metrics[mode].items()
        }  # average the metrics

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        super().log(logs, start_time)
        self._metrics[mode].clear()

        if self.accelerator.is_main_process and self.log_completions:
            if is_rich_available():
                print_prompt_completions_sample(
                    self._logs["prompt"],
                    self._logs["completion"],
                    self._logs["rewards"],
                    self._logs["advantages"],
                    self.state.global_step,
                    self.num_completions_to_print,
                )

            logging_backends = []
            if (
                self.args.report_to
                and "wandb" in self.args.report_to
                and wandb.run is not None
            ):
                logging_backends.append(wandb)
            if self.args.report_to and "trackio" in self.args.report_to:
                logging_backends.append(trackio)

            table = {
                "step": [self.state.global_step] * len(self._logs["prompt"]),
                "prompt": self._logs["prompt"],
                "completion": self._logs["completion"],
                **self._logs["rewards"],
                **self._logs["extra"],
                "advantage": self._logs["advantages"],
            }

            df_base = pd.DataFrame(table)
            df_base.to_parquet(
                os.path.join(
                    self.args.output_dir,
                    "completions",
                    f"completions_{self.state.global_step:05d}.parquet",
                )
            )

            images_raw = self._logs["images"] or []

            for logging_backend in logging_backends:
                if images_raw:
                    images = []
                    for image_list in self._logs["images"]:
                        images.append(
                            [logging_backend.Image(image) for image in image_list]
                        )
                    df = pd.concat(
                        [df_base, pd.Series(images, name="image")],
                        axis=1,
                        copy=False,
                    )
                else:
                    df = df_base

                if self.log_unique_prompts:
                    df = df.drop_duplicates(subset=["prompt"])

                logging_backend.log(
                    {"completions": logging_backend.Table(dataframe=df)}
                )

    # Ensure the model card is saved along with the checkpoint
    def _save_checkpoint(self, model, trial):
        if self.args.hub_model_id is None:
            model_name = Path(self.args.output_dir).name
        else:
            model_name = self.args.hub_model_id.split("/")[-1]
        self.create_model_card(model_name=model_name)
        super()._save_checkpoint(model, trial)
