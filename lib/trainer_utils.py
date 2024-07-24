from functools import reduce
from transformers import Seq2SeqTrainer
from transformers.integrations import WandbCallback

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch import nn

from transformers.generation.configuration_utils import GenerationConfig
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled

class Seq2SeqTrainerNoEvalLoss(Seq2SeqTrainer):
    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
        **gen_kwargs,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            gen_kwargs:
                Additional `generate` specific kwargs.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """

        if not self.args.predict_with_generate or prediction_loss_only:
            return super().super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        # Priority (handled in generate):
        # non-`None` gen_kwargs > model.generation_config > default GenerationConfig()
        if len(gen_kwargs) == 0 and hasattr(self, "_gen_kwargs"):
            gen_kwargs = self._gen_kwargs.copy()
        if "num_beams" in gen_kwargs and gen_kwargs["num_beams"] is None:
            gen_kwargs.pop("num_beams")
        if "max_length" in gen_kwargs and gen_kwargs["max_length"] is None:
            gen_kwargs.pop("max_length")

        if has_labels:
            # Don't constrain the min new tokens because the label might be shorter than max
            gen_kwargs["max_new_tokens"] = inputs['labels'].shape[1]

        default_synced_gpus = True if is_deepspeed_zero3_enabled() else False
        gen_kwargs["synced_gpus"] = (
            gen_kwargs["synced_gpus"] if gen_kwargs.get("synced_gpus") is not None else default_synced_gpus
        )

        generation_inputs = inputs.copy()
        # If the `decoder_input_ids` was created from `labels`, evict the former, so that the model can freely generate
        # (otherwise, it would continue generating from the padded `decoder_input_ids`)
        if (
            "labels" in generation_inputs
            and "decoder_input_ids" in generation_inputs
            and generation_inputs["labels"].shape == generation_inputs["decoder_input_ids"].shape
        ):
            generation_inputs = {
                k: v for k, v in inputs.items() if k not in ("decoder_input_ids", "decoder_attention_mask")
            }
        generated_tokens = self.model.generate(**generation_inputs, **gen_kwargs)

        # Temporary hack to ensure the generation config is not initialized for each iteration of the evaluation loop
        # TODO: remove this hack when the legacy code that initializes generation_config from a model config is
        # removed in https://github.com/huggingface/transformers/blob/98d88b23f54e5a23e741833f1e973fdf600cc2c5/src/transformers/generation/utils.py#L1183
        if self.model.generation_config._from_model_config:
            self.model.generation_config._from_model_config = False

        # Retrieves GenerationConfig from model.generation_config
        gen_config = self.model.generation_config
        # in case the batch is shorter than max length, the output should be padded
        max_length = max(
            max(gen_config.max_length or 0, gen_kwargs.get("max_length", 0)),
            generation_inputs['input_ids'].shape[-1] + max(gen_config.max_new_tokens or 0, gen_kwargs.get('max_new_tokens', 0))
        )
        if generated_tokens.shape[-1] < max_length:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, max_length)

        # with torch.no_grad():
        #     if has_labels:
        #         with self.compute_loss_context_manager():
        #             outputs = model(**inputs)
        #         if self.label_smoother is not None:
        #             loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
        #         else:
        #             loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
        #     else:
        #         loss = None
        loss = None

        if self.args.prediction_loss_only:
            return loss, None, None

        if has_labels:
            labels = inputs["labels"]
            # if labels.shape[-1] < gen_config.max_length:
            #     labels = self._pad_tensors_to_max_len(labels, gen_config.max_length)
            # elif gen_config.max_new_tokens is not None and labels.shape[-1] < gen_config.max_new_tokens + 1:
            #     labels = self._pad_tensors_to_max_len(labels, gen_config.max_new_tokens + 1)
        else:
            labels = None

        if torch.any(generated_tokens == -100):
            breakpoint()
        return loss, generated_tokens, labels


class AddWandbConfigCallback(WandbCallback):
    def __init__(self, extra_configs=[], **kwargs):
        super().__init__(**kwargs)
        self.extra_configs = extra_configs

    def setup(self, args, state, model, **kwargs):
        super().setup(args, state, model, **kwargs)
        new_config = reduce(lambda x, y: {**x, **y}, self.extra_configs)
        self._wandb.config.update(new_config, allow_val_change=True)
