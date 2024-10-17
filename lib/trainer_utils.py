from collections import defaultdict
from dataclasses import dataclass
from functools import partial, reduce
from datasets import Dataset
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, TrainerCallback, TrainerControl, TrainerState
from transformers.integrations import WandbCallback
from transformers.training_args import TrainingArguments
from trl import DPOTrainer, DPOConfig

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np

from transformers.generation.configuration_utils import GenerationConfig
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from lib.data_utils import PromptAnswerDataCollator

import wandb

class Seq2SeqTrainerNoEvalLoss(Seq2SeqTrainer):
    num_tokens_seen = defaultdict(int)

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        if self.args.track_num_tokens_seen_by_task:
            task_ids = inputs.pop('task_id')
            for tid in task_ids.unique():
                main_input = inputs['input_ids'][task_ids == tid]
                attn_mask = inputs['attention_mask'][task_ids == tid]
                self.num_tokens_seen[tid.item()] += (
                    torch.sum(
                        self.accelerator.gather(
                            torch.tensor(
                                main_input.numel(), device=self.args.device, dtype=torch.int64
                            ) - torch.sum(attn_mask)
                        )
                    )
                    .cpu()
                    .item()
                )
        return super().training_step(model, inputs)
    
    def log(self, logs: Dict[str, float]) -> None:
        if self.args.track_num_tokens_seen_by_task:
            logs |= {f'tokens_seen_{tid}': self.num_tokens_seen[tid] for tid in self.num_tokens_seen}
        return super().log(logs)
    
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
            return super().prediction_step(
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
            # gen_kwargs["max_length"] = inputs['labels'].shape[1] + inputs['input_ids'].shape[1]

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

        loss_mask = generation_inputs['loss_mask']
        del generation_inputs['loss_mask']
        if self.args.do_beam_search:
            gen_kwargs['num_beams'] = self.args.num_beams
            assert self.args.num_beams > 1, "num_beams must be greater than 1 for beam search"
            gen_kwargs['early_stopping'] = self.args.early_stopping
        
        if self.args.do_backtrack_decoding or self.args.do_backtrack_eval or self.args.do_backtrack_decoding2:
            gen_kwargs['max_new_tokens'] = gen_kwargs['max_new_tokens'] * self.args.backtrack_decoding_multiplier
            gen_kwargs['eos_token_id']=self.tokenizer.eos_token_id
        
        if self.args.do_backtrack_decoding:
            backtrack_tok = self.tokenizer.backtrack_token_id
            logits_processor = BacktrackLogitsProcessor(generation_inputs['labels'], backtrack_tok, eos_tok=self.tokenizer.eos_token_id, tokenizer=self.tokenizer)
            generated_tokens = self.model.generate(**generation_inputs, **gen_kwargs, logits_processor=[logits_processor])
        elif self.args.do_backtrack_decoding2:
            backtrack_tok = self.tokenizer.backtrack_token_id
            logits_processor = BacktrackLogitsProcessorWithoutRetry(generation_inputs['labels'], backtrack_tok, eos_tok=self.tokenizer.eos_token_id, tokenizer=self.tokenizer)
            generated_tokens = self.model.generate(**generation_inputs, **gen_kwargs, logits_processor=[logits_processor])
        elif (loss_mask == 1).all():
            generated_tokens = self.model.generate(**generation_inputs, **gen_kwargs)
        else:
            logits_processor = TemplateLogitsProcessor(loss_mask, generation_inputs['labels'])
            generated_tokens = self.model.generate(**generation_inputs, **gen_kwargs, logits_processor=[logits_processor])

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

        return loss, generated_tokens, labels

@dataclass
class DPOSeq2SeqConfig(DPOConfig, Seq2SeqTrainingArguments):
    pass

class DPOTrainerDefaultEval(DPOTrainer, Seq2SeqTrainerNoEvalLoss):
    def __init__(self, *args, train_dataset=None, eval_dataset=None, **kwargs):
        dummy_train_dataset = Dataset.from_dict({})
        DPOTrainer.__init__(self, *args, train_dataset=dummy_train_dataset, eval_dataset=None, **kwargs)
        train_dataset = train_dataset.map(self.tokenize_row)
        kwargs.pop('ref_model')
        kwargs['data_collator'] = PromptAnswerDataCollator(pad_token_id=self.tokenizer.pad_token_id, label_pad_token_id=self.label_pad_token_id)
        Seq2SeqTrainerNoEvalLoss.__init__(self, *args, train_dataset=train_dataset, eval_dataset=eval_dataset, **kwargs)

    def evaluate(self, *args, **kwargs):
        self.evaluation_loop = partial(Seq2SeqTrainerNoEvalLoss.evaluation_loop, self)
        self.prediction_step = partial(Seq2SeqTrainerNoEvalLoss.prediction_step, self)
        results = Seq2SeqTrainerNoEvalLoss.evaluate(self, *args, **kwargs)
        self.evaluation_loop = partial(DPOTrainer.evaluation_loop, self)
        self.prediction_step = partial(DPOTrainer.prediction_step, self)
        return results

class AddWandbConfigCallback(WandbCallback):
    def __init__(self, extra_configs=[], **kwargs):
        super().__init__(**kwargs)
        self.extra_configs = extra_configs

    def setup(self, args, state, model, **kwargs):
        super().setup(args, state, model, **kwargs)
        new_config = reduce(lambda x, y: {**x, **y}, self.extra_configs)
        self._wandb.config.update(new_config, allow_val_change=True)

class CustomParameterLoggingCallback(TrainerCallback):
    def __init__(self, model):
        self.model = model.model 

    def on_evaluate(self, args, state, control, **kwargs):
        # Let's log beta values
        beta_i_dict = {}
        for i in range(len(self.model.layers)):
            beta_i = self.model.layers[i].self_attn.temp_beta.detach().cpu()
            beta_i_dict[f"layer{i}_beta_value"] = beta_i.item()
            wandb.log({f"layer{i}_beta_value": beta_i})
        print(beta_i_dict)

import re

class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, metric_names, thresholds, patience):
        self.patience_counter = patience
        self.metric_names = metric_names
        self.thresholds = thresholds
        self.patience = patience

    def should_stop(self, state, metrics):
        matched_keys_list = [
            [ key for key in metrics.keys() if re.search(metric_name, key) ]
        for metric_name in self.metric_names ]
        if all(
            all(metrics[key] >= threshold for key in matched_keys)
            for matched_keys, threshold in zip(matched_keys_list, self.thresholds)
        ):
            self.patience_counter -= 1
        breakpoint()

        return self.patience_counter <= 0

    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, model, metrics, **kwargs):
        control.should_training_stop = self.should_stop(state, metrics)
    
    # def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
    #     if state.total_flos >= 511_920_053_762_457_600:
    #         control.should_training_stop = True
    #         control.should_evaluate = True

class DataMixtureSchedulingCallback(TrainerCallback):
    def __init__(self, init, end, schedule='cosine', wait_before=0, wait_after=0.3):
        self.init = np.array(init)
        self.end = np.array(end)
        self.schedule = schedule
        self.wait_before = wait_before
        self.wait_after = wait_after
    
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        wait_before = self.wait_before * args.max_steps
        wait_after = self.wait_after * args.max_steps
        total_steps = max(1, args.max_steps - wait_before - wait_after)
        step = np.clip(state.global_step - wait_before, 0, total_steps)
        if self.schedule == 'linear':
            mix = self.init + (self.end - self.init) * step / total_steps
        elif self.schedule == 'cosine':
            mix = self.end + (self.init - self.end) * (1 + np.cos(np.pi * step / total_steps)) / 2

        mix = mix / mix.sum()
        mix[np.argmax(mix)] += 1 - mix.sum()
        kwargs['train_dataloader'].dataset._ex_iterable.probabilities[:] = mix

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        mix = kwargs['train_dataloader'].dataset._ex_iterable.probabilities
        logs |= {f'mix_{i}': round(p, 3) for i, p in enumerate(mix)}

from transformers import Constraint, LogitsProcessor

class TemplateConstraint(Constraint):
    def __init__(self, template: List[int]):
        self.template = template
        self.count = 0
        self.test()
    
    def advance(self):
        tok = self.template[self.count]
        return None if tok == -100 else tok
    
    def does_advance(self, token_id: int):
        tok = self.template[self.count]
        return token_id == tok or tok == -100

    def update(self, token_id: int):
        stepped = self.does_advance(token_id)
        if stepped:
            self.count += 1
            completed = self.count == len(self.template)
            reset = False
        else:
            completed = False
            self.reset()
            reset = True
        
        return stepped, completed, reset

    def reset(self):
        self.count = 0
    
    def remaining(self):
        return len(self.template) - self.count

    def copy(self, stateful=False):
        c = TemplateConstraint(self.template)
        if stateful:
            c.count = self.count
        return c

class TemplateLogitsProcessor(LogitsProcessor):
    def __init__(self, loss_mask: torch.LongTensor, labels: torch.LongTensor):
        self.force_mask = loss_mask == 0
        self.labels = labels
        self.count = 0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        mask = self.force_mask[:, self.count] if self.count < self.force_mask.shape[1] else torch.zeros_like(self.force_mask[:, 0])
        labels = self.labels[:, self.count]
        scores[mask] = scores[mask].scatter(1, labels[mask][:, None], torch.inf)
        self.count += 1
        return scores

class BacktrackLogitsProcessor(LogitsProcessor):
    def __init__(self, labels, backtrack_tok, eos_tok, tokenizer=None):
        self.labels = labels
        self.backtrack_tok = torch.tensor(backtrack_tok)
        self.eos_tok = torch.tensor(eos_tok)
        self.count = 0
        self.label_count = torch.zeros(labels.shape[0], dtype=torch.long)
        self.tokenizer = tokenizer

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # if input_ids.shape[0] != self.labels.shape[0] (in the case of beam search), we repeat the labels
        if input_ids.shape[0] != self.labels.shape[0]:
            self.labels = self.labels.repeat_interleave(input_ids.shape[0] // self.labels.shape[0], dim=0)
            self.label_count = self.label_count.repeat_interleave(input_ids.shape[0] // self.label_count.shape[0], dim=0)

        # print(self.tokenizer.decode(input_ids[0]))
        # import ipdb; ipdb.set_trace()
        if self.count > 0:
            labels = self.labels[torch.arange(self.labels.shape[0]), self.label_count.clip(max=self.labels.shape[1]-1)]
            input_ids = input_ids[:, -1]
            
            self.label_count += (labels == input_ids).to(self.label_count.device) # when the model generates the correct token, increment the label count
            
            ended = self.label_count.to(input_ids.device) >= self.labels.shape[1] -1

            labels[ended] = input_ids[ended] # If there are no more labels to match, don't force anything
            mask = (labels != input_ids) & (input_ids != self.backtrack_tok)
            
            if mask.any():
                # when the model generates wrong tokens that aren't backtrack tokens, force a backtrack token
                scores[mask] = scores[mask].scatter(1, self.backtrack_tok.to(input_ids.device).expand_as(scores[mask]), 9e9)
                # If the model hasn't generated all the labels, prevent [EOS] from being generated
                scores[~ended] = scores[~ended].scatter(1, self.eos_tok.to(input_ids.device).expand_as(scores[~ended]), -9e9)

            # if end is reached, force [EOS] to be generated
            scores[ended] = scores[ended].scatter(1, self.eos_tok.to(input_ids.device).expand_as(scores[ended]), 9e9)

            # self.label_count += (labels == input_ids).to(self.label_count.device) # when the model generates the correct token, increment the label count

        self.count += 1

        return scores


class BacktrackLogitsProcessorWithoutRetry(LogitsProcessor):
    def __init__(self, labels, backtrack_tok, eos_tok, tokenizer=None):
        self.labels = labels # (batch_size, sequence_length):  the expected target tokens for the batch.
        self.backtrack_tok = torch.tensor(backtrack_tok)
        self.eos_tok = torch.tensor(eos_tok)
        self.count = 0
        self.label_count = torch.zeros(labels.shape[0], dtype=torch.long) # (batch_size)
        # Keep track of incorrect tokens generated for each sequence
        self.incorrect_tokens = [[] for _ in range(labels.shape[0])]
        self.tokenizer = tokenizer # for debugging purposes (used to decode tokens)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        device = input_ids.device  # Get the device of input_ids to ensure consistency
        self.label_count = self.label_count.to(device)  # Move label_count to the correct device

        # input_ids: (batch_size, current_sequence_length): the tokens generated by the model up to the current step
        
        # Ensure labels and label_count match input_ids shape (for beam search)
        if input_ids.shape[0] != self.labels.shape[0]:
            self.labels = self.labels.repeat_interleave(input_ids.shape[0] // self.labels.shape[0], dim=0).to(device)
            self.label_count = self.label_count.repeat_interleave(input_ids.shape[0] // self.label_count.shape[0], dim=0).to(device)
            self.incorrect_tokens = [self.incorrect_tokens[i % len(self.incorrect_tokens)] for i in range(input_ids.shape[0])]

        # import ipdb; ipdb.set_trace()
        if self.count > 0:            
            # Get current expected label for each sequence
            labels = self.labels[torch.arange(self.labels.shape[0], device=device), self.label_count.clip(max=self.labels.shape[1]-1)]
            input_ids = input_ids[:, -1]
            
            # Case 3: Correct token generated
            # ipdb.set_trace()
            correct_mask = (labels == input_ids)
            if correct_mask.any():
                # Correct token generated, increment label count
                self.label_count[correct_mask] += 1
                # Clear the list of incorrect tokens for those sequences since correct token was hit
                for i in torch.where(correct_mask)[0]:
                    self.incorrect_tokens[i] = []  # Reset incorrect tokens after hitting the correct label
                # surpress the backtrack token
                scores[correct_mask] = scores[correct_mask].scatter(1, self.backtrack_tok.to(input_ids.device).expand_as(scores[correct_mask]), -9e9)

            ended = self.label_count >= self.labels.shape[1] -1  # boolean mask for sequences that have exhausted their labels
            
            labels[ended] = input_ids[ended]  # Don't force labels if we've exhausted them
            mask = (labels != input_ids) & (input_ids != self.backtrack_tok) # (batch_size) boolean mask for incorrect tokens that aren't backtrack tokens

            # Case 1: Generated token is incorrect and not backtrack token
            if mask.any():
                # When the model generates a wrong token that's not the backtrack token, force a backtrack token
                scores[mask] = scores[mask].scatter(1, self.backtrack_tok.to(input_ids.device).expand_as(scores[mask]), 9e9)
                # If the model hasn't generated all the labels, prevent [EOS] from being generated
                scores[~ended] = scores[~ended].scatter(1, self.eos_tok.to(input_ids.device).expand_as(scores[~ended]), -9e9)

                # Update the list of incorrect tokens for sequences that generated incorrect tokens
                for i in range(input_ids.shape[0]):
                    if mask[i]:
                        self.incorrect_tokens[i].append(input_ids[i].item())  # Record the incorrect token

            # Case 2: Suppress previously incorrect tokens when backtrack token is generated
            for i in torch.where(input_ids == self.backtrack_tok)[0]:
                for incorrect_tok in self.incorrect_tokens[i]:
                    # Make the logits for previously incorrect tokens very low (suppress them)
                    scores[i] = scores[i].scatter(0, torch.tensor([incorrect_tok], device=device), -9e9)

        self.count += 1
        return scores
