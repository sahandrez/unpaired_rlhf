"""
RLOO trainer that uses unpaired preferences. 

Code adapted from the TRL library:
https://github.com/huggingface/trl/blob/v0.13.0/trl/trainer/rloo_trainer.py
"""

from typing import Optional, Union

import gc
import math
import time
from collections import defaultdict
from typing import Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from accelerate.utils import gather_object
from transformers import (
    BaseImageProcessor,
    DataCollatorWithPadding,
    FeatureExtractionMixin,
    GenerationConfig,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    TrainerCallback,
)
from trl import (
    RLOOTrainer,
    RLOOConfig,
)
from trl.models.utils import unwrap_model_for_generation
from trl.trainer.utils import (
    batch_generation,
    first_true_indices,
    forward,
    get_reward,
    print_rich_table,
    truncate_response,
    log_table_to_comet_experiment
)

INVALID_LOGPROB = 1.0


class UnpairedRLOOTrainer(RLOOTrainer):
    def __init__(
        self,
        config: RLOOConfig,
        processing_class: Optional[
            Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin]
        ],
        policy: nn.Module,
        ref_policy: nn.Module,
        reward_model: nn.Module,
        train_dataset: Dataset,
        data_collator: Optional[DataCollatorWithPadding] = None,
        eval_dataset: Optional[Union[Dataset, dict[str, Dataset]]] = None,
        # less commonly used
        optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        callbacks: Optional[list[TrainerCallback]] = None,
    ) -> None:
         super().__init__(
            config=config,
            processing_class=processing_class,
            policy=policy,
            ref_policy=ref_policy,
            reward_model=reward_model,
            train_dataset=train_dataset,
            data_collator=data_collator,
            eval_dataset=eval_dataset,
            optimizers=optimizers,
            callbacks=callbacks,
        )
    
    def train(self):
        args = self.args
        accelerator = self.accelerator
        optimizer = self.optimizer
        model = self.model
        self.model_wrapped = self.model
        ref_policy = self.ref_policy
        reward_model = self.reward_model
        processing_class = self.processing_class
        dataloader = self.dataloader
        device = accelerator.device

        def repeat_generator():
            while True:
                yield from dataloader

        iter_dataloader = iter(repeat_generator())
        generation_config = GenerationConfig(
            max_new_tokens=args.response_length,
            temperature=(args.temperature + 1e-7),
            top_k=0.0,
            top_p=1.0,
            do_sample=True,
        )

        accelerator.print("===training policy===")
        start_time = time.time()
        stats_shape = (args.num_ppo_epochs, args.num_mini_batches, args.gradient_accumulation_steps)
        approxkl_stats = torch.zeros(stats_shape, device=device)
        pg_clipfrac_stats = torch.zeros(stats_shape, device=device)
        pg_loss_stats = torch.zeros(stats_shape, device=device)
        vf_clipfrac_stats = torch.zeros(stats_shape, device=device)
        entropy_stats = torch.zeros(stats_shape, device=device)
        ratio_stats = torch.zeros(stats_shape, device=device)
        model.train()

        # trainer state initialization
        self.state.global_step = 0
        self.state.episode = 0
        self.state.max_steps = (args.num_total_batches * args.num_mini_batches) // 2
        self.state.num_train_epochs = args.total_episodes / self.train_dataset_len
        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps is not None:
            if args.logging_steps < 1:
                self.state.logging_steps = math.ceil(self.state.max_steps * args.logging_steps)
            else:
                self.state.logging_steps = args.logging_steps
        if args.eval_steps is not None:
            if args.eval_steps < 1:
                self.state.eval_steps = math.ceil(self.state.max_steps * args.eval_steps)
            else:
                self.state.eval_steps = args.eval_steps
        if args.save_steps is not None:
            if args.save_steps < 1:
                self.state.save_steps = math.ceil(self.state.max_steps * args.save_steps)
            else:
                self.state.save_steps = args.save_steps
        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        for update in range(1, args.num_total_batches + 1):
            self.state.episode += 1 * args.batch_size
            data = next(iter_dataloader)
            with torch.no_grad():
                queries = data["input_ids"].to(device)
                queries = queries.repeat(args.rloo_k, 1)
                context_length = queries.shape[1]
                responses = []
                postprocessed_responses = []
                logprobs = []
                ref_logprobs = []
                scores = []
                sequence_lengths = []
                with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
                    query_responses, logitss = batch_generation(
                        unwrapped_model,
                        queries,
                        args.local_rollout_forward_batch_size,
                        processing_class.pad_token_id,
                        generation_config,
                    )

                for i in range(0, queries.shape[0], args.local_rollout_forward_batch_size):
                    query = queries[i : i + args.local_rollout_forward_batch_size]
                    query_response = query_responses[i : i + args.local_rollout_forward_batch_size]
                    response = query_response[:, context_length:]
                    logits = logitss[i : i + args.local_rollout_forward_batch_size]
                    all_logprob = F.log_softmax(logits, dim=-1)
                    logprob = torch.gather(all_logprob, 2, response.unsqueeze(-1)).squeeze(-1)
                    del logits, all_logprob
                    torch.cuda.empty_cache()

                    ref_output = forward(ref_policy, query_response, processing_class.pad_token_id)
                    ref_logits = ref_output.logits[:, context_length - 1 : -1]
                    ref_logits /= args.temperature + 1e-7
                    ref_all_logprob = F.log_softmax(ref_logits, dim=-1)
                    ref_logprob = torch.gather(ref_all_logprob, 2, response.unsqueeze(-1)).squeeze(-1)
                    del ref_output, ref_logits, ref_all_logprob
                    torch.cuda.empty_cache()

                    # Response Processing 1. truncate response after the first occurrence of `stop_token_id`
                    postprocessed_response = response
                    if args.stop_token_id is not None:  # handle the edge case when stop_token_id exists but is 0
                        postprocessed_response = truncate_response(
                            args.stop_token_id, processing_class.pad_token_id, response
                        )

                    # Response Processing 2. run reward model on the truncated responses
                    postprocessed_query_response = torch.cat((query, postprocessed_response), 1)
                    sequence_length = first_true_indices(postprocessed_response == processing_class.pad_token_id) - 1
                    _, score, _ = get_reward(
                        reward_model, postprocessed_query_response, processing_class.pad_token_id, context_length
                    )

                    responses.append(response)
                    postprocessed_responses.append(postprocessed_response)
                    logprobs.append(logprob)
                    ref_logprobs.append(ref_logprob)
                    sequence_lengths.append(sequence_length)
                    scores.append(score)
                responses = torch.cat(responses, 0)
                postprocessed_responses = torch.cat(postprocessed_responses, 0)
                logprobs = torch.cat(logprobs, 0)
                ref_logprobs = torch.cat(ref_logprobs, 0)
                sequence_lengths = torch.cat(sequence_lengths, 0)
                scores = torch.cat(scores, 0)
                del (logprob, ref_logprob, score)
                torch.cuda.empty_cache()
                gc.collect()

                # Response Processing 3. filter response. Ensure that the sample contains stop_token_id
                # responses not passing that filter will receive a low (fixed) score
                # only query humans on responses that pass that filter
                contain_eos_token = torch.any(postprocessed_responses == processing_class.eos_token_id, dim=-1)
                if args.missing_eos_penalty is not None:
                    scores[~contain_eos_token] -= self.args.missing_eos_penalty
                # accelerator.print(f"{scores=}, {(contain_eos_token.sum() / len(contain_eos_token))=}")

                # be very careful with `padding_mask_p1`; see https://excalidraw.com/#json=LWnzG4w2k5DjF_EOL_xPt,e2w3a-hFJ_gX5vOfeyXGTw
                response_idxs = torch.arange(responses.shape[1], device=responses.device).repeat(responses.shape[0], 1)
                padding_mask = response_idxs > sequence_lengths.unsqueeze(1)
                logprobs = torch.masked_fill(logprobs, padding_mask, INVALID_LOGPROB)
                ref_logprobs = torch.masked_fill(ref_logprobs, padding_mask, INVALID_LOGPROB)

                # 4. compute rewards
                kl = logprobs - ref_logprobs
                non_score_reward = (-args.kl_coef * kl).sum(1)
                # the scores are 0/1 (thumbsed up or down)
                scores = scores.argmax(dim=1).float()
                rlhf_reward = scores + non_score_reward

                # vectorized RLOO advantages implementation
                rlhf_reward = rlhf_reward.reshape(args.rloo_k, -1)
                baseline = (rlhf_reward.sum(0) - rlhf_reward) / (args.rloo_k - 1)
                advantages = rlhf_reward - baseline
                advantages = advantages.flatten()
                torch.cuda.empty_cache()

            # Do multiple epochs of PPO training, with a fresh random shuffle in each epoch
            for ppo_epoch_idx in range(args.num_ppo_epochs):
                b_inds = np.random.permutation(args.local_batch_size)
                minibatch_idx = 0
                for mini_batch_start in range(0, args.local_batch_size, args.local_mini_batch_size):
                    mini_batch_end = mini_batch_start + args.local_mini_batch_size
                    mini_batch_inds = b_inds[mini_batch_start:mini_batch_end]
                    gradient_accumulation_idx = 0
                    for micro_batch_start in range(0, args.local_mini_batch_size, args.per_device_train_batch_size):
                        with accelerator.accumulate(model):
                            micro_batch_end = micro_batch_start + args.per_device_train_batch_size
                            micro_batch_inds = mini_batch_inds[micro_batch_start:micro_batch_end]
                            mb_advantage = advantages[micro_batch_inds]
                            mb_responses = responses[micro_batch_inds]
                            mb_query_responses = query_responses[micro_batch_inds]
                            mb_logprobs = logprobs[micro_batch_inds]

                            output = forward(model, mb_query_responses, processing_class.pad_token_id)
                            logits = output.logits[:, context_length - 1 : -1]
                            logits /= args.temperature + 1e-7
                            new_all_logprobs = F.log_softmax(logits, dim=-1)
                            new_logprobs = torch.gather(new_all_logprobs, 2, mb_responses.unsqueeze(-1)).squeeze(-1)
                            new_logprobs = torch.masked_fill(
                                new_logprobs, padding_mask[micro_batch_inds], INVALID_LOGPROB
                            )
                            new_ratio = (new_logprobs - mb_logprobs).exp()
                            new_logprobs = new_logprobs.sum(1)
                            mb_logprobs = mb_logprobs.sum(1)
                            logprobs_diff = new_logprobs - mb_logprobs
                            ratio = torch.exp(logprobs_diff)
                            pg_losses = -mb_advantage * ratio
                            pg_losses2 = -mb_advantage * torch.clamp(ratio, 1.0 - args.cliprange, 1.0 + args.cliprange)
                            pg_loss_max = torch.max(pg_losses, pg_losses2)
                            pg_loss = pg_loss_max.mean()
                            loss = pg_loss
                            accelerator.backward(loss)
                            optimizer.step()
                            optimizer.zero_grad()
                            with torch.no_grad():
                                pg_clipfrac = (pg_losses2 > pg_losses).float().mean()
                                prob_dist = torch.nn.functional.softmax(logits, dim=-1)
                                entropy = torch.logsumexp(logits, dim=-1) - torch.sum(prob_dist * logits, dim=-1)
                                approxkl = 0.5 * (logprobs_diff**2).mean()
                                approxkl_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = approxkl
                                pg_clipfrac_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = (
                                    pg_clipfrac
                                )
                                pg_loss_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = pg_loss
                                entropy_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = entropy.mean()
                                ratio_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = new_ratio.mean()
                        gradient_accumulation_idx += 1
                    minibatch_idx += 1
                    # del everything and empty cache
                    # fmt: off
                    del (
                        output, logits, new_all_logprobs, new_logprobs,
                        logprobs_diff, ratio, pg_losses, pg_losses2,
                        pg_loss, loss, pg_clipfrac, prob_dist, entropy, approxkl,
                        mb_advantage, mb_responses, mb_query_responses, mb_logprobs,
                    )
                    # fmt: on
                    torch.cuda.empty_cache()
            with torch.no_grad():
                mean_kl = kl.sum(1).mean()
                mean_entropy = (-logprobs).sum(1).mean()
                mean_non_score_reward = non_score_reward.mean()
                eps = int(self.state.episode / (time.time() - start_time))
                metrics = {}
                metrics["eps"] = eps
                metrics["objective/kl"] = self.accelerator.gather(mean_kl).mean().item()
                metrics["objective/entropy"] = self.accelerator.gather(mean_entropy).mean().item()
                metrics["objective/non_score_reward"] = self.accelerator.gather(mean_non_score_reward).mean().item()
                metrics["objective/rlhf_reward"] = self.accelerator.gather(rlhf_reward).mean().item()
                metrics["objective/scores"] = self.accelerator.gather(scores.mean()).mean().item()
                metrics["policy/approxkl_avg"] = self.accelerator.gather(approxkl_stats).mean().item()
                metrics["policy/clipfrac_avg"] = self.accelerator.gather(pg_clipfrac_stats).mean().item()
                metrics["loss/policy_avg"] = self.accelerator.gather(pg_loss_stats).mean().item()
                metrics["val/clipfrac_avg"] = self.accelerator.gather(vf_clipfrac_stats).mean().item()
                metrics["policy/entropy_avg"] = self.accelerator.gather(entropy_stats).mean().item()
                metrics["val/ratio"] = self.accelerator.gather(ratio_stats).mean().item()
                metrics["val/ratio_var"] = self.accelerator.gather(ratio_stats).var().item()
                metrics["val/num_eos_tokens"] = (responses == processing_class.eos_token_id).sum().item()
                metrics["lr"] = self.lr_scheduler.get_last_lr()[0]
                metrics["episode"] = self.state.episode
                self.state.epoch = self.state.episode / (args.rloo_k * self.train_dataset_len)  # used by self.log
                self.log(metrics)
            del kl, mean_kl, mean_entropy, scores

            self.lr_scheduler.step()
            self.state.global_step += 1
            self.control = self.callback_handler.on_step_end(args, self.state, self.control)
            if self.control.should_save:
                self._save_checkpoint(model, trial=None)
                self.control = self.callback_handler.on_save(self.args, self.state, self.control)
            torch.cuda.empty_cache()
            gc.collect()

            if args.num_sample_generations > 0 and (update - 1) % self.sample_generations_freq == 0:
                self.generate_completions(sampling=True)

        # HF trainer specifics
        self.control = self.callback_handler.on_train_end(args, self.state, self.control)
        if self.control.should_save:
            self._save_checkpoint(model, trial=None, metrics=None)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    def generate_completions(self, sampling: bool = False):
        args = self.args
        processing_class = self.processing_class
        generation_config = GenerationConfig(
            max_new_tokens=self.args.response_length,
            temperature=(0.01 + 1e-7),
            top_k=0.0,
            top_p=1.0,
            do_sample=True,
        )

        table = defaultdict(list)
        with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
            for batch in self.eval_dataloader:
                query = batch["input_ids"]
                with torch.no_grad():
                    context_length = query.shape[1]
                    query_response, _ = batch_generation(
                        unwrapped_model,
                        query,
                        query.shape[0],
                        processing_class.pad_token_id,
                        generation_config,
                    )
                    response = query_response[:, context_length:]
                    postprocessed_response = response
                    if args.stop_token_id is not None:  # handle the edge case when stop_token_id exists but is 0
                        postprocessed_response = truncate_response(
                            args.stop_token_id, processing_class.pad_token_id, response
                        )
                    table["query"].extend(
                        gather_object(processing_class.batch_decode(query, skip_special_tokens=True))
                    )
                    table["model response"].extend(
                        gather_object(processing_class.batch_decode(postprocessed_response))
                    )

                    postprocessed_query_response = torch.cat((query, postprocessed_response), 1)
                    _, score, _ = get_reward(
                        self.reward_model, postprocessed_query_response, processing_class.pad_token_id, context_length
                    )
                    table["score"].extend(self.accelerator.gather(score).float().cpu().numpy())

                if sampling:
                    break
        df = pd.DataFrame(table)

        if self.accelerator.is_main_process:
            print_rich_table(df.iloc[0 : 0 + 5])
            if "wandb" in args.report_to:
                import wandb

                if wandb.run is not None:
                    wandb.log({"completions": wandb.Table(dataframe=df)})

            if "comet_ml" in args.report_to:
                log_table_to_comet_experiment(
                    name="completions.csv",
                    table=df,
                )
