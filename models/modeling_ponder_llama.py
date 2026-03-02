# coding=utf-8
# Copyright 2024 PonderLM authors and the HuggingFace Inc. team. All rights reserved.
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
PonderLM implemented on top of LLaMA architecture.

This module ports the PonderLM iterative embedding refinement idea from the
original GPT-NeoX-based implementation to the LLaMA framework, enabling fair
comparison with other LLaMA-based models using the same tokenizer and vocab.
"""

from typing import List, Optional, Tuple, Union

import math
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from transformers.utils.generic import check_model_inputs

from transformers.models.llama.modeling_llama import LlamaModel, LlamaForCausalLM
from .configuration_ponder_llama import PonderLlamaConfig


logger = logging.get_logger(f"transformers.{__name__}")


class PonderLlamaModel(LlamaModel):
    """
    PonderLlama base model — identical to LlamaModel.
    The pondering logic lives entirely in PonderLlamaForCausalLM.
    """

    config_class = PonderLlamaConfig

    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            use_cache=use_cache,
            **kwargs,
        )


class PonderLlamaForCausalLM(LlamaForCausalLM):
    """
    PonderLM on LLaMA architecture with iterative embedding refinement.

    The core idea: instead of a single forward pass, PonderLM performs multiple
    passes through the transformer to iteratively refine "latent thought"
    embeddings that are interleaved with the original token embeddings.

    Training flow:
        1. Embed input tokens, optionally scale by sqrt(2.5 * H).
        2. Initial interpolation loop (num_latent_thoughts iterations):
           - Interleave current embedding stages.
           - Pass through transformer.
           - Extract hidden states to build new interpolated stages.
        3. Refinement loop (num_jacobi_iterations):
           - Interleave all stages, pass through transformer.
           - Update each interpolated stage with new hidden states.
        4. Final pass: interleave all stages, get logits from target stage.

    Inference flow (with KV cache):
        - Prefilling: same as training logic, then cache the KV states.
        - Decoding: two-pass per token (first gets pondered embedding,
          second generates final output).
    """

    config_class = PonderLlamaConfig

    def __init__(self, config: PonderLlamaConfig):
        # Call grandparent init to avoid LlamaForCausalLM creating a standard LlamaModel
        super(LlamaForCausalLM, self).__init__(config)
        self.model = PonderLlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    # ------------------------------------------------------------------ #
    #  Helper: interleave multiple embedding stages for joint processing  #
    # ------------------------------------------------------------------ #
    def prepare_inputs_for_stages(
        self,
        current_stages_list: List[torch.Tensor],
        batch_size: int,
        seq_len_orig: int,
        hidden_size: int,
        device: torch.device,
        orig_pos_ids: torch.Tensor,
        orig_attn_mask_2d: torch.Tensor,
    ):
        """
        Interleave N embedding stages into a single sequence of length N * seq_len.

        Given stages [S0, S1, ..., S_{N-1}] each of shape (B, L, H),
        produces an interleaved sequence where positions are:
            S0[0], S1[0], ..., S_{N-1}[0], S0[1], S1[1], ...

        Returns:
            input_embeds: (B, N*L, H)
            position_ids: (B, N*L) — each stage gets the original position ids
            attention_mask: (B, N*L)
        """
        num_stages = len(current_stages_list)
        interleaved_len = num_stages * seq_len_orig

        input_embeds = torch.empty(
            (batch_size, interleaved_len, hidden_size),
            dtype=current_stages_list[0].dtype,
            device=device,
        )
        pos_ids = torch.empty(
            (batch_size, interleaved_len),
            dtype=orig_pos_ids.dtype,
            device=device,
        )
        attn_mask = torch.empty(
            (batch_size, interleaved_len),
            dtype=orig_attn_mask_2d.dtype,
            device=device,
        )

        for stage_idx in range(num_stages):
            input_embeds[:, stage_idx::num_stages, :] = current_stages_list[stage_idx]
            pos_ids[:, stage_idx::num_stages] = orig_pos_ids
            attn_mask[:, stage_idx::num_stages] = orig_attn_mask_2d

        return input_embeds, pos_ids, attn_mask

    # ------------------------------------------------------------------ #
    #  Helper: extract hidden states for a specific stage                 #
    # ------------------------------------------------------------------ #
    @staticmethod
    def extract_hidden_for_stage(model_outputs, selector_slice):
        """Extract hidden states corresponding to a specific stage."""
        return model_outputs.last_hidden_state[:, selector_slice, :]

    # ------------------------------------------------------------------ #
    #  Helper: run backbone model                                         #
    # ------------------------------------------------------------------ #
    def _run_backbone(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        cache_position: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        """Run the LLaMA backbone model."""
        return self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )

    # ------------------------------------------------------------------ #
    #  Main forward                                                       #
    # ------------------------------------------------------------------ #
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        PonderLM forward pass with iterative embedding refinement.

        Training: multi-pass pondering with interleaved embedding stages.
        Inference (with past_key_values): two-pass pondering per token.
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = kwargs.get("output_attentions", False)
        output_hidden_states = kwargs.get("output_hidden_states", False)

        # ============================================================== #
        #  INFERENCE PATH (past_key_values is not None)                   #
        # ============================================================== #
        if past_key_values is not None:
            is_decoding = len(past_key_values) > 0

            if is_decoding:
                # --- Decoding phase: single token with pondering ---
                if inputs_embeds is None:
                    if input_ids is None:
                        raise ValueError("input_ids must be provided")
                    inputs_embeds = self.model.embed_tokens(input_ids)

                B, L_orig, H = inputs_embeds.shape
                device = inputs_embeds.device

                if self.config.scale_embeds:
                    embed_scale = torch.sqrt(
                        2.5 * torch.tensor(H, dtype=inputs_embeds.dtype, device=device)
                    )
                    inputs_embeds = inputs_embeds * embed_scale

                # First pass: get pondered embedding
                first_pass = self._run_backbone(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    cache_position=cache_position,
                )
                pondered_embeds = first_pass.last_hidden_state
                updated_kv = first_pass.past_key_values

                # Second pass: generate output using pondered embedding
                second_pass = self._run_backbone(
                    inputs_embeds=pondered_embeds,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=updated_kv,
                    use_cache=True,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    cache_position=cache_position,
                )

                logits = self.lm_head(second_pass.last_hidden_state)

                return CausalLMOutputWithPast(
                    loss=None,
                    logits=logits,
                    past_key_values=second_pass.past_key_values,
                    hidden_states=second_pass.hidden_states,
                    attentions=second_pass.attentions,
                )

            else:
                # --- Prefilling phase: process entire prompt ---
                return self._forward_pondering(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    inputs_embeds=inputs_embeds,
                    labels=None,  # No loss during inference
                    use_cache=True,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    cache_position=cache_position,
                    is_inference_prefill=True,
                    **kwargs,
                )

        # ============================================================== #
        #  TRAINING PATH                                                  #
        # ============================================================== #
        return self._forward_pondering(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            is_inference_prefill=False,
            **kwargs,
        )

    def _forward_pondering(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        is_inference_prefill: bool = False,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """
        Core pondering logic shared by training and inference-prefilling.

        Steps:
            1. Embed and optionally scale.
            2. Initial interpolation loop to build latent thought stages.
            3. Jacobi refinement loop to iteratively improve stages.
            4. Final pass to compute logits.
        """
        # --- Step 0: Get initial embeddings ---
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("input_ids must be provided")
            initial_embeds_raw = self.model.embed_tokens(input_ids)
        else:
            initial_embeds_raw = inputs_embeds

        B, L_orig, H = initial_embeds_raw.shape
        device = initial_embeds_raw.device

        # Prepare position_ids and attention_mask (2D)
        if position_ids is None:
            orig_pos_ids = torch.arange(
                0, L_orig, dtype=torch.long, device=device
            ).unsqueeze(0).expand(B, L_orig)
        else:
            orig_pos_ids = position_ids

        if attention_mask is None:
            orig_attn_mask_2d = torch.ones((B, L_orig), dtype=torch.long, device=device)
        else:
            if attention_mask.ndim == 4:
                orig_attn_mask_2d = attention_mask[:, 0, 0, :L_orig]
            elif attention_mask.ndim == 3:
                orig_attn_mask_2d = attention_mask[:, 0, :L_orig]
            elif attention_mask.ndim == 2:
                orig_attn_mask_2d = attention_mask
            else:
                raise ValueError(f"Unsupported attention_mask dimension: {attention_mask.ndim}")

        # --- Step 1: Scale embeddings ---
        embed_scale = None
        inputs_embeds0 = initial_embeds_raw
        if self.config.scale_embeds:
            embed_scale = torch.sqrt(
                2.5 * torch.tensor(H, dtype=initial_embeds_raw.dtype, device=device)
            )
            inputs_embeds0 = initial_embeds_raw * embed_scale

        num_latent = self.config.num_latent_thoughts
        all_stages: List[torch.Tensor] = [inputs_embeds0]  # Stage 0 = scaled original embeddings

        # --- Step 2: Initial Interpolation Loop ---
        for iter_k in range(num_latent):
            current_stages = list(all_stages)
            num_current = len(current_stages)

            iter_embeds, iter_pos, iter_mask = self.prepare_inputs_for_stages(
                current_stages, B, L_orig, H, device, orig_pos_ids, orig_attn_mask_2d,
            )

            iter_outputs = self._run_backbone(
                inputs_embeds=iter_embeds,
                attention_mask=iter_mask,
                position_ids=iter_pos,
                past_key_values=None,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=False,
            )

            # Extract hidden states for each existing stage
            computed_embeddings = []
            for stage_idx in range(num_current):
                selector = slice(stage_idx, None, num_current)
                hidden = self.extract_hidden_for_stage(iter_outputs, selector)
                computed_embeddings.append(hidden)

            if iter_k == 0:
                # First iteration: add first interpolated stage
                all_stages.append(computed_embeddings[0])
            else:
                # Refine existing interpolated stages
                for j in range(iter_k):
                    all_stages[j + 1] = computed_embeddings[j]
                # Add new interpolated stage
                all_stages.append(computed_embeddings[iter_k])

        # --- Step 3: Jacobi Refinement Loop ---
        if self.training:
            if getattr(self.config, "random_jacobi_iterations", False):
                num_refine = torch.randint(2, 5, (1,)).item()
            else:
                num_refine = self.config.num_jacobi_iterations
        else:
            num_refine = self.config.num_jacobi_iterations

        if num_latent > 0 and num_refine > 0:
            for _ref_idx in range(num_refine):
                current_stages = list(all_stages)
                num_current = len(current_stages)

                ref_embeds, ref_pos, ref_mask = self.prepare_inputs_for_stages(
                    current_stages, B, L_orig, H, device, orig_pos_ids, orig_attn_mask_2d,
                )

                ref_outputs = self._run_backbone(
                    inputs_embeds=ref_embeds,
                    attention_mask=ref_mask,
                    position_ids=ref_pos,
                    past_key_values=None,
                    use_cache=False,
                    output_attentions=False,
                    output_hidden_states=False,
                )

                # Refine each interpolated stage
                for i in range(num_latent):
                    selector = slice(i, None, num_current)
                    hidden = self.extract_hidden_for_stage(ref_outputs, selector)
                    all_stages[i + 1] = hidden

        # --- Step 4: Final pass ---
        final_stages = list(all_stages)
        num_total_stages = len(final_stages)

        final_embeds, final_pos, final_mask = self.prepare_inputs_for_stages(
            final_stages, B, L_orig, H, device, orig_pos_ids, orig_attn_mask_2d,
        )

        final_outputs = self._run_backbone(
            inputs_embeds=final_embeds,
            attention_mask=final_mask,
            position_ids=final_pos,
            past_key_values=None if not is_inference_prefill else DynamicCache(),
            use_cache=use_cache if not is_inference_prefill else True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
        )

        # Extract logits from the target stage (last interpolated stage)
        logits_start_idx = num_latent  # K-th stage in zero-indexed interleaving
        if num_latent == 0 and num_total_stages == 1:
            logits_start_idx = 0

        all_hs = final_outputs.last_hidden_state

        if is_inference_prefill:
            # For inference prefill, only need last token's logits
            logits = self.lm_head(all_hs[:, -1:, :])
        else:
            # Extract hidden states for the target stage
            target_hs = all_hs[:, logits_start_idx::num_total_stages, :]
            logits = self.lm_head(target_hs)

        # --- Compute loss ---
        loss = None
        if labels is not None and not is_inference_prefill:
            if logits.shape[1] == L_orig:
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(
                    shift_logits.view(-1, self.config.vocab_size),
                    shift_labels.view(-1),
                )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=final_outputs.past_key_values,
            hidden_states=final_outputs.hidden_states if output_hidden_states else None,
            attentions=final_outputs.attentions if output_attentions else None,
        )
