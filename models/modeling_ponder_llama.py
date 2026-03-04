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
PonderLM-2 implemented on the LLaMA architecture.

Ports the PonderLM-2 "horizontal scaling" idea from the original GPT-NeoX
implementation to LLaMA, enabling fair comparison with other LLaMA-based
models using the same tokenizer and vocabulary.

Key idea (from paper):
    Instead of predicting next token directly, each position first generates a
    latent thought h_t (the last-layer hidden state), which is then used to
    predict the actual next token.  During training this is parallelised via
    Jacobi iteration on the interleaved sequence
        [e(x_1), h_1, e(x_2), h_2, ...]

IMPORTANT — logit extraction position:
    In the interleaved sequence [e(x_1), h_1, e(x_2), h_2, ...],
    predictions are made from **e(x_t) positions** (even indices, 0::2),
    NOT from the h_t positions.  This is because e(x_t) appears AFTER
    h_{t-1} in the sequence, so under causal attention it integrates
    all previous e(x) and h information.

Flash-attention compatibility:
    The interleaved sequence uses *standard causal masking* — every token
    attends to all positions earlier in the sequence.  The 2D attention mask
    (1/0 for padding) is the only mask needed.  position_ids are repeated
    (for RoPE) but do not affect the attention pattern.  This is fully
    compatible with flash_attention_2, sdpa, and eager backends.
"""

from typing import List, Optional, Tuple, Union

import math
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from transformers.utils.generic import check_model_inputs

from transformers.models.llama.modeling_llama import LlamaModel, LlamaForCausalLM
from .configuration_ponder_llama import PonderLlamaConfig


logger = logging.get_logger(f"transformers.{__name__}")


# ====================================================================== #
#  Base model — identical to LlamaModel, just uses PonderLlamaConfig      #
# ====================================================================== #

class PonderLlamaModel(LlamaModel):
    """
    PonderLlama base transformer.  Architecturally identical to LlamaModel;
    the pondering logic lives entirely in `PonderLlamaForCausalLM`.
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


# ====================================================================== #
#  CausalLM wrapper with PonderLM-2 pondering logic                       #
# ====================================================================== #

class PonderLlamaForCausalLM(LlamaForCausalLM):
    """
    PonderLM-2 on LLaMA with iterative embedding refinement.

    Training flow (Jacobi-parallelised):
        1. Embed input tokens -> e(x), optionally scale by sqrt(2.5 * H).
        2. Initial forward: run transformer on e(x) alone to get h^0.
        3. Jacobi refinement: interleave [e(x), h^k], run transformer,
           extract hidden states at **e(x) positions** (0::2) as h^{k+1}.
           Repeat for random 2-5 iterations (training) or fixed iterations.
        4. Final pass: interleave [e(x), h^final], run transformer,
           extract logits from **e(x) positions** (0::2), compute CE loss.

    Inference flow:
        Prefill  - same as training minus the loss.
        Decode   - two transformer passes per new token:
                   pass-1 produces h_t, pass-2 uses h_t to predict x_{t+1}.

    All passes use standard causal attention -> flash_attention_2 works.
    """

    config_class = PonderLlamaConfig

    def __init__(self, config: PonderLlamaConfig):
        # Skip LlamaForCausalLM.__init__ so we can swap in PonderLlamaModel
        super(LlamaForCausalLM, self).__init__(config)
        self.model = PonderLlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    # -------------------------------------------------------------- #
    #  Helpers                                                         #
    # -------------------------------------------------------------- #

    @staticmethod
    def _interleave_stages(
        stages: List[torch.Tensor],
        batch_size: int,
        seq_len: int,
        hidden_size: int,
        device: torch.device,
        orig_pos_ids: torch.Tensor,
        orig_attn_mask_2d: torch.Tensor,
    ):
        """
        Interleave N embedding stages into one sequence of length N·L.

        Layout (N=2, L=3):
            [S0[0], S1[0],  S0[1], S1[1],  S0[2], S1[2]]

        position_ids and attention_mask are repeated per stage so that
        each stage sees the same RoPE position and padding mask.
        """
        N = len(stages)
        T = N * seq_len

        embeds = torch.empty((batch_size, T, hidden_size),
                             dtype=stages[0].dtype, device=device)
        pos    = torch.empty((batch_size, T),
                             dtype=orig_pos_ids.dtype, device=device)
        mask   = torch.empty((batch_size, T),
                             dtype=orig_attn_mask_2d.dtype, device=device)

        for i in range(N):
            embeds[:, i::N, :] = stages[i]
            pos[:, i::N]       = orig_pos_ids
            mask[:, i::N]      = orig_attn_mask_2d

        return embeds, pos, mask

    def _run_backbone(self, inputs_embeds, **kwargs) -> BaseModelOutputWithPast:
        """Thin wrapper around self.model (LLaMA backbone)."""
        return self.model(inputs_embeds=inputs_embeds, **kwargs)

    # -------------------------------------------------------------- #
    #  Forward                                                         #
    # -------------------------------------------------------------- #

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
        PonderLM-2 forward pass with iterative embedding refinement.
        Compatible with flash_attention_2, sdpa, and eager backends.
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = kwargs.get("output_attentions", False)
        output_hidden_states = kwargs.get("output_hidden_states", False)

        # ============================================================ #
        #  INFERENCE  (past_key_values supplied by generate())          #
        # ============================================================ #
        if past_key_values is not None:
            is_decoding = len(past_key_values) > 0

            if is_decoding:
                # ---------- Decode: 2-pass per token ----------
                if inputs_embeds is None:
                    if input_ids is None:
                        raise ValueError("input_ids must be provided")
                    inputs_embeds = self.model.embed_tokens(input_ids)

                B, L, H = inputs_embeds.shape
                if self.config.scale_embeds:
                    scale = math.sqrt(2.5 * H)
                    inputs_embeds = inputs_embeds * scale

                # Pass 1 → pondered embedding h_t
                out1 = self._run_backbone(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    cache_position=cache_position,
                    **{k: v for k, v in kwargs.items()
                       if k not in ("output_attentions", "output_hidden_states")},
                )
                pondered = out1.last_hidden_state

                # Pass 2 → final hidden state to predict x_{t+1}
                out2 = self._run_backbone(
                    inputs_embeds=pondered,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=out1.past_key_values,
                    use_cache=True,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    cache_position=cache_position,
                    **{k: v for k, v in kwargs.items()
                       if k not in ("output_attentions", "output_hidden_states")},
                )
                logits = self.lm_head(out2.last_hidden_state)
                return CausalLMOutputWithPast(
                    loss=None, logits=logits,
                    past_key_values=out2.past_key_values,
                    hidden_states=out2.hidden_states,
                    attentions=out2.attentions,
                )

            else:
                # ---------- Prefill ----------
                return self._forward_ponder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    inputs_embeds=inputs_embeds,
                    labels=None,
                    use_cache=True,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    cache_position=cache_position,
                    inference_prefill=True,
                    **kwargs,
                )

        # ============================================================ #
        #  TRAINING                                                     #
        # ============================================================ #
        return self._forward_ponder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            inference_prefill=False,
            **kwargs,
        )

    # -------------------------------------------------------------- #
    #  Core pondering logic (shared by train + prefill)                #
    # -------------------------------------------------------------- #

    def _forward_ponder(
        self,
        input_ids=None, attention_mask=None, position_ids=None,
        inputs_embeds=None, labels=None,
        use_cache=False, output_attentions=False, output_hidden_states=False,
        cache_position=None, inference_prefill=False,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """
        Full PonderLM-2 pipeline:
            Step 1: embed + scale
            Step 2: initial forward on e(x) alone -> h^0
            Step 3: Jacobi refinement (interleave [e(x), h^k],
                    extract h^{k+1} from e(x) positions 0::N)
            Step 4: final interleaved forward -> logits from e(x) positions

        CRITICAL: logits and Jacobi updates are extracted from e(x) positions
        (index 0::N in interleaved sequence), NOT from h positions.
        """
        # ---- 0. Embed ----------------------------------------------- #
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("Either input_ids or inputs_embeds must be given")
            initial_embeds = self.model.embed_tokens(input_ids)
        else:
            initial_embeds = inputs_embeds

        B, L, H = initial_embeds.shape
        device = initial_embeds.device

        # Position ids (original, before interleaving)
        if position_ids is None:
            orig_pos = torch.arange(L, dtype=torch.long, device=device
                                    ).unsqueeze(0).expand(B, -1)
        else:
            orig_pos = position_ids

        # 2-D attention mask (padding only; causal mask is added by model)
        if attention_mask is None:
            orig_mask = torch.ones((B, L), dtype=torch.long, device=device)
        else:
            if attention_mask.ndim == 2:
                orig_mask = attention_mask
            elif attention_mask.ndim == 4:
                orig_mask = attention_mask[:, 0, 0, :L]
            elif attention_mask.ndim == 3:
                orig_mask = attention_mask[:, 0, :L]
            else:
                raise ValueError(
                    f"Unsupported attention_mask ndim={attention_mask.ndim}")

        # ---- 1. Scale embeddings ------------------------------------ #
        e0 = initial_embeds
        if self.config.scale_embeds:
            scale = math.sqrt(2.5 * H)
            e0 = initial_embeds * scale

        K = self.config.num_latent_thoughts          # typically 1

        # strip pondering-unrelated kwargs
        backbone_extra = {k: v for k, v in kwargs.items()
                         if k not in ("output_attentions", "output_hidden_states")}

        # ---- 2. Initial forward: e(x) only -> h^0 ------------------- #
        # Run transformer on just e(x) (no interleaving).
        # The last-layer hidden states become the initial latent estimates.
        init_out = self._run_backbone(
            inputs_embeds=e0,
            attention_mask=orig_mask,
            position_ids=orig_pos,
            use_cache=False,
            **backbone_extra,
        )
        # h^0 for every latent thought stage (all initialised the same)
        latent_thoughts: List[torch.Tensor] = [init_out.last_hidden_state]

        # For K > 1: additional stages via interleaved forward passes
        for k in range(1, K):
            cur_stages = [e0] + latent_thoughts   # N = k+1 stages
            N = len(cur_stages)

            emb_il, pos_il, mask_il = self._interleave_stages(
                cur_stages, B, L, H, device, orig_pos, orig_mask)

            out = self._run_backbone(
                inputs_embeds=emb_il,
                attention_mask=mask_il,
                position_ids=pos_il,
                use_cache=False,
                **backbone_extra,
            )
            # Extract from e(x) positions (index 0 in each group of N)
            h_new = out.last_hidden_state[:, 0::N, :]
            # Update all existing + add new stage
            for j in range(len(latent_thoughts)):
                latent_thoughts[j] = h_new
            latent_thoughts.append(h_new)

        # ---- 3. Jacobi refinement loop ------------------------------ #
        if self.training and getattr(self.config, "random_jacobi_iterations", False):
            num_refine = torch.randint(2, 6, (1,)).item()  # 2,3,4,5
        else:
            num_refine = self.config.num_jacobi_iterations

        if K > 0 and num_refine > 0:
            for _ in range(num_refine):
                cur_stages = [e0] + latent_thoughts
                N = len(cur_stages)   # K + 1

                emb_il, pos_il, mask_il = self._interleave_stages(
                    cur_stages, B, L, H, device, orig_pos, orig_mask)

                out = self._run_backbone(
                    inputs_embeds=emb_il,
                    attention_mask=mask_il,
                    position_ids=pos_il,
                    use_cache=False,
                    **backbone_extra,
                )

                # KEY FIX: extract from e(x) positions (0::N), NOT h positions
                h_from_ex = out.last_hidden_state[:, 0::N, :]
                for i in range(K):
                    latent_thoughts[i] = h_from_ex

        # ---- 4. Final pass ------------------------------------------ #
        final_stages = [e0] + latent_thoughts
        N_final = len(final_stages)   # K + 1

        emb_il, pos_il, mask_il = self._interleave_stages(
            final_stages, B, L, H, device, orig_pos, orig_mask)

        final_out = self._run_backbone(
            inputs_embeds=emb_il,
            attention_mask=mask_il,
            position_ids=pos_il,
            past_key_values=DynamicCache() if inference_prefill else None,
            use_cache=True if inference_prefill else use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **backbone_extra,
        )

        all_hs = final_out.last_hidden_state

        # KEY FIX: extract logits from e(x) positions (index 0::N_final).
        # In [e(x_1), h_1, e(x_2), h_2, ...], e(x_t) at position 2t-2
        # sees all previous e(x) and h, so it integrates latent thoughts.
        target_hs = all_hs[:, 0::N_final, :]   # [B, L, H]

        if inference_prefill:
            logits = self.lm_head(target_hs[:, -1:, :])
        else:
            logits = self.lm_head(target_hs)

        # ---- 5. Loss ------------------------------------------------ #
        loss = None
        if labels is not None and not inference_prefill:
            if logits.shape[1] == L:
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
            past_key_values=final_out.past_key_values,
            hidden_states=final_out.hidden_states if output_hidden_states else None,
            attentions=final_out.attentions if output_attentions else None,
        )
