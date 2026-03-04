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
"""PonderLM-2 configuration based on LLaMA."""

from transformers.models.llama.configuration_llama import LlamaConfig


class PonderLlamaConfig(LlamaConfig):
    """
    Configuration class for PonderLM-2 implemented on top of LLaMA architecture.

    Extends LlamaConfig with PonderLM-specific parameters for iterative
    embedding refinement (horizontal scaling / pondering).

    Additional Args:
        num_latent_thoughts (`int`, *optional*, defaults to 1):
            Number of latent thought stages for initial interpolation.
            Each token gets this many "thinking" hidden states interleaved
            with the original embeddings.
        num_jacobi_iterations (`int`, *optional*, defaults to 4):
            Number of Jacobi refinement iterations applied after the
            initial interpolation stages. Paper shows ~3 is enough to converge.
        scale_embeds (`bool`, *optional*, defaults to True):
            Whether to scale input embeddings by sqrt(2.5 * hidden_size).
        softmax_temperature (`float`, *optional*, defaults to 1.0):
            Temperature for softmax (reserved for future use).
        use_all_logits (`bool`, *optional*, defaults to False):
            Whether to compute logits for all interleaved positions
            (not just the target stage).
        random_jacobi_iterations (`bool`, *optional*, defaults to True):
            Whether to use a random number of Jacobi iterations (2-5)
            during training for regularization.
    """

    model_type = "ponder-llama"

    def __init__(
        self,
        num_latent_thoughts: int = 1,
        num_jacobi_iterations: int = 4,
        scale_embeds: bool = True,
        softmax_temperature: float = 1.0,
        use_all_logits: bool = False,
        random_jacobi_iterations: bool = True,
        **kwargs,
    ):
        self.num_latent_thoughts = num_latent_thoughts
        self.num_jacobi_iterations = num_jacobi_iterations
        self.scale_embeds = scale_embeds
        self.softmax_temperature = softmax_temperature
        self.use_all_logits = use_all_logits
        self.random_jacobi_iterations = random_jacobi_iterations
        super().__init__(**kwargs)
