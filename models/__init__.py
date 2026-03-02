from .configuration_llama import MyLlamaConfig
from .modeling_llama import MyLlamaModel, MyLlamaForCausalLM

from .configuration_ponder_llama import PonderLlamaConfig
from .modeling_ponder_llama import PonderLlamaModel, PonderLlamaForCausalLM

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

AutoConfig.register("my-llama", MyLlamaConfig)
AutoModel.register(MyLlamaConfig, MyLlamaModel)
AutoModelForCausalLM.register(MyLlamaConfig, MyLlamaForCausalLM)

AutoConfig.register("ponder-llama", PonderLlamaConfig)
AutoModel.register(PonderLlamaConfig, PonderLlamaModel)
AutoModelForCausalLM.register(PonderLlamaConfig, PonderLlamaForCausalLM)
