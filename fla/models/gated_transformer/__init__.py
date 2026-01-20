
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.gated_transformer.configuration_gated_transformer import GatedTransformerConfig
from fla.models.gated_transformer.modeling_gated_transformer import (
    GatedTransformerForCausalLM,
    GatedTransformerModel,
)

AutoConfig.register(GatedTransformerConfig.model_type, GatedTransformerConfig, exist_ok=True)
AutoModel.register(GatedTransformerConfig, GatedTransformerModel, exist_ok=True)
AutoModelForCausalLM.register(GatedTransformerConfig, GatedTransformerForCausalLM, exist_ok=True)


__all__ = ['GatedTransformerConfig', 'GatedTransformerForCausalLM', 'GatedTransformerModel']
