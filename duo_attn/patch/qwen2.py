import warnings
from typing import Optional, Tuple
import os
import torch
from torch import nn
from transformers import Cache, DynamicCache

from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2ForCausalLM,
    Qwen2Model,
    repeat_kv,
    apply_rotary_pos_emb,
    CausalLMOutputWithPast,
    List,
    Union,
    CrossEntropyLoss,
    BaseModelOutputWithPast,
)
import types
from .utils import (
    reorder_linear_weights,
    reorder_full_attn_heads,
)
from .streaming_attn import (
    generate_streaming_mask,
    streaming_attn_sdpa,
    generate_streaming_info_blocksparse_flash_attn,
    streaming_attn_blocksparse_flash_attn,
)

from .static_kv_cache import (
    DuoAttentionStaticKVCache,
    enable_duo_attention_static_kv_cache_for_mistral,
)
from .tuple_kv_cache import enable_tuple_kv_cache_for_qwen2
from .flashinfer_utils import apply_rope_inplace, enable_flashinfer_rmsnorm

from tensor_parallel.pretrained_model import TensorParallelPreTrainedModel
from flash_attn import flash_attn_func, flash_attn_with_kvcache
from duo_attn.ulysses import UlyssesAttention


# From Huggingface's Transformers v4.37.2 This is the forward method of Qwen2Model using the tuple style KV cache.
# 模型向前传播
def old_qwen2_model_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:
    # 输出注意力权重，当 output_attentions=True 时，模型不仅会输出最终的结果，还会返回每一层或多头注意力机制中每个“头”的注意力权重
    # 输出注意力权重可能会增加计算开销和内存使用，对于每一个token，模型都要保存它与所有其他token之间的注意力得分
    # 并不是所有的模型实现都支持 output_attentions 参数，而且一些优化过的实现（如下面的SDPA）可能出于性能考虑而不支持直接输出注意力权重
    # 如果确实需要这些权重，可能需要采用替代的方法或回退到标准的实现方式
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    # 选择模型输出格式为字典（dict）或命名元组（named tuple）
    # 字典访问方式：outputs['last_hidden_state']
    # 元组访问方式：outputs[0]
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # retrieve input_ids and inputs_embeds
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape
    elif inputs_embeds is not None:
        # _表示忽略后续的值
        batch_size, seq_length, _ = inputs_embeds.shape
    else:
        raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

    # if self.gradient_checkpointing and self.training:
    #     if use_cache:
    #         logger.warning_once(
    #             "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
    #         )
    #         use_cache = False

    # 通常用于增量解码，它代表之前已经处理过的token数量
    past_key_values_length = 0

    # past_key_values表示KV缓存，Qwen的KV缓存可能使用了DynamicCache，所以需要进行转换
    # llama则是直接past_key_values[0][0].shap[2]，其中past_key_values[0]的表示取第一层KV缓存
    # past_key_values[0][0]，第一层保存有K和V的张量，则取的是K的张量
    # past_key_values[0][0].shap[2]，表示取张量里面的序列长度，也就是已生成或处理过的令牌数量
    if use_cache:
        use_legacy_cache = not isinstance(past_key_values, Cache)
        if use_legacy_cache:
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        # DynamicCache计算就是past_key_values[0][0].shap[2]
        past_key_values_length = past_key_values.get_usable_length(seq_length)

    # 表示每个token的位置信息，是一个二维张量，一维表示批次大小(一批处理的输入)，二维表示每个输入中的token的位置信息，是一个连续整数
    # 整数的起始位置是past_key_values_length，如果没有开启past_key_values，则默认是0
    if position_ids is None:
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        position_ids = torch.arange(
            past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
        )
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
    else:
        position_ids = position_ids.view(-1, seq_length).long()

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    # attention_mask作用是屏蔽填充的字符，支持因果编码(即确保模型在预测每个token时只能看到它之前的token)
    if attention_mask is not None and self._attn_implementation == "flash_attention_2" and use_cache:
        # attention_mask是一个二维张量，每一行代表一个输入序列，每一列表示每个输入序列对应token是否有效
        # 有效的token则为置为1，无效则为0，通过attention_mask[:, -1].sum().item()获取最后一列之和，判断是否是右侧填充
        is_padding_right = attention_mask[:, -1].sum().item() != batch_size
        if is_padding_right:
            # 怀疑flash_attention_2是左侧填充
            raise ValueError(
                "You are attempting to perform batched generation with padding_side='right'"
                " this may lead to unexpected behaviour for Flash Attention version of Qwen2. Make sure to "
                " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
            )

    # TODO 待确认Llama将attention_mask统一置为了None
    # if self._attn_implementation == "flash_attention_2":
    #     # 2d mask is passed through the layers
    #     attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
    # elif self._attn_implementation == "sdpa" and not output_attentions:
    #     # output_attentions=True can not be supported when using SDPA, and we fall back on
    #     # the manual implementation that requires a 4D causal mask in all cases.
    #     attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
    #         attention_mask,
    #         (batch_size, seq_length),
    #         inputs_embeds,
    #         past_key_values_length,
    #     )
    # else:
    #     # 4d mask is passed through the layers
    #     attention_mask = _prepare_4d_causal_attention_mask(
    #         attention_mask,
    #         (batch_size, seq_length),
    #         inputs_embeds,
    #         past_key_values_length,
    #         sliding_window=self.config.sliding_window,
    #     )

    # 如果attention_mask不等于None，并且 0 in attention_mask，则padding_mask=attention_mask
    # 确保填充的屏蔽信息不会遗失
    padding_mask = None
    if attention_mask is not None:
        if 0 in attention_mask:
            padding_mask = attention_mask
        else:
            padding_mask = None
    attention_mask = None

    hidden_states = inputs_embeds

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None

    for idx, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # if self.gradient_checkpointing and self.training:
        #     layer_outputs = self._gradient_checkpointing_func(
        #         decoder_layer.__call__,
        #         hidden_states,
        #         attention_mask,
        #         position_ids,
        #         past_key_values,
        #         output_attentions,
        #         use_cache,
        #     )
        # else:
        #     layer_outputs = decoder_layer(
        #         hidden_states,
        #         attention_mask=attention_mask,
        #         position_ids=position_ids,
        #         past_key_value=past_key_values,
        #         output_attentions=output_attentions,
        #         use_cache=use_cache,
        #     )

        past_key_value = past_key_values[idx] if past_key_values is not None else None

        # padding_mask用于屏蔽填充部分，attention_mask不仅可以用于屏蔽填充部分
        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            padding_mask=padding_mask
        )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache = layer_outputs[2 if output_attentions else 1]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = None
    if use_cache:
        next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache

    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )

# From Huggingface's Transformers v4.37.2 This is the forward method of Qwen2DecoderLayer using the tuple style KV cache.
# 解码器层向前传播
def old_qwen2_decoder_layer_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
    **kwargs,
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
    # 4.37 版本剔除了padding_mask，使用attention_mask替代
    if "padding_mask" in kwargs:
        warnings.warn(
            "Passing `padding_mask` is deprecated and will be removed in v4.37. "
            "Please make sure use `attention_mask` instead.`"
        )
    """
    Args:
        hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
        attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
            `(batch, sequence_length)` where padding elements are indicated by 0.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under
            returned tensors for more detail.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
            (see `past_key_values`).
        past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
    """

    # 每一层的输入，以及输出
    residual = hidden_states

    # 输入执行层归一化。对于每个token的嵌入向量（长度为 embed_dim），层归一化会计算其均值和标准差，并将该向量标准化为均值为0、标准差为1的分布。
    # 归一化后的结果会再乘以一个可学习的比例参数（gamma）并加上一个可学习的偏移参数（beta），以恢复一定的表达能力。
    hidden_states = self.input_layernorm(hidden_states)

    # Self Attention
    hidden_states, self_attn_weights, present_key_value = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
    )
    hidden_states = residual + hidden_states

    # Fully Connected 残差连接
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    # 多层感知器，进行非线性变换
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (self_attn_weights,)

    if use_cache:
        outputs += (present_key_value,)

    return outputs



def enable_qwen2_duo_attention_training(
    model: Qwen2ForCausalLM,
    sink_size,
    recent_size,
    max_length,
    initial_value=1.0,
    enable_ulysses_attention=False,
    streaming_attn_implementation="blocksparse",
):
    # 使用flash_attn替换Qwen2原模型的forward，attn等模块
    enable_tuple_kv_cache_for_qwen2(model)


    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    if streaming_attn_implementation == "blocksparse":
        num_sink_blocks = (sink_size + 127) // 128
        num_recent_blocks = (recent_size + 127) // 128
        num_heads_per_device = model.config.num_attention_heads // int(
            os.environ["WORLD_SIZE"]
        )
        print(
            f"Using blocksparse implementation with {num_sink_blocks} sink blocks, {num_recent_blocks} recent blocks, and {num_heads_per_device} heads per device"
        )
        streaming_mask = generate_streaming_info_blocksparse_flash_attn(
            num_sink_blocks, num_recent_blocks, num_heads_per_device, device
        )
        streaming_attn_func = streaming_attn_blocksparse_flash_attn
    elif streaming_attn_implementation == "sdpa":
        streaming_mask = generate_streaming_mask(
            max_length, sink_size, recent_size, device
        )
        streaming_attn_func = streaming_attn_sdpa
    else:
        raise ValueError(
            f"Unsupported streaming attention implementation: {streaming_attn_implementation}"
        )

    for layer in model.model.layers:
        module = layer.self_attn
        module.forward = types.MethodType(mistral_duo_attention_forward_two_way, module)
        module.sink_size = sink_size
        module.recent_size = recent_size
        module.register_parameter(
            "full_attention_heads",
            nn.Parameter(
                torch.ones(
                    module.num_key_value_heads,
                    device=device,
                    dtype=dtype,
                    requires_grad=True,
                )
                * initial_value
            ),
        )

        module.register_buffer("streaming_mask", streaming_mask)
        if not enable_ulysses_attention:
            module.streaming_attn_func = streaming_attn_func
            module.full_attn_func = flash_attn_func
        else:
            module.streaming_attn_func = UlyssesAttention(
                attn_func=streaming_attn_func,
            )
            module.full_attn_func = UlyssesAttention(
                attn_func=flash_attn_func,
            )





