import torch
from torch import nn as nn
from transformers import LlamaForCausalLM as HFLlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaPreTrainedModel, LlamaModel, repeat_kv
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.utils import logging
from transformers.cache_utils import DynamicCache, StaticCache
# from transformers.generation import GenerationMixin
from typing import List, Optional, Tuple, Union
from transformers.cache_utils import Cache
from torch.nn import CrossEntropyLoss
import os
import numpy as np
import math

logger = logging.get_logger(__name__)

class FastVLlamaModel(LlamaModel):
    def __init__(self, config):
        self.last_attention = None
        self.saved_selected_tokens = {prune_layer: [] for prune_layer in config.fastv_k}
        self.fastV_k = config.fastv_k
        self.fastV_ratio = 1 - config.fastv_ratio # reduction ratio
        self.fastV_use_rope = True
        self.use_predefined_mask = False
        self.fastV_sample_cluster = False
        self.stride = None
        if hasattr(config, 'fastv_predefined_mask') and config.fastv_predefined_mask is not None:
            if 'no_rope' in config.fastv_predefined_mask:
                if config.fastv_predefined_mask == 'no_rope':
                    self.fastV_use_rope = False
                    print(f'fastv model with k={self.fastV_k} and r={1 - self.fastV_ratio}, no rope')
                elif 'sample_cluster' in config.fastv_predefined_mask:
                    assert False
                else:
                    assert 's=' in config.fastv_predefined_mask
                    self.fastV_use_rope = False
                    self.use_predefined_mask = True
                    self.stride = int(config.fastv_predefined_mask.replace('no_rope_s=', ''))
                    print(f'feather model with k={self.fastV_k} and r={1 - self.fastV_ratio}, no rope, adding stride of {self.stride}')
            elif 'sample_cluster' in config.fastv_predefined_mask:
                self.use_predefined_mask = True
                cluster_info = config.fastv_predefined_mask[config.fastv_predefined_mask.index('sample_cluster') + len('sample_cluster_'):]
                num_clusters = cluster_info.split('_')[0]
                tokens_per_cluster = cluster_info.split('_')[1]
                self.fastV_sample_cluster = True
                print(f'fastv model with k={self.fastV_k} and r={1 - self.fastV_ratio}, additionally adding {num_clusters} clusters each with {tokens_per_cluster} tokens')
            else:
                self.use_predefined_mask = True
                assert config.fastv_predefined_mask.split('_')[0][:2] == 's=' and config.fastv_predefined_mask.split('_')[1][:2] == 'p='
                self.stride = int(config.fastv_predefined_mask.split('_')[0][2:])
                self.percent_last_tokens = float(config.fastv_predefined_mask.split('_')[1][2:])
                print(f'fastv model with k={self.fastV_k}, predefined mask of stride={self.stride} with {self.percent_last_tokens * 100} percent last tokens')
        else:
            print(f'fastv model with k={self.fastV_k} and r={1 - self.fastV_ratio}')
        self.dataset_example_num = 0

        super().__init__(config)
    
    # adopted from https://github.com/huggingface/transformers/blob/a0857740c0e6127485c11476650314df3accc2b6/src/transformers/models/llama/modeling_llama.py#L934C5-L1044C10
    def forward(
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
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        past_seen_tokens = 0
        if use_cache:  # kept for BC (cache positions)
            if not isinstance(past_key_values, StaticCache):
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_seen_tokens = past_key_values.get_seq_length()

        if cache_position is None:
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(attention_mask, inputs_embeds)

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                )
            else:
                ##### pruning code  #####
                # code adapted from https://github.com/pkunlp-icler/FastV/blob/5e8ca117619f67d0447f1065ac1c0b76b49e1357/src/FastV/lmms-eval/fastv_kvcache.py#L122
                seq_length = hidden_states.shape[1]
                device = hidden_states.device
                if decoder_layer.self_attn.layer_idx in self.fastV_k and seq_length > 1:
                    assert self.last_attention is not None
                    
                    if self.use_predefined_mask and not self.fastV_use_rope: # feather setup: using both tokens determined from attention and uniform stride
                        if decoder_layer.self_attn.layer_idx == self.fastV_k[0]: # first time, add the attention and uniform tokens
                            image_attention_score = self.last_attention
                            top_attention_rank_index = image_attention_score.topk(int(len(image_indices) * self.fastV_ratio)).indices + image_offset

                            grid_size = int(math.sqrt(len(image_indices)))
                            grid_indices = np.arange(len(image_indices)).reshape(grid_size, grid_size)
                            assert self.stride != 0
                            strided_indices = grid_indices[::self.stride, ::self.stride].flatten()
                            strided_indices = torch.from_numpy(strided_indices).to(device) + image_offset
                            combined_stride_fast_indices = torch.cat((top_attention_rank_index, strided_indices))
                            combined_stride_fast_indices, inverse_indices = torch.unique(combined_stride_fast_indices, sorted=True, return_inverse=True) # torch.unique sorts
                            
                            top_attention_rank_index = combined_stride_fast_indices
                            kept_tokens_original_positions = (top_attention_rank_index - image_offset).sort().values

                        else: # later time, only calculate attention from non-uniform tokens
                            assert decoder_layer.self_attn.layer_idx == self.fastV_k[1] # Only allow pruning at two layers (ratio exponentially increases)

                            image_attention_score = self.last_attention
                            top_attention_rank_index = image_attention_score.topk(int(len(image_indices) * (self.fastV_ratio * self.fastV_ratio))).indices + image_offset # increasing ratio

                            kept_tokens_original_positions = kept_tokens_original_positions[(top_attention_rank_index - image_offset).sort().values]
                    elif self.use_predefined_mask:
                        assert len(self.fastV_k) == 1 # only works when doing one pruning
                        
                        if self.fastV_sample_cluster:
                            top_attention_rank_index = kwargs['fastv_clusters'] + image_offset
                            kept_tokens_original_positions = top_attention_rank_index - image_offset
                        else:
                            grid_size = int(math.sqrt(len(image_indices)))
                            grid_indices = np.arange(len(image_indices)).reshape(grid_size, grid_size)
                            if self.stride != 0:
                                strided_indices = grid_indices[::self.stride, ::self.stride].flatten()
                            else:
                                strided_indices = np.asarray([], dtype=grid_indices.dtype)
                            last_image_indices = np.arange(int(len(image_indices) * (1 - self.percent_last_tokens)), len(image_indices))
                            uniform_and_last_indices = np.unique(np.concatenate((strided_indices, last_image_indices)))
                            top_attention_rank_index = torch.from_numpy(uniform_and_last_indices).to(device) + image_offset
                            kept_tokens_original_positions = top_attention_rank_index - image_offset
                    else:
                        image_attention_score = self.last_attention

                        top_attention_rank_index = image_attention_score.topk(int(len(image_indices) * self.fastV_ratio)).indices + image_offset

                        if decoder_layer.self_attn.layer_idx == self.fastV_k[0]:
                            kept_tokens_original_positions = (top_attention_rank_index - image_offset).sort().values
                        else:
                            kept_tokens_original_positions = kept_tokens_original_positions[(top_attention_rank_index - image_offset).sort().values]
                    keep_indexs = torch.cat((torch.arange(image_offset, device=device), top_attention_rank_index, torch.arange(image_indices[-1] + 1, seq_length, device=device)))
                    keep_indexs = keep_indexs.sort().values
                    hidden_states = hidden_states[:,keep_indexs,:]
                    causal_mask = causal_mask[:,:,:hidden_states.shape[1],:hidden_states.shape[1]]
                    if position_ids.dim() == 2:
                        position_ids = position_ids[:,keep_indexs]
                    else:
                        position_ids = position_ids[:,:,keep_indexs]
                    cache_position = cache_position[:len(keep_indexs)]

                    image_indices = torch.arange(len(top_attention_rank_index), device=device) + image_offset

                    self.saved_selected_tokens[decoder_layer.self_attn.layer_idx].append(kept_tokens_original_positions.to('cpu').numpy().astype(np.uint16))

                if decoder_layer.self_attn.layer_idx + 1 in self.fastV_k and seq_length > 1:
                    if decoder_layer.self_attn.layer_idx + 1 == self.fastV_k[0]:
                        assert 'image_indices' in kwargs
                        image_indices = kwargs['image_indices'].clone()
                        image_offset = image_indices[0]

                    self.last_attention = self.calculate_attention_weight(decoder_layer, hidden_states, causal_mask, position_ids, past_key_values, cache_position, image_indices)

                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = (
                next_decoder_cache.to_legacy_cache() if isinstance(next_decoder_cache, Cache) else next_decoder_cache
            )
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
    
    def calculate_attention_weight(self, decoder_layer, hidden_states, attention_mask, position_ids, past_key_value, cache_position, image_indices):
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
        bsz, q_len, _ = hidden_states.size()

        hidden_states = decoder_layer.input_layernorm(hidden_states)
        query_states = decoder_layer.self_attn.q_proj(hidden_states)
        key_states = decoder_layer.self_attn.k_proj(hidden_states)
        value_states = decoder_layer.self_attn.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, decoder_layer.self_attn.num_heads, decoder_layer.self_attn.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, decoder_layer.self_attn.num_key_value_heads, decoder_layer.self_attn.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, decoder_layer.self_attn.num_key_value_heads, decoder_layer.self_attn.head_dim).transpose(1, 2)

        if self.fastV_use_rope:
            past_key_value = getattr(decoder_layer.self_attn, "past_key_value", past_key_value)
            cos, sin = decoder_layer.self_attn.rotary_emb(value_states, position_ids)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        key_states = repeat_kv(key_states, decoder_layer.self_attn.num_key_value_groups)

        text_query_states = query_states[:,:,-1:,:]
        image_key_states = key_states[:,:,image_indices,:]

        attn_weights = torch.matmul(text_query_states, image_key_states.transpose(2, 3)) / math.sqrt(decoder_layer.self_attn.head_dim)

        # full weight calculation is 1 x num_heads x seq x seq, here it is 1 x num_heads x 1 x img_seq
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        attn_weights = attn_weights.mean(dim=1)[0][0]
        return attn_weights



class FastVLlamaForCausalLM(HFLlamaForCausalLM):
    # adapted from https://github.com/huggingface/transformers/blob/a0857740c0e6127485c11476650314df3accc2b6/src/transformers/models/llama/modeling_llama.py#L1090
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        LlamaPreTrainedModel.__init__(self, config)
        # experiments run with eager attention
        # config._attn_implementation = 'eager'
        self.model = FastVLlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs
        )
        
        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )