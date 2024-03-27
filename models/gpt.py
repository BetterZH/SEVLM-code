import torch
import torch.utils.checkpoint
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import GPT2PreTrainedModel
from transformers.activations import ACT2FN
from transformers.modeling_outputs import (BaseModelOutputWithPastAndCrossAttentions,CausalLMOutputWithCrossAttentions,)
from transformers.modeling_utils import Conv1D
from typing import Tuple
import numpy as np
from utils.data_utils import neg_ids
from torch.nn.modules.activation import Sigmoid
import math 

epsilon = 1e-8

def optional_repeat(value, times):
    """ helper function, to repeat a parameter's value many times
    :param value: an single basic python type (int, float, boolean, string), or a list with length equals to times
    :param times: int, how many times to repeat
    :return: a list with length equal to times
    """
    if type(value) is not list:
        value = [value]

    if len(value) != 1 and len(value) != times:
        raise ValueError('The value should be a singleton, or be a list with times length.')

    if len(value) == times:
        return value # do nothing

    return np.array(value).repeat(times).tolist()


class CustomMSELoss(nn.Module):
    def __init__(self):
        super(CustomMSELoss, self).__init__()

    def forward(self, pred_lable, true_lable):
        mask = torch.any(true_lable != 0, dim=-1)  
        diff = pred_lable - true_lable  
        diff = diff[mask]  
        return (diff ** 2).mean()  


class GPT2Attention(nn.Module):
    def __init__(self, config, is_cross_attention=False):
        super().__init__()

        max_positions = config.max_position_embeddings
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.uint8)).view(
                1, 1, max_positions, max_positions
            ),
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4))

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
            )

        self.scale_attn_weights = config.scale_attn_weights
        self.is_cross_attention = is_cross_attention

        if self.is_cross_attention:
            self.c_attn = Conv1D(2 * self.embed_dim, self.embed_dim)
            self.q_attn = Conv1D(self.embed_dim, self.embed_dim)
        else:
            self.c_attn = Conv1D(3 * self.embed_dim, self.embed_dim)
        self.c_proj = Conv1D(self.embed_dim, self.embed_dim)

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)


    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        if self.scale_attn_weights:
            attn_weights = attn_weights / (float(value.size(-1)) ** 0.5)

        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length].bool()
            attn_weights = torch.where(causal_mask, attn_weights, self.masked_bias.to(attn_weights.dtype))

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.Softmax(dim=-1)(attn_weights)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(*new_shape)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)

    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=False,
        output_attentions=False,
    ):
        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )

            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)
        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)


class GPT2MLP(nn.Module):
    def __init__(self, intermediate_size, config):
        super().__init__()
        embed_dim = config.hidden_size
        self.c_fc = Conv1D(intermediate_size, embed_dim)
        self.c_proj = Conv1D(embed_dim, intermediate_size)
        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states):
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class GPT2Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = GPT2Attention(config)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        if config.add_cross_attention:
            self.crossattention = GPT2Attention(config, is_cross_attention=True)
            self.ln_cross_attn = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        self.mlp = GPT2MLP(inner_dim, config)

    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=False,
        output_attentions=False,
    ):
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]
        # residual connection
        hidden_states = attn_output + residual

        if encoder_hidden_states is not None:
            # add one self-attention block for cross-attention
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
                    "cross-attention layers by setting `config.add_cross_attention=True`"
                )
            residual = hidden_states
            hidden_states = self.ln_cross_attn(hidden_states)
            cross_attn_outputs = self.crossattention(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )
            attn_output = cross_attn_outputs[0]
            # residual connection
            hidden_states = residual + attn_output
            outputs = outputs + cross_attn_outputs[2:]  # add cross attentions if we output attention weights

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + feed_forward_hidden_states

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs  # hidden_states, present, (attentions, cross_attentions)


class Mappting(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(Mappting, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)


class ContrastiveHead(nn.Module):
    def __init__(self, im_dim, hidden_dim, lang_dim):
        super().__init__()
        self.pred = nn.Sequential(nn.Linear(lang_dim * 2 + im_dim, hidden_dim),
                                nn.ReLU(inplace=True),
                                nn.Linear(hidden_dim, hidden_dim),
                                nn.ReLU(inplace=True),
                                nn.Linear(hidden_dim, hidden_dim),
                                nn.ReLU(inplace=True),
                                nn.Linear(hidden_dim, hidden_dim),
                                nn.ReLU(inplace=True),
                                nn.Linear(hidden_dim, 1))
        
    def forward(self, img, emo, exp):
        info = {}
        out = self.pred(torch.cat([img, emo, exp], -1)).squeeze()
        return out, info
    
 
class GPT2Model(GPT2PreTrainedModel):
    _keys_to_ignore_on_load_missing = ["attn.masked_bias"]

    def __init__(self, config):
        super().__init__(config)

        self.embed_dim = config.hidden_size 
        self.vocab_size = config.vocab_size
        self.max_position_embeddings = config.max_position_embeddings
        self.wte = nn.Embedding(self.vocab_size, self.embed_dim) 
        self.gpt_embedding_size = self.wte.weight.shape[1]
        self.len_prefix=config.len_prefix
        self.VAD_linear = nn.Linear(self.gpt_embedding_size+3, self.gpt_embedding_size)
        
        if self.len_prefix > 0:
            self.set_prefix_embeddings(config)
          
        self.wpe = nn.Embedding(self.max_position_embeddings, self.embed_dim)  
        self.drop = nn.Dropout(config.embd_pdrop) # 0.1
        self.h = nn.ModuleList([GPT2Block(config) for _ in range(config.num_hidden_layers)])  
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        self.init_weights()


    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings
        
    
    def set_prefix_embeddings(self,config):
        self.clip_project = Mappting((config.prefix_size, (self.gpt_embedding_size * config.len_prefix) // 2,
                                     self.gpt_embedding_size * config.len_prefix))
    

    def forward(
        self,
        input_ids=None, #
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,  # 
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None, # 
        encoder_attention_mask=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None, #
        emo_features =None, #
    ):
        if isinstance(encoder_hidden_states, Tuple):
            prefix_hidden = encoder_hidden_states[0]
            encoder_hidden_states = encoder_hidden_states[1]
        else:
            prefix_hidden=encoder_hidden_states
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions 
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        ) # False
        use_cache = use_cache if use_cache is not None else self.config.use_cache 
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict  

       
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h)) 
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1]) 

        # GPT2Attention mask.
        if attention_mask is not None: # False
            assert batch_size > 0, "batch_size has to be defined and > 0"
            attention_mask = attention_mask.view(batch_size, -1)
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.add_cross_attention and encoder_hidden_states is not None: 

            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask) 
        else:
            encoder_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer) 
        word_embeds = self.wte(input_ids[:,self.len_prefix:])
      
        if inputs_embeds is None:
            if self.len_prefix > 0 :
                prefix_embeds = self.clip_project(prefix_hidden).view(-1, self.len_prefix, self.gpt_embedding_size) 
                inputs_embeds = torch.cat((prefix_embeds,word_embeds),1)

            else:
                inputs_embeds = word_embeds
                prefix_embeds=None

        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        if emo_features is not None:
            emo_prefix_hidden_states = hidden_states[:,:self.len_prefix+6,:]
            exp_hidden_states = hidden_states[:,self.len_prefix+6:,:]

            
            exp_hidden_states = torch.cat((exp_hidden_states, emo_features[:,self.len_prefix+6:,:]),dim=2) 
            exp_hidden_states =self.VAD_linear(exp_hidden_states)

            hidden_states = torch.cat((emo_prefix_hidden_states, exp_hidden_states),1)


        output_shape = input_shape + (hidden_states.size(-1),)

        presents = () if use_cache else None  
        all_self_attentions = () if output_attentions else None  
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None  
        all_hidden_states = () if output_hidden_states else None 
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            outputs = block(
                hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                head_mask=head_mask[i],
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            hidden_states = outputs[0]
            if use_cache is True: 
                presents = presents + (outputs[1],)

            if output_attentions: 
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[3 if use_cache else 2],)


        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(*output_shape)
    

        # Add last hidden state
        if output_hidden_states: 
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions,prefix_embeds] if v is not None)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=(hidden_states,prefix_embeds),
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class GPT2LMHeadModel(GPT2PreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"attn.masked_bias", r"attn.bias", r"lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) 
        self.len_prefix=config.len_prefix
        self.use_distribution_loss = config.use_distribution_loss
        try:
            self.use_VAD_loss = config.use_VAD_loss 
        except:
            self.use_VAD_loss =  None
        self.encoder_dim = config.encoder_dim
       
        try:
            self.use_cl_loss=config.use_cl_loss
        except:
            self.use_cl_loss=None
        

        if self.use_cl_loss and self.use_cl_loss is not None:
            self.sim_mlp = ContrastiveHead(self.encoder_dim, self.transformer.gpt_embedding_size, self.transformer.gpt_embedding_size)

        if self.use_VAD_loss and self.use_VAD_loss is not None:
            self.VAD_head = nn.Linear(config.n_embd, 3)
            self.loss_mse = CustomMSELoss()

        self.init_weights()
     
    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        # only last token for inputs_ids if past is defined in kwargs
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None
        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

    def forward(
                self,
                input_ids=None, #
                past_key_values=None,
                attention_mask=None,
                token_type_ids=None, #
                position_ids=None, 
                head_mask=None,
                inputs_embeds=None,
                encoder_hidden_states=None, #
                encoder_attention_mask=None,
                labels=None,#
                use_cache=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None, #
                alpha = 1., # XE weight
                cl_Loss_weight=1.0,#
                emo_features =None, #
                VAD_labels = None, #
                VAD_Loss_weight=1.0, #
            ):   
       
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        transformer_outputs = self.transformer(
            input_ids, # input_ids
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids, # segment_ids
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states, # img_embeddings
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True, # True
            emo_features = emo_features,
        )
        hidden_states = transformer_outputs.last_hidden_state[0]  
        lm_logits = self.lm_head(hidden_states[:,self.len_prefix:,:])  


        loss = 0
        loss_exp_contrasive = None

        if True:
            lang_embeded = hidden_states[:,self.len_prefix:,:]
            emotion_embeded=lang_embeded[:,:6,:]
            emotion_embeded = emotion_embeded.mean(1)
            if lang_embeded.shape[1]>6:
                exp_embeded=lang_embeded[:,6:,:]
                exp_embeded = exp_embeded.mean(1)
            else:
                exp_embeded=None


            if isinstance(encoder_hidden_states, Tuple):
                encoder_mean = encoder_hidden_states[1].mean(1)
            else:
                encoder_mean = encoder_hidden_states.mean(1)
            
            encoder_embeded = encoder_mean

        if self.use_cl_loss and labels is not None:   
            pos_sim = self.sim_mlp(encoder_embeded, emotion_embeded, exp_embeded)[0]

            neg_sim=[]
            num_neg_exp=1
            for _ in range(num_neg_exp):
                size=exp_embeded.size()[0]
                    
                ## explanation contrasive learning for emotional perception 
                negvidid_emo=[]
                for idx in range(int(size/2)):
                    negvidid_emo.append(idx*2+1)
                    negvidid_emo.append(idx*2)
                exp_embeded_shuf=exp_embeded[negvidid_emo]

                neg_sim.append(self.sim_mlp(encoder_embeded, emotion_embeded, exp_embeded_shuf)[0])
               
                ## explanation contrasive learning for visual semantics 
                negvidid_0=neg_ids(int(size/2))*2
                negvidid_1=neg_ids(int(size/2))*2+1
                neg_list=[]
                for i in range(int(size/2)):
                    neg_list.append(negvidid_0[i])
                    neg_list.append(negvidid_1[i])
                negvidid_exp=torch.tensor(neg_list)
                exp_embeded_shuf=exp_embeded[negvidid_exp]
                neg_sim.append(self.sim_mlp(encoder_embeded, emotion_embeded, exp_embeded_shuf)[0])  
                

            neg_sim = torch.stack(neg_sim,-1)
            loss_exp_contrasive = -torch.log(epsilon + (torch.exp(pos_sim) / (epsilon + torch.exp(pos_sim)+ torch.exp(neg_sim).sum(-1))))
            loss_exp_contrasive =cl_Loss_weight *(loss_exp_contrasive.mean())
            loss = loss + loss_exp_contrasive

        loss_xe_text = None
        loss_exp_position=None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous() 
            shift_labels = labels[..., self.len_prefix+1:].contiguous()  
            # Flatten the tokens
            
            loss_fct = CrossEntropyLoss()
            loss_xe_text = alpha * loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = loss + loss_xe_text
        loss_xe=[]
        loss_xe.append(loss_xe_text)
        loss_xe.append(loss_exp_position)
        
        loss_VAD = None
        if self.use_VAD_loss and labels is not None:
            VAD_pres =self.VAD_head(hidden_states[:,self.len_prefix+5:,:])  
            shift_vadpres = VAD_pres[..., :-1, :].contiguous()  
            shift_vadlabels = VAD_labels[..., self.len_prefix+6:,:].contiguous()  

            loss_VAD = self.loss_mse(shift_vadpres, shift_vadlabels) 
            loss_VAD = loss_VAD * VAD_Loss_weight
            loss = loss + loss_VAD
            

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
           
            return ((loss,) + output) if loss is not None else output
        
        return CausalLMOutputWithCrossAttentions(
            loss=(loss,loss_xe,loss_exp_contrasive,loss_VAD),
            logits=(lm_logits,encoder_embeded,exp_embeded),
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

    @staticmethod
    def _reorder_cache(past: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the :obj:`past_key_values` cache if
        :meth:`~transformers.PreTrainedModel.beam_search` or :meth:`~transformers.PreTrainedModel.beam_sample` is
        called. This is required to match :obj:`past_key_values` with the correct beam_idx at every generation step.
        """
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past
        )
