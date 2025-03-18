# Custom ViT from T5
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py

from transformers.models.t5.modeling_t5 import (
    T5Model,
    T5Config,
    T5Stack,
    T5PreTrainedModel,
    T5Block,
    T5LayerNorm,
    T5LayerFF,
    T5LayerSelfAttention,
    T5Attention,
    T5LayerCrossAttention,
)

from transformers.modeling_outputs import (
    CausalLMOutputWithPast,
    BaseModelOutputWithPastAndCrossAttentions,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)



import math
import torch
from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
#encoder related code starts here 
# Unified Vision Transformer Embedding class
class VisionTransformerEmbedding(nn.Module):
    def __init__(self, embed_dim, config):
        super(VisionTransformerEmbedding, self).__init__()
        self.config = config
        self.embed_dim = embed_dim

        # Learnable scaling factors for the learnable normalization option
        if self.config.PE_mix_strategy in ['learnable_scaling_vec', 'weighted_sum_vec', 'weighted_sum_no_norm_vec']:
            self.position_scale = nn.Parameter(torch.ones(1, embed_dim))
            self.input_weight = nn.Parameter(torch.ones(1,embed_dim))
            self.position_weight = nn.Parameter(torch.ones(1,embed_dim))

        if self.config.PE_mix_strategy in ['learnable_scaling', 'weighted_sum', 'weighted_sum_no_norm']:
            self.position_scale = nn.Parameter(torch.ones(1))
            self.input_weight = nn.Parameter(torch.ones(1))
            self.position_weight = nn.Parameter(torch.ones(1))

        # Positional attention mechanism for the positional attention option
        if self.config.PE_mix_strategy == 'positional_attention':
            self.attention = nn.MultiheadAttention(embed_dim, num_heads=8)

        # Layer normalization for the layer normalization option
        if self.config.PE_mix_strategy == 'layer_norm':
            self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, inputs_embeds, position_embeds):
        strategy = self.config.PE_mix_strategy

        if strategy == 'hardcoded_normalization':
            inputs_embeds_norm = F.normalize(inputs_embeds, p=2, dim=-1)
            position_embeds_norm = F.normalize(position_embeds, p=2, dim=-1)
            output_embeds = inputs_embeds_norm + position_embeds_norm

        elif strategy in ['learnable_scaling','learnable_scaling_vec']:
            scaled_position_embeds = self.position_scale * position_embeds
            output_embeds = inputs_embeds + scaled_position_embeds

        elif strategy in ['weighted_sum','weighted_sum_vec']:
            inputs_embeds_norm = F.normalize(inputs_embeds, p=2, dim=-1)
            position_embeds_norm = F.normalize(position_embeds, p=2, dim=-1)
            output_embeds = (self.input_weight * inputs_embeds_norm) + (self.position_weight * position_embeds_norm)

        elif strategy in ['weighted_sum_no_norm','weighted_sum_no_norm_vec']:
            # Directly apply the weights without normalization
            output_embeds = (self.input_weight * inputs_embeds) + (self.position_weight * position_embeds)

        elif strategy == 'positional_attention':
            # Expanding position_embeds to match the batch size of inputs_embeds
            position_embeds_expanded = position_embeds.expand(inputs_embeds.shape[0], -1, -1)

            # Ensure the inputs are in the correct shape for MultiheadAttention (3D: [seq_len, batch_size, embed_dim])
            inputs_embeds_reshaped = inputs_embeds.transpose(0, 1)  # [batch_size, seq_len, embed_dim] -> [seq_len, batch_size, embed_dim]
            position_embeds_reshaped = position_embeds_expanded.transpose(0, 1)  # [batch_size, seq_len, embed_dim] -> [seq_len, batch_size, embed_dim]

            attn_output, _ = self.attention(inputs_embeds_reshaped, position_embeds_reshaped, position_embeds_reshaped)
            output_embeds = inputs_embeds_reshaped + attn_output

            # Transpose back to original shape
            output_embeds = output_embeds.transpose(0, 1)  # [seq_len, batch_size, embed_dim] -> [batch_size, seq_len, embed_dim]

        elif strategy == 'layer_norm':
            combined_embeds = inputs_embeds + position_embeds
            # Default comes with Learnable Scaling and Shifting
            output_embeds = self.layer_norm(combined_embeds)

        elif strategy == 'default':
            output_embeds = inputs_embeds + position_embeds

        else:
            raise ValueError(f"Unsupported PE_mix_strategy: {strategy}")

        return output_embeds


# https://github.com/McGill-NLP/length-generalization/blob/main/src/models/custom_t5_decoder_only.py
class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super().__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if bsz is not None:
            return pos_emb[None, :, :].expand(bsz, -1, -1)
        else:
            return pos_emb[None, :, :]


class FixedAbsolutePositionalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(16384).type_as(inv_freq)
        sinusoid_inp = torch.einsum("i , j -> i j", t, inv_freq)
        emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        self.embed = nn.Embedding.from_pretrained(emb, freeze=True)

    def forward(self, position_ids: torch.Tensor):
        return self.embed(position_ids.long())


class FixedRotaryPositionalEmbedding(nn.Module):
    def __init__(
        self, rotary_dim: int, rotary_base: int = 10000, max_position: int = 16384
    ):
        super().__init__()
        # This is an inverse frequency tensor
        # Each dimension has a higher denominator than the previous one
        # So, the frequency will be lower for higher dimensions
        inv_freq = 1.0 / (
            rotary_base ** (torch.arange(0, rotary_dim, 2).float() / rotary_dim)
        )  # [rotary_dim/2]

        # Now, we create frequencies for each position
        t = torch.arange(max_position, device=inv_freq.device, dtype=inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, inv_freq)  # [max_position, rotary_dim/2]

        sins = torch.sin(freqs)
        coss = torch.cos(freqs)

        emb = torch.cat([sins, coss], dim=-1)  # [max_position, rotary_dim]
        self.embed = nn.Embedding.from_pretrained(emb, freeze=True)

    def forward(self, position_ids: torch.Tensor):
        return self.embed(position_ids.long())

def fixed_pos_embedding(x, seq_dim=1, seq_len=None):
    dim = x.shape[-1]
    if seq_len is None:
        seq_len = x.shape[seq_dim]
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2) / dim))
    sinusoid_inp = (
        torch.einsum("i , j -> i j", torch.arange(seq_len), inv_freq)
        .to(x.device)
        .float()
    )
    return torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)


def rotate_every_two(x):
    """
    Example: [a, b, c, d] -> [-b, a, -d, c]
    """
    x1 = x[:, :, :, ::2]
    x2 = x[:, :, :, 1::2]
    x = torch.stack((-x2, x1), axis=-1)
    return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')


def apply_rotary_pos_emb(x, sincos, offset=0):
    sin, cos = map(
        lambda t: t[None, offset : x.shape[1] + offset, None, :].repeat_interleave(
            2, 3
        ),
        sincos,
    )
    # einsum notation for lambda t: repeat(t[offset:x.shape[1]+offset,:], "n d -> () n () (d j)", j=2)
    return (x * cos) + (rotate_every_two(x) * sin)


def apply_rotary_pos_emb_new(x, sincos, offset=0):
    sin, cos = map(
        lambda t: t[:, :, None, :].repeat_interleave(2, 3),
        sincos,
    )
    # einsum notation for lambda t: repeat(t[offset:x.shape[1]+offset,:], "n d -> () n () (d j)", j=2)
    return (x * cos) + (rotate_every_two(x) * sin)


class CustomT5Attention(T5Attention):
    def __init__(self, config: T5Config, has_relative_attention_bias=False, pos_enc_type="RPE", attn_type="self", rpe_type="abs"):
        super().__init__(config)

        #self.pos_enc_type = pos_enc_type
        # Alibi-rpe_sbias
        if "-" in pos_enc_type:
            pos_enc_split = pos_enc_type.split("-")
            self.pos_enc_type = pos_enc_split[0]
            self.struct_attn_type = pos_enc_split[1]
        else:
            self.pos_enc_type = pos_enc_type
            self.struct_attn_type = ""

        self.d_head = config.d_kv
        self.attn_type = attn_type
        self.rpe_type = rpe_type
        self.has_relative_attention_bias = has_relative_attention_bias

        if self.pos_enc_type == "RoPE":
            self.rotary_dim = None
            if getattr(config, "rotary_dim", None) is not None:
                self.rotary_dim = config.rotary_dim
            self.rotary_dim = int(0.25 * self.d_head)

        # Get the device from the configuration
        #device = torch.device("cuda" if torch.cuda.is_available() and config.device == 'cuda' else "cpu")
        if self.pos_enc_type != "RPE":
            self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.n_heads)
            device = self.relative_attention_bias.weight.device

        #print(f"has_relative_attention_bias:{has_relative_attention_bias}")
        if self.has_relative_attention_bias:
            if self.pos_enc_type == "RPE":
                self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.n_heads)
            elif self.pos_enc_type in ["Alibi","APEAlibi"]:
                #print(f"device:{device}")
                if self.struct_attn_type == "duo":
                    self.slopes_l = torch.Tensor(self.get_slopes(self.n_heads)).to(device)*-1
                    self.slopes_r = torch.Tensor(self.get_slopes(self.n_heads)).to(device)*-1
                elif self.struct_attn_type == "rpe_sbias":
                    self.slopes = torch.Tensor(self.get_slopes(self.n_heads)).to(device)*-1
                    self.struct_slopes = torch.Tensor(self.get_slopes(self.n_heads)).to(device)*-1
                else:
                    self.slopes = torch.Tensor(self.get_slopes(self.n_heads)).to(device)*-1
            elif self.pos_enc_type == "KerpleLog":
                self.eps = 1e-2
                self.bias_p = self.get_kerple_parameter(2, 'uniform',device)
                self.bias_a = self.get_kerple_parameter(1, 'uniform',device)
            elif self.pos_enc_type in ["NoPE", "LearnedAPE", "SinusoidalAPE","SinusoidalAPE2D", "RoPE"]:
                #self.relative_attention_bias = None  # No positional encoding bias
                pass
            else:
                self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.n_heads)
        # Add more types if necessary

    # Allocate weights and initialize.
    # The kernel has the form -p*log(1+a*|m-n|)
    def get_kerple_parameter(self,scale, init_method, device):
        if init_method == 'ones':
            return Parameter(torch.ones(
                            self.n_heads,
                            device=device,
                            )[:,None,None]*scale )
        elif init_method == 'uniform':
            return Parameter(torch.rand(
                            self.n_heads,
                            device=device,
                            )[:,None,None]*scale )

    # https://github.com/ofirpress/attention_with_linear_biases/issues/5
    def get_slopes(self, n):
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]

        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)                   #In the paper, we only train models that have 2^a heads for some a. This function has
        else:                                                 #some good properties that only occur when the input is a power of 2. To maintain that even
            closest_power_of_2 = 2**math.floor(math.log2(n))  #when the number of heads is not a power of 2, we use this workaround.
            return get_slopes_power_of_2(closest_power_of_2) + self.get_slopes(2*closest_power_of_2)[0::2][:n-closest_power_of_2]

    def compute_struct_bias(self, query_length, key_length, device=None, relative_position=None):
        """Compute binned relative position bias"""
        if device is None:
            device = self.relative_attention_bias.weight.device

        #print("#### Compute bias")
        if self.pos_enc_type in ["NoPE", "LearnedAPE", "SinusoidalAPE","SinusoidalAPE2D", "RoPE"]:
            return torch.zeros((1, self.n_heads, query_length, key_length), device=device)
        #elif self.pos_enc_type == "Alibi":
        elif self.pos_enc_type in ["Alibi","APEAlibi"]:
            if self.struct_attn_type == "duo":
                if relative_position is None:
                    context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
                    memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
                    relative_position = memory_position - context_position  # shape (query_length, key_length)

                if self.rpe_type == "abs":
                    relative_position = torch.abs(relative_position).unsqueeze(0).expand(self.n_heads, -1,-1)
                else:
                    relative_position = relative_position.unsqueeze(0).expand(self.n_heads, -1,-1)

                self.slopes_l = self.slopes_l.to(device)
                self.slopes_r = self.slopes_r.to(device)

                alibi_left = self.slopes_l.unsqueeze(1).unsqueeze(1) * relative_position
                alibi_right = self.slopes_r.unsqueeze(1).unsqueeze(1) * relative_position

                values = torch.triu(alibi_right) + torch.tril(alibi_left)
                values = values.view(1, self.n_heads, query_length, key_length) # shape (1, num_heads, query_length, key_length)
                return values
            else:
                if relative_position is None:
                    context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
                    memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
                    relative_position = memory_position - context_position  # shape (query_length, key_length)
                #else:
                    #Simple case here, every tree has the same distance matrix
                    #relative_position = relative_position.repeat(1, self.n_heads, 1, 1)

                if self.rpe_type == "abs":
                    relative_position = torch.abs(relative_position).unsqueeze(0).expand(self.n_heads, -1,-1)
                else:
                    relative_position = relative_position.unsqueeze(0).expand(self.n_heads, -1,-1)

                #print(f"relative_position.shape:{relative_position.shape}")
                #print(f"relative_position:{relative_position}")
                self.struct_slopes = self.struct_slopes.to(device)

                values = self.struct_slopes.unsqueeze(1).unsqueeze(1) * relative_position
                values = values.view(1, self.n_heads, query_length, key_length) # shape (1, num_heads, query_length, key_length)
                return values
        elif self.pos_enc_type == "KerpleLog":
            if relative_position is None:
                context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
                memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
                relative_position = memory_position - context_position  # shape (query_length, key_length)
            if self.rpe_type == "abs":
                relative_position = torch.abs(relative_position).unsqueeze(0).expand(self.n_heads, -1,-1)
            else:
                relative_position = relative_position.unsqueeze(0).expand(self.n_heads, -1,-1)

            self.bias_p.data = self.bias_p.data.clamp(min=self.eps)
            self.bias_a.data = self.bias_a.data.clamp(min=self.eps)

            self.bias_p = self.bias_p.to(device)
            self.bias_a = self.bias_a.to(device)

            values = -self.bias_p*torch.log(1+self.bias_a*relative_position) # log kernel # shape (num_heads, query_length, key_length)
            values = values.unsqueeze(0) # shape (1, num_heads, query_length, key_length)
            return values
        else:
            #context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
            #memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
            #relative_position = memory_position - context_position  # shape (query_length, key_length)
            if relative_position is None:
                context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
                memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
                relative_position = memory_position - context_position  # shape (query_length, key_length)
            relative_position_bucket = self._relative_position_bucket(
                relative_position,  # shape (query_length, key_length)
                bidirectional=(not self.is_decoder),
                num_buckets=self.relative_attention_num_buckets,
                max_distance=self.relative_attention_max_distance,
            )
            values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
            values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
            return values

    def compute_bias(self, query_length, key_length, device=None, relative_position=None):
        """Compute binned relative position bias"""
        if device is None:
            device = self.relative_attention_bias.weight.device

        #print("query_length",query_length)
        #print("key_length",key_length)

        #print("#### Compute bias")
        if self.pos_enc_type in ["NoPE", "LearnedAPE", "SinusoidalAPE","SinusoidalAPE2D", "RoPE"]:
            return torch.zeros((1, self.n_heads, query_length, key_length), device=device)
        #elif self.pos_enc_type == "Alibi":
        elif self.pos_enc_type in ["Alibi","APEAlibi"]:
            if self.struct_attn_type == "duo":
                relative_position = relative_position.to(device)

                if self.rpe_type == "abs":
                    relative_position = torch.abs(relative_position).unsqueeze(0).expand(self.n_heads, -1,-1)
                else:
                    relative_position = relative_position.unsqueeze(0).expand(self.n_heads, -1,-1)

                self.slopes_l = self.slopes_l.to(device)
                self.slopes_r = self.slopes_r.to(device)

                alibi_left = self.slopes_l.unsqueeze(1).unsqueeze(1) * relative_position
                alibi_right = self.slopes_r.unsqueeze(1).unsqueeze(1) * relative_position

                values = torch.triu(alibi_right) + torch.tril(alibi_left)
                # Slice the relevant part of the bias before reshaping
                values = values[:, :query_length, :key_length]  # Slicing the tensor before reshaping

                values = values.view(1, self.n_heads, query_length, key_length)  # shape (1, num_heads, query_length, key_length)
                #print(f"values.shape:{values.shape}")

                return values
            else:
                if relative_position is None:
                    context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
                    memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
                    relative_position = memory_position - context_position  # shape (query_length, key_length)
                #else:
                    #Simple case here, every tree has the same distance matrix
                    #relative_position = relative_position.repeat(1, self.n_heads, 1, 1)

                if self.rpe_type == "abs":
                    relative_position = torch.abs(relative_position).unsqueeze(0).expand(self.n_heads, -1,-1)
                else:
                    relative_position = relative_position.unsqueeze(0).expand(self.n_heads, -1,-1)

                #print(f"relative_position.shape:{relative_position.shape}")
                #print(f"relative_position:{relative_position}")
                self.slopes = self.slopes.to(device)

                values = self.slopes.unsqueeze(1).unsqueeze(1) * relative_position
                values = values.view(1, self.n_heads, query_length, key_length) # shape (1, num_heads, query_length, key_length)
                return values
        elif self.pos_enc_type == "KerpleLog":
            if relative_position is None:
                context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
                memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
                relative_position = memory_position - context_position  # shape (query_length, key_length)
            if self.rpe_type == "abs":
                relative_position = torch.abs(relative_position).unsqueeze(0).expand(self.n_heads, -1,-1)
            else:
                relative_position = relative_position.unsqueeze(0).expand(self.n_heads, -1,-1)

            self.bias_p.data = self.bias_p.data.clamp(min=self.eps)
            self.bias_a.data = self.bias_a.data.clamp(min=self.eps)

            self.bias_p = self.bias_p.to(device)
            self.bias_a = self.bias_a.to(device)

            values = -self.bias_p*torch.log(1+self.bias_a*relative_position) # log kernel # shape (num_heads, query_length, key_length)
            values = values.unsqueeze(0) # shape (1, num_heads, query_length, key_length)
            return values
        else:
            #context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
            #memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
            #relative_position = memory_position - context_position  # shape (query_length, key_length)
            if relative_position is None:
                context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
                memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
                relative_position = memory_position - context_position  # shape (query_length, key_length)
            relative_position_bucket = self._relative_position_bucket(
                relative_position,  # shape (query_length, key_length)
                bidirectional=(not self.is_decoder),
                num_buckets=self.relative_attention_num_buckets,
                max_distance=self.relative_attention_max_distance,
            )
            values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
            values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
            return values

    def forward(
        self,
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
        past_key_value=None,
        layer_head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
        relative_position=None,
        struct_position_bias=None,
    ):
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        """
        # Input is (batch_size, seq_length, dim)
        # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
        # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
        batch_size, seq_length = hidden_states.shape[:2]


        real_seq_length = seq_length

        if past_key_value is not None:
            if len(past_key_value) != 2:
                raise ValueError(
                    f"past_key_value should have 2 past states: keys and values. Got { len(past_key_value)} past states"
                )
            real_seq_length += past_key_value[0].shape[2] if query_length is None else query_length

        key_length = real_seq_length if key_value_states is None else key_value_states.shape[1]


        def shape(states):
            """projection"""
            return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

        def unshape(states):
            """reshape"""
            return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)

        def project(hidden_states, proj_layer, key_value_states, past_key_value):
            """projects hidden states correctly to key/query states"""
            if key_value_states is None:
                # self-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(hidden_states))
            elif past_key_value is None:
                # cross-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(key_value_states))

            if past_key_value is not None:
                if key_value_states is None:
                    # self-attn
                    # (batch_size, n_heads, key_length, dim_per_head)
                    hidden_states = torch.cat([past_key_value, hidden_states], dim=2)
                elif past_key_value.shape[2] != key_value_states.shape[1]:
                    # checking that the `sequence_length` of the `past_key_value` is the same as
                    # the provided `key_value_states` to support prefix tuning
                    # cross-attn
                    # (batch_size, n_heads, seq_length, dim_per_head)
                    hidden_states = shape(proj_layer(key_value_states))
                else:
                    # cross-attn
                    hidden_states = past_key_value
            return hidden_states

        #print(f"\nattn_type:{self.attn_type}")
        #print(f"hidden_states.shape:{hidden_states.shape}")
        #if key_value_states is not None:
        #    print(f"key_value_states.shape:{key_value_states.shape}")
        #if past_key_value is not None:
        #    print(f"past_key_value[0].shape:{past_key_value[0].shape}")

        # get query states
        query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
        #print(f"query_states.shape (before RoPE): {query_states.shape}")  # Check shape before RoPE

        # get key/value states
        if self.pos_enc_type == "RoPE":
            #key_states = shape(self.k(hidden_states))
            #findme
            key_states = project(
                hidden_states, self.k, key_value_states, past_key_value[0] if past_key_value is not None else None
            )

            #print(f"key_states2.shape (before RoPE): {key_states2.shape}")
        else:
            key_states = project(
                hidden_states, self.k, key_value_states, past_key_value[0] if past_key_value is not None else None
            )

        #print(f"key_states.shape (before RoPE): {key_states.shape}")

        value_states = project(
            hidden_states, self.v, key_value_states, past_key_value[1] if past_key_value is not None else None
        )

        attention_output_dict = {}

        #print(f"orig, key_states.shape:{key_states.shape}")
        #print(f"orig, query_states.shape:{query_states.shape}")

        #print(f"has_relative_attention_bias:{self.has_relative_attention_bias}")
        #print(f"attn_type:{self.attn_type}")
        #print(f"pos_enc_type:{self.pos_enc_type}")
        #print(f"rpe_type:{self.rpe_type}")

        if self.pos_enc_type == "RoPE":
            r_seq_len = hidden_states.shape[1]
            r_offset = 0

            if past_key_value is not None:
                # This is considering seq2seq auto-regressive generation case, while the absolute position is offset by + input_len
                # Can be turned off to test
                #print(f"past_key_value[0].shape:{past_key_value[0].shape}")
                r_offset = past_key_value[0].shape[2]
                r_seq_len += r_offset

            query_states = query_states.permute(0, 2, 1, 3)
            key_states = key_states.permute(0, 2, 1, 3)

            if self.rotary_dim is not None:

                k_rot = key_states[:, :, :, : self.rotary_dim]
                k_pass = key_states[:, :, :, self.rotary_dim :]

                q_rot = query_states[:, :, :, : self.rotary_dim]
                q_pass = query_states[:, :, :, self.rotary_dim :]

                sincos = fixed_pos_embedding(k_rot, 1, seq_len=r_seq_len)
                k_rot = apply_rotary_pos_emb(k_rot, sincos, offset=r_offset)
                q_rot = apply_rotary_pos_emb(q_rot, sincos, offset=r_offset)

                if output_attentions:
                    scores_pass = torch.matmul(
                        q_pass.permute(0, 2, 1, 3),
                        k_pass.permute(0, 2, 1, 3).transpose(3, 2),
                    )
                    attention_output_dict["scores_pass"] = scores_pass

                    scores_rot = torch.matmul(
                        q_rot.permute(0, 2, 1, 3),
                        k_rot.permute(0, 2, 1, 3).transpose(3, 2),
                    )
                    attention_output_dict["scores_rot"] = scores_rot

                key_states = torch.cat([k_rot, k_pass], dim=-1)
                query_states = torch.cat([q_rot, q_pass], dim=-1)
            else:
                sincos = fixed_pos_embedding(key_states, 1, seq_len=r_seq_len)
                key_states = apply_rotary_pos_emb(key_states, sincos, offset=r_offset)
                query_states = apply_rotary_pos_emb(
                    query_states, sincos, offset=r_offset
                )

            #print(f"inner,before_permute, key_states.shape:{key_states.shape}")
            #print(f"inner,before_permute, query_states.shape:{query_states.shape}")
            """
            inner,before_permute, key_states.shape:torch.Size([1, 2, 8, 64])
            inner,before_permute, query_states.shape:torch.Size([1, 1, 8, 64])
            """

            query_states = query_states.permute(0, 2, 1, 3)
            key_states = key_states.permute(0, 2, 1, 3)

            #Ignore this if it's already taken care of in project(hidden_states, proj_layer, key_value_states, past_key_value)
            """
            if past_key_value is not None:
                print(f"past_key_value[0].shape before concat: {past_key_value[0].shape}")
                key_states = torch.cat([past_key_value[0], key_states], dim=2)
            """

            #print(f"inner, key_states.shape:{key_states.shape}")
            #print(f"inner, key_states.transpose(3, 2).shape:{key_states.transpose(3, 2).shape}")
            #print(f"inner, query_states.shape:{query_states.shape}")
            """
            # At decoder for 3rd token self-attn
            attn_type:self
            hidden_states.shape:torch.Size([1, 1, 128])
            query_states.shape (before RoPE): torch.Size([1, 8, 1, 64])
            key_states.shape (before RoPE): torch.Size([1, 8, 2, 64])
            orig, key_states.shape:torch.Size([1, 8, 2, 64])
            orig, query_states.shape:torch.Size([1, 8, 1, 64])
            inner, key_states.shape:torch.Size([1, 8, 3, 64])  <- this should be [1, 8, 2, 64]
            inner, query_states.shape:torch.Size([1, 8, 1, 64])
            scores.shape:torch.Size([1, 8, 1, 3])
            mask.shape:torch.Size([1, 1, 1, 2])
            """


            scores = torch.matmul(
                query_states, key_states.transpose(3, 2)
            )  # equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9

            #print(f"scores.shape:{scores.shape}")
            #scores.shape:torch.Size([480, 8, 64, 64])
            #mask.shape:torch.Size([480, 1, 1, 64])

            # At 1st layer cross attn
            # scores.shape:torch.Size([1, 8, 1, 1])!!! for the first token it could be key_length=1 but why seq_length = 1 ??
            if mask is not None:
                #print(f"mask.shape:{mask.shape}")
                #scores += mask  # (batch_size, n_heads, seq_length, key_length)
                #scores = scores+mask  # (batch_size, n_heads, seq_length, key_length)
                expanded_mask = mask.expand_as(scores) # expand mask tensor to all heads
                #print(f"expanded_mask.shape:{expanded_mask.shape}")
                #print("mask",mask)
                #print("expanded_mask",expanded_mask)
                scores += expanded_mask
                #print("scores",scores)
                #print(f"scores.shape:{scores.shape}")
                #RuntimeError: output with shape [512, 8, 1, 1] doesn't match the broadcast shape [512, 8, 1, 64]

        else:
            # compute scores
            scores = torch.matmul(
                query_states, key_states.transpose(3, 2)
            )  # equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9

            #print(f"scores.shape:{scores.shape}")
            #scores.shape:torch.Size([480, 8, 64, 64])
            #print(f"self.attn_type",self.attn_type)

            if self.struct_attn_type == "rpe_sbias":
                if struct_position_bias is None:
                    if not self.has_relative_attention_bias:
                        #print("not has_relative_attention_bias")
                        struct_position_bias = torch.zeros(
                            (1, self.n_heads, real_seq_length, key_length), device=scores.device, dtype=scores.dtype
                        )
                        if self.gradient_checkpointing and self.training:
                            struct_position_bias.requires_grad = True
                    else:
                        struct_position_bias = self.compute_struct_bias(real_seq_length, key_length, device=scores.device, relative_position=relative_position)

                    # if key and values are already calculated
                    # we want only the last query position bias
                    if past_key_value is not None:
                        struct_position_bias = struct_position_bias[:, :, -hidden_states.size(1) :, :]

                    #print("struct_position_bias.shape:", position_bias.shape)
                    #struct_position_bias.shape: torch.Size([1, 8, 64, 64])
                    if mask is not None:
                        #print(f"mask.shape:{mask.shape}")
                        #mask.shape:torch.Size([480, 1, 1, 64])
                        struct_position_bias = struct_position_bias + mask  # (batch_size, n_heads, seq_length, key_length)
                        #print(f"position_bias.shape:{position_bias.shape}")
                        # torch.Size([480, 8, 64, 64])

                if self.pruned_heads:
                    mask = torch.ones(struct_position_bias.shape[1])
                    mask[list(self.pruned_heads)] = 0
                    struct_position_bias_masked = struct_position_bias[:, mask.bool()]
                else:
                    struct_position_bias_masked = struct_position_bias

                #print(f"struct_position_bias.shape:{struct_position_bias.shape}")
                #print(f"struct_position_bias_masked.shape:{struct_position_bias_masked.shape}")

            if position_bias is None:
                if not self.has_relative_attention_bias:
                    #print("not has_relative_attention_bias")
                    position_bias = torch.zeros(
                        (1, self.n_heads, real_seq_length, key_length), device=scores.device, dtype=scores.dtype
                    )
                    if self.gradient_checkpointing and self.training:
                        position_bias.requires_grad = True
                else:
                    if self.pos_enc_type in ["Alibi","APEAlibi"]:
                        position_bias = self.compute_bias(real_seq_length, key_length, device=scores.device, relative_position=relative_position)
                    else:
                        if self.struct_attn_type == "rpe_sbias":
                            position_bias = self.compute_bias(real_seq_length, key_length, device=scores.device, relative_position=None)
                        else:
                            position_bias = self.compute_bias(real_seq_length, key_length, device=scores.device, relative_position=None)
                #print(f"position_bias1.shape:{position_bias.shape}")

                # if key and values are already calculated
                # we want only the last query position bias
                if past_key_value is not None:
                    position_bias = position_bias[:, :, -hidden_states.size(1) :, :]

                #print(f"position_bias2.shape:{position_bias.shape}")

                #print("position_bias.shape:", position_bias.shape)
                #position_bias.shape: torch.Size([1, 8, 64, 64])
                if mask is not None:
                    #print(f"mask.shape:{mask.shape}")
                    #mask.shape:torch.Size([480, 1, 1, 64])
                    position_bias = position_bias + mask  # (batch_size, n_heads, seq_length, key_length)
                    #print(f"masked position_bias.shape:{position_bias.shape}")
                    # torch.Size([480, 8, 64, 64])

                #print(f"position_bias3.shape:{position_bias.shape}")

            if self.pruned_heads:
                mask = torch.ones(position_bias.shape[1])
                mask[list(self.pruned_heads)] = 0
                position_bias_masked = position_bias[:, mask.bool()]
            else:
                position_bias_masked = position_bias

            #print(f"position_bias.shape:{position_bias.shape}")
            #print(f"position_bias_masked.shape:{position_bias_masked.shape}")
            #print(f"scores.shape:{scores.shape}")

            if self.struct_attn_type == "rpe_sbias" and self.attn_type == "self":
                scores += position_bias_masked + struct_position_bias_masked
            else:
                scores += position_bias_masked

        attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
            scores
        )  # (batch_size, n_heads, seq_length, key_length)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )  # (batch_size, n_heads, seq_length, key_length)

        # Mask heads if we want to
        if layer_head_mask is not None:
            attn_weights = attn_weights * layer_head_mask

        attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
        attn_output = self.o(attn_output)

        present_key_value_state = (key_states, value_states) if (self.is_decoder and use_cache) else None
        """
        if self.struct_attn_type == "rpe_sbias":
            outputs = (attn_output,) + (present_key_value_state,) + (position_bias,) + (struct_position_bias,)
        else:
            outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)
        """

        outputs = (attn_output,) + (present_key_value_state,) + (position_bias,) + (struct_position_bias,)

        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs


from transformers.models.t5.modeling_t5 import T5LayerSelfAttention, T5LayerCrossAttention
import copy

class CustomT5LayerSelfAttention(T5LayerSelfAttention):
    def __init__(self, config, has_relative_attention_bias=False, pos_enc_type="RPE", rpe_type="abs"):
        super().__init__(config, has_relative_attention_bias)
        self.pos_enc_type=pos_enc_type
        self.rpe_type=rpe_type
        self.SelfAttention = CustomT5Attention(config, has_relative_attention_bias=has_relative_attention_bias, pos_enc_type=pos_enc_type, attn_type="self", rpe_type=rpe_type)
        self.is_decoder = config.is_decoder

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
        relative_position=None,
        struct_position_bias=None,
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.SelfAttention(
            normed_hidden_states,
            mask=attention_mask,
            position_bias=position_bias,
            struct_position_bias=struct_position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
            relative_position=relative_position,
        )
        hidden_states = hidden_states + self.dropout(attention_output[0])
        outputs = (hidden_states,) + attention_output[1:]  # add attentions if we output them
        return outputs

class CustomT5LayerCrossAttention(T5LayerCrossAttention):
    def __init__(self, config, pos_enc_type="RPE", rpe_type="abs"):
        super().__init__(config)
        self.pos_enc_type=pos_enc_type
        self.rpe_type=rpe_type
        self.EncDecAttention = CustomT5Attention(config, has_relative_attention_bias=False, pos_enc_type=pos_enc_type, attn_type="cross", rpe_type=rpe_type)
        self.is_decoder = config.is_decoder

    def forward(
        self,
        hidden_states,
        key_value_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        query_length=None,
        output_attentions=False,
        relative_position=None,
        struct_position_bias=None,
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.EncDecAttention(
            normed_hidden_states,
            mask=attention_mask,
            key_value_states=key_value_states,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            query_length=query_length,
            output_attentions=output_attentions,
            relative_position=relative_position,
            struct_position_bias=struct_position_bias,
        )
        layer_output = hidden_states + self.dropout(attention_output[0])
        outputs = (layer_output,) + attention_output[1:]  # add attentions if we output them
        return outputs

from transformers.models.t5.modeling_t5 import T5Block, T5LayerFF

class CustomT5Block(T5Block):
    def __init__(self, config, has_relative_attention_bias=False, pos_enc_type="RPE", rpe_type="abs"):
        super().__init__(config, has_relative_attention_bias)
        self.pos_enc_type=pos_enc_type
        self.rpe_type=rpe_type
        self.layer = nn.ModuleList()
        self.layer.append(CustomT5LayerSelfAttention(config, has_relative_attention_bias=has_relative_attention_bias, pos_enc_type=pos_enc_type, rpe_type=rpe_type))
        if self.is_decoder:
            self.layer.append(CustomT5LayerCrossAttention(config, pos_enc_type=pos_enc_type, rpe_type=rpe_type))
        self.layer.append(T5LayerFF(config))

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_decoder_position_bias=None,
        encoder_decoder_struct_position_bias=None,
        layer_head_mask=None,
        cross_attn_layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
        return_dict=True,
        relative_position=None,
        struct_position_bias=None,
    ):
        if past_key_value is not None:
            if not self.is_decoder:
                logger.warning("`past_key_values` is passed to the encoder. Please make sure this is intended.")
            expected_num_past_key_values = 2 if encoder_hidden_states is None else 4

            if len(past_key_value) != expected_num_past_key_values:
                raise ValueError(
                    f"There should be {expected_num_past_key_values} past states. "
                    f"{'2 (key / value) for cross attention. ' if expected_num_past_key_values == 4 else ''}"
                    f"Got {len(past_key_value)} past key / value states"
                )

            self_attn_past_key_value = past_key_value[:2]
            cross_attn_past_key_value = past_key_value[2:]
        else:
            self_attn_past_key_value, cross_attn_past_key_value = None, None

        self_attention_outputs = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=self_attn_past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
            relative_position=relative_position,
            struct_position_bias=struct_position_bias,
        )
        hidden_states, present_key_value_state = self_attention_outputs[:2]
        attention_outputs = self_attention_outputs[2:]  # Keep self-attention outputs and relative position weights

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16:
            clamp_value = torch.where(
                torch.isinf(hidden_states).any(),
                torch.finfo(hidden_states.dtype).max - 1000,
                torch.finfo(hidden_states.dtype).max,
            )
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        do_cross_attention = self.is_decoder and encoder_hidden_states is not None
        if do_cross_attention:
            # the actual query length is unknown for cross attention
            # if using past key value states. Need to inject it here
            if present_key_value_state is not None:
                query_length = present_key_value_state[0].shape[2]
            else:
                query_length = None

            cross_attention_outputs = self.layer[1](
                hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                position_bias=encoder_decoder_position_bias,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                query_length=query_length,
                use_cache=use_cache,
                output_attentions=output_attentions,
                struct_position_bias=encoder_decoder_struct_position_bias,
                relative_position=relative_position,
            )
            hidden_states = cross_attention_outputs[0]

            # clamp inf values to enable fp16 training
            if hidden_states.dtype == torch.float16:
                clamp_value = torch.where(
                    torch.isinf(hidden_states).any(),
                    torch.finfo(hidden_states.dtype).max - 1000,
                    torch.finfo(hidden_states.dtype).max,
                )
                hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

            # Combine self attn and cross attn key value states
            if present_key_value_state is not None:
                present_key_value_state = present_key_value_state + cross_attention_outputs[1]

            # Keep cross-attention outputs and relative position weights
            attention_outputs = attention_outputs + cross_attention_outputs[2:]

        # Apply Feed Forward layer
        hidden_states = self.layer[-1](hidden_states)

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16:
            clamp_value = torch.where(
                torch.isinf(hidden_states).any(),
                torch.finfo(hidden_states.dtype).max - 1000,
                torch.finfo(hidden_states.dtype).max,
            )
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if use_cache:
            outputs = outputs + (present_key_value_state,) + attention_outputs
        else:
            outputs = outputs + attention_outputs

        return outputs  # hidden-states, present_key_value_states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)


from transformers.models.t5.modeling_t5 import T5Stack
import numpy as np
from pathlib import Path
import logging
import os
logger = logging.getLogger("debug")

class CustomT5Stack(T5Stack):
    def __init__(self, config, embed_tokens=None, pos_enc_type="RPE", rpe_type="abs"):
        super().__init__(config, embed_tokens)
        #self.pos_enc_type=pos_enc_type

        # Alibi-rpe_sbias
        if "-" in pos_enc_type:
            pos_enc_split = pos_enc_type.split("-")
            self.pos_enc_type = pos_enc_split[0]
            self.struct_attn_type = pos_enc_split[1]
        else:
            self.pos_enc_type = pos_enc_type
            self.struct_attn_type = ""

        self.rpe_type=rpe_type
        self.block = nn.ModuleList(
            [CustomT5Block(config, has_relative_attention_bias=bool(i == 0), pos_enc_type=pos_enc_type, rpe_type=rpe_type) for i in range(config.num_layers)]
        )

        self.PE_mixer = VisionTransformerEmbedding(config.d_model, config)
        self.config = config

        if self.pos_enc_type == "LearnedAPE":
            self.wpe = nn.Embedding(2048, config.d_model)
            self.wpe.weight.data.normal_(
                    mean=0.0, std=config.initializer_factor * 1.0
            )

            """
            parent_dir = Path(os.path.dirname(os.path.abspath(__file__)))
            learned_embed_file = parent_dir / "gpt_neo_125m_pos_embed.npy"
            if learned_embed_file.exists():
                logger.info(
                    "Loading position embedding from {}".format(learned_embed_file)
                )

                weight = np.load(str(learned_embed_file))
                self.wpe.weight.data.copy_(torch.from_numpy(weight))
                self.wpe.weight.requires_grad = False
            else:
                self.wpe.weight.data.normal_(
                    mean=0.0, std=config.initializer_factor * 1.0
                )
            """

        if self.pos_enc_type == "SinusoidalAPE":
            self.wpe = FixedAbsolutePositionalEmbedding(config.d_model)
        
        if self.pos_enc_type in ["SinusoidalAPE2D","APEAlibi-duo","APEAlibi"]:
            # 2D APE for encoder and cross attn
            # A norminate obj_id just to test
            if config.use_objidx=="yes":
                self.wpe_obj_enc = FixedAbsolutePositionalEmbedding(config.d_model/2) # 128/2 -> 64
                self.wpe_x_enc = FixedAbsolutePositionalEmbedding(config.d_model/4) # 128/4 -> 32
                self.wpe_y_enc = FixedAbsolutePositionalEmbedding(config.d_model/4) # 128/4 -> 32

            # Decoder is the same old 2D
            self.wpe_x = FixedAbsolutePositionalEmbedding(config.d_model/2) # 128/2 -> 64
            self.wpe_y = FixedAbsolutePositionalEmbedding(config.d_model/2) # 128/2 -> 64

            # 1D APE for decoder/ non-2d positions
            self.wpe = FixedAbsolutePositionalEmbedding(config.d_model)

        if self.pos_enc_type in ["Alibi-duo", "Alibi", "APEAlibi-duo", "APEAlibi"]:
            # Calculate relative positions for the 2D grid
            grid_height = self.config.grid_max_height
            grid_width = self.config.grid_max_width
            large_dist = max(grid_height,grid_width)+2
            relative_position_2d = self.calculate_2d_relative_positions(grid_height, grid_width)

            # Create a relative position matrix for the full sequence including <s> and </s>
            total_length = grid_height * grid_width + 2  # +2 for <s> and </s>
            distance_matrix = torch.full((total_length, total_length), fill_value=large_dist)  # 100 as a large distance

            # Assign the 2D relative positions to the correct part of the matrix
            distance_matrix[1:1 + grid_height * grid_width, 1:1 + grid_height * grid_width] = relative_position_2d

            # Optionally handle <s> and </s> relative positions
            distance_matrix[0, :] = large_dist  # <s> is far from everything
            distance_matrix[:, 0] = large_dist
            distance_matrix[-1, :] = large_dist+1  # </s> is far from everything
            distance_matrix[:, -1] = large_dist+1

            self.distance_matrix_2D = distance_matrix
            #self.register_buffer("distance_matrix", self.distance_matrix)

    def calculate_2d_relative_positions(self, grid_height, grid_width):
        # Create grid coordinates
        x_coords, y_coords = torch.meshgrid(
            torch.arange(grid_height, dtype=torch.long),
            torch.arange(grid_width, dtype=torch.long),
            indexing='ij'
        )

        # Flatten the 2D grid coordinates
        x_flat = x_coords.flatten()
        y_flat = y_coords.flatten()

        # Initialize the relative position matrix
        num_positions = grid_height * grid_width
        relative_position = torch.zeros((num_positions, num_positions), dtype=torch.long)

        # Calculate Manhattan distance between each pair of points
        for i in range(num_positions):
            for j in range(num_positions):
                relative_position[i, j] = abs(x_flat[i] - x_flat[j]) + abs(y_flat[i] - y_flat[j])

        return relative_position


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        position_ids=None,
        return_dict=None,
        relative_position=None,
        object_idx=None,
    ):
        # Model parallel
        if self.model_parallel:
            torch.cuda.set_device(self.first_device)
            self.embed_tokens = self.embed_tokens.to(self.first_device)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds")

        if self.pos_enc_type in ["Alibi-duo", "Alibi", "APEAlibi-duo", "APEAlibi"]:
            relative_position = self.distance_matrix_2D

        #print(f"input_ids.shape:{input_ids.shape}")
        # Print the shape of the embedding matrix
        #print(f"Embedding matrix shape: {self.embed_tokens.weight.shape}")
        # Print unique values in input_ids
        #unique_input_ids = torch.unique(input_ids)
        #print(f"Unique input IDs: {unique_input_ids}")
        #print(f"Max input ID: {torch.max(unique_input_ids)}")
        #print(f"Min input ID: {torch.min(unique_input_ids)}")

        if inputs_embeds is None:
            if self.embed_tokens is None:
                raise ValueError("You have to initialize the model with valid token embeddings")
            inputs_embeds = self.embed_tokens(input_ids)

        #print(f"inputs_embeds.shape:{inputs_embeds.shape}")

        batch_size, seq_length = input_shape

        # required mask seq length can be calculated via length of past
        mask_seq_length = past_key_values[0][0].shape[2] + seq_length if past_key_values is not None else seq_length
        #print(f"mask_seq_length:{mask_seq_length}")


        # Add 2D position embeddings, but only on input seq
        if self.pos_enc_type in [
            "SinusoidalAPE2D","APEAlibi-duo","APEAlibi"
        ]:
            if self.is_decoder or self.config.use_objidx!="yes":
                if position_ids is not None:
                    position_ids = position_ids.view(-1, input_shape[-1])

                if past_key_values is None:
                    past_length = 0
                else:
                    past_length = past_key_values[0][0].size(-2)

                device = input_ids.device if input_ids is not None else inputs_embeds.device
                if position_ids is None:
                    position_ids = torch.arange(
                        past_length,
                        input_shape[-1] + past_length,
                        dtype=torch.long,
                        device=device,
                    )
                    position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

                #print(f"position_ids.shape:{position_ids.shape}")
                #print(f"position_ids:{position_ids}")

                if position_ids.shape[-1] == 1024 or position_ids.shape[-1] == 1025 or True:
                #if position_ids.shape[-1] == 1024 or position_ids.shape[-1] == 1025:
                    # Desired dimensions for ARC IO, individually
                    # For decoder because we have <pad> as first token
                    rows = self.config.grid_max_height
                    cols = self.config.grid_max_width

                    # Flatten the position_ids tensor to remove batch dimension
                    flat_position_ids = position_ids.view(-1)
                    #print(f"flat_position_ids.shape:{flat_position_ids.shape}")
                    #print(f"flat_position_ids:{flat_position_ids}")

                    # Generate position_ids_x
                    position_ids_x = torch.arange(cols, device=device).repeat(rows)

                    # Generate position_ids_y
                    position_ids_y = torch.arange(rows, device=device).repeat_interleave(cols)

                    # Handling batch size, repeat for each batch
                    batch_size = position_ids.shape[0]
                    position_ids_x = position_ids_x.repeat(batch_size, 1)
                    position_ids_y = position_ids_y.repeat(batch_size, 1)

                    #position_embeds = self.wpe(position_ids)
                    position_embeds_x = self.wpe_x(position_ids_x)
                    position_embeds_y = self.wpe_y(position_ids_y)
                    #print(f"position_embeds_x.shape:{position_embeds_x.shape}")

                    #position_embeds
                    position_embeds_2d = torch.cat((position_embeds_x, position_embeds_y), dim=-1)
                    # Apply 1D sinAPE for the <pad> token and tokens beyond 2+1024
                    position_embeds_1d = self.wpe(position_ids)
                    if self.is_decoder:
                        # Combine embeddings
                        position_embeds = position_embeds_1d.clone()
                        #print(f"position_embeds=position_embeds_1d.clone().shape:{position_embeds.shape}")

                        p_seq_len = position_ids.shape[-1]
                        #print(f"p_seq_len:{p_seq_len}")
                        if p_seq_len >= 1123:
                          position_embeds[:, 1:1123] = position_embeds_2d[:, :1122]
                        elif p_seq_len == 1:
                          pos_index = flat_position_ids[0]
                          if pos_index == 0:
                            # <pad> for 1d APE
                            pass
                          elif pos_index>1 and pos_index<=1122:
                            # For model.generate() this will always be 1, but position_ids=(bs, pos_index)
                            position_embeds[:, 0] = position_embeds_2d[:, pos_index-1]
                          else:
                            # > 1025
                            pass
                        else:
                          #print(f"position_embeds.shape:{position_embeds.shape}")
                          #print(f"position_embeds_2d.shape:{position_embeds_2d.shape}")
                          #print(f"position_embeds[:, 1:p_seq_len].shape:{position_embeds[:, 1:p_seq_len].shape}")
                          #print(f"position_embeds_2d[:, :p_seq_len-1].shape:{position_embeds_2d[:, :p_seq_len-1].shape}")
                          position_embeds[:, 1:p_seq_len] = position_embeds_2d[:, :p_seq_len-1]
                    else:
                        position_embeds = position_embeds_1d.clone()
                        position_embeds[:, 1:1123] = position_embeds_2d[:, :1122]
                else:
                    # 1D sinAPE
                    position_embeds = self.wpe(position_ids)
            else:
                # if NOT self.is_decoder:
                if position_ids is not None:
                    position_ids = position_ids.view(-1, input_shape[-1])

                if past_key_values is None:
                    past_length = 0
                else:
                    past_length = past_key_values[0][0].size(-2)

                device = input_ids.device if input_ids is not None else inputs_embeds.device
                if position_ids is None:
                    position_ids = torch.arange(
                        past_length,
                        input_shape[-1] + past_length,
                        dtype=torch.long,
                        device=device,
                    )
                    position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

                #print(f"position_ids.shape:{position_ids.shape}")
                #print(f"position_ids:{position_ids}")

                if position_ids.shape[-1] == 1024 or position_ids.shape[-1] == 1025 or True:
                #if position_ids.shape[-1] == 1024 or position_ids.shape[-1] == 1025:
                    # Desired dimensions for ARC IO, individually
                    # For decoder because we have <pad> as first token
                    rows = self.config.grid_max_height
                    cols = self.config.grid_max_width

                    # Flatten the position_ids tensor to remove batch dimension
                    flat_position_ids = position_ids.view(-1)
                    #print(f"flat_position_ids.shape:{flat_position_ids.shape}")
                    #print(f"flat_position_ids:{flat_position_ids}")

                    # Generate position_ids_x
                    position_ids_x = torch.arange(cols, device=device).repeat(rows)

                    # Generate position_ids_y
                    position_ids_y = torch.arange(rows, device=device).repeat_interleave(cols)

                    # Handling batch size, repeat for each batch
                    batch_size = position_ids.shape[0]
                    position_ids_x = position_ids_x.repeat(batch_size, 1)
                    position_ids_y = position_ids_y.repeat(batch_size, 1)

                    # Get the object embeddings
                    object_embeds = self.wpe_obj_enc(object_idx[:, 1:-1])  # Assuming `object_idx` is passed in
                    #print(f"object_idx.shape:{object_idx.shape}")
                    #print(f"object_embeds.shape:{object_embeds.shape}")

                    #position_embeds = self.wpe(position_ids)
                    position_embeds_x = self.wpe_x_enc(position_ids_x)
                    #print(f"position_ids_x.shape:{position_ids_x.shape}")
                    #print(f"position_embeds_x.shape:{position_embeds_x.shape}")
                    position_embeds_y = self.wpe_y_enc(position_ids_y)

                    # Expand position_embeds_x and position_embeds_y to match the batch size
                    position_embeds_x = position_embeds_x.expand(object_embeds.size(0), -1, -1)  # Expand along the batch size
                    position_embeds_y = position_embeds_y.expand(object_embeds.size(0), -1, -1)  # Expand along the batch size

                    #position_embeds
                    #position_embeds_2d = torch.cat((position_embeds_x, position_embeds_y), dim=-1)
                    position_embeds_2d = torch.cat((object_embeds, position_embeds_x, position_embeds_y), dim=-1)

                    # Apply 1D sinAPE for the <pad> token and tokens beyond 2+1024
                    position_embeds_1d = self.wpe(position_ids)
                    position_embeds_1d = position_embeds_1d.expand(object_embeds.size(0), -1, -1)  # Expand along the batch size

                    position_embeds = position_embeds_1d.clone()
                    position_embeds[:, 1:1123] = position_embeds_2d[:, :1122]
                else:
                    # 1D sinAPE
                    position_embeds = self.wpe(position_ids)

            #print(f"position_embeds.shape:{position_embeds.shape}")
            #print(f"position_embeds:{position_embeds}")
            #inputs_embeds += position_embeds
            inputs_embeds = self.PE_mixer(inputs_embeds, position_embeds)

        if self.pos_enc_type in [
            "SinusoidalAPE",
            "LearnedAPE",
        ]:
            if position_ids is not None:
                position_ids = position_ids.view(-1, input_shape[-1])

            if past_key_values is None:
                past_length = 0
            else:
                past_length = past_key_values[0][0].size(-2)

            device = input_ids.device if input_ids is not None else inputs_embeds.device
            if position_ids is None:
                position_ids = torch.arange(
                    past_length,
                    input_shape[-1] + past_length,
                    dtype=torch.long,
                    device=device,
                )
                position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

            #print(f"position_ids.shape:{position_ids.shape}")
            position_embeds = self.wpe(position_ids)
            #print(f"position_embeds.shape:{position_embeds.shape}")
            inputs_embeds += position_embeds

            if self.struct_attn_type == "ape_sbias":
                # Extra APE, naive trial
                if relative_position is not None:
                    struct_position_ids = relative_position.view(-1, input_shape[-1])
                    #print(relative_position)
                    #print(f"struct_position_ids.shape:{struct_position_ids.shape}")
                    #print(struct_position_ids)
                    struct_position_embeds = self.wpe(struct_position_ids)
                    #print(f"struct_position_embeds.shape:{struct_position_embeds.shape}")
                    inputs_embeds += struct_position_embeds

        if use_cache is True:
            if not self.is_decoder:
                raise ValueError(f"`use_cache` can only be set to `True` if {self} is used as a decoder")

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(
                    encoder_hidden_shape, device=inputs_embeds.device, dtype=torch.long
                )
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        cross_attn_head_mask = self.get_head_mask(cross_attn_head_mask, self.config.num_layers)
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None
        position_bias = None
        struct_position_bias = None
        encoder_decoder_position_bias = None
        encoder_decoder_struct_position_bias = None

        hidden_states = self.dropout(inputs_embeds)

        for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):
            layer_head_mask = head_mask[i]
            cross_attn_layer_head_mask = cross_attn_head_mask[i]
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if position_bias is not None:
                    position_bias = position_bias.to(hidden_states.device)
                if struct_position_bias is not None:
                    struct_position_bias = struct_position_bias.to(hidden_states.device)
                if encoder_hidden_states is not None:
                    encoder_hidden_states = encoder_hidden_states.to(hidden_states.device)
                if encoder_extended_attention_mask is not None:
                    encoder_extended_attention_mask = encoder_extended_attention_mask.to(hidden_states.device)
                if encoder_decoder_position_bias is not None:
                    encoder_decoder_position_bias = encoder_decoder_position_bias.to(hidden_states.device)
                if encoder_decoder_struct_position_bias is not None:
                    encoder_decoder_struct_position_bias = encoder_decoder_struct_position_bias.to(hidden_states.device)
                if layer_head_mask is not None:
                    layer_head_mask = layer_head_mask.to(hidden_states.device)
                if cross_attn_layer_head_mask is not None:
                    cross_attn_layer_head_mask = cross_attn_layer_head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.forward,
                    hidden_states,
                    extended_attention_mask,
                    position_bias,
                    encoder_hidden_states,
                    encoder_extended_attention_mask,
                    encoder_decoder_position_bias,
                    layer_head_mask,
                    cross_attn_layer_head_mask,
                    None,  # past_key_value is always None with gradient checkpointing
                    use_cache,
                    output_attentions,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=extended_attention_mask,
                    position_bias=position_bias,
                    struct_position_bias=struct_position_bias,  # Pass the struct_position_bias to the layer
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_extended_attention_mask,
                    encoder_decoder_position_bias=encoder_decoder_position_bias,
                    encoder_decoder_struct_position_bias=encoder_decoder_struct_position_bias,
                    layer_head_mask=layer_head_mask,
                    cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    relative_position=relative_position,  # Pass the relative_position to the layer
                )

            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)
            # hidden-states, key-value-states, (self-attention position bias), (self-attention struct position bias), (self-attention weights),
            #                                  (cross-attention position bias), (cross-attention struct position bias), (cross-attention weights)
            if use_cache is False:
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

            hidden_states, present_key_value_state = layer_outputs[:2]

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention position bias), (self-attention weights),
            # (cross-attention position bias), (cross-attention weights)
            position_bias = layer_outputs[2]
            struct_position_bias = layer_outputs[3]
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[5 if output_attentions else 4]
                encoder_decoder_struct_position_bias = layer_outputs[7 if output_attentions else 5]

            # append next layer key value states
            if use_cache:
                present_key_value_states = present_key_value_states + (present_key_value_state,)

            """
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[3],)
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[5],)
            """

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[4],)
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[6],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    present_key_value_states,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )


from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration, T5Config

import copy
import math
import os
import warnings
from typing import List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
    Seq2SeqQuestionAnsweringModelOutput,
    Seq2SeqSequenceClassifierOutput,
    TokenClassifierOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS, find_pruneable_heads_and_indices, prune_linear_layer
from transformers.utils import (
    DUMMY_INPUTS,
    DUMMY_MASK,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_torch_fx_proxy,
    logging,
    replace_return_docstrings,
)
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map
from transformers.models.t5.configuration_t5 import T5Config

# Warning message for FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
__HEAD_MASK_WARNING_MSG = """
The input argument `head_mask` was split into two arguments `head_mask` and `decoder_head_mask`. Currently,
`decoder_head_mask` is set to copy `head_mask`, but this feature is deprecated and will be removed in future versions.
If you do not want to use any `decoder_head_mask` now, please set `decoder_head_mask = torch.ones(num_layers,
num_heads)`.
"""

class CustomT5ForConditionalGeneration(T5ForConditionalGeneration):
    def __init__(self, config: T5Config, pos_enc_type="RPE", rpe_type="abs"):
        super().__init__(config)
        self.model_dim = config.d_model
        self.pos_enc_type=pos_enc_type
        self.rpe_type=rpe_type

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = CustomT5Stack(encoder_config, self.shared, pos_enc_type=pos_enc_type, rpe_type=rpe_type)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = CustomT5Stack(decoder_config, self.shared, pos_enc_type=pos_enc_type, rpe_type=rpe_type)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None
 
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # customized distance_matrix w.r.t to encoder self-attention
        distance_matrix: Optional[torch.FloatTensor] = None,
        object_idx: Optional[torch.FloatTensor] = None,
        # unlike nlp [0,..n] natural sequence, customized struct_position_indexs
        # For now, just re-use distance_matrix if APE-sbias
        #struct_position_indexs: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to `-100` are ignored (masked), the loss is only computed for
            labels in `[0, ..., config.vocab_size]`

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, T5ForConditionalGeneration

        >>> tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
        >>> model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-small")

        >>> # training
        >>> input_ids = tokenizer("The <extra_id_0> walks in <extra_id_1> park", return_tensors="pt").input_ids
        >>> labels = tokenizer("<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", return_tensors="pt").input_ids
        >>> outputs = model(input_ids=input_ids, labels=labels)
        >>> loss = outputs.loss
        >>> logits = outputs.logits

        >>> # inference
        >>> input_ids = tokenizer(
        ...     "summarize: studies have shown that owning a dog is good for you", return_tensors="pt"
        ... ).input_ids  # Batch size 1
        >>> outputs = model.generate(input_ids)
        >>> print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        >>> # studies have shown that owning a dog is good for you.
        ```"""
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                relative_position=distance_matrix,  # Pass the distance_matrix here
                object_idx=object_idx,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            # move labels to correct device to enable PP
            labels = labels.to(lm_logits.device)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
