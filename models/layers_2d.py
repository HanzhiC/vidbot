from torch import nn
from torch.nn import functional as F
import numpy as np

from typing import Optional, Tuple
import warnings
import torch
from torch import Tensor
from torch.nn import Linear
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
from torch.nn.init import xavier_normal_
from torch.nn.parameter import Parameter
from torch.nn import Module
from torch.nn import functional as F

from models.layers_3d import RotaryPositionEncoding
import models.clip.clip as clip
from models.clip.model import ModifiedResNet
import math
from einops.layers.torch import Rearrange
from torch.utils.checkpoint import checkpoint
from torchvision.models import resnet50, ResNet50_Weights


 
def load_clip():
    clip_model, clip_transforms = clip.load("RN50")
    state_dict = clip_model.state_dict()
    layers = tuple(
        [
            len(
                set(
                    k.split(".")[2]
                    for k in state_dict
                    if k.startswith(f"visual.layer{b}")
                )
            )
            for b in [1, 2, 3, 4]
        ]
    )
    output_dim = state_dict["text_projection"].shape[1]
    heads = state_dict["visual.layer1.0.conv1.weight"].shape[0] * 32 // 64
    backbone = ModifiedResNetFeatures(layers, output_dim, heads)
    backbone.load_state_dict(clip_model.visual.state_dict())
    normalize = clip_transforms.transforms[-1]
    return backbone, normalize


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    """
    Conv1d --> GroupNorm --> Mish
    """

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(
                inp_channels, out_channels, kernel_size, padding=kernel_size // 2
            ),
            Rearrange("batch channels horizon -> batch channels 1 horizon"),
            nn.GroupNorm(n_groups, out_channels),
            Rearrange("batch channels 1 horizon -> batch channels horizon"),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ModifiedResNetFeatures(ModifiedResNet):
    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__(layers, output_dim, heads, input_resolution, width)

    def forward(self, x: torch.Tensor):
        x = x.type(self.conv1.weight.dtype)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x0 = self.relu3(self.bn3(self.conv3(x)))
        x = self.avgpool(x0)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return {
            "res1": x0,
            "res2": x1,
            "res3": x2,
            "res4": x3,
            "res5": x4,
        }


class MultiheadCustomAttention(Module):
    r"""Allows the model to jointly attend to information
    from different representation subspaces.
    See reference: Attention Is All You Need
    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O
        \text{where} head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
    Args:
        embed_dim: total dimension of the model.
        num_heads: parallel attention heads.
        dropout: a Dropout layer on attn_output_weights. Default: 0.0.
        bias: add bias as module parameter. Default: True.
        add_bias_kv: add bias to the key and value sequences at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        kdim: total number of features in key. Default: None.
        vdim: total number of features in key. Default: None.
        Note: if kdim and vdim are None, they will be set to embed_dim such that
        query, key, and value have the same number of features.
    Examples::
        >>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value)
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        kdim=None,
        vdim=None,
        slot_competition=False,
        return_kv=False,
        gate_attn=False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        self.in_proj_weight = Parameter(torch.empty(3 * embed_dim, embed_dim))
        ##### Custom
        self.slot_competition = slot_competition
        self.return_kv = return_kv
        self.gate_attn = None
        if gate_attn:
            self.gate_attn = Parameter(torch.randn(num_heads))  # randn
        #####
        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))
            self.k_proj_weight = Parameter(torch.Tensor(embed_dim, self.kdim))
            self.v_proj_weight = Parameter(torch.Tensor(embed_dim, self.vdim))

        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter("in_proj_bias", None)
        self.out_proj = Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = Parameter(torch.empty(1, 1, embed_dim))
            self.bias_v = Parameter(torch.empty(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.0)
            constant_(self.out_proj.bias, 0.0)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

    def forward(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        need_weights=True,
        attn_mask=None,
        k_mem=None,
        v_mem=None,
        mem_mask=None,
        rotary_pe=None,
    ):
        r"""
        Args:
            query, key, value: map a query and a set of key-value pairs to an output.
                See "Attention Is All You Need" for more details.
            key_padding_mask: if provided, specified padding elements in the key will
                be ignored by the attention. This is an binary mask. When the value is True,
                the corresponding value on the attention layer will be filled with -inf.
            need_weights: output attn_output_weights.
            attn_mask: mask that prevents attention to certain positions. This is an additive mask
                (i.e. the values will be added to the attention layer).
        Shape:
            - Inputs:
            - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
              the embedding dimension.
            - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
              the embedding dimension.
            - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
              the embedding dimension.
            - key_padding_mask: :math:`(N, S)`, ByteTensor, where N is the batch size, S is the source sequence length.
            - attn_mask: :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
            - Outputs:
            - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
              E is the embedding dimension.
            - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
              L is the target sequence length, S is the source sequence length.
        """
        if hasattr(self, "_qkv_same_embed_dim") and self._qkv_same_embed_dim is False:
            return multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                self.in_proj_weight,
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj.weight,
                self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight,
                k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight,
                slot_competition=self.slot_competition,
                return_kv=self.return_kv,
                k_mem=k_mem,
                v_mem=v_mem,
                gate_attn=self.gate_attn,
                mem_mask=mem_mask,
                rotary_pe=rotary_pe,
            )
        else:
            if not hasattr(self, "_qkv_same_embed_dim"):
                warnings.warn(
                    "A new version of MultiheadAttention module has been implemented. \
                    Please re-train your model with the new module",
                    UserWarning,
                )

            return multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                self.in_proj_weight,
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj.weight,
                self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                slot_competition=self.slot_competition,
                return_kv=self.return_kv,
                k_mem=k_mem,
                v_mem=v_mem,
                gate_attn=self.gate_attn,
                mem_mask=mem_mask,
                rotary_pe=rotary_pe,
            )


def multi_head_attention_forward(
    query,  # type: Tensor
    key,  # type: Tensor
    value,  # type: Tensor
    embed_dim_to_check,  # type: int
    num_heads,  # type: int
    in_proj_weight,  # type: Tensor
    in_proj_bias,  # type: Tensor
    bias_k,  # type: Optional[Tensor]
    bias_v,  # type: Optional[Tensor]
    add_zero_attn,  # type: bool
    dropout_p,  # type: float
    out_proj_weight,  # type: Tensor
    out_proj_bias,  # type: Tensor
    training=True,  # type: bool
    key_padding_mask=None,  # type: Optional[Tensor]
    need_weights=True,  # type: bool
    attn_mask=None,  # type: Optional[Tensor]
    use_separate_proj_weight=False,  # type: bool
    q_proj_weight=None,  # type: Optional[Tensor]
    k_proj_weight=None,  # type: Optional[Tensor]
    v_proj_weight=None,  # type: Optional[Tensor]
    static_k=None,  # type: Optional[Tensor]
    static_v=None,  # type: Optional[Tensor]
    slot_competition=False,
    rotary_pe=None,
    return_kv=False,
    k_mem=None,
    v_mem=None,
    gate_attn=None,
    mem_mask=None,
):
    # type: (...) -> Tuple[Tensor, Optional[Tensor]]
    r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        embed_dim_to_check: total dimension of the model.
        num_heads: parallel attention heads.
        in_proj_weight, in_proj_bias: input projection weight and bias.
        bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        dropout_p: probability of an element to be zeroed.
        out_proj_weight, out_proj_bias: the output projection weight and bias.
        training: apply dropout if is ``True``.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
        attn_mask: mask that prevents attention to certain positions. This is an additive mask
            (i.e. the values will be added to the attention layer).
        use_separate_proj_weight: the function accept the proj. weights for query, key,
            and value in differnt forms. If false, in_proj_weight will be used, which is
            a combination of q_proj_weight, k_proj_weight, v_proj_weight.
        q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.
        static_k, static_v: static key and value used for attention operators.
    Shape:
        Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)`, ByteTensor, where N is the batch size, S is the source sequence length.
        - attn_mask: :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
        - static_k: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        - static_v: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
    """

    qkv_same = torch.equal(query, key) and torch.equal(key, value)
    kv_same = torch.equal(key, value)

    tgt_len, bsz, embed_dim = query.size()
    assert embed_dim == embed_dim_to_check
    assert list(query.size()) == [tgt_len, bsz, embed_dim]
    assert key.size() == value.size()

    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
    scaling = float(head_dim) ** -0.5

    if use_separate_proj_weight is not True:
        if qkv_same:
            # self-attention
            q, k, v = F.linear(query, in_proj_weight, in_proj_bias).chunk(3, dim=-1)

        elif kv_same:
            # encoder-decoder attention
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = F.linear(query, _w, _b)

            if key is None:
                assert value is None
                k = None
                v = None
            else:

                # This is inline in_proj function with in_proj_weight and in_proj_bias
                _b = in_proj_bias
                _start = embed_dim
                _end = None
                _w = in_proj_weight[_start:, :]
                if _b is not None:
                    _b = _b[_start:]
                k, v = F.linear(key, _w, _b).chunk(2, dim=-1)

        else:
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = F.linear(query, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim
            _end = embed_dim * 2
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            k = F.linear(key, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim * 2
            _end = None
            _w = in_proj_weight[_start:, :]
            if _b is not None:
                _b = _b[_start:]
            v = F.linear(value, _w, _b)
    else:
        q_proj_weight_non_opt = torch.jit._unwrap_optional(q_proj_weight)
        len1, len2 = q_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == query.size(-1)

        k_proj_weight_non_opt = torch.jit._unwrap_optional(k_proj_weight)
        len1, len2 = k_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == key.size(-1)

        v_proj_weight_non_opt = torch.jit._unwrap_optional(v_proj_weight)
        len1, len2 = v_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == value.size(-1)

        if in_proj_bias is not None:
            q = F.linear(query, q_proj_weight_non_opt, in_proj_bias[0:embed_dim])
            k = F.linear(
                key, k_proj_weight_non_opt, in_proj_bias[embed_dim : (embed_dim * 2)]
            )
            v = F.linear(value, v_proj_weight_non_opt, in_proj_bias[(embed_dim * 2) :])
        else:
            q = F.linear(query, q_proj_weight_non_opt, in_proj_bias)
            k = F.linear(key, k_proj_weight_non_opt, in_proj_bias)
            v = F.linear(value, v_proj_weight_non_opt, in_proj_bias)
    q = q * scaling

    if bias_k is not None and bias_v is not None:
        if static_k is None and static_v is None:
            k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat(
                    [
                        attn_mask,
                        torch.zeros(
                            (attn_mask.size(0), 1),
                            dtype=attn_mask.dtype,
                            device=attn_mask.device,
                        ),
                    ],
                    dim=1,
                )
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [
                        key_padding_mask,
                        torch.zeros(
                            (key_padding_mask.size(0), 1),
                            dtype=key_padding_mask.dtype,
                            device=key_padding_mask.device,
                        ),
                    ],
                    dim=1,
                )
        else:
            assert static_k is None, "bias cannot be added to static key."
            assert static_v is None, "bias cannot be added to static value."
    else:
        assert bias_k is None
        assert bias_v is None

    if rotary_pe is not None:  # rotary pe ROPE disentangeld
        qp, kvp = rotary_pe
        q_cos, q_sin = qp[..., 0], qp[..., 1]
        k_cos, k_sin = kvp[..., 0], kvp[..., 1]
        q = RotaryPositionEncoding.embed_rotary(
            q.transpose(0, 1), q_cos, q_sin
        ).transpose(0, 1)
        k = RotaryPositionEncoding.embed_rotary(
            k.transpose(0, 1), k_cos, k_sin
        ).transpose(0, 1)

    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if k is not None:
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    if v is not None:
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

    if static_k is not None:
        assert static_k.size(0) == bsz * num_heads
        assert static_k.size(2) == head_dim
        k = static_k

    if static_v is not None:
        assert static_v.size(0) == bsz * num_heads
        assert static_v.size(2) == head_dim
        v = static_v

    src_len = k.size(1)

    if key_padding_mask is not None:
        assert key_padding_mask.size(0) == bsz
        assert key_padding_mask.size(1) == src_len

    if add_zero_attn:
        src_len += 1
        k = torch.cat(
            [
                k,
                torch.zeros(
                    (k.size(0), 1) + k.size()[2:], dtype=k.dtype, device=k.device
                ),
            ],
            dim=1,
        )
        v = torch.cat(
            [
                v,
                torch.zeros(
                    (v.size(0), 1) + v.size()[2:], dtype=v.dtype, device=v.device
                ),
            ],
            dim=1,
        )
        if attn_mask is not None:
            attn_mask = torch.cat(
                [
                    attn_mask,
                    torch.zeros(
                        (attn_mask.size(0), 1),
                        dtype=attn_mask.dtype,
                        device=attn_mask.device,
                    ),
                ],
                dim=1,
            )
        if key_padding_mask is not None:
            key_padding_mask = torch.cat(
                [
                    key_padding_mask,
                    torch.zeros(
                        (key_padding_mask.size(0), 1),
                        dtype=key_padding_mask.dtype,
                        device=key_padding_mask.device,
                    ),
                ],
                dim=1,
            )

    attn_output_weights = torch.bmm(q, k.transpose(1, 2))
    assert list(attn_output_weights.size()) == [bsz * num_heads, tgt_len, src_len]

    if attn_mask is not None:
        attn_mask = attn_mask.unsqueeze(0)
        attn_output_weights += attn_mask

    if key_padding_mask is not None:
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        attn_output_weights = attn_output_weights.masked_fill(
            key_padding_mask.unsqueeze(1).unsqueeze(2),
            float("-inf"),
        )
        attn_output_weights = attn_output_weights.view(
            bsz * num_heads, tgt_len, src_len
        )

    if slot_competition:
        attn_output_weights = F.softmax(attn_output_weights, dim=-2) + 1e-8
        attn_output_weights = attn_output_weights / attn_output_weights.sum(
            dim=-1, keepdim=True
        )
    else:
        attn_output_weights = F.softmax(attn_output_weights, dim=-1)

    attn_output_weights = F.dropout(attn_output_weights, p=dropout_p, training=training)

    attn_output = torch.bmm(attn_output_weights, v)
    assert list(attn_output.size()) == [bsz * num_heads, tgt_len, head_dim]

    # do memorizing transformer gating
    if (gate_attn is not None) and (k_mem is not None) and (v_mem is not None):
        k_mem = k_mem.permute((2, 0, 1))
        key_mem_len = k_mem.shape[0]
        k_mem = (
            k_mem.contiguous()
            .view(key_mem_len, bsz * num_heads, head_dim)
            .transpose(0, 1)
        )
        v_mem = v_mem.permute((2, 0, 1))
        v_mem = (
            v_mem.contiguous()
            .view(key_mem_len, bsz * num_heads, head_dim)
            .transpose(0, 1)
        )
        #         if True:
        #             k_mem = F.normalize(k_mem, dim = -1)

        attn_output_weights_mem = torch.bmm(q, k_mem.transpose(1, 2))  # [24, 16, 110]
        # bcz correspondance b/w key key is good not query, key visually
        #         attn_output_weights_mem = torch.bmm(k, k_mem.transpose(1, 2))
        attn_output_weights_mem = F.softmax(attn_output_weights_mem, dim=-1)
        if mem_mask is not None:
            mem_mask = mem_mask.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, key_mem_len]
            attn_output_weights_mem = attn_output_weights_mem.reshape(
                bsz, num_heads, tgt_len, key_mem_len
            )
            attn_output_weights_mem = attn_output_weights_mem * mem_mask
            attn_output_weights_mem = attn_output_weights_mem.reshape(
                bsz * num_heads, tgt_len, key_mem_len
            )

        attn_output_weights_mem = F.dropout(
            attn_output_weights_mem, p=dropout_p, training=training
        )
        attn_output_mem = torch.bmm(
            attn_output_weights_mem, v_mem
        )  # [bsz * num_heads, tgt_len, head_dim]

        # gated learnable attention like memorizing transformers
        print("gate_attn ", torch.sigmoid(gate_attn))
        gate = torch.sigmoid(gate_attn).reshape(-1, 1, 1, 1)  # (n_head, 1, 1, 1)
        attn_output_mem = attn_output_mem.view(
            bsz, num_heads, tgt_len, head_dim
        ).transpose(
            0, 1
        )  # [num_heads, bsz, tgt_len, head_dim]
        attn_output = attn_output.view(bsz, num_heads, tgt_len, head_dim).transpose(
            0, 1
        )  # [num_heads, bsz, tgt_len, head_dim]
        attn_output = gate * attn_output_mem + (1.0 - gate) * attn_output
        attn_output = attn_output.transpose(1, 0).view(
            bsz * num_heads, tgt_len, head_dim
        )

    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)

    if return_kv:
        return attn_output, q, k, v
    elif need_weights:
        # average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        #         return attn_output, attn_output_weights.sum(dim=1) / num_heads
        return attn_output, attn_output_weights
    else:
        return attn_output, None


class ParallelAttentionLayer(nn.Module):
    """Self-/Cross-attention between two sequences."""

    def __init__(
        self,
        d_model=256,
        dropout=0.1,
        n_heads=8,
        pre_norm=False,
        self_attention1=True,
        self_attention2=True,
        cross_attention1=True,
        cross_attention2=True,
        apply_ffn=True,
        slot_attention12=False,
        slot_attention21=False,
        rotary_pe=False,
        use_adaln=False,
    ):
        """Initialize layers, d_model is the encoder dimension."""
        super().__init__()
        self.pre_norm = pre_norm
        self.self_attention1 = self_attention1
        self.self_attention2 = self_attention2
        self.cross_attention1 = cross_attention1
        self.cross_attention2 = cross_attention2
        self.apply_ffn = apply_ffn
        self.rotary_pe = rotary_pe

        # Self-attention for seq1
        if self.self_attention1:
            self.adaln_1 = None
            if use_adaln:
                self.adaln_1 = AdaLN(d_model)
            self.sa1 = MultiheadCustomAttention(d_model, n_heads, dropout=dropout)
            self.dropout_1 = nn.Dropout(dropout)
            self.norm_1 = nn.LayerNorm(d_model)

        # Self-attention for seq2
        if self.self_attention2:
            self.adaln_2 = None
            if use_adaln:
                self.adaln_2 = AdaLN(d_model)
            self.sa2 = MultiheadCustomAttention(d_model, n_heads, dropout=dropout)
            self.dropout_2 = nn.Dropout(dropout)
            self.norm_2 = nn.LayerNorm(d_model)

        # Cross attention from seq1 to seq2
        self.norm_12 = None
        if cross_attention1:
            self.adaln_12 = None
            if use_adaln:
                self.adaln_12 = AdaLN(d_model)
            self.cross_12 = MultiheadCustomAttention(
                d_model, n_heads, dropout=dropout, slot_competition=slot_attention12
            )
            self.dropout_12 = nn.Dropout(dropout)
            self.norm_12 = nn.LayerNorm(d_model)

        # Cross attention from seq2 to seq1
        self.norm_21 = None
        if cross_attention2:
            self.adaln_21 = None
            if use_adaln:
                self.adaln_21 = AdaLN(d_model)
            self.cross_21 = MultiheadCustomAttention(
                d_model, n_heads, dropout=dropout, slot_competition=slot_attention21
            )
            self.dropout_21 = nn.Dropout(dropout)
            self.norm_21 = nn.LayerNorm(d_model)

        # FFN-1
        if self_attention1 or cross_attention1:
            self.adaln_ff1 = None
            if use_adaln:
                self.adaln_ff1 = AdaLN(d_model)
            self.ffn_12 = nn.Sequential(
                nn.Linear(d_model, 4 * d_model),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(4 * d_model, d_model),
                nn.Dropout(dropout),
            )
            self.norm_122 = nn.LayerNorm(d_model)

        # FFN-2
        if self_attention2 or cross_attention2:
            self.adaln_ff2 = None
            if use_adaln:
                self.adaln_ff2 = AdaLN(d_model)
            self.ffn_21 = nn.Sequential(
                nn.Linear(d_model, 4 * d_model),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(4 * d_model, d_model),
                nn.Dropout(dropout),
            )
            self.norm_212 = nn.LayerNorm(d_model)

    def _norm(self, x, layer, normalize=True):
        if normalize and layer is not None:
            return layer(x)
        return x

    def with_pos_embed(self, tensor, pos=None):
        return tensor if pos is None else tensor + pos

    def _adaln(self, x, layer, ada_sgnl):
        if layer is not None and ada_sgnl is not None:
            return layer(x, ada_sgnl)
        return x

    def forward(
        self,
        seq1,
        seq1_key_padding_mask,
        seq2,
        seq2_key_padding_mask,
        seq1_pos=None,
        seq2_pos=None,
        seq1_sem_pos=None,
        seq2_sem_pos=None,
        ada_sgnl=None,
    ):
        """Forward pass, seq1 (B, S1, F), seq2 (B, S2, F)."""
        rot_args = {}

        # Create key, query, value for seq1, seq2
        q1 = k1 = v1 = self._norm(seq1, self.norm_12, self.pre_norm)
        q2 = k2 = v2 = self._norm(seq2, self.norm_21, self.pre_norm)
        if not self.rotary_pe:
            q1 = k1 = self.with_pos_embed(seq1, seq1_pos)
            q2 = k2 = self.with_pos_embed(seq2, seq2_pos)
        q1 = self.with_pos_embed(q1, seq1_sem_pos)
        k1 = self.with_pos_embed(k1, seq1_sem_pos)
        q2 = self.with_pos_embed(q2, seq2_sem_pos)
        k2 = self.with_pos_embed(k2, seq2_sem_pos)

        # Cross-attention from seq1 to seq2
        if self.cross_attention1:
            if self.rotary_pe:
                rot_args["rotary_pe"] = (seq1_pos, seq2_pos)
            seq1b = self.cross_12(
                query=self._adaln(q1, self.adaln_12, ada_sgnl).transpose(0, 1),
                key=k2.transpose(0, 1),
                value=v2.transpose(0, 1),
                attn_mask=None,
                key_padding_mask=seq2_key_padding_mask,  # (B, S2)
                **rot_args,
            )[0].transpose(0, 1)
            seq1 = seq1 + self.dropout_12(seq1b)
            seq1 = self._norm(seq1, self.norm_12, not self.pre_norm)

        # Cross-attention from seq2 to seq1
        if self.cross_attention2:
            if self.rotary_pe:
                rot_args["rotary_pe"] = (seq2_pos, seq1_pos)
            seq2b = self.cross_21(
                query=self._adaln(q2, self.adaln_21, ada_sgnl).transpose(0, 1),
                key=k1.transpose(0, 1),
                value=v1.transpose(0, 1),
                attn_mask=None,
                key_padding_mask=seq1_key_padding_mask,  # (B, S1)
                **rot_args,
            )[0].transpose(0, 1)
            seq2 = seq2 + self.dropout_21(seq2b)
            seq2 = self._norm(seq2, self.norm_21, not self.pre_norm)

        # Self-attention for seq1
        if self.self_attention1:
            q1 = k1 = v1 = self._norm(seq1, self.norm_1, self.pre_norm)
            if self.rotary_pe:
                rot_args["rotary_pe"] = (seq1_pos, seq1_pos)
            else:
                q1 = k1 = self.with_pos_embed(seq1, seq1_pos)
            q1 = self.with_pos_embed(q1, seq1_sem_pos)
            k1 = self.with_pos_embed(k1, seq1_sem_pos)
            seq1b = self.sa1(
                query=self._adaln(q1, self.adaln_1, ada_sgnl).transpose(0, 1),
                key=self._adaln(k1, self.adaln_1, ada_sgnl).transpose(0, 1),
                value=self._adaln(v1, self.adaln_1, ada_sgnl).transpose(0, 1),
                attn_mask=None,
                key_padding_mask=seq1_key_padding_mask,  # (B, S1)
                **rot_args,
            )[0].transpose(0, 1)
            seq1 = seq1 + self.dropout_1(seq1b)
            seq1 = self._norm(seq1, self.norm_1, not self.pre_norm)

        # Self-attention for seq2
        if self.self_attention2:
            q2 = k2 = v2 = self._norm(seq2, self.norm_2, self.pre_norm)
            if self.rotary_pe:
                rot_args["rotary_pe"] = (seq2_pos, seq2_pos)
            else:
                q2 = k2 = self.with_pos_embed(seq2, seq2_pos)
            q2 = self.with_pos_embed(q2, seq2_sem_pos)
            k2 = self.with_pos_embed(k2, seq2_sem_pos)
            seq2b = self.sa2(
                query=self._adaln(q2, self.adaln_2, ada_sgnl).transpose(0, 1),
                key=self._adaln(k2, self.adaln_2, ada_sgnl).transpose(0, 1),
                value=self._adaln(v2, self.adaln_2, ada_sgnl).transpose(0, 1),
                attn_mask=None,
                key_padding_mask=seq2_key_padding_mask,  # (B, S2)
                **rot_args,
            )[0].transpose(0, 1)
            seq2 = seq2 + self.dropout_2(seq2b)
            seq2 = self._norm(seq2, self.norm_2, not self.pre_norm)

        # FFN-1
        if (self.self_attention1 or self.cross_attention1) and self.apply_ffn:
            seq1 = self._norm(seq1, self.norm_122, self.pre_norm)
            seq1 = self._adaln(seq1, self.adaln_ff1, ada_sgnl)
            seq1 = seq1 + self.ffn_12(seq1)
            seq1 = self._norm(seq1, self.norm_122, not self.pre_norm)

        # FFN-2
        if (self.self_attention2 or self.cross_attention2) and self.apply_ffn:
            seq2 = self._norm(seq2, self.norm_212, self.pre_norm)
            seq2 = self._adaln(seq2, self.adaln_ff2, ada_sgnl)
            seq2 = seq2 + self.ffn_21(seq2)
            seq2 = self._norm(seq2, self.norm_212, not self.pre_norm)

        return seq1, seq2


class ParallelAttention(nn.Module):
    """Self-/Cross-attention between two sequences."""

    def __init__(
        self,
        num_layers=1,
        d_model=256,
        dropout=0.1,
        n_heads=8,
        pre_norm=False,
        self_attention1=True,
        self_attention2=True,
        cross_attention1=True,
        cross_attention2=True,
        apply_ffn=True,
        slot_attention12=False,
        slot_attention21=False,
        rotary_pe=False,
        use_adaln=False,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.update_seq1 = self_attention1 or cross_attention1
        self.update_seq2 = self_attention2 or cross_attention2
        for _ in range(num_layers):
            self.layers.append(
                ParallelAttentionLayer(
                    d_model=d_model,
                    dropout=dropout,
                    n_heads=n_heads,
                    pre_norm=pre_norm,
                    self_attention1=self_attention1,
                    self_attention2=self_attention2,
                    cross_attention1=cross_attention1,
                    cross_attention2=cross_attention2,
                    apply_ffn=apply_ffn,
                    slot_attention12=slot_attention12,
                    slot_attention21=slot_attention21,
                    rotary_pe=rotary_pe,
                    use_adaln=use_adaln,
                )
            )

    def forward(
        self,
        seq1,
        seq1_key_padding_mask,
        seq2,
        seq2_key_padding_mask,
        seq1_pos=None,
        seq2_pos=None,
        seq1_sem_pos=None,
        seq2_sem_pos=None,
        ada_sgnl=None,
    ):
        """Forward pass, seq1 (B, S1, F), seq2 (B, S2, F)."""
        for layer in self.layers:
            seq1_, seq2_ = layer(
                seq1=seq1,
                seq1_key_padding_mask=seq1_key_padding_mask,
                seq2=seq2,
                seq2_key_padding_mask=seq2_key_padding_mask,
                seq1_pos=seq1_pos,
                seq2_pos=seq2_pos,
                seq1_sem_pos=seq1_sem_pos,
                seq2_sem_pos=seq2_sem_pos,
                ada_sgnl=ada_sgnl,
            )
            if self.update_seq1:
                seq1 = seq1_
            if self.update_seq2:
                seq2 = seq2_
        return seq1, seq2


class AdaLN(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(embedding_dim, 2 * embedding_dim, bias=True)
        )
        nn.init.constant_(self.modulation[-1].weight, 0)
        nn.init.constant_(self.modulation[-1].bias, 0)

    def forward(self, x, t):
        """
        Args:
            x: A tensor of shape (B, N, C)
            t: A tensor of shape (B, C)
        """
        scale, shift = self.modulation(t).chunk(2, dim=-1)  # (B, C), (B, C)
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return x  # (B, N, C)


class RelativeCrossAttentionLayer(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout=0.0):
        super().__init__()
        self.multihead_attn = MultiheadCustomAttention(
            embedding_dim, num_heads, dropout=dropout
        )
        self.norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, value, query_pos=None, value_pos=None, pad_mask=None):
        attn_output, attn_output_weights = self.multihead_attn(
            query=query,
            key=value,
            value=value,
            rotary_pe=(query_pos, value_pos) if query_pos is not None else None,
            key_padding_mask=pad_mask,
        )
        output = query + self.dropout(attn_output)
        output = self.norm(output)
        return output, attn_output_weights.mean(dim=1)


class FeedforwardLayer(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.linear1 = nn.Linear(embedding_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim)
        self.activation = F.relu
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        output = self.linear2(self.dropout(self.activation(self.linear1(x))))
        output = x + self.dropout(output)
        output = self.norm(output)
        return output


class RelativeCrossAttentionModule(nn.Module):
    def __init__(self, embedding_dim, num_attn_heads, num_layers):
        super().__init__()

        self.attn_layers = nn.ModuleList()
        self.ffw_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.attn_layers.append(
                RelativeCrossAttentionLayer(embedding_dim, num_attn_heads)
            )
            self.ffw_layers.append(FeedforwardLayer(embedding_dim, embedding_dim))

    def forward(self, query, value, query_pos=None, value_pos=None):
        output = []
        for i in range(len(self.attn_layers)):
            query, _ = self.attn_layers[i](query, value, query_pos, value_pos)
            query = self.ffw_layers[i](query)
            output.append(query)
        return output


class BackprojectDepth(nn.Module):
    """Layer to transform a depth image into a point cloud"""

    def __init__(self, height, width):
        super(BackprojectDepth, self).__init__()

        # self.batch_size = batch_size
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing="xy")
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(
            torch.from_numpy(self.id_coords), requires_grad=False
        )
        self.pix_coords = torch.unsqueeze(
            torch.stack([self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0
        )

    def forward(self, depth, K):
        if isinstance(K, np.ndarray):
            assert K.shape == (3, 3)
            K = torch.from_numpy(K).float().to(depth.device)[None]

        batch_size = depth.shape[0]
        ones = torch.ones(batch_size, 1, self.height * self.width).to(depth.device)
        inv_K = torch.inverse(K).to(depth.device)  # [B, 3, 3]

        pix_coords = self.pix_coords.clone().to(depth.device)
        pix_coords = pix_coords.repeat(batch_size, 1, 1)
        pix_coords = torch.cat([pix_coords, ones], 1)  # [B, 3, H*W]

        cam_points = torch.matmul(inv_K, pix_coords)  # [B, 3, 3] @ [B, 3, H*W]
        cam_points = (
            depth.view(batch_size, 1, -1) * cam_points
        )  # [B, 1, H*W] * [B, 3, H*W]
        return cam_points


class Project3D(nn.Module):
    """Layer which projects 3D points into a camera with intrinsics K and at position T"""

    def __init__(self, eps=1e-7):
        super(Project3D, self).__init__()
        self.eps = eps

    def forward(self, points, K, T=None):
        """_summary_

        Parameters
        ----------
        points : torch.Tensor
            [B, H*W, 3]
        K : torch.Tensor / np.ndarray
            [3, 3]
        T : torch.Tensor
            [B, 4, 4]

        Returns
        -------
        torch.Tensor
            [B, 2, H*W]
        """
        if isinstance(K, np.ndarray):
            K = torch.from_numpy(K).float().to(points.device)
            if len(K.shape) == 2:
                K = K[None]
        if T is None:
            T = torch.eye(4).float().to(points.device)[None]
        points = points.float()
        batch_size = points.shape[0]
        points_cam = points @ T[:, :3, :3].transpose(-1, -2) + T[:, :3, 3].unsqueeze(
            -2
        )  # [B, N, 3] @ [B, 3, 3] + [B, 1, 3]

        pix_coords = points_cam @ K.transpose(
            -1, -2
        )  # [B, N, 3] @ [B, 3, 3] => [B, N, 3]
        pix_coords = pix_coords[..., :2] / (
            pix_coords[..., 2:3] + self.eps
        )  # [B, N, 2]
        pix_coords = pix_coords.permute(0, 2, 1)  # [B, 2, N]
        return pix_coords


class MLP(nn.Module):
    """
    Base class for simple Multi-Layer Perceptrons.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        layer_dims: tuple = (),
        layer_func=nn.Linear,
        layer_func_kwargs=None,
        activation=nn.ReLU,
        dropouts=None,
        normalization=False,
        output_activation=None,
    ):
        """
        Args:
            input_dim (int): dimension of inputs
            output_dim (int): dimension of outputs
            layer_dims ([int]): sequence of integers for the hidden layers sizes
            layer_func: mapping per layer - defaults to Linear
            layer_func_kwargs (dict): kwargs for @layer_func
            activation: non-linearity per layer - defaults to ReLU
            dropouts ([float]): if not None, adds dropout layers with the corresponding probabilities
                after every layer. Must be same size as @layer_dims.
            normalization (bool): if True, apply layer normalization after each layer
            output_activation: if provided, applies the provided non-linearity to the output layer
        """
        super(MLP, self).__init__()
        layers = []
        dim = input_dim
        if layer_func_kwargs is None:
            layer_func_kwargs = dict()
        if dropouts is not None:
            assert len(dropouts) == len(layer_dims)
        for i, l in enumerate(layer_dims):
            layers.append(layer_func(dim, l, **layer_func_kwargs))
            if normalization:
                layers.append(nn.LayerNorm(l))
            layers.append(activation())
            if dropouts is not None and dropouts[i] > 0.0:
                layers.append(nn.Dropout(dropouts[i]))
            dim = l
        layers.append(layer_func(dim, output_dim))
        if output_activation is not None:
            layers.append(output_activation())
        self._layer_func = layer_func
        self.nets = layers
        self._model = nn.Sequential(*layers)

        self._layer_dims = layer_dims
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._dropouts = dropouts
        self._act = activation
        self._output_act = output_activation

    def output_shape(self, input_shape=None):
        """
        Function to compute output shape from inputs to this module.
        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend
                on the size of the input, or if they assume fixed size input.
        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """
        return [self._output_dim]

    def forward(self, inputs):
        """
        Forward pass.
        """
        return self._model(inputs)


"""borrowed and modified from https://github.com/CompVis/taming-transformers"""


def nonlinearity(x):
    # swish
    return F.silu(x, inplace=True)  # x*torch.sigmoid(x)


def Normalize(in_channels):
    return torch.nn.GroupNorm(
        num_groups=32, num_channels=in_channels, eps=1e-6, affine=True
    )


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1
            )

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=2, padding=0
            )

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout,
        temb_channels=512,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.checkpointing = False

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout, inplace=True)
        self.conv2 = torch.nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(
                    in_channels, out_channels, kernel_size=3, stride=1, padding=1
                )
            else:
                self.nin_shortcut = torch.nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=1, padding=0
                )

    def _forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h

    def forward(self, x, temb):
        if self.checkpointing and self.training:
            out = checkpoint(self._forward, x, temb)
        else:
            out = self._forward(x, temb)
        return out


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.k = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.v = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.proj_out = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)  # b,hw,c
        k = k.reshape(b, c, h * w)  # b,c,hw
        w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x + h_


class Encoder(nn.Module):
    def __init__(
        self,
        *,
        ch,
        out_ch,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks,
        attn_resolutions,
        dropout=0.0,
        resamp_with_conv=True,
        in_channels,
        resolution,
        z_channels,
        double_z=True,
        **ignore_kwargs,
    ):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.h, self.w = resolution[0], resolution[1]

        # downsampling
        self.conv_in = torch.nn.Conv2d(
            in_channels, self.ch, kernel_size=3, stride=1, padding=1
        )

        curr_res = self.h
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(
            block_in,
            2 * z_channels if double_z else z_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x):
        # timestep embedding
        temb = None

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    def __init__(
        self,
        *,
        ch,
        out_ch,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks,
        attn_resolutions,
        dropout=0.0,
        resamp_with_conv=True,
        in_channels,
        resolution,
        z_channels,
        give_pre_end=False,
        **ignorekwargs,
    ):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.h, self.w = resolution[0], resolution[1]

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res_h = self.h // 2 ** (self.num_resolutions - 1)
        curr_res_w = self.w // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res_h, curr_res_w)
        # z to block_in
        self.conv_in = torch.nn.Conv2d(
            z_channels, block_in, kernel_size=3, stride=1, padding=1
        )

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )

        # upsampling
        curr_res = curr_res_h
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(
            block_in, out_ch, kernel_size=3, stride=1, padding=1
        )

    def forward(self, z):
        # assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            # (Mohit): argh... forgot to remove this batchnorm
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            # (Mohit): argh... forgot to remove this batchnorm
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2=None):
        """
        U-Net forward by concatenating input feature (x1) with mirroring encoding feature maps channel-wise (x2)
        Args:
            x1 (torch.Tensor): [B, C1, H1, W1]
            x2 (torch.Tensor): [B, C2, H2, W2]

        Returns:
            output (torch.Tensor): [B, out_channels, H2, W2]
        """
        x1 = self.up(x1)
        # input is CHW
        if x2 is not None:
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]

            x1 = F.pad(
                x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2]
            )
            # if you have padding issues, see
            # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
            # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
            x = torch.cat([x2, x1], dim=1)
        else:
            x = x1
        return self.conv(x)


class ConvBlock(nn.Module):
    def __init__(
        self, in_planes, filters, kernel_size, stride=1, final_relu=True, batchnorm=True
    ):
        super(ConvBlock, self).__init__()
        self.final_relu = final_relu
        self.batchnorm = batchnorm

        filters1, filters2, filters3 = filters
        self.conv1 = nn.Conv2d(in_planes, filters1, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(filters1) if self.batchnorm else nn.Identity()
        self.conv2 = nn.Conv2d(
            filters1,
            filters2,
            kernel_size=kernel_size,
            dilation=1,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(filters2) if self.batchnorm else nn.Identity()
        self.conv3 = nn.Conv2d(filters2, filters3, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(filters3) if self.batchnorm else nn.Identity()

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_planes, filters3, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(filters3) if self.batchnorm else nn.Identity(),
        )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        if self.final_relu:
            out = F.relu(out)
        return out


class ResNet50Encoder(nn.Module):
    DEPTH = 6

    def __init__(self, input_channels=3):
        super().__init__()
        resnet = resnet50(pretrained=True)
        down_blocks = []
        self.input_channels = input_channels
        if input_channels != 3:
            self.input_conv = nn.Conv2d(
                input_channels,
                64,
                kernel_size=(7, 7),
                stride=(2, 2),
                padding=(3, 3),
                bias=False,
            )
        else:
            self.input_conv = nn.Sequential(*list(resnet.children()))[0]

        self.input_block = nn.Sequential(*list(resnet.children()))[1:3]

        self.input_pool = list(resnet.children())[3]
        for bottleneck in list(resnet.children()):
            if isinstance(bottleneck, nn.Sequential):
                down_blocks.append(bottleneck)
        self.down_blocks = nn.ModuleList(down_blocks)

    def forward(self, x, with_output_feature_map=False):
        pre_pools = dict()
        pre_pools[f"res0"] = x
        x = self.input_block(self.input_conv(x))
        pre_pools[f"res1"] = x
        x = self.input_pool(x)

        for i, block in enumerate(self.down_blocks, 2):
            x = block(x)
            # if i == (ResNet50Encoder.DEPTH - 1):
            #     continue
            pre_pools[f"res{i}"] = x
        return pre_pools


class ResNet50Decoder(nn.Module):
    FeatureKey2Channels = {
        "res1": 64,
        "res2": 256,
        "res3": 512,
        "res4": 1024,
        "res5": 2048,
    }
    SKIP_FEATURE_KEYS = ["res1", "res2", "res3"]

    def __init__(
        self,
        output_channels=2,
        use_skip=True,
        bottleneck_feature_key="res5",
    ):
        super().__init__()
        self.outpu_channels = output_channels
        self.bottleneck_feature_key = bottleneck_feature_key
        self.use_skip = use_skip
        self.latent_dim = ResNet50Decoder.FeatureKey2Channels[bottleneck_feature_key]
 
        up_blocks = []
        up_blocks.append(Up(self.latent_dim, 1024))
        up_blocks.append(Up(1024, 512))
        if self.use_skip:
            skip_feat_key = ResNet50Decoder.SKIP_FEATURE_KEYS[2]
            skip_feat_dim = ResNet50Decoder.FeatureKey2Channels[skip_feat_key]
        else:
            skip_feat_dim = 0
        up_blocks.append(
            Up(
                512 + skip_feat_dim,
                512,
            )
        )

        if self.use_skip:
            skip_feat_key = ResNet50Decoder.SKIP_FEATURE_KEYS[1]
            skip_feat_dim = ResNet50Decoder.FeatureKey2Channels[skip_feat_key]
        else:
            skip_feat_dim = 0

        up_blocks.append(
            Up(
                512 + skip_feat_dim,
                256,
            )
        )

        if self.use_skip:
            skip_feat_key = ResNet50Decoder.SKIP_FEATURE_KEYS[0]
            skip_feat_dim = ResNet50Decoder.FeatureKey2Channels[skip_feat_key]
        else:
            skip_feat_dim = 0

        up_blocks.append(
            Up(
                256 + skip_feat_dim,
                64,
            )
        )
        if self.use_skip:
            up_blocks.append(Up(64, 64))
        else:
            up_blocks.append(DoubleConv(64, 64))

        self.up_blocks = nn.ModuleList(up_blocks)
        self.out = nn.Conv2d(64, output_channels, kernel_size=1, stride=1)

    def forward(self, features):
        x = features[self.bottleneck_feature_key]
        x = self.up_blocks[0](x)
        x = self.up_blocks[1](x)
        if self.use_skip:
            x = self.up_blocks[2](x, features[ResNet50Decoder.SKIP_FEATURE_KEYS[2]])
            x = self.up_blocks[3](x, features[ResNet50Decoder.SKIP_FEATURE_KEYS[1]])
            x = self.up_blocks[4](x, features[ResNet50Decoder.SKIP_FEATURE_KEYS[0]])
        else:
            x = self.up_blocks[2](x)
            x = self.up_blocks[3](x)
            x = self.up_blocks[4](x)
        x = self.up_blocks[5](x)
        x = self.out(x)
        return x


class PositionalEmbeddingV2(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)  # (N,1)
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()[None]

        pe[:, 0::2] = torch.sin(position * div_term)  # (N, d_model/2)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)  # (1, max_len, D)

    def forward(self, x):
        """
        @x: (B,N,D)
        """
        return x + self.pe[:, : x.size(1)]
