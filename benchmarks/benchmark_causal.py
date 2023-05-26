from functools import partial
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

from flash_attn.utils.benchmark import benchmark_forward, benchmark_all, pytorch_profiler, benchmark_memory
from flash_attn.flash_attn_interface import flash_attn_unpadded_qkvpacked_func
# from flash_attn.triton.fused_attention import attention as attention
from flash_attn.flash_attn_triton import flash_attn_qkvpacked_func
from flash_attn.flash_attn_triton_onewritehead import flash_attn_qkvpacked_func_onewritehead
from flash_attn.flash_attn_triton_og import attention as attention_og

'''
repeats = 30
batch_size = 8 #2
seqlen = 128 #4096
nheads = 12 #12
# headdim = 128 # <= original
headdim=64 # 128 # note: headdim=64, nheads=16 implies n=1024
'''

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-r","--repeats", type=int,
                    default=30)
parser.add_argument("-b", "--batch-size", type=int,
                    default=8)
parser.add_argument("-m", "--seqlen", type=int,
                    default=1024)
parser.add_argument("-e", "--nheads", type=int,
                    default=64)
parser.add_argument("-k", "--headdim", type=int,
                    default=64)

args = parser.parse_args()

torch.manual_seed(0)
repeats = args.repeats
batch_size = args.batch_size
seqlen = args.seqlen
nheads = args.nheads
headdim=args.headdim

try:
    from flash_attn.fused_softmax import scaled_upper_triang_masked_softmax
except ImportError:
    scaled_upper_triang_masked_softmax = None


def attention_pytorch(qkv, dropout_p=0.0, causal=True):
    """
    Arguments:
        qkv: (batch_size, seqlen, 3, nheads, head_dim)
        dropout_p: float
    Output:
        output: (batch_size, seqlen, nheads, head_dim)
    """
    batch_size, seqlen, _, nheads, d = qkv.shape
    q, k, v = qkv.unbind(dim=2)
    q = rearrange(q, 'b t h d -> (b h) t d')
    k = rearrange(k, 'b s h d -> (b h) d s')
    softmax_scale = 1.0 / math.sqrt(d)
    # Preallocate attn_weights for `baddbmm`
    scores = torch.empty(batch_size * nheads, seqlen, seqlen, dtype=qkv.dtype, device=qkv.device)
    scores = rearrange(torch.baddbmm(scores, q, k, beta=0, alpha=softmax_scale),
                       '(b h) t s -> b h t s', h=nheads)
    if causal:
        # "triu_tril_cuda_template" not implemented for 'BFloat16'
        # So we have to construct the mask in float
        causal_mask = torch.triu(torch.full((seqlen, seqlen), -10000.0, device=scores.device), 1)
        # TD [2022-09-30]: Adding is faster than masked_fill_ (idk why, just better kernel I guess)
        scores = scores + causal_mask.to(dtype=scores.dtype)
    attention = torch.softmax(scores, dim=-1)
    attention_drop = F.dropout(attention, dropout_p)
    output = torch.einsum('bhts,bshd->bthd', attention_drop , v)
    return output.to(dtype=qkv.dtype)


def attention_megatron(qkv):
    """
    Arguments:
        qkv: (batch_size, seqlen, 3, nheads, head_dim)
    Output:
        output: (batch_size, seqlen, nheads, head_dim)
    """
    batch_size, seqlen, _, nheads, d = qkv.shape
    q, k, v = qkv.unbind(dim=2)
    q = rearrange(q, 'b t h d -> (b h) t d')
    k = rearrange(k, 'b s h d -> (b h) d s')
    softmax_scale = 1.0 / math.sqrt(d)
    # Preallocate attn_weights for `baddbmm`
    scores = torch.empty(batch_size * nheads, seqlen, seqlen, dtype=qkv.dtype, device=qkv.device)
    scores = rearrange(torch.baddbmm(scores, q, k, beta=0, alpha=softmax_scale),
                       '(b h) t s -> b h t s', h=nheads)
    attention = scaled_upper_triang_masked_softmax(scores, None, scale=1.0)
    output = torch.einsum('bhts,bshd->bthd', attention, v)
    return output.to(dtype=qkv.dtype)

# Modification by abf149: match noncausal configuration
#torch.manual_seed(0)
#repeats = 30
#batch_size = 8 #2
#seqlen = 128 #4096
#nheads = 12 #12
# headdim = 128 # <= original
#headdim=64 # 128 # note: headdim=64, nheads=16 implies n=1024
# batch_size = 64
# seqlen = 512
# nheads = 8
# headdim = 128
dropout_p = 0.0 #Requirement for triton
causal = True
dtype = torch.float16 #torch.bfloat16
device = 'cuda'

qkv = torch.randn(batch_size, seqlen, 3, nheads, headdim, device=device, dtype=dtype,
                  requires_grad=True)
cu_seqlens = torch.arange(0, (batch_size + 1) * seqlen, step=seqlen, dtype=torch.int32,
                          device=qkv.device)

# FlashAttention (CUDA)
benchmark_all(flash_attn_unpadded_qkvpacked_func, rearrange(qkv, 'b s ... -> (b s) ...'),
              cu_seqlens, seqlen, dropout_p, causal=causal, repeats=repeats, desc='FlashAttention', return_attn_probs=True)
benchmark_memory(flash_attn_unpadded_qkvpacked_func, rearrange(qkv, 'b s ... -> (b s) ...'), cu_seqlens, seqlen, dropout_p, causal=causal, desc='MaxMemory', return_attn_probs=True)
# Standard PyTorch
benchmark_all(attention_pytorch, qkv, dropout_p, causal=causal,
              repeats=repeats, desc='PyTorch Attention')
benchmark_memory(attention_pytorch, qkv, dropout_p, causal=causal, desc='MaxMemory')

# FlashAttention (Triton)
benchmark_all(flash_attn_qkvpacked_func, qkv, None, causal, repeats=repeats, desc='FlashAttention Triton')
pytorch_profiler(flash_attn_qkvpacked_func, qkv, None, causal, backward=True)
benchmark_memory(flash_attn_qkvpacked_func, qkv, None, causal, desc='MaxMemory')

# FlashAttention + one write (OWH) (Triton)
benchmark_all(flash_attn_qkvpacked_func_onewritehead, qkv, None, causal, repeats=repeats, desc='FlashAttention Triton OWH')
pytorch_profiler(flash_attn_qkvpacked_func_onewritehead, qkv, None, causal, backward=True)
benchmark_memory(flash_attn_qkvpacked_func_onewritehead, qkv, None, causal, desc='MaxMemory')

#q, k, v = [torch.randn(batch_size, nheads, seqlen, headdim, device=device, dtype=dtype,
#                       requires_grad=True) for _ in range(3)]
#benchmark_all(attention_og, q, k, v, 1.0, repeats=repeats, desc='FlashAttention Triton OG')
## pytorch_profiler(attention, q, k, v, 1.0, backward=True)
#
#if scaled_upper_triang_masked_softmax is not None:
#    benchmark_all(attention_megatron, qkv, repeats=repeats, desc='Megatron Attention')
