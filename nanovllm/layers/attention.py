import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    """Triton kernel：将key和value存储到KV缓存
    
    Args:
        key_ptr: key张量的指针
        key_stride: key张量在N维度的步长
        value_ptr: value张量的指针
        value_stride: value张量在N维度的步长
        k_cache_ptr: K缓存指针
        v_cache_ptr: V缓存指针
        slot_mapping_ptr: slot映射表指针（每个token对应的缓存位置）
        D: 每个token的key/value维度（num_heads * head_dim）
    """
    # 获取当前kernel实例处理的token索引
    idx = tl.program_id(0)
    # 获取该token对应的KV缓存slot
    slot = tl.load(slot_mapping_ptr + idx)
    # slot为-1表示无效（padding），直接返回
    if slot == -1: return
    # 计算key和value的偏移量
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    # 加载key和value
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    # 计算缓存中的偏移量
    cache_offsets = slot * D + tl.arange(0, D)
    # 存储到KV缓存
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    """将key和value存储到KV缓存
    
    Args:
        key: key张量，形状为 [N, num_heads, head_dim]
        value: value张量，形状为 [N, num_heads, head_dim]
        k_cache: K缓存张量
        v_cache: V缓存张量
        slot_mapping: slot映射表，长度为N
    """
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim  # 每个token的总维度
    # 检查内存布局：最后一维必须是连续的
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    # 启动Triton kernel，每个token一个kernel实例
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


class Attention(nn.Module):
    """多头注意力层，支持KV缓存和Flash Attention"""

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale  # attention的缩放因子（通常是1/sqrt(head_dim)）
        self.num_kv_heads = num_kv_heads  # KV头数（可能小于Q头数，用于GQA）
        # KV缓存（会在ModelRunner.allocate_kv_cache中分配）
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """前向传播
        
        Args:
            q: query张量 [N, num_heads, head_dim] 或 [N, 1, num_heads, head_dim]
            k: key张量 [N, num_kv_heads, head_dim]
            v: value张量 [N, num_kv_heads, head_dim]
            
        Returns:
            attention输出
        """
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        
        # 如果有KV缓存，将新计算的k,v存储到缓存
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        
        if context.is_prefill:
            # ========== 预填充阶段 ==========
            if context.block_tables is not None:  # 有前缀缓存
                # 使用缓存的KV（prefix caching）
                k, v = k_cache, v_cache
            # 使用Flash Attention的varlen版本（处理变长序列）
            o = flash_attn_varlen_func(
                q, k, v,
                max_seqlen_q=context.max_seqlen_q,
                cu_seqlens_q=context.cu_seqlens_q,
                max_seqlen_k=context.max_seqlen_k,
                cu_seqlens_k=context.cu_seqlens_k,
                softmax_scale=self.scale,
                causal=True,  # 因果注意力（只能看到前面的token）
                block_table=context.block_tables  # PagedAttention的块表
            )
        else:
            # ========== 解码阶段 ==========
            # 使用Flash Attention的kvcache版本（优化的decode）
            # q需要unsqueeze(1)变成 [N, 1, num_heads, head_dim]
            o = flash_attn_with_kvcache(
                q.unsqueeze(1), k_cache, v_cache,
                cache_seqlens=context.context_lens,  # 每个序列的上下文长度
                block_table=context.block_tables,  # PagedAttention的块表
                softmax_scale=self.scale,
                causal=True
            )
        return o
