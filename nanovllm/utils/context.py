from dataclasses import dataclass
import torch


@dataclass(slots=True)
class Context:
    """全局上下文，存储当前推理步骤的元信息
    
    这个上下文在model_runner的prepare阶段设置，供attention等层使用。
    使用slots=True可以减少内存占用。
    """
    is_prefill: bool = False  # 是否是预填充阶段
    cu_seqlens_q: torch.Tensor | None = None  # query的累积序列长度（用于Flash Attention varlen）
    cu_seqlens_k: torch.Tensor | None = None  # key的累积序列长度（用于Flash Attention varlen，支持prefix caching）
    max_seqlen_q: int = 0  # 最大query序列长度
    max_seqlen_k: int = 0  # 最大key序列长度
    slot_mapping: torch.Tensor | None = None  # token到KV缓存slot的映射
    context_lens: torch.Tensor | None = None  # 每个序列的上下文长度（用于decode阶段）
    block_tables: torch.Tensor | None = None  # PagedAttention的块表


# 全局上下文实例
_CONTEXT = Context()


def get_context():
    """获取当前全局上下文"""
    return _CONTEXT


def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None):
    """设置全局上下文
    
    Args:
        is_prefill: 是否是预填充阶段
        cu_seqlens_q: query累积序列长度
        cu_seqlens_k: key累积序列长度
        max_seqlen_q: 最大query序列长度
        max_seqlen_k: 最大key序列长度
        slot_mapping: slot映射表
        context_lens: 上下文长度
        block_tables: 块表
    """
    global _CONTEXT
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables)


def reset_context():
    """重置全局上下文（回到初始状态）"""
    global _CONTEXT
    _CONTEXT = Context()
