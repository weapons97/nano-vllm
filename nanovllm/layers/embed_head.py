import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.utils.context import get_context


class VocabParallelEmbedding(nn.Module):
    """词表并行的Embedding层，用于张量并行
    
    将词表按张量并行度切分到不同GPU上。
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        super().__init__()
        # 当前进程的张量并行rank
        self.tp_rank = dist.get_rank()
        # 张量并行总大小
        self.tp_size = dist.get_world_size()
        # 词表大小必须能被tp_size整除
        assert num_embeddings % self.tp_size == 0
        self.num_embeddings = num_embeddings
        # 每个分区（GPU）负责的词表大小
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size
        # 当前分区负责的词表起始索引
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        # 当前分区负责的词表结束索引（不包含）
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        # 创建Embedding权重（只保存当前分区部分）
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))
        # 设置权重加载函数（用于加载检查点）
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """加载权重到当前分区
        
        Args:
            param: 当前分区的参数
            loaded_weight: 完整的权重
        """
        param_data = param.data
        shard_size = param_data.size(0)
        # 计算当前分区在完整权重中的起始位置
        start_idx = self.tp_rank * shard_size
        # 提取当前分区的权重
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        """前向传播
        
        Args:
            x: token id张量，可能包含其他分区的token id
            
        Returns:
            embedding后的张量
        """
        if self.tp_size > 1:
            # 创建mask：只保留当前分区负责的token
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            # 将其他分区的token id映射到当前分区（减去偏移）
            x = mask * (x - self.vocab_start_idx)
        y = F.embedding(x, self.weight)
        if self.tp_size > 1:
            # 将其他分区的token的embedding置零
            y = mask.unsqueeze(1) * y
            # All-reduce：汇总所有分区的embedding
            dist.all_reduce(y)
        return y


class ParallelLMHead(VocabParallelEmbedding):
    """并行的LM头（语言模型输出头）
    
    将词汇表并行化，每个GPU只计算部分词汇的logits。
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
    ):
        # LM头不支持bias
        assert not bias
        super().__init__(num_embeddings, embedding_dim)

    def forward(self, x: torch.Tensor):
        """前向传播，计算logits
        
        Args:
            x: 隐藏状态张量 [N, hidden_dim] 或 [batch, seq_len, hidden_dim]
            
        Returns:
            完整的logits张量（只有rank 0返回，其他为None）
        """
        context = get_context()
        # 预填充阶段：只取每个序列的最后一个token（用于生成下一个token）
        if context.is_prefill:
            last_indices = context.cu_seqlens_q[1:] - 1
            x = x[last_indices].contiguous()
        
        # 线性变换：hidden -> vocab logits
        logits = F.linear(x, self.weight)
        
        if self.tp_size > 1:
            # 收集所有分区的logits（只有rank 0需要完整logits）
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None
            dist.gather(logits, all_logits, 0)
            # rank 0拼接所有分区的logits
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None
        return logits
