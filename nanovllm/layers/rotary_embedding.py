from functools import lru_cache
import torch
from torch import nn


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """应用旋转位置编码（RoPE）
    
    将输入x在最后一维分成两半，分别应用旋转：
    x' = [x1 * cos - x2 * sin, x2 * cos + x1 * sin]
    
    Args:
        x: 输入张量，形状为 [..., head_size]
        cos: 余弦部分
        sin: 正弦部分
        
    Returns:
        应用RoPE后的张量
    """
    # 在最后一维将x分成两半：x1和x2
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    # 应用旋转公式
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    # 拼接回原形状
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):
    """旋转位置编码（RoPE）
    
    对query和key应用基于位置的正弦/余弦旋转。
    这是LLaMA、Qwen等模型使用的位置编码方法。
    """

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        # rotary_dim应该等于head_size
        assert rotary_dim == head_size
        
        # 计算频率：1 / (base^(i / rotary_dim)), i = 0, 2, 4, ...
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        # 位置索引：0, 1, 2, ..., max_position_embeddings-1
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        # 计算频率矩阵：[max_position, rotary_dim/2]
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        # 计算cos和sin：[max_position, rotary_dim/2]
        cos = freqs.cos()
        sin = freqs.sin()
        # 拼接cos和sin，并添加一个维度：[max_position, 1, rotary_dim]
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        # 注册为buffer（不作为模型参数，但会随模型保存/加载）
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """前向传播，对query和key应用RoPE
        
        Args:
            positions: 位置索引张量，形状为 [N]
            query: query张量，形状为 [N, num_heads, head_size]
            key: key张量，形状为 [N, num_kv_heads, head_size]
            
        Returns:
            (旋转后的query, 旋转后的key)
        """
        # 根据位置索引获取对应的cos/sin
        cos_sin = self.cos_sin_cache[positions]
        # 分成cos和sin两部分
        cos, sin = cos_sin.chunk(2, dim=-1)
        # 应用旋转
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        return query, key


@lru_cache(1)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
):
    """获取或创建RotaryEmbedding实例（带缓存）
    
    使用lru_cache缓存最近一次创建的实例，避免重复创建。
    """
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb
