import torch
from torch import nn
import torch.nn.functional as F


class SiluAndMul(nn.Module):
    """SiLU激活函数与乘法门控的结合
    
    用于SwiGLU（Swish-Gated Linear Unit）结构。
    输入被分成两半，一半经过SiLU激活后与另一半相乘。
    这是LLaMA、Qwen等模型使用的激活函数变体。
    """

    @torch.compile  # 使用PyTorch编译优化，加速执行
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入张量，最后一维会被分成两半
            
        Returns:
            经过SiLU激活和乘法后的张量
        """
        # 在最后一维将输入分成两半：x和y
        x, y = x.chunk(2, -1)
        # SiLU(x) * y：门控机制
        return F.silu(x) * y
