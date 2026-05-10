import torch
from torch import nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization（均方根层归一化）
    
    相比普通LayerNorm，RMSNorm计算更简单，只使用方差归一化，不使用均值中心化。
    在LLaMA、Qwen等模型中广泛使用。
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.eps = eps  # 数值稳定性常数
        # 可学习的缩放参数
        self.weight = nn.Parameter(torch.ones(hidden_size))

    @torch.compile  # 使用PyTorch编译优化
    def rms_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """不带残差连接的RMSNorm前向传播
        
        Args:
            x: 输入张量，形状为 [..., hidden_size]
            
        Returns:
            归一化后的张量
        """
        orig_dtype = x.dtype
        # 转换为float32计算（数值稳定）
        x = x.float()
        # 计算均方值：E[x^2]
        var = x.pow(2).mean(dim=-1, keepdim=True)
        # RMS归一化：x / sqrt(E[x^2] + eps)
        x.mul_(torch.rsqrt(var + self.eps))
        # 应用缩放参数，并转换回原始dtype
        x = x.to(orig_dtype).mul_(self.weight)
        return x

    @torch.compile  # 使用PyTorch编译优化
    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """带残差连接的RMSNorm前向传播
        
        将残差加法和归一化融合在一起，减少内存读写。
        
        Args:
            x: 输入张量
            residual: 残差张量（会被原地修改）
            
        Returns:
            (归一化后的x, 残差residual)
        """
        orig_dtype = x.dtype
        # 将x和residual都转为float32，并原地相加
        x = x.float().add_(residual.float())
        # residual也转为原始dtype（用于后续使用）
        residual = x.to(orig_dtype)
        # 计算均方值
        var = x.pow(2).mean(dim=-1, keepdim=True)
        # RMS归一化
        x.mul_(torch.rsqrt(var + self.eps))
        # 应用缩放参数，并转换回原始dtype
        x = x.to(orig_dtype).mul_(self.weight)
        return x, residual

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """前向传播入口
        
        Args:
            x: 输入张量
            residual: 可选的残差张量
            
        Returns:
            如果residual为None，返回归一化后的x
            否则返回 (归一化后的x, residual)
        """
        if residual is None:
            return self.rms_forward(x)
        else:
            return self.add_rms_forward(x, residual)
