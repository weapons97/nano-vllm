import torch
from torch import nn


class Sampler(nn.Module):
    """采样器，根据logits和温度采样下一个token
    
    使用Gumbel-Softmax技巧进行采样：
    通过除以温度和使用指数分布噪声来实现采样。
    """

    @torch.compile  # 使用PyTorch编译优化
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        """前向传播，采样token
        
        Args:
            logits: 模型输出的logits，形状为 [batch_size, vocab_size]
            temperatures: 每个序列的温度，形状为 [batch_size]
            
        Returns:
            采样的token id，形状为 [batch_size]
        """
        # 除以温度（温度越低，分布越尖锐；温度越高，分布越平坦）
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        # 计算softmax得到概率
        probs = torch.softmax(logits, dim=-1)
        # Gumbel-Softmax采样：
        # 1. 生成指数分布噪声（模拟Gumbel分布）
        # 2. 除以概率并取argmax（等价于Gumbel-max技巧）
        sample_tokens = probs.div_(
            torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)
        ).argmax(dim=-1)
        return sample_tokens
