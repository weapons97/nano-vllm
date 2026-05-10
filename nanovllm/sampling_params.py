from dataclasses import dataclass


@dataclass(slots=True)
class SamplingParams:
    """采样参数配置
    
    控制文本生成时的采样行为。
    使用slots=True可以减少内存占用。
    """
    temperature: float = 1.0  # 采样温度（越高越随机，越低越确定）
    max_tokens: int = 64  # 每个序列最大生成的token数
    ignore_eos: bool = False  # 是否忽略结束符（持续生成）

    def __post_init__(self):
        """参数验证"""
        # 温度必须为正（不支持贪心采样，因为会有数值问题）
        assert self.temperature > 1e-10, "greedy sampling is not permitted"
