import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    """nano-vllm的配置类
    
    包含模型路径、推理参数、内存管理等所有配置项。
    使用slots=True可以减少内存占用。
    """
    model: str  # 模型路径或HuggingFace模型名
    max_num_batched_tokens: int = 16384  # 单次调度最大处理的token数（用于chunked prefill）
    max_num_seqs: int = 512  # 最大同时处理的序列数
    max_model_len: int = 4096  # 模型支持的最大序列长度
    gpu_memory_utilization: float = 0.9  # GPU内存利用率（0-1之间）
    tensor_parallel_size: int = 1  # 张量并行度
    enforce_eager: bool = False  # 是否强制使用eager模式（不使用CUDA Graph）
    hf_config: AutoConfig | None = None  # HuggingFace配置对象
    eos: int = -1  # 结束符token id（会在LLMEngine中设置）
    kvcache_block_size: int = 256  # KV缓存块大小
    num_kvcache_blocks: int = -1  # KV缓存块数量（会在ModelRunner中计算）

    def __post_init__(self):
        """初始化后的验证和配置"""
        assert os.path.isdir(self.model)  # 确保模型路径存在
        assert self.kvcache_block_size % 256 == 0  # 块大小必须是256的倍数
        assert 1 <= self.tensor_parallel_size <= 8  # 张量并行度在1-8之间
        # 加载HuggingFace配置
        self.hf_config = AutoConfig.from_pretrained(self.model)
        # 使用配置中的max_position_embeddings和实际配置的较小值
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
