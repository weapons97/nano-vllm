import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist


def divide(numerator, denominator):
    """整除，确保可以整除"""
    assert numerator % denominator == 0
    return numerator // denominator


class LinearBase(nn.Module):
    """线性层基类，支持张量并行"""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_dim: int | None = None,
    ):
        super().__init__()
        # 张量并行的维度（0表示输出维度切分，1表示输入维度切分）
        self.tp_dim = tp_dim
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        # 创建权重参数
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        # 设置权重加载函数
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ReplicatedLinear(LinearBase):
    """复制线性层（每个GPU都有完整的权重，不做切分）"""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        # 不需要张量并行，所以tp_dim为None
        super().__init__(input_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """直接复制权重（不需要切分）"""
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class ColumnParallelLinear(LinearBase):
    """列并行线性层
    
    输出维度（列）被切分到不同GPU上。
    每个GPU计算部分输出，需要all-reduce来汇总。
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        # 输出维度切分，所以tp_dim=0
        super().__init__(input_size, divide(output_size, tp_size), bias, 0)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """加载权重：只取当前rank对应的部分"""
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class MergedColumnParallelLinear(ColumnParallelLinear):
    """合并的列并行线性层
    
    用于QKV投影等需要多个输出矩阵的情况。
    将这些矩阵在输出维度拼接后一起做列并行。
    """

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
    ):
        self.output_sizes = output_sizes
        # 输出维度是各个输出尺寸之和
        super().__init__(input_size, sum(output_sizes), bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        """加载特定shard的权重
        
        Args:
            loaded_shard_id: shard的索引（表示加载的是哪个输出矩阵）
        """
        param_data = param.data
        # 计算该shard在拼接后的偏移量
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        # 定位到正确的位置
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        # 从完整权重中取出对应shard
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class QKVParallelLinear(ColumnParallelLinear):
    """QKV并行的线性层
    
    将Q、K、V的投影合并到一个线性层中，输出维度为：
    (num_heads + 2 * num_kv_heads) * head_size
    支持GQA（Grouped Query Attention）。
    """

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.head_size = head_size
        # 每个GPU负责的Q头数和KV头数
        self.num_heads = divide(total_num_heads, tp_size)
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)
        # 总输出维度：Q + K + V
        output_size = (total_num_heads + 2 * total_num_kv_heads) * head_size
        super().__init__(hidden_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        """加载Q、K或V的权重
        
        Args:
            loaded_shard_id: "q"、"k"或"v"，表示加载哪个投影的权重
        """
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:  # "v"
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = (self.num_heads + self.num_kv_heads) * self.head_size
        # 定位到正确的位置
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        # 从完整权重中取出对应shard
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class RowParallelLinear(LinearBase):
    """行并行线性层
    
    输入维度（行）被切分到不同GPU上。
    每个GPU计算部分输入的结果，需要all-reduce来汇总。
    通常跟在ColumnParallelLinear后面使用。
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        # 输入维度切分，所以tp_dim=1
        super().__init__(divide(input_size, tp_size), output_size, bias, 1)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """加载权重：只取当前rank对应的部分"""
        param_data = param.data
        if param_data.ndim == 1:  # bias是一维的，直接复制
            param_data.copy_(loaded_weight)
            return
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 注意：RowParallel的输出通常需要all-reduce，但这里假设外层处理
        return F.linear(x, self.weight, self.bias)
