from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    """序列状态枚举"""
    WAITING = auto()   # 等待调度（新请求或被抢占）
    RUNNING = auto()   # 正在运行（预填充或解码中）
    FINISHED = auto()  # 已完成（生成结束）


class Sequence:
    """表示一个推理序列，包含token和相关状态"""
    
    # 类变量：KV缓存块大小（由Config设置）
    block_size = 256
    # 序列ID生成器
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        # 唯一序列ID
        self.seq_id = next(Sequence.counter)
        # 序列状态
        self.status = SequenceStatus.WAITING
        # 所有token id（prompt + 生成的completion）
        self.token_ids = copy(token_ids)
        # 最后一个token（用于解码阶段快速访问）
        self.last_token = token_ids[-1]
        # 总token数
        self.num_tokens = len(self.token_ids)
        # prompt的token数
        self.num_prompt_tokens = len(token_ids)
        # 已缓存到KV缓存的token数（用于prefix caching）
        self.num_cached_tokens = 0
        # 本次调度要处理的token数
        self.num_scheduled_tokens = 0
        # 是否是预填充阶段
        self.is_prefill = True
        # 块表：映射逻辑块到物理块id
        self.block_table = []
        # 采样温度
        self.temperature = sampling_params.temperature
        # 最大生成token数
        self.max_tokens = sampling_params.max_tokens
        # 是否忽略eos（持续生成）
        self.ignore_eos = sampling_params.ignore_eos

    def __len__(self):
        """返回总token数"""
        return self.num_tokens

    def __getitem__(self, key):
        """支持索引访问token_ids"""
        return self.token_ids[key]

    @property
    def is_finished(self):
        """检查序列是否已完成"""
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        """返回已生成的completion token数"""
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        """返回prompt部分的token ids"""
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        """返回completion部分的token ids"""
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_blocks(self):
        """计算需要多少个KV缓存块"""
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        """返回最后一个块中的token数量"""
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        """获取第i个块的token ids
        
        Args:
            i: 块索引
            
        Returns:
            该块的token id列表
        """
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int):
        """追加一个新生成的token
        
        Args:
            token_id: 要追加的token id
        """
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def __getstate__(self):
        """序列化状态（用于多进程共享内存传递）
        
        只传递必要的状态，不传递整个token_ids（减少数据传输）
        """
        last_state = self.last_token if not self.is_prefill else self.token_ids
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, 
                self.num_scheduled_tokens, self.block_table, last_state)

    def __setstate__(self, state):
        """反序列化状态"""
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, \
            self.num_scheduled_tokens, self.block_table, last_state = state
        if isinstance(last_state, list):
            # 预填充阶段：恢复完整token_ids
            self.token_ids = last_state
            self.last_token = self.token_ids[-1]
        else:
            # 解码阶段：只恢复last_token
            self.token_ids = []
            self.last_token = last_state
