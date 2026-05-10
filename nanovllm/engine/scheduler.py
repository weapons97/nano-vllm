from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:
    """调度器，负责管理序列的状态转换（等待->运行->完成）"""

    def __init__(self, config: Config):
        # 最大同时处理的序列数
        self.max_num_seqs = config.max_num_seqs
        # 单次调度最大处理的token数（用于chunked prefill）
        self.max_num_batched_tokens = config.max_num_batched_tokens
        # 结束符token id
        self.eos = config.eos
        # KV缓存块大小
        self.block_size = config.kvcache_block_size
        # 块管理器
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        # 等待调度的队列（新请求或抢占的请求）
        self.waiting: deque[Sequence] = deque()
        # 正在运行的队列（正在解码的序列）
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        """检查是否所有请求都已完成"""
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        """添加新序列到等待队列"""
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        """调度序列执行，返回要执行的序列列表和是否是预填充阶段
        
        Returns:
            scheduled_seqs: 被调度的序列列表
            is_prefill: 是否是预填充阶段（True表示prefill，False表示decode）
        """
        scheduled_seqs = []
        num_batched_tokens = 0

        # ========== 预填充阶段 ==========
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens
            # 如果已无剩余token配额，停止调度
            if remaining == 0:
                break
            
            # 计算需要多少token
            if not seq.block_table:
                # 新序列：检查是否可以分配块，计算已缓存的块数
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    # 内存不足，停止调度
                    break
                # 实际需要处理的token数（减去已缓存的）
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                # 被抢占后重新调度：只需要处理未缓存的token
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            
            # chunked prefill：如果剩余token不足，且已有调度的序列，则停止
            # （只允许第一个序列进行chunked prefill）
            if remaining < num_tokens and scheduled_seqs:
                break
            
            # 分配KV缓存块（如果是新序列）
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
            
            # 设置本次要处理的token数
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens
            
            # 如果完成了所有token的处理，移到running队列
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            
            scheduled_seqs.append(seq)

        # 如果有预填充的序列，返回
        if scheduled_seqs:
            return scheduled_seqs, True

        # ========== 解码阶段 ==========
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            
            # 检查是否可以追加新token（可能需要新块）
            while not self.block_manager.can_append(seq):
                # 内存不足，需要抢占一个序列
                if self.running:
                    # 抢占最后一个序列（FIFO策略）
                    self.preempt(self.running.pop())
                else:
                    # 没有其他序列可抢占，只能抢占自己
                    self.preempt(seq)
                    break
            else:
                # 可以追加，设置解码参数
                seq.num_scheduled_tokens = 1  # 解码阶段每次处理1个token
                seq.is_prefill = False
                # 如果需要，分配新的块
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        
        # 解码阶段必须至少有一个序列
        assert scheduled_seqs
        # 将序列按原顺序放回running队列头部
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        """抢占序列，将其从running移到waiting，释放KV缓存
        
        Args:
            seq: 要抢占的序列
        """
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True  # 下次调度时需要重新prefill
        self.block_manager.deallocate(seq)  # 释放KV缓存
        self.waiting.appendleft(seq)  # 放到等待队列头部（优先调度）

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        """后处理：更新序列状态，检查是否完成
        
        Args:
            seqs: 刚执行完的序列列表
            token_ids: 模型采样得到的token id列表
            is_prefill: 是否是预填充阶段
        """
        for seq, token_id in zip(seqs, token_ids):
            # 为已生成的token计算哈希（用于prefix caching）
            self.block_manager.hash_blocks(seq)
            # 更新已缓存的token数
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            
            # 预填充阶段：如果还有未处理的token，继续等待下次调度
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue
            
            # 追加生成的token
            seq.append_token(token_id)
            
            # 检查是否结束：遇到eos或达到最大生成长度
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)  # 释放KV缓存
                self.running.remove(seq)  # 从running队列移除
