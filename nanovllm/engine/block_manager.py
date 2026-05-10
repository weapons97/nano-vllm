from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:
    """表示一个KV缓存块，用于存储序列的token和对应的哈希值"""

    def __init__(self, block_id):
        # 块的唯一标识符
        self.block_id = block_id
        # 引用计数，表示该块被多少个序列共享
        self.ref_count = 0
        # 块的哈希值，用于快速查找和去重
        self.hash = -1
        # 该块中存储的token id列表
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        """更新块的哈希值和token id列表"""
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        """重置块状态，设置引用计数为1（新分配时）"""
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:
    """管理KV缓存块的分配、释放和哈希查找"""

    def __init__(self, num_blocks: int, block_size: int):
        # 每个块可以存储的token数量
        self.block_size = block_size
        # 所有可用的块列表，预先创建好
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        # 哈希值到块id的映射，用于快速查找已缓存的块
        self.hash_to_block_id: dict[int, int] = dict()
        # 空闲块id队列，用于快速分配
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        # 正在使用的块id集合
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        """计算token序列的哈希值，支持前缀哈希（用于增量计算）
        
        Args:
            token_ids: 要计算哈希的token id列表
            prefix: 前一个块的哈希值，用于增量计算（-1表示无前缀）
        
        Returns:
            计算得到的64位哈希值
        """
        h = xxhash.xxh64()
        # 如果有前缀哈希，先更新前缀
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        # 将token ids转换为字节数组并更新哈希
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self) -> int:
        """从空闲队列中分配一个块
        
        Returns:
            分配得到的块id
        """
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        # 确保要分配的块没有被引用
        assert block.ref_count == 0
        # 如果该块有旧的哈希值，从哈希表中移除
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]
        # 重置块状态
        block.reset()
        # 加入到使用集合中
        self.used_block_ids.add(block_id)
        return block_id

    def _deallocate_block(self, block_id: int):
        """将块释放回空闲队列
        
        Args:
            block_id: 要释放的块id
        """
        # 确保块的引用计数为0
        assert self.blocks[block_id].ref_count == 0
        # 从使用集合中移除
        self.used_block_ids.remove(block_id)
        # 加入空闲队列
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> int:
        """检查是否可以为序列分配块，并返回可以复用的缓存块数量
        
        Args:
            seq: 要分配块的序列
            
        Returns:
            可以复用的缓存块数量，如果空间不足返回-1
        """
        h = -1
        num_cached_blocks = 0
        num_new_blocks = seq.num_blocks
        # 遍历序列的所有块（除了最后一个，因为最后一个可能不完整）
        for i in range(seq.num_blocks - 1):
            token_ids = seq.block(i)
            # 增量计算哈希值
            h = self.compute_hash(token_ids, h)
            # 查找是否有相同哈希的块
            block_id = self.hash_to_block_id.get(h, -1)
            # 如果没有找到，或者token不匹配（哈希冲突），停止查找
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                break
            num_cached_blocks += 1
            # 如果该块已经被使用，不需要新分配
            if block_id in self.used_block_ids:
                num_new_blocks -= 1
        # 检查是否有足够的空闲块
        if len(self.free_block_ids) < num_new_blocks:
            return -1
        return num_cached_blocks

    def allocate(self, seq: Sequence, num_cached_blocks: int):
        """为序列分配块，包括复用缓存块和分配新块
        
        Args:
            seq: 要分配块的序列
            num_cached_blocks: 可以复用的缓存块数量
        """
        assert not seq.block_table
        h = -1
        # 先处理可以复用的缓存块
        for i in range(num_cached_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id[h]
            block = self.blocks[block_id]
            # 如果该块已被使用，增加引用计数（共享）
            if block_id in self.used_block_ids:
                block.ref_count += 1
            else:
                # 否则设置为1，并从空闲队列移除
                block.ref_count = 1
                self.free_block_ids.remove(block_id)
                self.used_block_ids.add(block_id)
            seq.block_table.append(block_id)
        # 为剩余的块分配新的块
        for i in range(num_cached_blocks, seq.num_blocks):
            seq.block_table.append(self._allocate_block())
        # 记录已缓存的token数量
        seq.num_cached_tokens = num_cached_blocks * self.block_size

    def deallocate(self, seq: Sequence):
        """释放序列占用的所有块
        
        Args:
            seq: 要释放块的序列
        """
        # 逆序释放，避免共享块的问题
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            # 引用计数为0时，真正释放块
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        # 重置序列状态
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        """检查是否可以追加一个新的token（需要新块）
        
        Args:
            seq: 要检查序列
            
        Returns:
            如果有空间可以追加返回True
        """
        # 当序列长度对block_size取模等于1时，说明需要新块
        # 因为第0个token在第一个块，第block_size+1个token需要新块
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        """如果需要，为序列追加新的块
        
        Args:
            seq: 要追加块的序列
        """
        # 当序列长度对block_size取模等于1时，分配新块
        if len(seq) % self.block_size == 1:
            seq.block_table.append(self._allocate_block())

    def hash_blocks(self, seq: Sequence):
        """为序列新生成的token块计算哈希值并存储到哈希表
        
        Args:
            seq: 要哈希的序列
        """
        # 计算需要哈希的块范围（从已缓存的块之后开始）
        start = seq.num_cached_tokens // self.block_size
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
        if start == end: return
        # 获取前一个块的哈希值作为前缀
        h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
        # 为新的块计算哈希
        for i in range(start, end):
            block = self.blocks[seq.block_table[i]]
            token_ids = seq.block(i)
            # 增量计算哈希
            h = self.compute_hash(token_ids, h)
            block.update(h, token_ids)
            # 存储到哈希表
            self.hash_to_block_id[h] = block.block_id
