import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


class ModelRunner:
    """模型运行器，负责模型推理、KV缓存管理和多进程通信"""

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        # KV缓存块大小
        self.block_size = config.kvcache_block_size
        # 是否强制使用eager模式（不使用CUDA Graph）
        self.enforce_eager = config.enforce_eager
        # 张量并行世界大小
        self.world_size = config.tensor_parallel_size
        # 当前进程的rank
        self.rank = rank
        # 用于进程间同步的事件（rank 0是列表，其他是单个事件）
        self.event = event

        # 初始化分布式进程组（使用NCCL后端）
        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        # 设置当前CUDA设备
        torch.cuda.set_device(rank)
        # 保存原来的默认dtype
        default_dtype = torch.get_default_dtype()
        # 设置模型对应的dtype
        torch.set_default_dtype(hf_config.dtype)
        # 设置默认设备为cuda（简化后续代码）
        torch.set_default_device("cuda")
        # 创建模型
        self.model = Qwen3ForCausalLM(hf_config)
        # 加载模型权重
        load_model(self.model, config.model)
        # 创建采样器
        self.sampler = Sampler()
        # 模型预热（触发CUDA内核编译等）
        self.warmup_model()
        # 分配KV缓存
        self.allocate_kv_cache()
        # 如果不强制eager模式，捕获CUDA Graph
        if not self.enforce_eager:
            self.capture_cudagraph()
        # 恢复默认设置
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        # 多进程共享内存通信设置
        if self.world_size > 1:
            if rank == 0:
                # rank 0创建共享内存
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                # 其他rank连接到共享内存
                self.shm = SharedMemory(name="nanovllm")
                # 进入事件循环，等待rank 0的指令
                self.loop()

    def exit(self):
        """退出清理：关闭共享内存，销毁进程组"""
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        # 删除CUDA Graph相关对象
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        """工作进程的事件循环，等待并执行rank 0发来的指令"""
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        """从共享内存读取方法名和参数（工作进程调用）"""
        assert self.world_size > 1 and self.rank > 0
        # 等待rank 0发出信号
        self.event.wait()
        # 读取数据长度
        n = int.from_bytes(self.shm.buf[0:4], "little")
        # 反序列化数据
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        # 清除事件，允许下一次信号
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        """向共享内存写入方法名和参数（rank 0调用）"""
        assert self.world_size > 1 and self.rank == 0
        # 序列化数据
        data = pickle.dumps([method_name, *args])
        n = len(data)
        # 先写入长度，再写入数据
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        # 通知所有工作进程
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        """调用指定方法，如果是rank 0且多进程则先通知工作进程"""
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        # 获取并执行方法
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        """模型预热：运行一次以触发CUDA内核编译，避免后续延迟"""
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        # 构造一个最大的batch进行预热
        seq_len = min(max_num_batched_tokens, max_model_len)
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        """分配KV缓存张量，并计算可用的块数量"""
        config = self.config
        hf_config = config.hf_config
        # 获取当前GPU内存信息
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        # 计算每个块的字节数
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.dtype.itemsize
        # 根据可用内存计算可以分配多少块
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0
        # 分配KV缓存张量 [2(key&value), num_layers, num_blocks, block_size, num_kv_heads, head_dim]
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        # 将KV缓存分配给每个attention层
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        """准备块表张量（用于attention的块索引查找）
        
        Args:
            seqs: 序列列表
            
        Returns:
            填充后的块表张量 [num_seqs, max_num_blocks]
        """
        max_len = max(len(seq.block_table) for seq in seqs)
        # 将每个序列的块表填充到相同长度（-1表示无效块）
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        """为预填充阶段准备输入数据
        
        Args:
            seqs: 要预填充的序列列表
            
        Returns:
            input_ids, positions, 以及设置context
        """
        input_ids = []
        positions = []
        # 预填充阶段的query序列长度累计（用于flash attention的varlen接口）
        cu_seqlens_q = [0]
        # 预填充阶段的key序列长度累计（包含缓存的token）
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        
        for seq in seqs:
            start = seq.num_cached_tokens  # 已缓存的token数
            seqlen_q = seq.num_scheduled_tokens  # 本次要处理的token数
            end = start + seqlen_q
            seqlen_k = end  # key的长度是总长度
            
            # 收集input_ids和positions
            input_ids.extend(seq[start:end])
            positions.extend(range(start, end))
            
            # 更新cumsum长度
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            
            # 如果没有块表，说明是warmup阶段
            if not seq.block_table:
                continue
            
            # 计算slot mapping（token到KV缓存位置的映射）
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
            for i in range(start_block, end_block):
                slot_start = seq.block_table[i] * self.block_size
                if i == start_block:
                    slot_start += start % self.block_size
                if i != end_block - 1:
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
                slot_mapping.extend(range(slot_start, slot_end))
        
        # 如果有前缀缓存（部分token已缓存），需要块表
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:
            block_tables = self.prepare_block_tables(seqs)
        
        # 转换为CUDA张量
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        
        # 设置context（供attention层使用）
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        """为解码阶段准备输入数据（每次只处理一个token）
        
        Args:
            seqs: 要解码的序列列表
            
        Returns:
            input_ids, positions, 以及设置context
        """
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        
        for seq in seqs:
            # 每次只取最后一个token
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            # context长度（用于attention的因果掩码）
            context_lens.append(len(seq))
            # 计算KV缓存的slot位置
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1)
        
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        
        block_tables = self.prepare_block_tables(seqs)
        # 设置context（解码阶段）
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        """准备采样参数
        
        Args:
            seqs: 序列列表
            
        Returns:
            温度张量
        """
        temperatures = [seq.temperature for seq in seqs]
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        """运行模型推理
        
        Args:
            input_ids: 输入的token ids
            positions: 位置编码的位置
            is_prefill: 是否是预填充阶段
            
        Returns:
            计算得到的logits
        """
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            # 预填充阶段、强制eager、或batch size太大时，直接运行
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            # 解码阶段使用CUDA Graph加速
            bs = input_ids.size(0)
            context = get_context()
            # 找到大于等于当前batch size的graph
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            
            # 填充输入数据到graph的固定缓冲区
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            
            # 重放CUDA Graph
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        """执行推理并返回采样的token ids
        
        Args:
            seqs: 要推理的序列列表
            is_prefill: 是否是预填充阶段
            
        Returns:
            采样的token id列表（仅rank 0返回，其他rank返回None）
        """
        # 准备输入
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        # 只有rank 0需要准备采样参数
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        # 运行模型
        logits = self.run_model(input_ids, positions, is_prefill)
        # 采样（只有rank 0执行）
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        # 重置context
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        """捕获CUDA Graph以加速解码阶段
        
        为不同的batch size捕获多个CUDA Graph，覆盖常见的解码batch size。
        """
        config = self.config
        hf_config = config.hf_config
        # 最大batch size
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        
        # 创建固定缓冲区（用于graph的输入输出）
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        
        # 要捕获的batch size列表
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        # 从大到小捕获（让小batch可以共享大batch的pool）
        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            # 设置context
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            # 预热
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])
            # 捕获graph
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])
            # 保存第一个graph的pool供后续共享
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        # 保存graph使用的变量引用
        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
