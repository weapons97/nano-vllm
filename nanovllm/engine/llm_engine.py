import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:
    """LLM推理引擎，负责管理模型、调度请求和执行推理"""

    def __init__(self, model, **kwargs):
        # 从kwargs中提取属于Config的配置项
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        # 创建配置对象
        config = Config(model, **config_kwargs)
        # 设置序列的块大小（KV缓存块大小）
        Sequence.block_size = config.kvcache_block_size
        
        # 用于管理张量并行的工作进程和同步事件
        self.ps = []
        self.events = []
        # 使用spawn方式创建子进程（更安全，会重新导入模块）
        ctx = mp.get_context("spawn")
        # 为每个张量并行worker创建进程（rank 1到N-1）
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()  # 用于进程间同步的事件
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        
        # 主进程中的ModelRunner（rank 0）
        self.model_runner = ModelRunner(config, 0, self.events)
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        # 设置配置的eos token
        config.eos = self.tokenizer.eos_token_id
        # 创建调度器
        self.scheduler = Scheduler(config)
        # 注册退出时的清理函数
        atexit.register(self.exit)

    def exit(self):
        """清理资源，停止所有工作进程"""
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        """添加一个推理请求
        
        Args:
            prompt: 输入提示，可以是字符串或token id列表
            sampling_params: 采样参数
        """
        # 如果是字符串，先分词
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        # 创建序列对象
        seq = Sequence(prompt, sampling_params)
        # 添加到调度器
        self.scheduler.add(seq)

    def step(self):
        """执行一步推理（调度-执行-后处理）
        
        Returns:
            outputs: 已完成的序列列表 [(seq_id, completion_token_ids), ...]
            num_tokens: 预填充阶段的token数（正数）或解码阶段的序列数（负数）
        """
        # 调度器选择要执行的序列
        seqs, is_prefill = self.scheduler.schedule()
        # 计算token数量：预填充阶段是所有token数，解码阶段是序列数的负数
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
        # 执行模型推理
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        # 后处理：更新序列状态，检查是否完成
        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        # 收集已完成的序列输出
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs, num_tokens

    def is_finished(self):
        """检查是否所有请求都已完成"""
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        """批量生成文本
        
        Args:
            prompts: 输入提示列表，每个可以是字符串或token id列表
            sampling_params: 采样参数（可以统一指定或每个prompt单独指定）
            use_tqdm: 是否显示进度条
            
        Returns:
            生成结果列表，每个元素包含text和token_ids
        """
        # 创建进度条
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        
        # 如果采样参数是单个对象，复制给所有prompt
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        
        # 添加所有请求
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        
        # 存储输出结果，key为seq_id
        outputs = {}
        # 吞吐量统计
        prefill_throughput = decode_throughput = 0.
        
        # 循环执行推理直到所有请求完成
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            # 根据阶段更新吞吐量统计
            if num_tokens > 0:
                # 预填充阶段：计算token/s
                prefill_throughput = num_tokens / (perf_counter() - t)
            else:
                # 解码阶段：计算序列数/s
                decode_throughput = -num_tokens / (perf_counter() - t)
            # 更新进度条显示
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            # 收集完成的输出
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)
        
        pbar.close()
        
        # 按seq_id排序输出
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        # 解码token ids为文本
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs
