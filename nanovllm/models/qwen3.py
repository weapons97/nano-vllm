import torch
from torch import nn
import torch.distributed as dist
from transformers import Qwen3Config

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead


class Qwen3Attention(nn.Module):
    """Qwen3的注意力层"""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_theta: float = 10000,
        rope_scaling: dict | None = None,
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        # attention的缩放因子
        self.scaling = self.head_dim ** -0.5
        self.qkv_bias = qkv_bias

        # QKV投影（合并为一个线性层，支持GQA）
        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
        )
        # 输出投影（行并行）
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )
        # 处理RoPE的scaling参数
        if isinstance(rope_scaling, dict):
            rope_theta = rope_scaling.get("rope_theta", rope_theta)
        # 创建旋转位置编码
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
        )
        # 注意力计算层
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )
        # Qwen3在QKV投影后也有RMSNorm（如果没有bias）
        if not self.qkv_bias:
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """前向传播
        
        Args:
            positions: 位置索引
            hidden_states: 输入隐藏状态 [N, hidden_size]
            
        Returns:
            注意力输出
        """
        # QKV投影
        qkv = self.qkv_proj(hidden_states)
        # 分割成Q、K、V
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        # 重塑形状：[N, num_heads, head_dim]
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        # Qwen3特有的QKV归一化
        if not self.qkv_bias:
            q = self.q_norm(q)
            k = self.k_norm(k)
        # 应用旋转位置编码
        q, k = self.rotary_emb(positions, q, k)
        # 计算注意力
        o = self.attn(q, k, v)
        # 输出投影（展平多头）
        output = self.o_proj(o.flatten(1, -1))
        return output


class Qwen3MLP(nn.Module):
    """Qwen3的MLP（前馈网络）"""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ) -> None:
        super().__init__()
        # gate和up投影合并（SwiGLU需要两个输入）
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
        )
        # down投影
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        # Qwen3使用SiLU激活
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, x):
        # gate和up投影
        gate_up = self.gate_up_proj(x)
        # SwiGLU激活
        x = self.act_fn(gate_up)
        # down投影
        x = self.down_proj(x)
        return x


class Qwen3DecoderLayer(nn.Module):
    """Qwen3的解码器层（Transformer block）"""

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', True),
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        # LayerNorm在attention之前
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # LayerNorm在attention之后、MLP之前
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """前向传播
        
        Args:
            positions: 位置索引
            hidden_states: 输入隐藏状态
            residual: 残差连接（None表示第一层）
            
        Returns:
            (输出隐藏状态, 残差)
        """
        # 输入LayerNorm + 残差处理
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        # 自注意力
        hidden_states = self.self_attn(positions, hidden_states)
        # 后注意力LayerNorm + 残差处理
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        # MLP
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3Model(nn.Module):
    """Qwen3的Transformer模型（不包含LM头）"""

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        # Token embedding（词表并行）
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        # Transformer layers
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        # 最终LayerNorm
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """前向传播
        
        Args:
            input_ids: token id张量
            positions: 位置索引
            
        Returns:
            最后的隐藏状态
        """
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        # 最终LayerNorm
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3ForCausalLM(nn.Module):
    """Qwen3的因果语言模型（完整模型）"""
    
    # 模块映射：将检查点中的模块名映射到实际的模块和shard_id
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: Qwen3Config
    ) -> None:
        super().__init__()
        self.model = Qwen3Model(config)
        # LM头（词表并行）
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        # 如果设置了权重共享，将embedding和LM头的权重绑定
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """前向传播，返回隐藏状态"""
        return self.model(input_ids, positions)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """计算logits（在forward之后调用）"""
        return self.lm_head(hidden_states)
