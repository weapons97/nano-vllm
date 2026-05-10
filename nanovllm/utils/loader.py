import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    """默认权重加载函数：直接复制权重"""
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    """从safetensors文件加载模型权重
    
    支持合并的模块（如QKV合并层），通过packed_modules_mapping
    来将检查点中的权重加载到正确的参数位置。
    
    Args:
        model: 要加载权重的模型
        path: 模型目录路径（包含safetensors文件）
    """
    # 获取模型的合并模块映射（如果存在）
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    
    # 遍历目录下所有safetensors文件
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                # 检查是否是合并模块（如q_proj在检查点中，但模型中叫qkv_proj）
                for k in packed_modules_mapping:
                    if k in weight_name:
                        # 找到对应的合并模块名和shard_id
                        v, shard_id = packed_modules_mapping[k]
                        # 将检查点中的模块名替换为模型中的模块名
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        # 使用参数的weight_loader加载特定shard
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    # 不是合并模块，直接加载
                    param = model.get_parameter(weight_name)
                    # 使用参数的weight_loader，如果没有则使用默认加载器
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))
