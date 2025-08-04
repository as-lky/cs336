import torch
import torch.distributed as dist
from torch import nn
from typing import Any

class DDP(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        print(f"WORLD_SIZE : {self.world_size}")
        # Broadcast initial parameters from rank 0 to all other ranks
        self._broadcast_parameters()
        
        # Store handles for gradient synchronization
        self._gradient_handles = []
        
        # Register post accumulate grad hooks for overlapping
        self._register_post_accumulate_grad_hooks()

    def _broadcast_parameters(self):
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)

    def _register_post_accumulate_grad_hooks(self):
        def hook(param):
            # 直接修改梯度张量本身
            param.grad.div_(self.world_size)  # 就地(in-place)平均梯度
            
            # 启动异步all-reduce
            handle = dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=True)
            self._gradient_handles.append(handle)
            
            # 必须返回None
            return None
        
        for param in self.module.parameters():
            if param.requires_grad:
                # Using the post-accumulate grad hook
                param.register_post_accumulate_grad_hook(hook)

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        # 等待所有异步梯度归约完成
        for handle in self._gradient_handles:
            handle.wait()
        self._gradient_handles.clear()