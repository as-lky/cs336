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
        
class DDPBucketed(nn.Module):
    def __init__(self, module: nn.Module, bucket_size_mb: float = 25.0):
        super().__init__()
        self.module = module
        self.world_size = dist.get_world_size()
        self.bucket_size_mb = bucket_size_mb
        
        # 分桶结构
        self._buckets = []
        self._bucket_states = {}
        self._bucket_handles = []
        
        # 1. 同步初始参数
        self._sync_initial_parameters()
        
        # 2. 创建参数分桶
        self._create_buckets()
        
        # 3. 注册梯度钩子
        self._register_gradient_hooks()

    def _sync_initial_parameters(self):
        with torch.no_grad():
            for p in self.module.parameters():
                dist.broadcast(p, src=0)

    def _create_buckets(self):
        current_bucket = []
        current_size = 0
        
        # 反向遍历参数(因为梯度计算顺序是反向的)
        for param in reversed(list(self.module.parameters())):
            if not param.requires_grad:
                continue
                
            param_size = param.numel() * param.element_size()
            
           # if current_bucket and (current_size + param_size) > self.bucket_size_mb * 1024 * 1024:
            if len(current_bucket) == 1:
                self._buckets.append(current_bucket)
                current_bucket = []
                current_size = 0
                
            current_bucket.append(param)
            current_size += param_size
        
        if current_bucket:
            self._buckets.append(current_bucket)
        
        # 初始化每个桶的状态跟踪器
        for bucket_idx, bucket in enumerate(self._buckets):
            self._bucket_states[bucket_idx] = {
                'ready_count': 0,
                'total_params': len(bucket),
                'grads_ready': [False] * len(bucket),
                'flat_grads': None
            }

    def _register_gradient_hooks(self):
        for bucket_idx, bucket in enumerate(self._buckets):
            for param_idx, param in enumerate(bucket):
                # 为每个参数注册post-accumulate钩子
                param.register_post_accumulate_grad_hook(
                    self._make_grad_hook(bucket_idx, param_idx))
    
    def _make_grad_hook(self, bucket_idx: int, param_idx: int):
        def hook(param):
            # 1. 就地平均梯度
            param.grad.div_(self.world_size)
            handle = dist.all_reduce(
                param.grad,
                op=dist.ReduceOp.SUM,
                async_op=True
            )
            self._bucket_handles.append(handle)

            # # 2. 更新桶状态
            # bucket_state = self._bucket_states[bucket_idx]
            # bucket_state['grads_ready'][param_idx] = True
            # bucket_state['ready_count'] += 1
            
            # # 3. 如果整个桶就绪，触发通信
            # if bucket_state['ready_count'] == bucket_state['total_params']:
            #     # 准备扁平化的梯度张量
            #     grads = []
            #     for p in self._buckets[bucket_idx]:
            #         grads.append(p.grad.flatten())
            #     flat_grads = torch.cat(grads)
                
            #     # 异步all-reduce
            #     handle = dist.all_reduce(
            #         flat_grads,
            #         op=dist.ReduceOp.SUM,
            #         async_op=True
            #     )
            #     self._bucket_handles.append(handle)
        
            return None  # 必须返回None
    
        return hook

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        """等待所有异步通信完成并重置桶状态"""
        # 等待所有通信完成
        for handle in self._bucket_handles:
            handle.wait()
        self._bucket_handles.clear()
        
        for bucket_idx in self._bucket_states:
            self._bucket_states[bucket_idx]['ready_count'] = 0
            self._bucket_states[bucket_idx]['grads_ready'] = [
                False] * self._bucket_states[bucket_idx]['total_params']