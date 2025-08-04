# import torch
# from torch.multiprocessing import spawn

# # a = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# # a = a.to('cuda')
# # b = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# # b = b.to('cuda')
# # print((a + b).device)

# def co(rank, world_size):   
#     torch.dist.barrier()
#     print(f"rank: {rank}")
#     torch.dist.barrier()
        
# spawn(co, args=(4,) )
# torch.dist.reduce

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def distributed_demo(rank, world_size):
    setup(rank, world_size)
    # data = torch.randint(0, 10, (5,), device='cuda')
    data = torch.randint(0, 10, (5,))
    print(f"rank {rank} data (before all-reduce): {data}   device : {data.device}")
    dist.all_reduce(data, async_op=True)
   # dist.barrier()  # Ensure all processes complete the all-reduce before printing
    print(f"rank {rank} data (after all-reduce): {data}")

if __name__ == "__main__":
    world_size = 4
    mp.spawn(fn=distributed_demo, args=(world_size, ), nprocs=world_size, join=True)