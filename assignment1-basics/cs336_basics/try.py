# # import regex as re
# # #PAT = ' '
# # #PAT2 = 's'
# # #P = '|'.join([PAT, PAT2])
# # PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
# # a = re.findall(PAT, 'iron asd wae sda sd')
# # print(a)
# # # a = re.split(P, "some text that i'll pre-tokenize")
# # # print(a)
# # #dd = {'a': 1, 'v' : 2}
# # #A = 2, 3
# # #print(type(A))
# # # print(list('牛'.encode()), list('a'.encode()))
# # # a = bytes([0xe7, 0x89, 0x9b, 97])
# # # print(a.decode())
# # #rint('牛'.encode() < 'a'.encode())
# # #a = '牛ad@'
# # # #print(list(a))
# # # for i in range(10):
# # #     print(i)
# # #     if i % 2 == 0:
# # #         i += 1
# # #         print(i)

# # #PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    
# # a = 'asd'
# # b = list(a.encode())
# # c = [_.encode() for _ in a]
# # print(type(b[0]))
# # print(c[0] + c[1])
# # print(b == c)

# # with open('cs336_basics/a.txt', 'r') as f:
# #     lines = f.readlines()
# # cnt = 0
# # for line in lines:
# #     cnt += 1
# #     if cnt > 5:
# #         break
# #     line = line.strip()
# #     line = line.split(' ')
# #     print(line[0].encode(), line[1].encode())

# #import regex as re
# #a = 'tesxt'
# #PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
# #print(re.findall(PAT, a))
# # a = [1, 2, 3]
# # b = ['a', 'b', 'c']
# # print(a + b)
# # b[0].encode().decode(errors=)
# import regex as re
# a = "a"
# # print(len(a))
# # for _ in a:
# #     print(_)
# b = 'tesadysakdsad'
# print(re.split(a, b))
# print(len('🙃'))
# print('🙃'.encode())
# print(ord('🙃'))
# print('🙃')
# a = '\u00f0\u0141\u013b\u0125'
# print(a.encode())
# b = bytes([_.encode() for _ in a]).decode()
# print(b)
# import regex as re
# PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
# a = "iron asd wae"
# a = re.findall(PAT, 'iron asd wae sda sd')
# for _ in a:
#     print(list(_.encode()))
# print(bytes([140]))
# a = '\u00d2'
# b = a.encode()
# print(b)
# def w(f):
#     for _ in f:
#         yield _
#         return 1
# a = []
# with open("./cs336_basics/test.txt") as f:
#     print(type(w(f)))
#     for _id in w(f):
#         print(type(w(f)))
#         a.append(_id)
# print(a)
#print(list('!'.encode()))
#print(ord('¸'))
#print(b'\xf0'.decode(), b'\x9f'.decode())
# import regex as re
# a = re.splititer('( )', 'adssada awd awd adw sa d')
# for _ in a:
#     print(_, end='')
# a = [1, 2, 3]
# b = ['a', 'v']
# a.append(b)
# print(a)
import torch
# a = [[1, 2, 3], [2, 4, 6]]
# a = torch.tensor(a)
# b = torch.LongTensor(torch.tensor([[0, 1, 0], [1, 1, 1], [0, 0, 0]]))
# print(a[b])
# b = torch.tensor([0] * 34)
# a = torch.ones(34, device=None, dtype=None)
# print(a)
# print(b)
# class w(torch.nn.Module):
#     def __init__(self, x):
#         super().__init__()
#         self.xx = x
#     def forward(self, w):
#         return self.xx * w
# A = w(10)
# print(A(10))
# def ww(a):
#     a = a * 2
#     return a
# def ss(a):
#     a = ['1', '2']
#     return a

# A = torch.tensor([1, 2, 3])
# B = ss(A)
# print(B)
# print(A)

# GG = 10
# def func():
#     print(GG)
# GG = 5
# # func()
# a = torch.tensor([[1, 2, 3], [3, 2, 1]])
# b = torch.tensor([[0], [1]])
# print(a[b])


# from collections.abc import Callable, Iterable
# from typing import Optional
# import torch
# import math

# class SGD(torch.optim.Optimizer):
#     def __init__(self, params, lr=1e-3):
#         if lr < 0:
#             raise ValueError(f"Invalid learning rate: {lr}")
#         defaults = {"lr": lr}
#         super().__init__(params, defaults)
#     def step(self, closure: Optional[Callable] = None):
#         loss = None if closure is None else closure()
#         for group in self.param_groups:
#             lr = group["lr"] # Get the learning rate.
#             for p in group["params"]:
#                 if p.grad is None:
#                     continue
#                 state = self.state[p] # Get state associated with p.
#                 t = state.get("t", 0) # Get iteration number from the state, or initial value.
#                 grad = p.grad.data # Get the gradient of loss with respect to p.
#                 p.data -= lr / math.sqrt(t + 1) * grad # Update weight tensor in-place.
#                 state["t"] = t + 1 # Increment iteration number.
#         return loss

# weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
# opt = SGD([weights], lr=1000)
# for t in range(100):
#     opt.zero_grad() # Reset the gradients for all learnable parameters.
#     loss = (weights**2).mean() # Compute a scalar loss value.
#     print(loss.cpu().item())
#     loss.backward() # Run backward pass, which computes gradients.
#     opt.step() # Run optimizer step.
import torch
A = torch.tensor([1, 2, 3])
print(A.grad)