# import cs336_basics
# import torch
# a = torch.tensor([1, 2, 3])
# b = torch.tensor([2, 3, 4])
# print(a.device, b.device)
# a = a.to('cuda')
# b = b.to('cuda')
# print(a.device)
# c = a + b
# print(c, c.device)
import timeit
a = timeit.default_timer()
print(a)
sum = 0
for i in range(100000000):
    sum += i
b = timeit.default_timer()
print(b - a)