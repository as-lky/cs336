import torch

a = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
a = a.to('cuda')
b = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
b = b.to('cuda')
print((a + b).device)