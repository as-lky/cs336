from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import cross_entropy, clip_gradient
from cs336_basics.data import get_batch
from tqdm import tqdm

import torch
import numpy as np
import timeit 

Model_settings = {'small' : {'vocab_size':10000, 'context_length':16, 'd_model':768, 'd_ff':3072, 'num_layers':12, 'num_heads':12, 'rope_theta':10000},
                  'large' : {'vocab_size':10000, 'context_length':16, 'd_model':1280, 'd_ff':5120, 'num_layers':36, 'num_heads':20, 'rope_theta':10000},
                 # '2.7B' : {'vocab_size':10000, 'context_length':16, 'd_model':2560, 'd_ff':10240, 'num_layers':32, 'num_heads':32, 'rope_theta':10000},}
               }
device = 'cuda'

model = BasicsTransformerLM(**(Model_settings['small'])).to(device)
dataset = np.random.randint(0, 10000, 16 * 1024 * 256, dtype=np.int32)
optimizer = AdamW(model.parameters())

# for epoch in tqdm(range(1000)):
# #   for j in tqdm(range(1024 * 256)):
#    for j in range(1024 * 256):
#       data_now, targets_now = get_batch(dataset, batch_size=16, context_length=16, device=device)
#       optimizer.zero_grad()
#       logits_now = model(data_now)
#       loss = cross_entropy(logits_now, targets_now)
#       print(loss.item())
#       loss.backward()
#       optimizer.step()

def sta(lis):
   sum = 0
   for _ in lis:
      sum += _
   mea = sum / len(lis)
   sum = 0
   for _ in lis:
      sum += (_ - mea) ** 2
   return ( sum / len(lis) ) ** 0.5

def benchmark_forward():
   sum = 0.0
   sum2 = 0.0
   lis1 = []
   lis2 = []
   for j in tqdm(range(2)):
      data_now, targets_now = get_batch(dataset, batch_size=4, context_length=16, device=device)
      optimizer.zero_grad()
      logits_now = model(data_now)
      loss = cross_entropy(logits_now, targets_now)
      loss.backward()
      optimizer.step()

   for j in tqdm(range(10)):
      data_now, targets_now = get_batch(dataset, batch_size=4, context_length=16, device=device)
      optimizer.zero_grad()
      torch.cuda.synchronize()
      start = timeit.default_timer()
      logits_now = model(data_now)
      torch.cuda.synchronize()
      end = timeit.default_timer()
      lis1.append(end - start)
      sum += end - start
      loss = cross_entropy(logits_now, targets_now)
#      print(loss.item())
#      clip_gradient(model.parameters(), )
      torch.cuda.synchronize()
      start = timeit.default_timer()
      loss.backward()
      torch.cuda.synchronize()
      sum2 += timeit.default_timer() - start
      lis2.append(timeit.default_timer() - start)
      optimizer.step()
      
   print(sum / 10, sta(lis1))
   print(sum2 / 10, sta(lis2))
   print(lis1)
   print(lis2)

benchmark_forward()