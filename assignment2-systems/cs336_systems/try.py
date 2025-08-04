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
from openai import OpenAI
from PyPDF2 import PdfReader
from tqdm import tqdm

client = OpenAI(api_key="sk-0bc782de86094dddb4270b261c7d322a", base_url="https://api.deepseek.com")
reader = PdfReader("a.pdf")
num_pages = len(reader.pages)

def list_l_r(l, r, id=1):
    text = ""
    for i in range(l-1, r):
        text += reader.pages[i].extract_text()

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": f"你好,我需要你帮助我完成任务。从下列文本当中的word list提取出我指定格式的内容，具体而言为每个list的单词加空格加单词的中文释义，两个list之间留出区分。当前请你提取list {id} 的内容"},
            {"role": "user", "content": "规则如下：\n 1.以以下格式反馈给我： \n abandon v.抛弃 \n acute j.剧烈的 \n 2.把adj简写为j.;adv简写为adv;\n 3.词性缩写与中文释义直接不要加空格，比如n. 卷改为n.卷 \n 4.如果有多个词性的单词,在两个词性的中文释义之间加入分号;而不是空格"},
            {"role": "user", "content": f"内容如下: \n {text}"},
        ],
        stream=False
    )
    content = response.choices[0].delta.content 
    print(content)
    print("\n")

def word_list(id):
    start = (id - 1) * 19 + 13 - 5
    end = start + 40
    list_l_r(start, end, id)

word_list(4)