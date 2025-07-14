from openai import OpenAI
from PyPDF2 import PdfReader
from tqdm import tqdm
from pdf2image import convert_from_path
import pytesseract

client = OpenAI(api_key="sk-0bc782de86094dddb4270b261c7d322a", base_url="https://api.deepseek.com")

response = client.chat.completions.create(
    model="deepseek-reasoner",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
#        {"role": "user", "content": f"import torch\na = torch.tensor([1, 2, 3])\nb = torch.tensor([2, 3, 4])\nprint(a.device, b.device)a.to('cuda')\nb.to('cuda')\nprint(a.device)\nc = a + b\nprint(c, c.device)\n为什么a.to('cuda')过后a的device依然不是cuda\n"}, ],
        {"role": "user", "content": f"你好"}, ],

    stream=True
)
for chunk in response:
    content = chunk.choices[0].delta.content 
    print(content, end="", flush=True)
