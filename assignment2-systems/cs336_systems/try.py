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