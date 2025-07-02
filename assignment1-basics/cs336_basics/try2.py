import tiktoken

# 加载 GPT-2 的编码器
enc = tiktoken.get_encoding("gpt2")

# 获取词表（字典形式：{token_bytes: token_id}）
vocab = enc._mergeable_ranks  # 核心词表（BPE merges 的结果）

# 查看前10个token和对应的ID
# for token, idx in list(vocab.items())[7190:7210]:
#    print(f"Token ID: {idx}, Token: {token}")
#print(vocab)
print(list(vocab.items())[41615])
#print(enc.encode('\u0142'))
print(enc.encode('\u00c4'))
print(list('\u0120'.encode()))
# print(enc.encode('\u00c5\u0124'))
# print('\u0142', '\u0142'.encode(), '\u0142'.encode('utf-8'))
# print(enc.decode([254]))
# print(enc.decode([41615]))
# #print(vocab[7200])
# # 查看词表大小
# print(f"词表大小: {len(vocab)}")