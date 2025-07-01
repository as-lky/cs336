# import regex as re
# #PAT = ' '
# #PAT2 = 's'
# #P = '|'.join([PAT, PAT2])
# PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
# a = re.findall(PAT, 'iron asd wae sda sd')
# print(a)
# # a = re.split(P, "some text that i'll pre-tokenize")
# # print(a)
# #dd = {'a': 1, 'v' : 2}
# #A = 2, 3
# #print(type(A))
# # print(list('牛'.encode()), list('a'.encode()))
# # a = bytes([0xe7, 0x89, 0x9b, 97])
# # print(a.decode())
# #rint('牛'.encode() < 'a'.encode())
# #a = '牛ad@'
# # #print(list(a))
# # for i in range(10):
# #     print(i)
# #     if i % 2 == 0:
# #         i += 1
# #         print(i)

# #PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    
# a = 'asd'
# b = list(a.encode())
# c = [_.encode() for _ in a]
# print(type(b[0]))
# print(c[0] + c[1])
# print(b == c)

with open('cs336_basics/a.txt', 'r') as f:
    lines = f.readlines()
cnt = 0
for line in lines:
    cnt += 1
    if cnt > 5:
        break
    line = line.strip()
    line = line.split(' ')
    print(line[0].encode(), line[1].encode())