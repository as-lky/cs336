from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import EncodingDetector
from fastwarc.warc import ArchiveIterator, WarcRecordType
from itertools import combinations
import fasttext
import re
import nltk
import mmh3
import os
import random

def extract_text_from_html_bytes(html_bytes):
    a = EncodingDetector()
    a.update(html_bytes)
    html_str = html_bytes.decode(encoding=a.encoding())
    return extract_plain_text(html_str)

def identify_language(text):
    model = fasttext.load_model('./cs336_data/lid.176.bin')
    text = text.replace('\n', '')
    predict = model.predict(text)    
    return predict[0][0].replace('__label__', ''), predict[1][0]

def classify_toxic_speech(text):
    model = fasttext.load_model('./jigsaw_fasttext_bigrams_hatespeech_final.bin')
    predict = model.predict(text)
    return predict[0][0].replace('__label__', ''), predict[1][0]

def classify_nsfw(text):
    model = fasttext.load_model('./jigsaw_fasttext_bigrams_nsfw_final.bin')
    predict = model.predict(text)
    return predict[0][0].replace('__label__', ''), predict[1][0]

def mask_emails(text):
    PAT = r'[\w\.]+@[\w\.]+'
    a = re.findall(PAT, text)
    text = re.sub(PAT, r"|||EMAIL_ADDRESS|||", text)
    return text, len(a)

def mask_phone_numbers(text):
    # '+1 123-456-7890'
    # '123-456-7890'
    # '(123) 456-7890'
    # '1234567890'
    PAT = r'(\+1 )?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}' # .在字符集中不需要转义
    a = re.findall(PAT, text)
    text = re.sub(PAT, r"|||PHONE_NUMBER|||", text)
    return text, len(a)

def mask_ips(text):
    PAT = r'(2[0-5][0-5]|1\d\d|[1-9]\d|\d)\.(2[0-5][0-5]|1\d\d|[1-9]\d|\d)\.(2[0-5][0-5]|1\d\d|[1-9]\d|\d)\.(2[0-5][0-5]|1\d\d|[1-9]\d|\d)'
    a = re.findall(PAT, text)
    text = re.sub(PAT, r"|||IP_ADDRESS|||", text)
    return text, len(a)

def gopher_quality_filter(text):
    words = nltk.word_tokenize(text)
    if len(words) < 50 or len(words) > 100000:
        return False
    sum = 0.0
    num = 0.0
    for word in words:
        sum += len(word)
        if re.match(r'[a-zA-Z]', word) is not None:
            num += 1
    sum /= len(words)
    if sum < 3.0 or sum > 10.0:
        return False
    if num / len(words) < 0.8:
        return False
    
    sentences = nltk.sent_tokenize(text)
    num = 0.0
    for sentence in sentences:
        if sentence[-3:] == '...':
            num += 1
    if num / len(sentences) > 0.3:
        return False
    return True

def exact_line_deduplication(input_files, output_directory):
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)
    hash_set = {}
    for file in input_files:
        with open(file, 'r') as f:
            for line in f:
                hash_value = mmh3.hash64(line, seed=123)
                if hash_value in hash_set:
                    hash_set[hash_value] += 1
                else:
                    hash_set[hash_value] = 1

    for file in input_files:
        with open(file, 'r') as f:
            with open(os.path.join(output_directory, os.path.basename(file)), 'w') as ff:
                for line in f:
                    hash_value = mmh3.hash64(line, seed=123)
                    if hash_set[hash_value] >= 2:
                        continue
                    ff.write(line)

def jaccard_similarity(documentA, documentB, ngrams):
    if len(documentA) < ngrams or len(documentB) < ngrams:
        ngrams = 1
    dict_both = {}
    dict_any = {}
    for i in range(len(documentA) - ngrams + 1):
        ss = ' '.join([_ for _ in documentA[i:i+ngrams]])
        hash_ss = mmh3.hash64(ss)
        dict_any[hash_ss] = 1
        dict_both[hash_ss] = 1
    for i in range(len(documentB) - ngrams + 1):
        ss = ' '.join([_ for _ in documentB[i:i+ngrams]])
        hash_ss = mmh3.hash64(ss)
        dict_any[hash_ss] = 1
        dict_both[hash_ss] = dict_both.get(hash_ss, 0) | 2
    sum_any = len(dict_any)
    sum_both = 0.0
    for val in dict_both.values():
        if val == 3:
            sum_both += 1
    return sum_both / sum_any


def minhash_deduplication(
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
):
    assert num_hashes % num_bands == 0

    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)

    random.seed(123)
    seed_set = [random.randint(1, 1230213) for _ in range(num_hashes)]

    file_end_document_id = []

    documents = []
    for file in input_files:
        with open(file, 'r') as f:
            lines = f.readlines()
        document_now = []
        for line in lines:
            if line.strip() == '':
                if len(document_now) == 0:
                    continue
                documents.append(document_now.copy())
                document_now = []
            else:
                document_now += nltk.word_tokenize(line.strip())
        if len(document_now) != 0:
            documents.append(document_now.copy())
        file_end_document_id.append(len(documents))

    hash_2_document_cluster = [{} for _ in range(num_bands)]
    document_candidate_cluster = [set() for _ in range(num_bands)]
    band_size = num_hashes // num_bands

    for document_num in range(len(documents)):
        document = documents[document_num]
        if len(document) < ngrams:
            ngrams_up = ngrams
            ngrams = 1
        else:
            ngrams_up = ngrams
        
        hhash_now = []
        for i in range(num_hashes):
            MIN = 2e30
            for kk in range(len(document) - ngrams + 1):
                ss = ' '.join([_ for _ in document[kk:kk+ngrams]])
                hash_now = mmh3.hash(ss, seed_set[i])
                MIN = min(MIN, hash_now)
            hhash_now.append(MIN)
        for i in range(num_bands):
            w = tuple(hhash_now[i*band_size:(i+1)*band_size])
            if w in hash_2_document_cluster[i]:
                if hash_2_document_cluster[i][w] not in document_candidate_cluster[i]:
                    document_candidate_cluster[i].add(hash_2_document_cluster[i][w])
                document_candidate_cluster[i].add(document_num)
            else:
                hash_2_document_cluster[i][w] = document_num
        ngrams = ngrams_up

    edges = [[] for _ in range(len(documents))]
    used = [0 for _ in range(len(documents))]
    keep = [1 for _ in range(len(documents))]

    for i in range(num_bands):
        tmp = list(document_candidate_cluster[i])
        for aa in range(len(tmp)):
            for bb in range(aa + 1, len(tmp)):
                js = jaccard_similarity(documents[tmp[aa]], documents[tmp[bb]], ngrams)
                if js > jaccard_threshold:
                    edges[tmp[aa]].append(tmp[bb])
                    edges[tmp[bb]].append(tmp[aa])
 
    for i in range(len(documents)):
        if used[i] == 1:
            continue
        cluster = []
        queue = [i]
        used[i] = 1
        pl = 0
        while pl < len(queue):
            now = queue[pl]    
            cluster.append(now)
            for to in edges[now]:
                if used[to] == 1:
                    continue
                used[to] = 1 
                queue.append(to)
            pl += 1
        rr = random.randint(0, len(cluster) - 1)
        for cc in cluster:
            keep[cc] = 0
        keep[cluster[rr]] = 1
    
    print(keep)

    document_now_id = 0
    for file in input_files:
        with open(file, 'r') as f:
            lines = f.readlines()
        document_now = []
        WRITE_LINES = []
        for line in lines:
            if line.strip() == '':
                if keep[document_now_id] == 1:
                    for lineline in document_now:
                        WRITE_LINES.append((lineline))
                    WRITE_LINES.append('\n')
                document_now_id += 1
                document_now = []
                continue
            else:
                document_now.append(line.strip())
        
        if len(document_now) != 0:
            if keep[document_now_id] == 1:
                for lineline in document_now:
                    WRITE_LINES.append((lineline))
            document_now_id += 1
        if len(WRITE_LINES) != 0:
            with open(os.path.join(output_directory, os.path.basename(file)), 'w') as ff:
                print(os.path.join(output_directory, os.path.basename(file)))
                for lines in WRITE_LINES:
                    ff.write(lines)
                    ff.write('\n')