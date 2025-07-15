from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import EncodingDetector
from fastwarc.warc import ArchiveIterator, WarcRecordType
import fasttext
import re
import nltk
import mmh3


with open('./cs336_data/tmp.txt', 'r') as f:
    for a in f:
        print(a, end='')
    f.seek(0)
    for b in f:
        print(b)