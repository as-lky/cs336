from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import EncodingDetector
from fastwarc.warc import ArchiveIterator, WarcRecordType
import fasttext
import re

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