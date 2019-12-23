import re

def striphtml(text):
    p = re.compile(r'<.*?>')
    return p.sub('',text)

def tokenizer(text):
    text = striphtml(text)
    text = re.sub(r'[^A-Za-z0-9]+', ' ', text) 
    text = re.sub(r'https?:/\/\S+', ' ', text)
    return text.lower().strip().split()