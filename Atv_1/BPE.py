import unicodedata
from collections import defaultdict

class BPE_Tokenizer:
    def __init__(self):
        self.vocab = {}
        self.merges = {}  

    def get_stats(ids):
        counts = {}
        for pair in zip(ids, ids[1:]): 
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def merge(ids, pair, idx):
    
        newids = []
        i = 0
        while i < len(ids):
            
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                newids.append(idx)
                i += 2
            else:
                newids.append(ids[i])
                i += 1
        return newids

    def replace_control_characters(s: str) -> str:
        # troca caracteres de controle por sua representação unicode
        chars = []
        for ch in s:
            if unicodedata.category(ch)[0] != "C":
                chars.append(ch) 
            else:
                chars.append(f"\\u{ord(ch):04x}") 
        return "".join(chars)

    def render_token(t: bytes) -> str:
        #
        s = t.decode('utf-8', errors='replace')
        s = replace_control_characters(s)
        return s

