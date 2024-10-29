import unicodedata


def get_stats(tokens):
    counts = {}
    for pair in zip(tokens, tokens[1:]): 
        counts[pair] = counts.get(pair, 0) + 1
    return counts
    
def merge(tokens, pair, idx):
    
    newids = []
    i = 0
    while i < len(tokens):
        
        if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(tokens[i])
            i += 1
    return newids
    