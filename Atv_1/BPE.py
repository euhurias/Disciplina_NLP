import unicodedata
from collections import defaultdict

from base import get_stats, merge



class BPE_Tokenizer:

    def __init__(self, num_merges: int):

        self.byte_value_offset = 256

        self.num_merges = num_merges
        self.vocab_size = num_merges + self.byte_value_offset

        self.tokens = []
        self.processed_tokens = []

        self.merges = {}
        self.vocab = {} 

    
    def train(self, text):
        
        # input do texto
        print(f"Tamanho do texto em caracteres: {len(text)}")
        self.tokens = text.encode("utf-8") # raw bytes
        self.tokens = list(map(int, self.tokens)) # list of integers in range 0..255

        print(f"Tamanho do texto em tokens: {len(self.tokens)}")

        self.processed_tokens = self.tokens
       # int -> bytes
        for i in range(self.num_merges):
            
            stats = get_stats(self.processed_tokens)
            # resgata o par com maior frequência
            pair = max(stats, key=stats.get)
            
            idx = self.byte_value_offset + i
            # merge
            ids = merge(self.processed_tokens, pair, idx)
            self.processed_tokens = ids
            # save the merge
            self.merges[pair] = idx

        print(f"Tamanho da de tokens após BPE: {len(ids)}")

        taxa = len(self.tokens)/len(self.processed_tokens)
        print(f"Taxa de compressão: {taxa:.2f}X")
        self.vocab = {idx: bytes([idx]) for idx in range(self.byte_value_offset)}
        for(pair, idx) in self.merges.items():
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]
            
        
    def encode(self, text):
        print("Conteúdo de self.merges:", self.merges)
        
        tokens = self.tokenize_text(text)
        
        while True:
            stats = get_stats(tokens)
            
            # Verifique se stats está vazio e trate o caso
            if not stats:
                print("Aviso: 'stats' está vazio, nenhum par encontrado.")
                break
            
            # Encontre o par com menor valor em stats usando self.merges
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            
            if pair not in self.merges:
                break

            # Atualize tokens ao realizar a fusão do par encontrado
            tokens = merge(tokens, pair)

        # Retorne os tokens processados ou um token desconhecido se estiver vazio
        return tokens if tokens else ["<UNK>"]


    def save(self, file_prefix, new_tokens=True, verbose=True):
        file = file_prefix + "_tokens" + str(self.num_merges) + ".txt"

        inverted_merges = {idx: pair for pair, idx in self.merges.items()}

        vocab_list = list(self.vocab.items())

        with open(file, "w", encoding="utf-8") as f:
            for idx, token in vocab_list:
                s = token.decode("utf-8", errors="replace")

                output = False
                # verifica se o indice está no dicionário de merges
                if idx in inverted_merges:
                    idx0, idx1 = inverted_merges[idx]

                    s0 = self.vocab.get(idx0, b"").decode("utf-8", errors="replace")
                    s1 = self.vocab.get(idx1, b"").decode("utf-8", errors="replace")

                    output = f"{idx}:[{s0}][{s1}] -> {s}"
                elif not new_tokens:
                    output = f"[{s}] -> {idx}"  

                # Imprime o output e escreve no arquivo, se existir algo a ser exibido
                if output:
                    if verbose:
                        print(output)
                    f.write(output + "\n")
                else:
                    if verbose:
                        print(f"DEBUG - Sem output gerado para idx {idx}, token {s}")
                        

