"""
Contém a classe base Tokenizer e algumas funções auxiliares comuns. 
A classe base também contém a funcionalidade (comum) salvar/carregar.
"""
import unicodedata

# -----------------------------------------------------------------------------
# algumas funções auxiliares úteis para BasicTokenizer e RegexTokenizer

def get_stats(ids, counts=None):
    """
    Dada uma lista de inteiros, retorna um dicionário de contagens de pares consecutivos
    Exemplo: [1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}
    Opcionalmente permite atualizar um dicionário existente de contagens
    """
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]): # iterar elementos consecutivos
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids, pair, idx):
    """
    Na lista de inteiros (ids), substitua todas as ocorrências consecutivas
    de pair pelo novo token inteiro idx
    Exemplo: ids=[1, 2, 3, 1, 2], pair=(1, 2), idx=4 -> [4, 3, 4]
    """
    newids = []
    i = 0
    while i < len(ids):
        # se não estiver na última posição e o par corresponder, substitua-o
        if ids[i] == pair[0] and i < len(ids) - 1 and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids

# primeiras duas funções auxiliares...
def replace_control_characters(s: str) -> str:
    # substitua caracteres de controle por suas representações de escape
    # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python/19016117#19016117
    # http://www.unicode.org/reports/tr44/#GC_Values_Table
    chars = []
    for ch in s:
        if unicodedata.category(ch)[0] != "C":
            chars.append(ch) # não é um caractere de controle
        else:
            chars.append(f"\\u{ord(ch):04x}") # escape
    return "".join(chars)

def render_token(t: bytes) -> str:
    # renderiza um token de bytes em uma string legível
    s = t.decode('utf-8', errors='replace')
    s = replace_control_characters(s)
    return s

# -----------------------------------------------------------------------------
# a classe base Tokenizer

class Tokenizer:
    """ Classe base Tokenizer, que define a interface comum para todos os tokenizers """

    def __init__(self):
        #   merges: dicionário de pares de índices -> novo índice
        self.merges = {} # (int, int) -> int
        self.pattern = "" # str
        self.special_tokens = {} # str -> int, e.g. {'<|endoftext|>': 100257}
        self.vocab = self._build_vocab() # int -> bytes

    def train(self, text, vocab_size, verbose=False):
        # Treina o tokenizer em um texto, com um vocab_size alvo
        raise NotImplementedError

    def encode(self, text):
        # Tokenizer pode codificar uma string em uma lista de inteiros
        raise NotImplementedError

    def decode(self, ids):
        # Tokenizer pode decodificar uma lista de inteiros em uma string
        raise NotImplementedError

    def _build_vocab(self):
        # Constrói o vocabulário a partir dos merges e tokens especiais
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab

    def save(self, file_prefix):
        """
        Salva o modelo e o vocabulário em dois arquivos separados
        O modelo contém a versão, o padrão, os tokens especiais e os merges
        O vocabulário contém os tokens e seus índices

        """
        # salva o modelo
        model_file = file_prefix + ".model"
        with open(model_file, 'w') as f:
            # escreve a versão, o padrão e as mesclagens, isso é tudo o que é necessário
            f.write("minbpe v1\n")
            f.write(f"{self.pattern}\n")
            # escreve os tokens especiais, primeiro o número deles e depois cada um
            f.write(f"{len(self.special_tokens)}\n")
            for special, idx in self.special_tokens.items():
                f.write(f"{special} {idx}\n")
            # o dicionário de mesclagens
            for idx1, idx2 in self.merges:
                f.write(f"{idx1} {idx2}\n")
        # escreve o vocabulario
        vocab_file = file_prefix + ".vocab"
        inverted_merges = {idx: pair for pair, idx in self.merges.items()}
        with open(vocab_file, "w", encoding="utf-8") as f:
            for idx, token in self.vocab.items():
                # nota: os tokens são bytes, mas alguns deles não são válidos utf-8
                # e podem ser substituídos por �. Portanto, decodificamos com
                # errors='replace' para que possamos renderizar o token de volta
                # porque a decodificação dessa maneira é uma operação de perda!
                # (não podemos reverter para o token original)
                s = render_token(token)
                # se este índice é um merge, renderize-o como um merge
                if idx in inverted_merges:
                    # se este token tiver filhos, renderize-o bem como um merge
                    idx0, idx1 = inverted_merges[idx]
                    s0 = render_token(self.vocab[idx0])
                    s1 = render_token(self.vocab[idx1])
                    f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
                else:
                    # caso contrário, apenas renderize o token
                    # (não há necessidade de renderizar os tokens especiais)
                    f.write(f"[{s}] {idx}\n")

    def load(self, model_file):
        """ Inverso de save() mas apenas para o arquivo de modelo"""
        assert model_file.endswith(".model")
        # carrega o modelo
        merges = {}
        special_tokens = {}
        idx = 256
        with open(model_file, 'r', encoding="utf-8") as f:
            # leia a versão
            version = f.readline().strip()
            assert version == "minbpe v1"
            # leia o padrão
            self.pattern = f.readline().strip()
            # leia os tokens especiais
            num_special = int(f.readline().strip())
            for _ in range(num_special):
                special, special_idx = f.readline().strip().split()
                special_tokens[special] = int(special_idx)
            # leia os merges
            for line in f:
                idx1, idx2 = map(int, line.split())
                merges[(idx1, idx2)] = idx
                idx += 1
        self.merges = merges
        self.special_tokens = special_tokens
        self.vocab = self._build_vocab()