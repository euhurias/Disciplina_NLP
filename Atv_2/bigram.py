from collections import defaultdict, Counter

class BigramModel:
    def __init__(self):
        self.bigram_counts = defaultdict(Counter)
        self.unigram_counts = Counter()
        self.total_bigrams = 0
    
    def train(self, tokens):
        # Conta unigramas e bigramas no corpus de treino
        for i in range(len(tokens) - 1):
            self.unigram_counts[tokens[i]] += 1
            self.bigram_counts[tokens[i]][tokens[i + 1]] += 1
            self.total_bigrams += 1

    def bigram_prob(self, word1, word2):
        # Probabilidade condicional P(word2 | word1)
        if self.unigram_counts[word1] == 0:
            return 0  # evita divisão por zero
        return self.bigram_counts[word1][word2] / self.unigram_counts[word1]