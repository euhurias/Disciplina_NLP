import unicodedata


def get_stats(tokens):
    print("Tokens recebidos:", tokens)
    stats = {}
    for i in range(len(tokens) - 1):
        pair = (tokens[i], tokens[i + 1])
        stats[pair] = stats.get(pair, 0) + 1
    print("Estatísticas de pares de tokens:", stats) 
    return stats
    
def merge(tokens, pair):
    """Função para mesclar um par específico de tokens."""
    new_token = ''.join(pair)  # Junte o par em um único token
    new_tokens = []
    skip = False

    for i in range(len(tokens) - 1):
        if skip:
            skip = False
            continue

        if tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
            new_tokens.append(new_token)  # Adiciona o token mesclado
            skip = True  # Pula o próximo token
        else:
            new_tokens.append(tokens[i])

    if not skip:  # Adiciona o último token, caso ele não tenha sido mesclado
        new_tokens.append(tokens[-1])

    return new_tokens
    