
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import string
# Importa as funções do script preprocessamento.py
from preprocessamento import load_data, split_data, vectorize_text

# Função de pré-processamento de texto
def preprocess_text(text):
    # Verifica se o valor é NaN ou não é uma string e substitui por uma string vazia
    if pd.isna(text) or not isinstance(text, str):
        text = ""  # Substitui valores NaN ou não-strings por uma string vazia
    
    # Converte o texto para minúsculas e remove pontuação
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    
    # Divida o texto em tokens (palavras)
    tokens = text.split()
    
    return tokens

# Função de divisão de dados
def split_data(df, test_size=0.2, random_state=14):
    # Aplica a função de pré-processamento de texto na coluna 'text'
    X = df['text'].apply(lambda text: ' '.join(preprocess_text(text)))  # Aplica a função de pré-processamento
    y = df['class']
    
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

# Função principal de treinamento e avaliação
def train_and_evaluate_model(corpus_path, model_name, output_dir="outputs", perform_grid_search_flag=False, results_file=None):
    print(f"\nTreinando e avaliando o modelo com a base: {corpus_path}")
    
    # Carregar e dividir os dados
    df = load_data(corpus_path)
    X_train, X_test, y_train, y_test = split_data(df)

    # Vetorizar os textos
    X_train_csr, X_test_csr, tfidf = vectorize_text(X_train, X_test)

    # Cria o modelo de acordo com a escolha do nome do modelo
    if model_name == "MultinomialNB":
        model = MultinomialNB()
    elif model_name == "LogisticRegression":
        model = LogisticRegression(max_iter=2000)

    if perform_grid_search_flag:
        # Verifica se o diretório de saída existe
        if not os.path.exists(output_dir):
            print(f"Diretório {output_dir} não encontrado. Criando diretório...")
            os.makedirs(output_dir)

        # Se nenhum arquivo de resultados for especificado, exibe os disponíveis
        if not results_file:
            results_files = [f for f in os.listdir(output_dir) if f.endswith("_grid_results.csv")]
            if results_files:
                # Exibe os arquivos encontrados
                print("\nArquivos de resultados encontrados:")
                for file in results_files:
                    print(file)

                # Permite ao usuário escolher um arquivo
                selected_file = input("Escolha o arquivo para usar (digite o nome completo): ")

                if selected_file in results_files:
                    results_file = selected_file
                else:
                    print("Arquivo não encontrado, usando o modelo padrão.")
                    return
            else:
                print(f"Nenhum arquivo de resultados encontrado no diretório {output_dir}.")
                return

        # Processamento do arquivo de resultados escolhido
        print(f"\nProcessando o arquivo: {results_file}")
        
        # Carrega os resultados do CSV
        result_df = pd.read_csv(os.path.join(output_dir, results_file))

        # Exibe os 5 melhores resultados
        print(result_df[['params', 'mean_test_score', 'rank_test_score']].sort_values(by='rank_test_score').head())

        # Reajusta o modelo com os melhores parâmetros no conjunto de teste
        best_params = result_df.loc[result_df['rank_test_score'] == 1, 'params'].values[0]
        if model_name == "MultinomialNB":
            model = MultinomialNB(**eval(best_params))
        elif model_name == "LogisticRegression":
            model = LogisticRegression(max_iter=2000, **eval(best_params))

        # Treina o modelo com os melhores parâmetros
        model.fit(X_train_csr, y_train)
        y_pred = model.predict(X_test_csr)

        # Exibe o relatório de classificação
        print("\nRelatório de Classificação:")
        print(classification_report(y_test, y_pred))

        # Calcula e exibe a matriz de confusão
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=model.classes_, yticklabels=model.classes_)
        plt.title(f'Matriz de Confusão - {model_name}')
        plt.xlabel('Predito')
        plt.ylabel('Verdadeiro')
        plt.show()

    else:
        # Treina o modelo normalmente se Grid Search não for executado
        model.fit(X_train_csr, y_train)

        # Realiza a previsão
        X_test_csr = tfidf.transform(X_test)
        y_pred = model.predict(X_test_csr)

        # Exibe o relatório de classificação
        print("\nRelatório de Classificação:")
        print(classification_report(y_test, y_pred))

        # Calcula e exibe a matriz de confusão
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=model.classes_, yticklabels=model.classes_)
        plt.title(f'Matriz de Confusão - {model_name}')
        plt.xlabel('Predito')
        plt.ylabel('Verdadeiro')
        plt.show()
