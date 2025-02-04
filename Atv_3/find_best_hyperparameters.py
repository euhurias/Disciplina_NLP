from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

import pandas as pd
import time
import os

def load_data(corpus_path):
    """Carrega o conjunto de dados a partir do arquivo CSV."""
    df = pd.read_csv(corpus_path)
    return df

def split_data(df, test_size=0.2, random_state=14):
    """Divide o conjunto de dados em treino e teste."""
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['class'],
        test_size=test_size, random_state=random_state,
        stratify=df['class']
    )
    return X_train, X_test, y_train, y_test

def vectorize_text(X_train, X_test, ngram_range=(1, 1)):

     # Verifica e trata valores NaN substituindo por uma string vazia
    X_train = X_train.fillna('')
    X_test = X_test.fillna('')

    """Vetoriza o texto usando TfidfVectorizer, aplicando para treino e teste."""
    tfidf = TfidfVectorizer(ngram_range=ngram_range)
    X_train_csr = tfidf.fit_transform(X_train)  # Para treino, usamos fit_transform
    X_test_csr = tfidf.transform(X_test)        # Para teste, usamos apenas transform
    return X_train_csr, X_test_csr, tfidf

def perform_grid_search(X_train_csr, y_train, output_dir="Atv_3/outputs1"):
    """Executa a busca de hiperparâmetros usando GridSearchCV e salva os resultados."""
    
    # Definindo os parâmetros dos modelos
    model_parameters = {
        MultinomialNB(): {
            "alpha": [0.1, 0.2, 0.5, 0.75, 1, 1.5, 2, 5]
        },
        LogisticRegression(max_iter=2000): {  # Ajuste para garantir convergência
            "solver": ["liblinear", "lbfgs"],
            "penalty": ["l1", "l2"],
            "tol": [1e-4, 1],
            "C": [1e-4, 1]
        }
    }

    # Criar pasta de saída, se não existir
    os.makedirs(output_dir, exist_ok=True)

    # Loop para cada modelo e seus parâmetros
    for model, parameters in model_parameters.items():
        start_time = time.time()
        model_name = type(model).__name__

        param_grid = []

        # Corrigir para garantir o uso adequado dos solvers e penalizações
        if model_name == 'LogisticRegression':
            
            for penalty in parameters['penalty']:
                if penalty == 'l1':
                    param_grid.append({'penalty': [penalty], 'solver': ['liblinear']})  # Usar 'liblinear' para 'l1'
                else:
                    param_grid.append({'penalty': [penalty], 'solver': ['lbfgs']})
         # Para o MultinomialNB, só utilizamos alpha
        elif model_name == 'MultinomialNB':
            param_grid.append(parameters)

        # Executa o GridSearchCV em paralelo
        for grid in param_grid:
            clf = GridSearchCV(model, parameters, n_jobs=-1)
            clf.fit(X_train_csr, y_train)

            # Calcula o tempo decorrido
            elapsed_time = time.time() - start_time
            print(f"Elapsed time of {model_name}: {elapsed_time:.2f}s")

            # Salva os resultados do GridSearch em um arquivo CSV
            result_df = pd.DataFrame(clf.cv_results_)
            result_df.to_csv(f"{output_dir}/{model_name}_grid_results.csv", index=False)

            print(f"Best parameters for {model_name}: {clf.best_params_}")
            print(result_df[['params', 'mean_test_score', 'rank_test_score']])

if __name__ == "__main__":
    # Caminho do corpus
    #corpus_path = "/home/hurias/Documentos/Disciplina_NLP/Atv_3/data/review_polarity.csv"
    corpus_path = "/home/hurias/Documentos/Disciplina_NLP/Atv_3/data/SyskillWebert.csv"

    # Carregar e dividir os dados
    df = load_data(corpus_path)
    X_train, X_test, y_train, y_test = split_data(df)

    # Vetorizar os textos
    X_train_csr, X_test_csr, tfidf = vectorize_text(X_train, X_test, ngram_range=(1, 1))


    # Executar a Grid Search
    perform_grid_search(X_train_csr, y_train)
