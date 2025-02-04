import pandas as pd
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

# Carregar dados
def load_data(file_path, file_type='csv'):
    if file_type == 'csv':
        return pd.read_csv(file_path)
    elif file_type == 'json':
        return pd.read_json(file_path)
    else:
        raise ValueError("Formato de arquivo não suportado!")

# Pré-processar texto
def preprocess_text(text, language='english'):
    stop_words = stopwords.words(language)
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    
    # Aplicar stemming e lematização
    tokens = [stemmer.stem(word) for word in tokens]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return ' '.join(tokens)

# Dividir dados
def split_data(df, test_size=0.2, random_state=14):
    X = df['text'].apply(preprocess_text)  # Aplica a função de pré-processamento
    y = df['class']
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

# Vetorizar texto
def vectorize_text(X_train, X_test, ngram_range=(1, 3), stop_words='english'):
    tfidf = TfidfVectorizer(ngram_range=ngram_range, stop_words=stop_words)
    X_train_csr = tfidf.fit_transform(X_train)
    X_test_csr = tfidf.transform(X_test)   
    return X_train_csr, X_test_csr, tfidf
