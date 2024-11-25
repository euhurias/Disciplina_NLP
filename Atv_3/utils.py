import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

def evaluate_model(model, X_test, y_test):
    # Realiza a previsão no conjunto de teste
    y_pred = model.predict(X_test)
    
    # Calcula métricas
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_micro = f1_score(y_test, y_pred, average='micro')
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Exibe as métricas
    print(f"Acurácia: {acc:.4f}")
    print(f"F1 Score (Macro): {f1_macro:.4f}")
    print(f"F1 Score (Micro): {f1_micro:.4f}")
    print("\nMatriz de Confusão:")
    print(conf_matrix)
    
    # Exibe o relatório de classificação detalhado
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred))
