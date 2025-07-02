import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

from .preprocessamento import classificar_sentimento


def treinar_modelos_avancado(path='data/Amazon_Products.csv', salvar_em='ml/'):
   
    # Carregamento e limpeza
   
    df = pd.read_csv(path)
    df = df[['reviews.text', 'reviews.rating']].dropna()
    df.columns = ['review', 'rating']
    df['sentimento'] = df['rating'].apply(classificar_sentimento)
    df = df[df['sentimento'] != 'neutro']

    X = df['review']
    y = df['sentimento']

   
    # Vetorização
  
    tfidf = TfidfVectorizer(stop_words='english')
    X_tfidf = tfidf.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_tfidf, y, test_size=0.2, random_state=42
    )

  
    # Modelos
    
    modelos = {
        'LogisticRegression': LogisticRegression(class_weight='balanced', max_iter=1000),
        'LinearSVC': LinearSVC(class_weight='balanced'),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    }

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # 4. Treinamento e Salvamento
    
    for nome, modelo in modelos.items():
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)

        # Salva modelo + tfidf
        joblib.dump((modelo, tfidf), os.path.join(salvar_em, f"modelo_{nome}_{timestamp}.pkl"))

        # Matriz de confusão
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Matriz de Confusão - {nome}')
        plt.xlabel('Predito')
        plt.ylabel('Real')
        plt.savefig(os.path.join(salvar_em, f'matriz_confusao_{nome}_{timestamp}.png'))
        plt.close()

        #Relatório no terminal
        print(f"\n{nome}:\n", classification_report(y_test, y_pred))
