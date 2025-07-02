import joblib
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from ml.preprocessamento import classificar_sentimento


def carregar_modelo():
    """
    Carrega o modelo salvo com base no nome escolhido na sessão do Streamlit.
    Se não encontrar, carrega um modelo padrão.
    """
    nome_modelo = st.session_state.get("modelo_escolhido", "LogisticRegression")
    nome_base = nome_modelo.replace("modelo_", "")
    arquivos = sorted(glob.glob(f"ml/modelo_{nome_base}*.pkl"), reverse=True)

    if not arquivos:
        st.warning(f"Modelo {nome_modelo} não encontrado. Usando modelo padrão.")
        try:
            return joblib.load("ml/modelo.pkl")
        except FileNotFoundError:
            st.error("Modelo padrão não encontrado.")
            return None, None

    modelo_carregado = joblib.load(arquivos[0])
    return modelo_carregado if isinstance(modelo_carregado, tuple) else (modelo_carregado, None)


def salvar_modelo(modelo, tfidf):
    """
    Salva o modelo e o vetor de TF-IDF juntos em um arquivo.
    """
    joblib.dump((modelo, tfidf), "ml/modelo.pkl")


def treinar_modelo():
    """
    Treina um modelo de regressão logística com base nos dados da Amazon,
    salvando o modelo e exibindo a matriz de confusão.
    """
    # Carregamento e pré-processamento dos dados
    df = pd.read_csv("data/Amazon_Products.csv")
    df = df[["reviews.text", "reviews.rating"]].dropna()
    df.rename(columns={"reviews.text": "review", "reviews.rating": "rating"}, inplace=True)

    # Classificação de sentimentos
    df["sentimento"] = df["rating"].apply(classificar_sentimento)
    df = df[df["sentimento"] != "neutro"]

    # Vetorização e divisão dos dados
    X = df["review"]
    y = df["sentimento"]
    tfidf = TfidfVectorizer(stop_words="english")
    X_tfidf = tfidf.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

    # Treinamento do modelo
    modelo = LogisticRegression(class_weight="balanced", max_iter=1000)
    modelo.fit(X_train, y_train)

    # Salvar modelo treinado
    salvar_modelo(modelo, tfidf)

    # Avaliação do modelo
    y_pred = modelo.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    # Plot da matriz de confusão
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Matriz de Confusão")
    plt.xlabel("Predito")
    plt.ylabel("Verdadeiro")
    plt.tight_layout()
    plt.savefig("ml/matriz_confusao.png")
    plt.close()

    return classification_report(y_test, y_pred)
