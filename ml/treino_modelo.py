import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from ml.preprocessamento import classificar_sentimento

# Carregar dataset
df = pd.read_csv('data/Amazon_Products.csv')
df = df[['reviews.text', 'reviews.rating']].dropna()
df = df.rename(columns={'reviews.text': 'review', 'reviews.rating': 'rating'})

# Aplicar classificação de sentimento
df['sentimento'] = df['rating'].apply(classificar_sentimento)
df = df[df['sentimento'] != 'neutro']

# Separar features e labels
X = df['review']
y = df['sentimento']

#  Vetorização TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
X_tfidf = tfidf.fit_transform(X)

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

#  Otimizar hiperparâmetros
param_grid = {'C': [0.1, 1, 10], 'solver': ['liblinear']}
grid = GridSearchCV(LogisticRegression(class_weight='balanced'), param_grid, cv=5, scoring='f1_macro')
grid.fit(X_train, y_train)
modelo = grid.best_estimator_

# Avaliar modelo
y_pred = modelo.predict(X_test)
print("Relatório de Classificação:\n")
print(classification_report(y_test, y_pred))

#  Matriz de confusão
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=modelo.classes_, yticklabels=modelo.classes_)
plt.xlabel("Predito")
plt.ylabel("Verdadeiro")
plt.title("Matriz de Confusão")
plt.tight_layout()
plt.savefig("ml/matriz_confusao.png")
plt.show()

# Salvar modelo 
joblib.dump((modelo, tfidf), 'ml/modelo.pkl')
print("Modelo salvo em ml/modelo.pkl")
