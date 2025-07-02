import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
from sklearn.exceptions import ConvergenceWarning
import pandas as pd


from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

from ml.preprocessamento import classificar_sentimento

# Ignorar avisos de convergência
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Função nomeada para evitar erro de serialização com lambda
def to_dense(x):
    return x.toarray()

# Carregar e preparar os dados
df = pd.read_csv('data/Amazon_Products.csv')
df = df[['reviews.text', 'reviews.rating']].dropna()
df = df.rename(columns={'reviews.text': 'review', 'reviews.rating': 'rating'})
df['sentimento'] = df['rating'].apply(classificar_sentimento)
df = df[df['sentimento'] != 'neutro']

X = df['review']
y = df['sentimento']

# Codificar rótulos como inteiros para XGBoost
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Dividir os dados
X_train, X_test, y_train_raw, y_test_raw = train_test_split(X, y, test_size=0.2, random_state=42)
y_train = label_encoder.transform(y_train_raw)
y_test = label_encoder.transform(y_test_raw)

# Definir os modelos a testar
modelos = {
    'LogisticRegression': LogisticRegression(max_iter=10000, class_weight='balanced'),
    'LinearSVC': LinearSVC(class_weight='balanced'),
    'XGBoost': XGBClassifier(eval_metric='logloss')
}

# Resultados
resultados = {}

for nome, clf in modelos.items():
    print(f"\nTreinando modelo: {nome}")

    steps = [
        ('tfidf', TfidfVectorizer(
            stop_words='english',
            max_df=0.9,
            min_df=5,
            ngram_range=(1, 2),
            sublinear_tf=True
        ))
    ]

    # Se for XGBoost, transformar para denso e usar y codificado
    if nome == 'XGBoost':
        steps.append(('to_dense', FunctionTransformer(to_dense, accept_sparse=True)))
        y_train_model = y_train
        y_test_model = y_test
    else:
        y_train_model = y_train_raw
        y_test_model = y_test_raw

    steps.append(('clf', clf))
    pipeline = Pipeline(steps)

    # Grid Search apenas para LogisticRegression
    if nome == 'LogisticRegression':
        param_grid = {
            'clf__C': [0.1, 1, 10],
            'clf__penalty': ['l2'],
            'clf__solver': ['liblinear', 'saga']
        }
        grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1_macro')
        grid.fit(X_train, y_train_model)
        melhor_modelo = grid.best_estimator_
    else:
        pipeline.fit(X_train, y_train_model)
        melhor_modelo = pipeline

    y_pred = melhor_modelo.predict(X_test)

    # Se for XGBoost, decodificar predições
    if nome == 'XGBoost':
        y_pred_decoded = label_encoder.inverse_transform(y_pred)
        y_test_decoded = label_encoder.inverse_transform(y_test_model)
    else:
        y_pred_decoded = y_pred
        y_test_decoded = y_test_model

    print(classification_report(y_test_decoded, y_pred_decoded))

    cm = confusion_matrix(y_test_decoded, y_pred_decoded, labels=label_encoder.classes_)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title(f"Matriz de Confusão - {nome}")
    plt.xlabel("Predito")
    plt.ylabel("Verdadeiro")
    plt.tight_layout()

    nome_arquivo_img = f"ml/matriz_confusao_{nome}.png"
    plt.savefig(nome_arquivo_img)
    plt.close()

    # Salvar modelo final
    nome_modelo = f"ml/modelo_{nome}_{datetime.now().strftime('%Y%m%d_%H%M')}.pkl"
    joblib.dump(melhor_modelo, nome_modelo)
    print(f"Modelo salvo em {nome_modelo}")

    resultados[nome] = {
        'modelo': melhor_modelo,
        'arquivo': nome_modelo,
        'matriz_confusao': nome_arquivo_img
    }

print("\nTodos os modelos treinados com sucesso.")
