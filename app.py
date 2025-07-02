import streamlit as st
from app import model_handler, dashboard
from app.game_launcher import run_game

def to_dense(X):
    return X.todense()

#Configuração do layout do app
st.set_page_config(
    page_title="Análise de Sentimentos — Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Carregar modelo e vetorizar
modelo, tfidf = model_handler.carregar_modelo()

# Menu de navegação
aba = st.sidebar.radio(
    "📑 Menu de Navegação",
    [
        "🏠 Home", 
        "📥 Upload CSV", 
        "📊 Dashboard", 
        "🧠 Predição Manual", 
        "⚙️ Gerenciar Modelo", 
        "📈 Comparar Modelos",
        "🎮 Jogar Rompe Blocos", 
        "ℹ️ Sobre"
    ]
)

# -------------------------------------------
# 🏠 HOME
# -------------------------------------------
if aba == "🏠 Home":
    st.title("🧠 Análise de Sentimentos — Reviews de Produtos")

    st.markdown("""
    Bem-vindo ao seu painel de **Análise de Sentimentos**!  
    Este aplicativo permite que você classifique automaticamente avaliações de produtos como **positivas** ou **negativas**, com visualizações completas e interativas.

    ### 🔧 Funcionalidades disponíveis:
    - 📥 **Upload de CSV**: Envie um arquivo com avaliações e classifique automaticamente.
    - 📊 **Dashboard**: Veja gráficos, contadores e nuvens de palavras com os resultados.
    - 🧠 **Predição Manual**: Digite uma review e veja a classificação instantânea.
    - ⚙️ **Re-treinamento do Modelo**: Atualize o modelo com novos dados (se desejar).
    - 💾 **Salvar no MongoDB**: Armazene os dados analisados em um banco na nuvem.

    ### 📁 Requisitos do CSV:
    - Deve conter uma coluna chamada **`review`** ou **`reviews.text`**.

    ---

    👇 Comece navegando no menu lateral à esquerda.
    """)

    st.info("🔐 Seus dados são processados localmente. Nenhum texto é enviado para terceiros.")
    st.markdown("---")
    st.caption("Desenvolvido por Felipe Barbosa • Projeto de Machine Learning e NLP com Streamlit")


# Upload CSV

elif aba == "📥 Upload CSV":
    dashboard.upload_csv(modelo, tfidf)

# Dashboard

elif aba == "📊 Dashboard":
    dashboard.exibir_dashboard()


# Predição Manual

elif aba == "🧠 Predição Manual":
    dashboard.predicao_manual(modelo, tfidf)


# Gerenciar Modelo

elif aba == "⚙️ Gerenciar Modelo":
    dashboard.gerenciar_modelo()

#  Comparar Modelos

elif aba == "📈 Comparar Modelos":
    dashboard.comparar_modelos()
    

#  Jogar Rompe Blocos
elif aba == "🎮 Jogar Rompe Blocos":
    st.title("🎮 Rompe Blocos — Controle por Mão")
    st.markdown("""
    Experimente o mini-jogo **Rompe Blocos**, controlado por gestos de mão via MediaPipe.
    Uma janela será aberta fora do navegador.
    """)
    if st.button("🚀 Abrir Jogo"):
        run_game()
        st.success("✅ O jogo foi aberto em uma nova janela!")


# Sobre

elif aba == "ℹ️ Sobre":
    dashboard.sobre()
