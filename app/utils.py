import streamlit as st

def validar_csv(df):
    """Valida se o CSV possui a coluna 'review' ou 'reviews.text'."""
    if 'review' in df.columns:
        return True
    elif 'reviews.text' in df.columns:
        df.rename(columns={'reviews.text': 'review'}, inplace=True)
        return True
    else:
        st.error("O arquivo CSV deve conter uma coluna chamada 'review' ou 'reviews.text'.")
        return False

def to_dense(X):
    return X.todense()
