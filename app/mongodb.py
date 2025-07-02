import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv() 

def salvar_no_mongo(dados):
    """Salva os dados (lista de dicionários) no MongoDB."""
    uri = os.getenv('MONGO_URI')
    if not uri:
        raise ValueError("MONGO_URI não encontrado no arquivo .env")

    client = MongoClient(uri)
    db = client['analise_sentimentos']
    colecao = db['resultados']

    colecao.insert_many(dados)
