import logging
import os

# Criar pasta de logs se não existir
os.makedirs("logs", exist_ok=True)

# Configurar logger
logging.basicConfig(
    filename="logs/atividades.log",
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def log_info(mensagem):
    logging.info(mensagem)

def log_erro(mensagem):
    logging.error(mensagem)
