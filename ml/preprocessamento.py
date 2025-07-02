def classificar_sentimento(nota):
    if nota >= 4:
        return 'positivo'
    elif nota <= 2:
        return 'negativo'
    else:
        return 'neutro'
