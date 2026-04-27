import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from preparacao import Preparador
from normalizacao import Normalizador
from treinamento import Treinador
from predicao import Preditor

if __name__ == '__main__':
    # carregando o df
    df = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')

    # preparação do df
    preparador = Preparador(df)
    df_prep = preparador.preparar()
    preparador.salvar(df_prep)
    
    # normalizacao do df
    normalizador = Normalizador(df_prep)
    df_norm = normalizador.normalizar()
    normalizador.salvar(df_norm)
    
    # treinamento do df
    treinador = Treinador(df_norm)
    treinador_titanic = treinador.treinar()
    treinador.salvar(treinador_titanic)
    
    # inferência
    novo_passageiro = [900, 0, None, "Astor, Col. John Jacob", "male", None, 1, 0, "PC 17757", 227.52, "C62", None]
    
    # predicao
    preditor = Preditor(normalizador, treinador_titanic)