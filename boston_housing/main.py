import pandas as pd
from preparacao import Preparar
from normalizacao import Normalizar
from treinamento import Treinar

if __name__ == '__main__':
    # carregamento do dataset
    df_housing = pd.read_csv('../datasets/boston_housing/HousingData.csv')
    
    # preparação dos dados e atribuindo valores aos campos NA
    preparador = Preparar()
    df_housing_prep = preparador.preparar(df_housing)
    
    # normalização dos dados
    normalizador = Normalizar()
    df_housing_norm = normalizador.normalizar(df_housing_prep)
    normalizador.salvar(df_housing_norm, normalizador)
    
    # treinamento
    treinador = Treinar()
    df_housing_treinado = treinador.treinar(df_housing_norm)
    treinador.salvar(df_housing_treinado)
    print(df_housing_treinado)