import pandas as pd
from preparacao import Preparar
from normalizacao import Normalizar
from treinamento import Treinar
from predicao import Predizer
from descricao import Descrever

if __name__ == '__main__':
    # carregamento do dataset
    df_housing = pd.read_csv('../datasets/boston_housing/HousingData.csv')
    
    # preparação dos dados e atribuindo valores aos campos NA
    preparador = Preparar()
    df_housing_prep = preparador.preparar(df_housing)
    preparador.salvar(df_housing_prep)
    
    # normalização dos dados
    normalizador = Normalizar()
    bh_normalizado = normalizador.normalizar(df_housing_prep)
    normalizador_bh = normalizador.normalizador_bh
    normalizador.salvar(bh_normalizado)
    
    # treinamento
    treinador = Treinar(bh_normalizado)
    treinador_bh = treinador.treinar()
    treinador.salvar(treinador_bh)
    
    # inferência
    novo_dado = [0.23124, 12.5, 7.18, 0.0, 0.590, 5.638, 100.0, 5.9878, 5, 312, 14.5, 386.6, 29.9, 16.5]
    
    # predição
    preditor = Predizer(normalizador_bh, treinador_bh, novo_dado)
    dado_predito = preditor.predizer()
    
    # descricao
    descritor = Descrever(normalizador_bh, treinador_bh, dado_predito)
    cluster_descrito = descritor.descrever()