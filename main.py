import pandas as pd
import pickle
from predicao import Predizer
from treinamento import Treinar
from normalizacao import Normalizar
from descricao import Descrever

if __name__ == '__main__':
    # carregamento dos dados
    df = pd.read_csv('./datasets/iris.csv', sep=';')
    
    # normalização
    normalizador = Normalizar(df)
    iris_normalizado = normalizador.normalizar()
    minmax_norm = normalizador.minmax_norm
    codificador_ohe = normalizador.codificador_ohe
    normalizador.salvar(iris_normalizado)

    # treinamento do cluster
    treinador = Treinar(iris_normalizado)
    k_otimo = treinador.calcular_elbow()
    cluster_treinador = treinador.treinar(k_otimo)
    treinador.salvar(cluster_treinador)

    # inferência e predição
    nova_flor = [6.4, 2.8, 5.6, 2.1]
    flor_normalizada = Predizer(minmax_norm, codificador_ohe, cluster_treinador)
    df_flor = flor_normalizada.preparar(nova_flor)
    predicao_flor = flor_normalizada.predizer(df_flor)
    
    # descrição do cluster
    descritor = Descrever(minmax_norm, codificador_ohe, cluster_treinador)
    cluster_descrito = descritor.descrever(predicao_flor, df_flor)