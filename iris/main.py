import pandas as pd
from predicao import Predizer
from treinamento import Treinar
from normalizacao import Normalizar
from descricao import Descrever

if __name__ == '__main__':
    # carregamento dos dados
    df = pd.read_csv('../datasets/iris/iris.csv', sep=';')

    # normalização
    normalizador = Normalizar(df)
    iris_normalizado = normalizador.normalizar()
    normalizador_iris = normalizador.normalizador_iris
    codificador_ohe_iris = normalizador.codificador_ohe_iris
    normalizador.salvar(iris_normalizado)

    # treinamento do cluster
    treinador = Treinar(iris_normalizado)
    k_otimo = treinador.calcular_elbow()
    treinador_iris = treinador.treinar(k_otimo)
    treinador.salvar(treinador_iris)

    # inferência e predição e flores de outros clusters para teste
    # nova_flor = [7.0, 3.2, 4.7, 1.4] # K0: Iris-versicolor
    # nova_flor = [6.3, 3.3, 6.0, 2.5] # K1: Iris-virginica
    nova_flor = [4.9, 3.0, 1.4, 0.2] # K2: Iris-setosa
    # nova_flor = [5.5, 2.3, 4.0, 1.3] # K3: Iris-versicolor
    # nova_flor = [5.8, 2.7, 5.1, 1.9] # K4: Iris-virginica
    # nova_flor = [5.1, 3.5, 1.4, 0.2] # K5: Iris-setosa
    
    flor_normalizada = Predizer(normalizador_iris, codificador_ohe_iris, treinador_iris)
    df_flor = flor_normalizada.preparar(nova_flor)
    predicao_flor = flor_normalizada.predizer(df_flor)

    # descrição do cluster
    descritor = Descrever(normalizador_iris, codificador_ohe_iris, treinador_iris)
    cluster_descrito = descritor.descrever(predicao_flor)