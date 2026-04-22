import numpy as np

class Descrever():
    def __init__(self, normalizador_iris, codificador_ohe_iris, treinador_iris):
        self.treinador_iris = treinador_iris
        self.normalizador_iris = normalizador_iris
        self.codificador_ohe_iris = codificador_ohe_iris
    
    def descrever(self, index_cluster):
        centroide = self.treinador_iris.cluster_centers_[index_cluster[0]]
        n_ohe = len(self.codificador_ohe_iris.get_feature_names_out())
        
        parte_numerica = centroide[:-n_ohe].reshape(1, -1)
        valores_reais = self.normalizador_iris.inverse_transform(parte_numerica)
        colunas = self.normalizador_iris.feature_names_in_
        
        ohe_parte = centroide[-n_ohe:]
        especie_index = np.argmax(ohe_parte)
        especie = self.codificador_ohe_iris.get_feature_names_out()[especie_index].split('_', 1)[1]
        
        print(f'Esta flor pertence ao cluster: {index_cluster[0]}.')
        print(f'Neste cluster, a espécie dominante é a: {especie}.')
        print(f'Com as seguintes características:')
        for col, val in zip(colunas, valores_reais[0]):
            print(f'  {col}: {val:.2f}')