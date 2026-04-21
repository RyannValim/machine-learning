import pandas as pd
import numpy as np

class Descrever():
    def __init__(self, minmax_norm, codificador_ohe, cluster_treinador):
        self.cluster_treinador = cluster_treinador
        self.minmax_norm = minmax_norm
        self.codificador_ohe = codificador_ohe
    
    def descrever(self, index_cluster, dado_norm: pd.DataFrame):
        centroide = self.cluster_treinador.cluster_centers_[index_cluster[0]]
        n_ohe = len(self.codificador_ohe.get_feature_names_out())
        
        parte_numerica = centroide[:-n_ohe].reshape(1, -1)
        valores_reais = self.minmax_norm.inverse_transform(parte_numerica)
        colunas = self.minmax_norm.feature_names_in_
        
        ohe_parte = centroide[-n_ohe:]
        especie_index = np.argmax(ohe_parte)
        especie = self.codificador_ohe.get_feature_names_out()[especie_index].split('_', 1)[1]
        
        print(f'Esta flor pertence ao cluster: {index_cluster[0]}.')
        print(f'Neste cluster, a espécie dominante é a: {especie}.')
        print(f'Com as seguintes características:')
        for col, val in zip(colunas, valores_reais[0]):
            print(f'  {col}: {val:.2f}')