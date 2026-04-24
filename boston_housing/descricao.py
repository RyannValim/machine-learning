class Descrever():
    def __init__(self, normalizador_bh, treinador_bh, cluster_dado):
        self.normalizador_bh = normalizador_bh
        self.treinador_bh = treinador_bh
        self.cluster_dado = cluster_dado
    
    def descrever(self):
        print(f'Este dado pertence ao cluster: {self.cluster_dado[0]}')
        
        centroides = self.treinador_bh.cluster_centers_[self.cluster_dado[0]]
        valores_reais = self.normalizador_bh.inverse_transform(centroides.reshape(1, -1))
        colunas_originais_bh = self.normalizador_bh.feature_names_in_
        
        print(f'E estes são os valores reais do cluster {self.cluster_dado[0]}:')
        for col, val in zip(colunas_originais_bh, valores_reais[0]):
            print(f'    {col}: {val:.2f}')