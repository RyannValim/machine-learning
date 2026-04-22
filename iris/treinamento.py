import pickle
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

class Treinar:
    def __init__(self, df_iris_normalizado):
        self.df_iris_normalizado = df_iris_normalizado
    
    def calcular_elbow(self):
        distorcoes = []
        K = range(1, 101)
        for i in K:
            treinador_iris = KMeans(n_clusters=i, random_state=42).fit(self.df_iris_normalizado)
            distorcoes.append(
                sum(np.min(cdist(self.df_iris_normalizado, treinador_iris.cluster_centers_, 'euclidean'),
                           axis=1) / self.df_iris_normalizado.shape[0])
            )
        
        distancias = []
        for i in range(len(distorcoes)):
            x = K[i]
            y = distorcoes[i]
        
            x0 = K[0]
            y0 = distorcoes[0]
            xn = K[-1]
            yn = distorcoes[-1]
            
            distancias.append(abs(((
                (yn - y0)*x) - (xn - x0)*y) + (xn * y0) - (yn * x0)
            ) / sqrt((yn - y0)**2 + (xn - x0)**2))
            
        plt.plot(K, distorcoes)
        plt.savefig('../plotagens/iris/elbow_curve_iris.png')
        plt.close()
        
        k_otimo = K[distancias.index(np.max(distancias))]
        return k_otimo
    
    def treinar(self, k_otimo):
        treinador_iris = KMeans(n_clusters=k_otimo, random_state=42).fit(self.df_iris_normalizado)
        return treinador_iris
        
    def salvar(self, treinador_iris):
        pickle.dump(treinador_iris, open('../modelos/iris/treinador_iris.pkl', 'wb'))