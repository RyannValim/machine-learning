import pickle
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

class Treinador():
    def __init__(self, df):
        self.df = df
        self.treinador_kmeans = KMeans()
    
    def treinar(self):
        k_otimo = self.calc_elbow()
        return KMeans(k_otimo, random_state=42).fit(self.df)
    
    def calc_elbow(self):
        distorcoes = []
        K = range(1, 151)
        for i in K:
            treinador_iris = KMeans(n_clusters=i, random_state=42).fit(self.df)
            distorcoes.append(
                sum(np.min(cdist(self.df, treinador_iris.cluster_centers_, 'euclidean'),
                           axis=1) / self.df.shape[0])
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
        plt.savefig('../plotagens/titanic/elbow_curve_titanic.png')
        plt.close()
         
        return K[distancias.index(np.max(distancias))]
    
    def salvar(self, treinador):
        pickle.dump(treinador, open('../modelos/titanic/treinador_titanic.pkl', 'wb'))